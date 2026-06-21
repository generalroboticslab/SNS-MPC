from .base_policy import Policy
import jax
import jax.numpy as jnp
from Common import *
from jax.scipy.linalg import cho_solve, solve_triangular



def jacfwd_and_value(f_r_g, knots_flat):
    return jax.jacfwd(f_r_g)(knots_flat), f_r_g(knots_flat)

class Spline_Shooter_Policy(Policy):
    """
    Direct shooting policy with sequential quadratic programming, spline parameterization of actions and barrier method for constraints.
    """

    def __init__(self, nn_model, step_cost, residuals, initial_solution, initial_sigma,
                 action_bounds, global_residuals=None, commands_callback=None, spline_order=1, n_knots=40, horizon=19,
                 n_linesearch_steps=128, n_iters=5, qp_alpha_init=10.0, lm_damping=1e-8, _lambda_init=1e-3, rho_init=1.0,
                 verbose=False, seed=0, jit=True):
        """
        Initialize Policy.
        
        Args:
            model: Rollout model taking future actions and returning predicted states.
            step_cost: Step cost function.
            initial_solution: Initial solution for the policy.
            action_bounds: Action bounds for the policy.
            commands_callback: Callback function with the signature (states, actions, commands) that returns the commands to be used in the cost function.
            spline_order: Order of the spline used for action parametrization.
            n_knots: Number of knots in the spline.
            n_linesearch_steps: Number of rollouts to perform in during the forward pass.
            n_iters: Number of SQP iterations to perform.
            temperature: Temperature for the MPPI policy.
            verbose: Whether to print debug information.
            seed: Random seed for reproducibility.
            jit: Whether to use JIT compilation for the policy function.
        """
        
        super().__init__(nn_model, step_cost, jit)

        self.global_residuals = global_residuals
        self.residuals = residuals
        self.lm_damping = lm_damping
        self._lambda_init = _lambda_init
        self.rho_init = rho_init

        self.horizon = horizon
        self.spline_order = spline_order
        self.n_action_knots = n_knots
        self.n_linesearch_steps = n_linesearch_steps
        self.alpha_init = qp_alpha_init
        self.n_iters = n_iters
        self.verbose = verbose
        self.jit = jit
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        self.commands_callback = commands_callback

        self.n_states = nn_model.dynamics.n_states
        self.n_actions = nn_model.dynamics.n_actions

        self.initial_solution = initial_solution
        self.action_bounds = action_bounds

        self.initial_solution = jax.tree.map(lambda x: x.reshape(1, 1), initial_solution)
        self.initial_solution = jax.tree.map(lambda x: jnp.repeat(x, self.horizon-1, axis=0), self.initial_solution)
        self.initial_solution = Contiguous.from_struct(self.initial_solution)

        self.action_bounds = jax.tree.map(lambda x: x.reshape(2, 1, 1), action_bounds)
        self.action_bounds = Contiguous.from_struct(self.action_bounds)
        self.act_min = jax.tree.map(lambda x: x[0], self.action_bounds)
        self.act_max = jax.tree.map(lambda x: x[1], self.action_bounds)

        self.initial_sigma = jax.tree.map(lambda x: x.reshape(1, 1), initial_sigma)
        self.initial_sigma = Contiguous.from_struct(self.initial_sigma)

        self.reset()

        self.init_model(nn_model)
        
    def act(self, past_states, past_actions, commands, weights, key=None, vmap=False, initial_solve=False):
        """
        Compute the best action using the policy.
        """
        if key is None:
            self.key, key = jax.random.split(self.key)

        if vmap:
            if initial_solve:
                last_solution, min_costs, _commands = self.vmapped_initial_solver(self.model, key, self.last_solution, past_states, past_actions, commands, weights)
            else:
                last_solution, min_costs, _commands = self.vmapped_solver(self.model, key, self.last_solution, past_states, past_actions, commands, weights)
        else:
            if initial_solve:
                last_solution, min_costs, _commands = self.initial_solver(self.model, key, self.last_solution, past_states, past_actions, commands, weights)
            else:

                last_solution, min_costs, _commands = self.solver(self.model, key, self.last_solution, past_states, past_actions, commands, weights)


        self.last_solution = last_solution
        return last_solution, min_costs, _commands
    
    

    def process_rollouts(self, initial_state_array, rollouts, actions):
        """
        Process the rollouts and actions to match the expected structure.
        """
        rollouts.array = jnp.concatenate([initial_state_array[None, :], rollouts.array], axis=0)
        #rollouts.array = self.interpolate_50_to_100hz(rollouts.array)
        actions.array = jnp.concatenate([actions.array, actions.array[-1][None, :]], axis=0)

        return rollouts, actions


    def get_policy(self):
        """
        Create the compiled policy function to be called repeatedly during optimization.
        """

        def get_knots_array(actions_array):
            knots = jax.vmap(lambda x: self.actions_to_knots(self.spline_order, x, self.n_action_knots), in_axes=-1, out_axes=-1)(actions_array) # call .array
            knot_points = jnp.clip(knots, self.act_min.array, self.act_max.array)
            return knot_points

        def get_actions_array(knot_points_array):
            actions_array = jax.vmap(lambda x: self.knots_to_actions(self.spline_order, x, self.horizon-1), in_axes=-1, out_axes=-1)(knot_points_array)
            actions_array = jnp.clip(actions_array, self.act_min.array, self.act_max.array)
            return actions_array


        def policy_fn(model, key, last_solution_array, past_states, past_actions,
                    commands, weights, n_iters=5, mu=1e-8):
            
            # H_100hz = self.model.dynamics.H * 2
            # T_100hz = self.model.dynamics.T * 2
            lifted_states = LiftedStates(past_states, past_actions, model.dynamics.T, model.dynamics.H)
            # action_slice_starts = jnp.arange(H_100hz+2, self.horizon + H_100hz + 2, step=T_100hz)[:-1]
            # action_slice_starts = action_slice_starts - action_slice_starts[0]  # Start from 0


            if self.commands_callback is not None:
                commands = jax.vmap(self.commands_callback, in_axes=-1, out_axes=-1)(commands)

            cost_weights_flat, _ = jax.tree.flatten(weights.cost_weights)
            W_r = jnp.concatenate(cost_weights_flat, axis=-1).reshape(-1)

            if self.global_residuals is not None:
                global_cost_weights_flat, _ = jax.tree.flatten(weights.global_cost_weights)
                W_r = jnp.concatenate([W_r, jnp.concatenate(global_cost_weights_flat, axis=-1).reshape(-1)], axis=-1)

            constraint_weights_flat, treedef = jax.tree.flatten(weights.constraint_weights)
            W_g = jnp.concatenate(constraint_weights_flat, axis=-1).reshape(-1).reshape(-1)

            delta_flat, _ = jax.tree.flatten(weights.constraint_relaxation)
            delta = jnp.concatenate(delta_flat, axis=-1).reshape(-1)

            def residuals_fn(knots_flat):
                "returns (cost_residuals_flattened_array, constraint_residuals_flattened_array), states"
                knots = knots_flat.reshape(self.n_action_knots, self.n_actions)
                actions_array = get_actions_array(knots)
                #actions_chunked = jax.vmap(lambda start: jax.lax.dynamic_slice_in_dim(actions_array, start, T_100hz, axis=0))(action_slice_starts)
                print(f"actions_array.shape: {actions_array.shape}, past_actions.shape: {past_actions.array.shape}")
                actions = jax.tree.map(lambda x: actions_array[..., None, :], past_actions)
                rollouts, actions = model.dynamics.rollout(lifted_states, actions)
                #actions_original = jax.tree.map(lambda x: actions_array, past_actions)
                states, actions = self.process_rollouts(lifted_states.states.array[-1], rollouts, actions)
                r, g = jax.vmap(self.residuals, in_axes=(0, 0, 0), out_axes=0)(states, actions, commands)
                #outs = [r.reshape(-1), g.reshape(-1)]
                if self.global_residuals is not None:
                    r_global = self.global_residuals(states, actions, commands)
                    r = jnp.concatenate([r.reshape(-1), r_global.reshape(-1)], axis=0)
                return (r.reshape(-1), g.reshape(-1)), states

            def relaxed_log_barrier(g):
                def log_barrier(g):
                    return -W_g * jnp.log(-g)
                def quadratic(g):
                     return -W_g * jnp.log(delta) + 0.5 * W_g * (((g + 2 * delta) / delta) ** 2 - 1)
                return jnp.where(g < -delta, log_barrier(g), quadratic(g))
            
            def Lagrangian_barrier(knots):
                knots_flat = knots.reshape(-1)
                (r, g), states = residuals_fn(knots_flat)
                cost = r @ (W_r * r)
                constraints = relaxed_log_barrier(g)
                L = cost + constraints.sum()
                return L, cost, states

            def forward_pass_barrier(knots, dknots, key, iteration, mu):
                alphas = jnp.linspace(0, 1, self.n_linesearch_steps)
                knots_search = knots + alphas[:, None, None] * dknots

                # if all dknots are zero, use random perturbations
                noise = jax.random.normal(key, (self.n_linesearch_steps, knots.shape[0], knots.shape[1])) * self.initial_sigma.array[None, None, :] / (iteration + 1)
                knots_search = jax.lax.cond(jnp.all(dknots == 100.0), 
                                            lambda _: knots + noise.squeeze(0),
                                            lambda _: knots_search,
                                            operand=None)
                
                knots_search = jnp.clip(knots_search, self.act_min.array, self.act_max.array)

                Ls, costs, states = jax.vmap(Lagrangian_barrier, in_axes=(0))(knots_search)

                best_idx = jnp.argmin(Ls)
                mu_new = mu
                best_knots, best_L, best_cost = knots_search[best_idx], Ls[best_idx], costs[best_idx]
                return best_knots, best_L, best_cost, mu_new, jax.tree.map(lambda x: x[best_idx], states)

            def backward_pass_barrier(knots, mu):
                knots_flat = knots.reshape(-1)
                (Jr, Jg), (r, g) = jacfwd_and_value(lambda x: residuals_fn(x)[0], knots_flat)

                #total_residual = r.T @ (W_r * r)
                #jax.debug.print("Total residual: {res}", res=total_residual)

                # g < -delta
                q_vec_case_1 = -W_g * (1 / g)
                H_inner_case_1 = (W_g / (g ** 2))

                # g >= -delta
                q_vec_case_2 = W_g * ((g + 2 * delta) / (delta ** 2))
                H_inner_case_2 = (W_g / (delta ** 2))

                q_vec = jnp.where(g < -delta, q_vec_case_1, q_vec_case_2)
                q_constraint = Jg.T @ q_vec
                H_inner = jnp.where(g < -delta, H_inner_case_1, H_inner_case_2)
                H_constraint = Jg.T @ (H_inner[:, None] * Jg)

                H = Jr.T @ (W_r[:, None] * Jr) + H_constraint
                q = Jr.T @ (W_r * r) + q_constraint

                L = jnp.linalg.cholesky(H + mu * jnp.eye(H.shape[0]))
                y = solve_triangular(L, -q, lower=True)
                dknots_flat = solve_triangular(L.T, y, lower=False)

                dknots = dknots_flat.reshape(self.n_action_knots, self.n_actions)
                return dknots


            def spline_shooter_step_barrier(carry, _):
                i, knots, key, mu = carry
                d_knots = backward_pass_barrier(knots, mu)
                d_knots = jnp.nan_to_num(d_knots, nan=100.0, posinf=100.0, neginf=100.0)
                d_knots = jax.lax.cond(jnp.any(d_knots == 100.0),
                                        lambda _: jnp.ones_like(d_knots) * 100.0,
                                        lambda _: d_knots,
                                        operand=None)
                next_knots, min_L, min_cost, mu_next, best_states = forward_pass_barrier(knots, d_knots, key, i, mu)
                new_key, __ = jax.random.split(key)
                return [i, next_knots, new_key, mu_next], [min_cost, min_L, best_states]
            

            last_solution = jnp.roll(last_solution_array, shift=-1, axis=0)
            last_solution = last_solution.at[-1].set(last_solution_array[-1])
            #print(f"last_solution.shape: {last_solution.shape}, past_states.shape: {past_states.array.shape}, commands.shape: {commands.v_x.shape}, weights.shape: {weights.cost_weights.v_lin.shape}")
            knots_array = get_knots_array(last_solution)
            init_scan_state = [0, knots_array, key, mu]

            if self.jit:
                ctrl_outputs, min_costs = jax.lax.scan(spline_shooter_step_barrier, init_scan_state, None, length=n_iters)


            ########### for debugging ###########
            else:
                scan_state = init_scan_state 
                min_costs = []
                for _ in range(n_iters):
                    scan_state, min_cost_i = spline_shooter_step_barrier(scan_state, None)
                    min_costs.append(min_cost_i)
                ctrl_outputs = scan_state

            knots = ctrl_outputs[1]
            actions = get_actions_array(knots)

            return jax.tree.map(lambda x: actions, lifted_states.actions), min_costs, commands
        
        return policy_fn

    def init_model(self, model):
        """
        Update the model used for rollouts.
        """
        self.model = model
        self.policy_ = self.get_policy()

        mu = jnp.array([self.lm_damping])

        mu_initial_solve = mu

        self.policy = lambda *args, **kwargs: self.policy_(*args, **kwargs, n_iters=self.n_iters, mu=mu)
        self.init_policy = lambda *args, **kwargs: self.policy_(*args, **kwargs, n_iters=15, mu=mu_initial_solve)

        self.vmapped_initial_solver = lambda *args, **kwargs: self.solve_fn(self.init_policy, True, *args, **kwargs)
        self.vmapped_solver = lambda *args, **kwargs: self.solve_fn(self.policy, True, *args, **kwargs)

        self.initial_solver = lambda *args, **kwargs: self.solve_fn(self.init_policy, False, *args, **kwargs)
        self.solver = lambda *args, **kwargs: self.solve_fn(self.policy, False, *args, **kwargs)

        if self.jit:
            self.vmapped_initial_solver = jax.jit(self.vmapped_initial_solver)
            self.vmapped_solver = jax.jit(self.vmapped_solver)

            self.initial_solver = jax.jit(self.initial_solver)
            self.solver = jax.jit(self.solver)


    def update_model(self, model):
        """
        Update the model used for rollouts.
        """
        self.model = model

    def reset(self):
        """
        Reset the policy to its initial state.
        """
        self.key = jax.random.PRNGKey(self.seed)
        self.last_solution = jax.tree.map(lambda x: x, self.initial_solution)

    @staticmethod
    def solve_fn(policy, vmap, model, key, last_solution, past_states, past_actions, commands, weights):
        """
        Initial solve to get the first solution.
        """
        if vmap:
            if last_solution.array.shape[0] != past_states.array.shape[0]:
                last_solution = jax.tree.map(lambda x: jnp.repeat(x[None, ...], past_states.array.shape[0], axis=0), last_solution)
                #print(f"last_solution.shape: {last_solution.array.shape}, past_states.shape: {past_states.array.shape}, commands.shape: {commands.v_x.shape}, weights.shape: {weights.cost_weights.v_lin.shape}")
            last_solution, min_costs, commands = jax.vmap(policy, in_axes=(None, None, 0, 0, 0, 0, 0))(model, key, last_solution.array,  past_states, past_actions, commands, weights)
        else:
            last_solution, min_costs, commands = policy(model, key, last_solution.array, past_states, past_actions, commands, weights)

        return last_solution, min_costs, commands
