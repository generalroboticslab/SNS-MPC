from .base_policy import Policy
import jax
import jax.numpy as jnp
from Common import *
import time
from functools import partial

#@register_pytree_node_class
class MPPI_Policy(Policy):
    """
    MPPI based trajectory optimization policy using a spline parametrization of actions.
    """

    def __init__(self, nn_model, step_cost, initial_solution, initial_sigma,
                 action_bounds, commands_callback=None, spline_order=1, n_knots=40, 
                 traj_diffuse_fctr=0.5, horizon_diffuse_fctr=0.9, horizon=19,
                 n_rollouts=128, n_iters=5, temperature=0.1,
                 verbose=False, seed=0, jit=True):
        """
        Initialize MPPI Policy.
        
        Args:
            model: Rollout model taking future actions and returning predicted states.
            step_cost: Step cost function.
            initial_solution: Initial solution for the MPPI policy.
            initial_sigma: Initial sigma for the MPPI policy as it is annealed with each iteration via reverse diffusion.
            action_bounds: Action bounds for the MPPI policy.
            commands_callback: Callback function with the signature (states, actions, commands) that returns the commands to be used in the cost function.
            spline_order: Order of the spline used for action parametrization.
            n_knots: Number of knots in the spline.
            traj_diffuse_fctr: Factor for trajectory diffusion.
            horizon_diffuse_fctr: Factor for horizon diffusion.
            n_rollouts: Number of rollouts to perform in each MPPI iteration.
            n_iters: Number of MPPI iterations to perform.
            temperature: Temperature for the MPPI policy.
            verbose: Whether to print debug information.
            seed: Random seed for reproducibility.
            jit: Whether to use JIT compilation for the policy function.
        """
        
        super().__init__(nn_model, step_cost, jit)

        self.horizon = horizon
        self.spline_order = spline_order
        self.n_action_knots = n_knots
        self.n_rollouts = n_rollouts
        self.temperature = temperature
        self.n_iters = n_iters
        self.verbose = verbose
        self.jit = jit
        self.seed = seed
        self.key = jax.random.PRNGKey(seed)

        self.commands_callback = commands_callback

        self.n_states = nn_model.dynamics.n_states
        self.n_actions = nn_model.dynamics.n_actions

        self.initial_solution = initial_solution
        self.initial_sigma = initial_sigma
        self.traj_diffuse_fctr = traj_diffuse_fctr
        self.horizon_diffuse_fctr = horizon_diffuse_fctr
        self.action_bounds = action_bounds

        self.initial_solution = jax.tree.map(lambda x: x.reshape(1, 1), initial_solution)
        self.initial_solution = jax.tree.map(lambda x: jnp.repeat(x, self.horizon-1, axis=0), self.initial_solution)
        self.initial_solution = Contiguous.from_struct(self.initial_solution)
        self.initial_sigma = jax.tree.map(lambda x: x.reshape(1, 1), initial_sigma)
        self.initial_sigma = Contiguous.from_struct(self.initial_sigma)

        self.action_bounds = jax.tree.map(lambda x: x.reshape(2, 1, 1), action_bounds)
        self.action_bounds = Contiguous.from_struct(self.action_bounds)
        self.act_min = jax.tree.map(lambda x: x[0], self.action_bounds)
        self.act_max = jax.tree.map(lambda x: x[1], self.action_bounds)

        self._policy = self.get_policy()

        self.policy = lambda *args, **kwargs: self._policy(*args, **kwargs, n_iters=self.n_iters)
        self.init_policy = lambda *args, **kwargs: self._policy(*args, **kwargs, n_iters=15)

        self.reset()

        self.update_model(nn_model)
        
    def act(self, past_states, past_actions, commands, weights, key=None, vmap=False, initial_solve=False):
        """
        Compute the best action using MPPI.

        Returns either the best action or both the best action and rollout if return_best_rollout is True.
        """

        
        if key is None:
            key, self.key = jax.random.split(self.key, 2)
        if vmap:
            if initial_solve:
                last_solution, min_costs, _commands = self.vmapped_initial_solver(key, self.last_solution, past_states, past_actions, commands, weights)
            else:
                last_solution, min_costs, _commands = self.vmapped_solver(key, self.last_solution, past_states, past_actions, commands, weights)
        else:
            if initial_solve:
                last_solution, min_costs, _commands = self.initial_solver(key, self.last_solution, past_states, past_actions, commands, weights)
            else:
                last_solution, min_costs, _commands = self.solver(key, self.last_solution, past_states, past_actions, commands, weights)

        self.last_solution = last_solution
        return last_solution, min_costs, _commands
    
    

    def process_rollouts(self, initial_state_array, rollouts, actions):
        """
        Process the rollouts and actions to match the expected structure.
        """
        actions.array = actions.array.reshape(self.n_rollouts, self.horizon-1, self.n_actions)
        # rollouts.array = rollouts.array.reshape(self.n_rollouts, (self.horizon - 1)//2, self.n_states)

        rollouts.array = jnp.concatenate([initial_state_array[:, None, :], rollouts.array], axis=1)

        # rollouts.array = jax.vmap(self.interpolate_50_to_100hz)(rollouts.array)
        actions.array = jnp.concatenate([actions.array, actions.array[:, -1][:, None, :]], axis=1)

        return rollouts, actions

    def get_policy(self):
        """
        Create the compiled mppi policy function to be called repeatedly during optimization.
        """

        def perturb_actions(actions, key, i, N):
    
            knots = jax.vmap(lambda x: self.actions_to_knots(self.spline_order, x, self.n_action_knots), in_axes=-1, out_axes=-1)(actions.array)
            h_vals = jnp.arange(self.n_action_knots)

            beta1 = self.traj_diffuse_fctr
            beta2 = self.horizon_diffuse_fctr
            #i = N - i  # Reverse the iteration index for annealing

            anneal = jnp.exp(
                - (N - i) / (2 * beta1 * N)
                - (self.n_action_knots - 1 - h_vals) / (2 * beta2 * (self.n_action_knots-1))
            )  # shape: (n_action_knots + 1,)

            # Apply annealed noise
            noise = jax.random.normal(
                key,
                (self.n_rollouts, self.n_action_knots, actions.array.shape[-1])
            ) * self.initial_sigma.array[None, None, :]  * anneal[None, :, None] #/ (i + 1) #

            knot_points = knots + noise.squeeze(0)
            knot_points = jnp.clip(knot_points, self.act_min.array, self.act_max.array)

            actions_array = jax.vmap(jax.vmap(lambda x: self.knots_to_actions(self.spline_order, x, self.horizon-1), in_axes=-1, out_axes=-1))(knot_points)
            actions_array = jnp.clip(actions_array, self.act_min.array, self.act_max.array)

            return jax.tree.map(lambda x: actions_array, actions), knot_points


        def policy_fn(last_solution_array, key, past_states, past_actions, commands, weights, n_iters=5):
            
            # H_100hz = self.model.dynamics.H * 2
            # T_100hz = self.model.dynamics.T * 2
            lifted_states = LiftedStates(past_states, past_actions, self.model.dynamics.T, self.model.dynamics.H)
            lifted_states = jax.tree.map(lambda x: jnp.repeat(x[None, ...], self.n_rollouts, axis=0), lifted_states)
            # action_slice_starts = jnp.arange(H_100hz+2, self.horizon + H_100hz + 2, step=T_100hz)[:-1]
            # action_slice_starts = action_slice_starts - action_slice_starts[0]  # Start from 0


            last_solution = jnp.roll(last_solution_array, shift=-1, axis=0)
            last_solution = last_solution.at[-1].set(last_solution_array[-1])
            init_scan_state = [0, key, last_solution]


            if self.commands_callback is not None:
                commands = jax.vmap(self.commands_callback, in_axes=-1, out_axes=-1)(commands)


            def mppi_step(carry, _):
                """Single MPPI iteration with rollout and fitness evaluation."""
                i, rng, mu_i = carry
                rng, subkey = jax.random.split(rng, 2)

                mu_i = jax.tree.map(lambda x: mu_i, lifted_states.actions)

                sampled_actions, knots = perturb_actions(mu_i, subkey, i, n_iters)
                #actions = jax.tree.map(lambda x: jax.vmap(lambda start: jax.lax.dynamic_slice_in_dim(x, start, T_100hz, axis=1))(action_slice_starts), sampled_actions)
                actions = jax.tree.map(lambda x: x.transpose(1, 0, 2)[..., None, :], sampled_actions)
     
                rollouts, actions = jax.vmap(self.model.dynamics.rollout, in_axes=(0, 1))(lifted_states, actions)

                states, actions = self.process_rollouts(lifted_states.states.array[:, -1], rollouts, actions)

                _cost = lambda states, actions, commands, weights: self.step_cost(states, actions, commands, weights, self.verbose)

                cost = jax.vmap(jax.vmap(_cost, in_axes=(0, 0, 0, 0), out_axes=0), in_axes=(0, 0, None, None))

                if self.verbose:
                    step_cost, costs = cost(states, actions, commands, weights)
                    mean_cost = jax.tree.map(jnp.mean, costs)
                    jax.debug.print("Iteration: {i} \nMean Cost: {mean_cost} \nMean Terminal Cost: {mean_terminal_cost}\n", i=i, mean_cost=mean_cost)
                else:
                    step_cost = cost(states, actions, commands, weights)

                total_cost = jnp.sum(step_cost, axis=1)


                # Compute the weights for the MPPI update
                min_cost = jnp.min(total_cost)
                max_cost = jnp.max(total_cost)
                exp_weights = jnp.exp(-1 / self.temperature * ((total_cost - min_cost) / (max_cost - min_cost)))
                mu_knots = jnp.sum(exp_weights[:, None, None] * knots, axis=0) / jnp.sum(exp_weights)
                mu_knots = jnp.clip(mu_knots, self.act_min.array, self.act_max.array)
                mu = jax.vmap(self.knots_to_actions, in_axes=(None, -1, None), out_axes=-1)(self.spline_order, mu_knots, self.horizon-1)  
                mu = jnp.clip(mu, self.act_min.array, self.act_max.array)

                i += 1

                return [i, rng, mu], min_cost
            
            # Run MPPI for the specified number of iterations


            if self.jit:
                ctrl_outputs, min_costs = jax.lax.scan(mppi_step, init_scan_state, None, length=n_iters)


            ########### for debugging ###########
            else:
                scan_state = init_scan_state 
                min_costs = []
                for _ in range(n_iters):
                    scan_state, min_cost_i = mppi_step(scan_state, None)
                    min_costs.append(min_cost_i)
                ctrl_outputs = scan_state

            actions_array = jax.lax.cond(
                jnp.any(jnp.isnan(ctrl_outputs[2])),
                lambda x: last_solution,
                lambda x: x[2],
                ctrl_outputs
            )
            return jax.tree.map(lambda x: actions_array, lifted_states.actions), min_costs, commands

        # if self.jit:
        #     return jax.jit(policy_fn, static_argnames=('initial_solve'))
        # else:
        return policy_fn


    def update_model(self, model):
        """
        Update the model used for rollouts.
        """
        self.model = model
        self.policy_ = self.get_policy()

        self.policy = lambda *args, **kwargs: self.policy_(*args, **kwargs, n_iters=self.n_iters)
        self.init_policy = lambda *args, **kwargs: self.policy_(*args, **kwargs, n_iters=15)

        self.vmapped_initial_solver = lambda *args, **kwargs: self.solve_fn(self.init_policy, True, *args, **kwargs)
        self.vmapped_solver = lambda *args, **kwargs: self.solve_fn(self.policy, True, *args, **kwargs)

        self.initial_solver = lambda *args, **kwargs: self.solve_fn(self.init_policy, False, *args, **kwargs)
        self.solver = lambda *args, **kwargs: self.solve_fn(self.policy, False, *args, **kwargs)

        if self.jit:
            self.vmapped_initial_solver = jax.jit(self.vmapped_initial_solver)
            self.vmapped_solver = jax.jit(self.vmapped_solver)

            self.initial_solver = jax.jit(self.initial_solver)
            self.solver = jax.jit(self.solver)



    def reset(self):
        """
        Reset the policy to its initial state.
        """
        self.key = jax.random.PRNGKey(self.seed)
        self.last_solution = jax.tree.map(lambda x: x, self.initial_solution)

    @staticmethod
    def solve_fn(policy, vmap, key, last_solution, past_states, past_actions, commands, weights):
        """
        Initial solve to get the first solution.
        """
        if vmap:
            if last_solution.array.shape[0] != past_states.array.shape[0]:
                last_solution = jax.tree.map(lambda x: jnp.repeat(x[None, ...], past_states.array.shape[0], axis=0), last_solution)
                print(f"last_solution.shape: {last_solution.array.shape}, past_states.shape: {past_states.array.shape}, commands.shape: {commands.v_x.shape}")
            last_solution, min_costs, commands = jax.vmap(policy, in_axes=(0, None, 0, 0, 0, 0))(last_solution.array, key, past_states, past_actions, commands, weights)
        else:
            last_solution, min_costs, commands = policy(last_solution.array, key, past_states, past_actions, commands, weights)

        return last_solution, min_costs, commands


    # def tree_flatten(self):
    #     """
    #     Flatten the policy parameters for serialization.
    #     """
    #     static = (self.spline_order, self.n_action_knots, self.n_rollouts, self.temperature, self.n_iters, self.verbose, 
    #               self.jit, self.initial_solution, self.initial_sigma, self.action_bounds, self.act_min, self.act_max, 
    #               self.
    
    #     dynamic = (self.model.dynamics