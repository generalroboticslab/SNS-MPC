import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from interpax import Interpolator1D

def linear_interpolate(t, y, t_query):
    """
    Pure JAX linear interpolation for 1D values.
    
    Args:
        t: (N,) jnp array, strictly increasing time points.
        y: (N,) jnp array, values at time points.
        t_query: (M,) jnp array, query times in [t[0], t[-1]].

    Returns:
        Interpolated values at t_query: (M,)
    """
    idx = jnp.searchsorted(t, t_query, side="right", method='compare_all') - 1
    idx = jnp.clip(idx, 0, len(t) - 2)

    t0 = t[idx]
    t1 = t[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]

    alpha = (t_query - t0) / (t1 - t0 + 1e-8)
    return (1 - alpha) * y0 + alpha * y1

class Policy(ABC):
    """Abstract Base Policy class defining a rollout model and cost function interface."""
    
    def __init__(self, model, step_cost, jit):
        self.model = model
        self.step_cost = step_cost
        self.jit = jit
        self.evaluate_current_state_fn = self.get_evaluate_current_state_fn()

    @abstractmethod
    def act(self, data, key):
        """Compute an action given data and PRNG key."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset the policy state if needed."""
        pass

    @staticmethod
    def actions_to_knots(spline_order, actions, n_knots):
        """
        Convert a full action sequence into spline knots.
        
        Args:
            spline_order: Spline interpolation order (0, 1, or 3).
            actions: Action sequence (T x n_action).
            n_knots: Number of spline knots to extract.

        Returns:
            Reduced set of action knots.
        """
        t = jnp.linspace(0, 1, actions.shape[0])
        t_sample = jnp.linspace(0, 1, n_knots)
        #return linear_interpolate(t, actions, t_sample)
        if spline_order == 0:
            return actions
        elif spline_order == 1:
            #print(f"t: {t.shape}, actions: {actions.shape}, t_sample: {t_sample.shape}")
            #spline = Interpolator1D(t, actions, method="linear")
            #return spline(t_sample)
            return linear_interpolate(t, actions, t_sample)
        elif spline_order == 3:
            spline = Interpolator1D(t, actions, method="cubic")
            return spline(t_sample)
        else:
            raise ValueError(f"Unknown spline order: {spline_order}")
        
        
    @staticmethod
    def knots_to_actions(spline_order, knots, horizon):
        """
        Convert spline knots back to a full action sequence.
        
        Args:
            spline_order: Spline interpolation order (0, 1, or 3).
            knots: Knot points (n_knots x n_action).
            horizon: Desired output sequence length.

        Returns:
            Full interpolated action sequence (horizon x n_action).
        """

        t = jnp.linspace(0, 1, knots.shape[0])
        t_sample = jnp.linspace(0, 1, horizon)
        #return linear_interpolate(t, knots, t_sample)
        if spline_order == 0:
            return knots
        elif spline_order == 1:
            # spline = Interpolator1D(t, knots, method="linear")
            # return spline(t_sample)
            return linear_interpolate(t, knots, t_sample)
        elif spline_order == 3:
            spline = Interpolator1D(t, knots, method="cubic")
            return spline(t_sample)
        else:
            raise ValueError(f"Unknown spline order: {spline_order}")


    @staticmethod
    def interpolate_50_to_100hz(x_t):
        # x_t: shape (n+1, ...)
        mid = (x_t[:-1] + x_t[1:]) / 2
        stacked = jnp.empty((2 * x_t.shape[0] - 1,) + x_t.shape[1:], dtype=x_t.dtype)
        stacked = stacked.at[0::2].set(x_t)
        stacked = stacked.at[1::2].set(mid)
        return stacked

    
    def get_evaluate_current_state_fn(self):
        
        def _evaluate_current_state(data_wrangler, eval_mj_dataset, commands, weights, idx, debug=False):

            nn_data = data_wrangler.process_data_training(eval_mj_dataset)
            states = nn_data.past_states(idx)
            actions = nn_data.future_actions(idx)

            states = jax.tree.map(lambda x: x[:, -1], states)
            print(f"actions in eval shape: {actions.array.shape}")
            actions = jax.tree.map(lambda x: x[:, 0], actions)
            commands = jax.tree.map(lambda x: x[:, 0], commands)
            weights = jax.tree.map(lambda x: x[:, 0], weights)
            step_cost = jax.vmap(lambda *args: self.step_cost(*args, debug))(states, actions, commands, weights)

            return step_cost

        if self.jit:
            return jax.jit(_evaluate_current_state, static_argnames=('debug'))
        else:
            return _evaluate_current_state

    
    def evaluate_current_state(self, data_wrangler, mj_dataset, commands, weights, idx, debug):
        return self.evaluate_current_state_fn(data_wrangler, mj_dataset, commands, weights, idx, debug)

    
    def update_data_wrangler(self, data_wrangler):
        """
        Update the data wrangler with the given data wrangler.

        Args:
            data_wrangler: The data wrangler to be updated.
        """
        self.data_wrangler = data_wrangler

    
    @staticmethod
    @jax.jit
    def to_mj_action_array(mj_dataset, action_instance):
        actions_mj_index_map = mj_dataset.action_trajectory.index_map
        actions = jnp.zeros_like(action_instance.array)
        for _, (k, v) in enumerate(actions_mj_index_map.items()):
            i, j = v
            act = getattr(action_instance, k)
            actions = actions.at[..., i:j].set(act)
        return actions

    


