from .base_policy import Policy
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from mujoco import rollout
from Common import *
from Training import innovation_data


class NNDataCollection_Policy(Policy):
    """
   Data Collection Policy using  a neural network policy.
    """

    def __init__(self, nn_policy, data_wrangler, jit=True):

        step_cost = nn_policy.step_cost
        super().__init__(nn_policy.model, step_cost, jit)
        # Load task-specific configurations

        self.policy_type = "nn"
        self.horizon = nn_policy.model.dynamics.horizon
        self.data_wrangler = data_wrangler
        self.nn_policy = nn_policy
        self.key = jax.random.PRNGKey(0)

        self.step_cost = step_cost
        self.calls = 0

        self.reset()

        #self.setup_nn_policy()
    

    def update_nn_policy(self, nn_policy):
        """
        Update the neural network policy with the given policy.

        Args:
            nn_policy: The neural network policy to be updated.
        """
        self.nn_policy = nn_policy
        #self.setup_nn_policy()

    @staticmethod
    @jax.jit
    def process_inputs(data_wrangler, mj_dataset, current_idx):
        mj_dataset = jax.tree.map(lambda x: x, mj_dataset)
        nn_data = data_wrangler.process_data_training(mj_dataset)
        past_states = nn_data.past_states(current_idx)
        past_actions = nn_data.past_actions(current_idx)
        return past_states, past_actions
    
    @staticmethod
    @jax.jit
    def process_inputs_for_state_estimation(data_wrangler, mj_dataset, current_idx):
        mj_dataset = jax.tree.map(lambda x: x, mj_dataset)
        nn_data = data_wrangler.process_data_training(mj_dataset)
        past_obs = nn_data.past_obs(current_idx)
        past_actions = nn_data.past_actions(current_idx)
        past_states = nn_data.past_states(current_idx)
        prev_future_actions = nn_data.future_actions(current_idx-1)
        prev_past_actions = nn_data.past_actions(current_idx-1)
        prev_past_states = nn_data.past_states(current_idx-1)
        #past_obs = jax.tree.map(lambda x: x[:, -1], past_obs)
        return past_obs, past_actions, past_states, prev_future_actions, prev_past_actions, prev_past_states    


    @staticmethod
    @jax.jit
    def estimate(observer_model, past_states_t, past_observations_t, past_states_tm1, future_actions_tm1, past_actions_tm1, dynamics_model):
        """
        Estimate the states using the observer model.
        """
        dinnovation, innovation, states_t_pred = innovation_data(
            past_states_tm1, past_actions_tm1, future_actions_tm1, past_observations_t, dynamics_model)
        
        past_states_t_in = jax.tree.map(lambda x, y: jnp.concatenate([x[:, 1:], y], axis=1), past_states_t, states_t_pred)
        #past_states_t_in.array = past_states_t_in.array.at[:, -1].set(states_t_pred.array[:, 0]) # only the last state (most recent) is used for estimation
        past_states_hat = jax.vmap(observer_model.F_x)(past_states_t_in, past_observations_t, past_states_tm1, innovation, dinnovation)
        return past_states_hat

    def nn_policy_fn(self, data_wrangler, mj_dataset, commands, weights, key, current_idx, initial_solve=False, with_state_estimation=False):

        self.calls += 1
        if not with_state_estimation:
            past_states_t, past_actions_t = self.process_inputs(data_wrangler, mj_dataset, current_idx) # actually the gt states but keeping the naming the same
            past_states_est = past_states_t
        else:
            past_observations_t, past_actions_t, past_states_t, future_actions_tm1, past_actions_tm1, past_states_tm1 = self.process_inputs_for_state_estimation(data_wrangler, mj_dataset, current_idx)
            if self.calls == 1:
                self.past_states_t = past_states_t #jax.tree.map(lambda x: np.random.normal(0, 1, x.shape), past_states_t)
                past_states_est = past_states_t


            self.past_states_tm1 = self.past_states_t

            # dinnovation, innovation, states_t_pred = innovation_data(
            #     self.past_states_tm1, past_actions_tm1, future_actions_tm1, past_observations_t, self.nn_policy.model.dynamics
            # )
            past_states_est = self.estimate(self.nn_policy.model.observer, self.past_states_t, past_observations_t, self.past_states_tm1, future_actions_tm1, past_actions_tm1, self.nn_policy.model.dynamics)
            self.past_states_t = past_states_est

        est_mse = jax.tree.map(lambda gt, est: jnp.abs((gt-est)).mean(), past_states_t, past_states_est)
        actions, min_cost, _commands = self.nn_policy.act(past_states_est, past_actions_t, commands, weights, key, vmap=True, initial_solve=initial_solve)
        actions = self.to_mj_action_array(mj_dataset, actions)
        return actions, (min_cost, est_mse.array), _commands

    def act(self, *args, **kwargs):
        return self.nn_policy_fn(*args, **kwargs)
    
    def reset(self):
        self.nn_policy.reset()
        self.calls = 0
