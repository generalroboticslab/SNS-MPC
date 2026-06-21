from .base_policy import Policy
import jax
import jax.numpy as jnp
from Common import *
from Policies.data_collection_policy import innovation_data

@jax.jit
def get_future_actions_tm1(actions_object, past_observations_t):
    """
    Get the future actions from the dataset.
    """
    actions_object = jax.tree.map(lambda x: x [:, 0][:, None], actions_object)
    actions_object.fl_hip = past_observations_t.qd_fl_hip[:, -2][:, None]
    actions_object.fl_knee = past_observations_t.qd_fl_knee[:, -2][:, None]
    actions_object.fl_ankle = past_observations_t.qd_fl_ankle[:, -2][:, None]
    actions_object.fr_hip = past_observations_t.qd_fr_hip[:, -2][:, None]
    actions_object.fr_knee = past_observations_t.qd_fr_knee[:, -2][:, None]
    actions_object.fr_ankle = past_observations_t.qd_fr_ankle[:, -2][:, None]
    actions_object.rl_hip = past_observations_t.qd_rl_hip[:, -2][:, None]
    actions_object.rl_knee = past_observations_t.qd_rl_knee[:, -2][:, None]
    actions_object.rl_ankle = past_observations_t.qd_rl_ankle[:, -2][:, None]
    actions_object.rr_hip = past_observations_t.qd_rr_hip[:, -2][:, None]
    actions_object.rr_knee = past_observations_t.qd_rr_knee[:, -2][:, None]
    actions_object.rr_ankle = past_observations_t.qd_rr_ankle[:, -2][:, None]
    return actions_object

@jax.jit
def get_past_actions_t(actions_object, past_observations_t):
    """
    Get the past actions from the dataset.
    """
    actions_object = jax.tree.map(lambda x: x, actions_object)
    actions_object.fl_hip = past_observations_t.qd_fl_hip[:, -9:-1]
    actions_object.fl_knee = past_observations_t.qd_fl_knee[:, -9:-1]
    actions_object.fl_ankle = past_observations_t.qd_fl_ankle[:, -9:-1]
    actions_object.fr_hip = past_observations_t.qd_fr_hip[:, -9:-1]
    actions_object.fr_knee = past_observations_t.qd_fr_knee[:, -9:-1]
    actions_object.fr_ankle = past_observations_t.qd_fr_ankle[:, -9:-1]
    actions_object.rl_hip = past_observations_t.qd_rl_hip[:, -9:-1]
    actions_object.rl_knee = past_observations_t.qd_rl_knee[:, -9:-1]
    actions_object.rl_ankle = past_observations_t.qd_rl_ankle[:, -9:-1]
    actions_object.rr_hip = past_observations_t.qd_rr_hip[:, -9:-1]
    actions_object.rr_knee = past_observations_t.qd_rr_knee[:, -9:-1]
    actions_object.rr_ankle = past_observations_t.qd_rr_ankle[:, -9:-1]
    return actions_object



@staticmethod
@jax.jit
def overwrite_estimates(past_observations_t, past_states_est):
    past_states_est.q_fl_hip = past_observations_t.q_fl_hip[:, -8:]
    past_states_est.q_fl_knee = past_observations_t.q_fl_knee[:, -8:]
    past_states_est.q_fl_ankle = past_observations_t.q_fl_ankle[:, -8:]
    past_states_est.q_fr_hip = past_observations_t.q_fr_hip[:, -8:]
    past_states_est.q_fr_knee = past_observations_t.q_fr_knee[:, -8:]
    past_states_est.q_fr_ankle = past_observations_t.q_fr_ankle[:, -8:]
    past_states_est.q_rl_hip = past_observations_t.q_rl_hip[:, -8:]
    past_states_est.q_rl_knee = past_observations_t.q_rl_knee[:, -8:]
    past_states_est.q_rl_ankle = past_observations_t.q_rl_ankle[:, -8:]
    past_states_est.q_rr_hip = past_observations_t.q_rr_hip[:, -8:]
    past_states_est.q_rr_knee = past_observations_t.q_rr_knee[:, -8:]
    past_states_est.q_rr_ankle = past_observations_t.q_rr_ankle[:, -8:]
    past_states_est.v11 = past_observations_t.v11[:, -8:]
    past_states_est.v12 = past_observations_t.v12[:, -8:]
    past_states_est.v13 = past_observations_t.v13[:, -8:]
    past_states_est.v21 = past_observations_t.v21[:, -8:]
    past_states_est.v22 = past_observations_t.v22[:, -8:]
    past_states_est.v23 = past_observations_t.v23[:, -8:]
    past_states_est.v_roll = past_observations_t.v_roll[:, -8:]
    past_states_est.v_pitch = past_observations_t.v_pitch[:, -8:]
    past_states_est.v_yaw = past_observations_t.v_yaw[:, -8:]
    past_states_est.v_fl_hip = past_observations_t.v_fl_hip[:, -8:]
    past_states_est.v_fl_knee = past_observations_t.v_fl_knee[:, -8:]
    past_states_est.v_fl_ankle = past_observations_t.v_fl_ankle[:, -8:]
    past_states_est.v_fr_hip = past_observations_t.v_fr_hip[:, -8:]
    past_states_est.v_fr_knee = past_observations_t.v_fr_knee[:, -8:]
    past_states_est.v_fr_ankle = past_observations_t.v_fr_ankle[:, -8:]
    past_states_est.v_rl_hip = past_observations_t.v_rl_hip[:, -8:]
    past_states_est.v_rl_knee = past_observations_t.v_rl_knee[:, -8:]
    past_states_est.v_rl_ankle = past_observations_t.v_rl_ankle[:, -8:]
    past_states_est.v_rr_hip = past_observations_t.v_rr_hip[:, -8:]
    past_states_est.v_rr_knee = past_observations_t.v_rr_knee[:, -8:]
    past_states_est.v_rr_ankle = past_observations_t.v_rr_ankle[:, -8:]
    return past_states_est


def get_joint_data(past_observations_t):

    q_pos = past_observations_t.array[..., past_observations_t.index_map['q_fl_hip'][0]:past_observations_t.index_map['q_rr_ankle'][1]]
    q_pos_target = past_observations_t.array[-2, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]
    last_q_pos_target = past_observations_t.array[-3, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]
    last_last_q_pos_target = past_observations_t.array[-4, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]

    pos_error = q_pos[-1] - q_pos_target
    last_pos_error = q_pos[-2] - last_q_pos_target
    last_last_pos_error = q_pos[-3] - last_last_q_pos_target

    velocity = past_observations_t.array[..., past_observations_t.index_map['v_fl_hip'][0]:past_observations_t.index_map['v_rr_ankle'][1]]

    joint_data = jnp.stack([
        pos_error, last_pos_error, last_last_pos_error,
        velocity[-1], velocity[-2], velocity[-3],
    ], axis=-1)
    return joint_data

@jax.jit
def get_actuator_net_torque(actuator_net, past_observations_t):
    
    joint_data = jax.vmap(get_joint_data)(past_observations_t)

    torques = jax.vmap(jax.vmap(actuator_net.forward))(joint_data)

    return torques


class NNDeploy_Policy(Policy):
    """
   Deploy Policy using mpc.
    """

    def __init__(self, nn_policy, states, actions, obs, jit=True):

        step_cost = nn_policy.step_cost
        super().__init__(nn_policy.model, step_cost, jit)
        # Load task-specific configurations

        self.policy_type = "nn"
        self.horizon = nn_policy.model.dynamics.horizon
        self.nn_policy = nn_policy
        self.key = jax.random.PRNGKey(0)

        self.step_cost = step_cost
        self.calls = 0

        self.states = states
        self.actions = jax.tree.map(lambda x: x[0, :8][None, ...], actions)
        self.observations = obs
        self.initial_solve = True
        self.reset()

        self.past_actions_t = jax.tree.map(lambda x: x, self.actions)
        self.past_states_t = jax.tree.map(lambda x: x[0, :8][None, ...], states)
        self.past_states_tm1 = jax.tree.map(lambda x: x[0, :8][None, ...], states)
        self.past_actions_tm1 = jax.tree.map(lambda x: x, self.actions)
        self.past_states_t = jax.tree.map(lambda x: jnp.zeros_like(x), self.past_states_t)
        self.past_actions_tm1 = jax.tree.map(lambda x: jnp.zeros_like(x), self.past_actions_tm1)



        #self.setup_nn_policy()
    

    def update_nn_policy(self, nn_policy):
        """
        Update the neural network policy with the given policy.

        Args:
            nn_policy: The neural network policy to be updated.
        """
        self.nn_policy = nn_policy


    @staticmethod
    @jax.jit
    def estimate(observer_model, past_observations_t, past_states_tm1, past_actions_tm1, future_actions_tm1, dynamics_model):
        """
        Estimate the states using the observer model.
        """
        dinnovation, innovation, states_t_pred = innovation_data(past_states_tm1, past_actions_tm1, future_actions_tm1, past_observations_t, dynamics_model)
        past_states_t_in = jax.tree.map(lambda x, y: jnp.concatenate([x[:, 1:], y], axis=1), past_states_tm1, states_t_pred)
        #past_states_t_in.array = past_states_t_in.array.at[:, -1].set(states_t_pred.array[:, 0]) # only the last state (most recent) is used for estimation
        past_states_hat = jax.vmap(observer_model.F_x)(past_states_t_in, past_observations_t, past_states_tm1, innovation, dinnovation)
        return past_states_hat



    def nn_policy_fn(self, past_observations_t, commands, weights):

        self.calls += 1
        # past_observations_t, past_actions_t, past_states_t, future_actions_tm1, past_actions_tm1, past_states_tm1 = self.process_inputs_for_state_estimation(data_wrangler, mj_dataset, current_idx)
        self.past_actions_t = get_past_actions_t(self.actions, past_observations_t)
        future_actions_tm1 = get_future_actions_tm1(self.actions, past_observations_t)
        past_states_tm1 = self.past_states_t
        past_actions_tm1 = self.past_actions_tm1
        past_actions_t = self.past_actions_t


        if self.calls > 4:
            past_states_est = self.estimate(self.nn_policy.model.observer, past_observations_t, past_states_tm1, past_actions_tm1, future_actions_tm1, self.nn_policy.model.dynamics)
            torque_est = get_actuator_net_torque(self.nn_policy.model.dynamics.actuator_net, past_observations_t)
            actions, best_, _commands = self.nn_policy.act(past_states_est, past_actions_t, commands, weights, key=None, vmap=True, initial_solve=self.initial_solve)
            best = best_[2]
            #ff_torque = best_[3]
            #print("returned actions")
            if self.initial_solve:
                self.initial_solve = False
        else:
            past_states_est = past_states_tm1
            actions = get_future_actions_tm1(self.actions, past_observations_t)
            _commands = commands
            best = jax.tree.map(lambda x: x[0, :8][None, ...], past_states_est)
            torque_est = jnp.zeros((1, 12))  # Placeholder for torque estimates when not using the observer
            #ff_torque = jax.tree.map(lambda x: x, self.actions)


        self.past_states_t = past_states_est
        self.past_actions_tm1 = past_actions_t
        
        return actions, best, _commands, past_states_est, torque_est

    def act(self, *args, **kwargs):
        return self.nn_policy_fn(*args, **kwargs)

    def reset(self):
        self.nn_policy.reset()
        self.calls = 0
        self.initial_solve = True