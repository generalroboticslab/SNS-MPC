
import time
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import rollout
import mujoco.viewer
import tqdm
from abc import ABC, abstractmethod



@jax.jit
def update_history(mj_dataset, sensors, actions):
    mj_dataset.action_history.array = jnp.roll(mj_dataset.action_history.array, -1, axis=0)
    mj_dataset.sensor_history.array = jnp.roll(mj_dataset.sensor_history.array, -1, axis=0)

    mj_dataset.action_history.array = mj_dataset.action_history.array.at[-1].set(actions)
    mj_dataset.sensor_history.array = mj_dataset.sensor_history.array.at[-1].set(sensors)

    return mj_dataset

@jax.jit
def start_history(steps_since_reset, mj_dataset, sensors, actions):
    mj_dataset.action_history.array = mj_dataset.action_history.array.at[steps_since_reset].set(actions)
    mj_dataset.sensor_history.array = mj_dataset.sensor_history.array.at[steps_since_reset].set(sensors)
    return mj_dataset



class DeployModule(ABC):
    def __init__(self, mj_model, mj_data, mj_dataset_cls, agent, n_steps, n_history_steps, update_rate=1):


        self.mj_model = mj_model
        self.mj_data = mj_data
        self.mj_model.opt.enableflags = 1
        self.mj_model.opt.o_solref = np.array([0.02, 1.0])
        self.mj_model.opt.integrator = 3
        self.mj_model.opt.timestep = 0.005
        self.timestep = mj_model.opt.timestep
        self.viewer = None
        self.update_rate = update_rate

        self.n_steps = n_steps 
        self.stop_step = n_steps

        # self.save_dir = save_dir

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
            
        self.agent = agent

        self.step_count = 0 

        self.n_history_steps = n_history_steps
        self.n_actions = self.mj_model.nu


        action_history_shape = (self.n_history_steps, self.n_actions)
        sensor_history_shape = (self.n_history_steps, self.mj_model.nsensordata)

        shapes = jax.tree.map(lambda x: None, mj_dataset_cls)
        shapes.action_history = action_history_shape
        shapes.sensor_history = sensor_history_shape
        shapes.action_rollouts = (self.n_history_steps, self.n_actions)
        shapes.sensor_rollouts = (self.n_history_steps, self.mj_model.nsensordata)

        print("Shapes: ", shapes)

        self.mj_dataset = jax.tree.map(lambda cls, shape: cls.from_mjmodel(self.mj_model, shape), mj_dataset_cls, shapes)

        self.state_rollouts = np.zeros(
            (1, self.agent.horizon, mujoco.mj_stateSize(self.mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )

        self.sensor_rollouts = np.zeros(
            (1, self.agent.horizon, self.mj_model.nsensordata)
        )


    def deploy(self, headless=False):
        
        self.raw_action = self.mj_model.key_ctrl[0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        if headless == False:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            
            self.viewer.opt.sitegroup[0] = False
        
        for i in tqdm.tqdm(range(0, self.n_steps)):
            if headless == False:
                if not self.viewer.is_running():
                    break
            current_time = time.perf_counter()
            self.step_main_simulation()
            if headless == False:
                self.viewer.sync()
            self.step_count += 1

            wait_time = self.mj_model.opt.timestep - (time.perf_counter() - current_time)
            if wait_time > 0:
                time.sleep(wait_time)

    
    def step_main_simulation(self):

        
        mujoco.mj_forward(self.mj_model, self.mj_data)

        last_action = self.raw_action
        sensors = np.asarray(self.mj_data.sensordata)
        actions = np.asarray(last_action)

        if self.step_count >= self.n_history_steps:
            self.mj_dataset = update_history(self.mj_dataset, sensors, actions)

        else:
            self.mj_dataset = start_history(self.step_count, self.mj_dataset, sensors, actions)

        if self.step_count >= self.n_history_steps:
            self.nn_input_data, self.nn_output_data = self.get_nn_inputs_and_outputs(self.mj_dataset)
            commands, cost_weights, terminal_cost_weights = self.get_commands_and_weights()
            if self.agent.return_best_rollout:
                self.raw_action, self.best_rollout_pred = jax.block_until_ready(self.agent.act(self.nn_input_data, self.nn_output_data, commands, cost_weights, terminal_cost_weights))
                self._call_rollout(self.mj_data.qpos, self.mj_data.qvel, self.best_rollout_pred[1])
                self.post_rollout_callback()
            else:
                self.raw_action = jax.block_until_ready(self.agent.act(self.nn_input_data, self.nn_output_data, commands, cost_weights, terminal_cost_weights))
            self.action = self.raw_action
            self.post_agent_update_callback()
            self.mj_data.ctrl = self.action
            # else:
            #     self.mj_data.ctrl = self.raw_action

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.agent.data = self.mj_data

    def _call_rollout(self, qpos, qvel, ctrl):
        """
        Perform a rollout of the model given the initial state and control actions.

        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        initial_conditions = np.concatenate([[0], np.concatenate([qpos, qvel])])[None, ...]
        ctrl = ctrl.array[None, ...]
        data = mujoco.MjData(self.mj_model)
        rollout.rollout(self.mj_model, data, skip_checks=True,
                        nroll=1, nstep=self.agent.horizon,
                        initial_state=initial_conditions, control=ctrl, state=self.state_rollouts, sensordata=self.sensor_rollouts)

    def post_agent_update_callback(self):
        pass
    
    @abstractmethod
    def get_nn_inputs_and_outputs(self, mj_dataset):
        pass

    @abstractmethod
    def get_commands_and_weights(self):
        pass

    def post_rollout_callback(self):
        pass