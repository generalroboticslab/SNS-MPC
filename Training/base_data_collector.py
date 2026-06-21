import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mujoco.viewer
import mujoco.rollout as rollout
import threading
from Common.utils import Camera
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import concurrent
from interpax import interp1d
import time
from PIL import Image, ImageDraw, ImageFont
from Common.nn_model import MLPBase
from Common.runtime_paths import ACTUATOR_NETWORK_200HZ, OPEN_SANS_REGULAR_TTF
from scipy.spatial.transform import Rotation as R

@jax.jit
def update_mj_dataset_sensors(mj_dataset, sensors, index):
    mj_dataset.sensor_trajectory.array = mj_dataset.sensor_trajectory.array.at[:, index].set(sensors)
    return mj_dataset

@jax.jit
def update_mj_dataset_actions(mj_dataset, actions, index):
    mj_dataset.action_trajectory.array = mj_dataset.action_trajectory.array.at[:, index].set(actions)
    return mj_dataset

def cubic_spline_actions(key, n_envs, n_actions, n_knots, n_steps, act_min, act_max, nominal_ctrl):
    """
    Generate cubic spline actions for the given parameters.
    
    Args:
        key: JAX random key.
        n_actions: Number of actions.
        n_knots: Number of knots in the spline.
        n_steps: Number of steps in the trajectory.
        act_min: Minimum action value.
        act_max: Maximum action value.
        
    Returns:
        A JAX array of shape (n_steps, n_actions) containing the cubic spline actions.
    """
    key, subkey = jax.random.split(key)
    knots = jax.random.uniform(subkey, (n_envs, n_knots, n_actions), minval=act_min, maxval=act_max)
    #knots = jax.random.normal(subkey, (n_envs, n_knots, n_actions)) * 1.0 + nominal_ctrl

    t = jnp.linspace(0, 1, n_knots)
    t_eval = jnp.linspace(0, 1, n_steps)
    actions = jax.vmap(jax.vmap(lambda k: interp1d(t_eval, t, k, method='cubic'), in_axes=-1, out_axes=-1))(knots)
    actions = jnp.clip(actions, act_min, act_max)
    return actions


torque_limits = {"fl_hip": (-24.0, 24.0),
                    "fl_knee": (-24.0, 24.0),
                    "fl_ankle": (-45.0, 45.0),
                    "fr_hip": (-24.0, 24.0),
                    "fr_knee": (-24.0, 24.0),
                    "fr_ankle": (-45.0, 45.0),
                    "rl_hip": (-24.0, 24.0),
                    "rl_knee": (-24.0, 24.0),
                    "rl_ankle": (-45.0, 45.0),
                    "rr_hip": (-24.0, 24.0),
                    "rr_knee": (-24.0, 24.0),
                    "rr_ankle": (-45.0, 45.0)}
tau_min = jnp.array([low for low, high in torque_limits.values()])[None, :]
tau_max = jnp.array([high for low, high in torque_limits.values()])[None, :]

def sigmoid_saturate(x, limit, k=1.0):
    """
    Symmetric sigmoid-based soft clipping.
    Maps input to range (-limit, limit), centered at 0.
    """
    return 2 * limit * (1 / (1 + jnp.exp(-k * x / limit)) - 0.5)


    # TODO clip in eval and not in 
@jax.jit
def to_torque(actuator_network, pos_error, last_pos_error, last_last_pos_error, vel, last_vel, last_last_vel):
    """
    Convert lagged actions to torques using the actuator network.
    
    Args:
        actuator_network: The trained actuator network.
        lagged_actions: Lagged actions array of shape (n_envs, decimation + 1, n_actions).
        decimation: Decimation factor.
        
    Returns:
        A JAX array of shape (n_envs, decimation + 1, n_actions) containing the torques.
    """
    # Reshape lagged actions to match the input shape of the actuator network
    # pos_error *= 0.9
    # last_pos_error *= 0.9
    # last_last_pos_error *= 0.9
    inputs = jnp.stack([pos_error, last_pos_error, last_last_pos_error, vel, last_vel, last_last_vel], axis=-1)
    raw = jax.vmap(jax.vmap(actuator_network.forward))(inputs)
    # Clip the torques to be within the specified limits
    #torques = sigmoid_saturate(raw, limit=tau_max + 30)
    #raw = raw.at[::3].multiply(0.5)  # Scale down the first action (hip) by 0.5
    torques = jnp.clip(raw, tau_min, tau_max)
    return torques

@jax.jit
def to_torque_200hz(actuator_network, 
                    pos_error, last_pos_error, last_last_pos_error, last_last_last_pos_error, last_last_last_last_pos_error, last_last_last_last_last_pos_error, last_6_pos_error, last_7_pos_error, last_8_pos_error,
                    vel, last_vel, last_last_vel, last_last_last_vel, last_last_last_last_vel, last_last_last_last_last_vel, last_6_vel, last_7_vel, last_8_vel):
    """
    Convert lagged actions to torques using the actuator network.
    
    Args:
        actuator_network: The trained actuator network.
        lagged_actions: Lagged actions array of shape (n_envs, decimation + 1, n_actions).
        decimation: Decimation factor.
        
    Returns:
        A JAX array of shape (n_envs, decimation + 1, n_actions) containing the torques.
    """
    inputs = jnp.stack([3*pos_error, 3*last_pos_error, 3*last_last_pos_error, 3*last_last_last_pos_error, 3*last_last_last_last_pos_error, 3*last_last_last_last_last_pos_error, 3*last_6_pos_error, 3*last_7_pos_error, 3*last_8_pos_error,
                        vel, last_vel, last_last_vel, last_last_last_vel, last_last_last_last_vel, last_last_last_last_last_vel, last_6_vel, last_7_vel, last_8_vel], axis=-1)
    raw = jax.vmap(jax.vmap(actuator_network.forward))(inputs)
    return jnp.clip(raw, tau_min.reshape(1, 12, 1), tau_max.reshape(1, 12, 1))

class ThreadedOnlineDataCollectionModule(ABC):
    def __init__(self, data_collection_agent,
                 eval_agent,
                 mj_dataset_cls, 
                 mj_model_path,
                 steps_per_episode, 
                 n_history_steps,
                 n_history_steps_observer,
                 horizon,
                 seed=0,
                 save_video=False,
                 video_text=False,
                 n_workers=512,
                 n_envs=1024, 
                 headless=False, 
                 camera_kwargs=None, 
                 dtype=jnp.float32):
        
        self.model_path = mj_model_path
        self.video_text = video_text
        # xml = ET.parse(mj_model_path)
        # root = xml.getroot()
        # xml = ET.tostring(root, encoding='unicode', method='xml')
        # print(f"xml: {xml}")
        # mj_model = mujoco.MjModel.from_xml_string(xml)

        self.decimation = 4
        self.n_lags = 5

        horizon = horizon
        n_history_steps = n_history_steps
        n_history_steps_observer = n_history_steps_observer

        self.camera_kwargs = camera_kwargs
        
        # print(f"Original model NSENSORDATA: {mj_model.nsensordata}, NU: {mj_model.nu}, NV: {mj_model.nv}, NQ: {mj_model.nq}, NA: {mj_model.na}")

        # assert False

        self.rng_key = jax.random.PRNGKey(seed)
        print("Initializing MuJoCo models")
        mj_models =  []
        spawn_pos = []
        for _ in range(n_envs):
            model, spawn_pos_arr = self.new_mj_model()
            mj_models.append(model)
            spawn_pos.append(spawn_pos_arr)

        self.spawn_pos = np.array(spawn_pos)
        self.num_envs = n_envs

        self.actuator_network_path = ACTUATOR_NETWORK_200HZ
        self.actuator_net = MLPBase.load(self.actuator_network_path)
        self.actuator_net.Lipschitz_ub = 1e12
        params = self.actuator_net.params
        self.actuator_net.set_inference_mode(True, params)
        # set all models to the params
        # for i in range(n_envs):
        #     mj_models[i].opt.enableflags = 1
        #     mj_models[i].opt.o_solref = np.array([0.02, 1.0])
        #     mj_models[i].opt.integrator = 3
        #     mj_models[i].opt.timestep = 0.005

        self.mj_models = mj_models
        self.mj_data = mujoco.MjData(self.mj_models[0])

        self.blender_data_pos = []
        self.blender_data_quat = []
        self.tracking_error = []
        self.viewer = None

        assert int(n_history_steps + horizon + n_history_steps_observer) < steps_per_episode, "n_history_steps + horizon must be less than steps_per_episode"
        self.steps_per_episode = steps_per_episode 

            
        self.agent = data_collection_agent
        self.eval_agent = eval_agent
        self.n_workers = n_workers

        self.headless = headless

        self.n_history_steps = n_history_steps
        self.n_history_steps_observer = n_history_steps_observer
        self.horizon = horizon
        self.n_actions = self.mj_models[0].nu
        self.nominal_ctrl = self.mj_models[0].key_ctrl[0]
        self.mj_data_steps = 0

        self.torque_buffer = np.zeros((12, steps_per_episode))
        self.augmented_torque_buffer = np.zeros((12, steps_per_episode))

        joint_limits = {"fl_hip": (-1.0472, 1.0472),
                    "fl_knee": (-1.5708, 3.4907),
                    "fl_ankle": (-2.7227, -0.83776),
                    "fr_hip": (-1.0472, 1.0472),
                    "fr_knee": (-1.5708, 3.4907),
                    "fr_ankle": (-2.7227, -0.83776),
                    "rl_hip": (-1.0472, 1.0472),
                    "rl_knee": (-0.5236, 4.5379),
                    "rl_ankle": (-2.7227, -0.83776),
                    "rr_hip": (-1.0472, 1.0472),
                    "rr_knee": (-0.5236, 4.5379),
                    "rr_ankle": (-2.7227, -0.83776)}
    
        def inheritrange_bounds(joint_name, factor=1.0):
            q_min, q_max = joint_limits[joint_name]
            mid = 0.5 * (q_min + q_max)
            half_range = 0.5 * (q_max - q_min)
            relaxed_lb = mid - factor * half_range
            relaxed_ub = mid + factor * half_range
            return relaxed_lb, relaxed_ub
    
        action_limits = {k: inheritrange_bounds(k) for k in joint_limits.keys()}
        
        self.act_min = np.zeros(self.n_actions)
        self.act_max = np.zeros(self.n_actions)
        for i, (low, high) in enumerate(action_limits.values()):
            self.act_min[i] = low
            self.act_max[i] = high

        self.randomize_motor_offset = False
        self.randomize_motor_strength = False

        #     self.act_max = self.mj_models[0].actuator_ctrlrange[:, 1]
        #     self.act_min = self.mj_models[0].actuator_ctrlrange[:, 0]
        # except AttributeError:
        #     print("No actuator control range found, using default limits.")
        #     self.act_max = np.array([1.0] * self.n_actions)
            # self.act_min = np.array([-1.0] * self.n_actions)


        action_trajectory_shape = (self.num_envs, self.steps_per_episode, self.n_actions)
        sensor_trajectory_shape = (self.num_envs, self.steps_per_episode, self.mj_models[0].nsensordata)

        # print(f"Action trajectory shape: {action_trajectory_shape}")
        # print(f"Sensor trajectory shape: {sensor_trajectory_shape}")

        shapes = jax.tree.map(lambda x: None, mj_dataset_cls)
        shapes.action_trajectory = action_trajectory_shape
        shapes.sensor_trajectory = sensor_trajectory_shape
        self.mj_dataset = jax.tree.map(lambda cls, shape: cls.from_mjmodel(self.mj_models[0], shape, dtype), mj_dataset_cls, shapes)
        self.dtype = dtype
        self.step_envs_state_data, self.step_envs_sensor_data, self.step_envs_ctrl_data = self.init_mj_step_data(2)
        self.current_envs_state_data, self.current_envs_sensor_data, self.current_envs_ctrl_data = self.init_mj_step_data(1)
        self.current_envs_state_data = self.current_envs_state_data[:, 0, :]
        self.current_envs_sensor_data = self.current_envs_sensor_data[:, 0, :]
        self.current_envs_ctrl_data = self.current_envs_ctrl_data[:, 0, :]

        self.action_lag_buffer = np.repeat(self.nominal_ctrl[None, None, :], self.num_envs, axis=0)
        self.action_lag_buffer = np.repeat(self.action_lag_buffer, self.n_lags, axis=1)

        self.position_error_buffer = np.zeros((self.num_envs, 12, self.n_actions))
        self.velocity_buffer = np.zeros((self.num_envs, 12, self.n_actions))

        self.set_weights()

        self.thread_local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers, initializer=self.thread_initializer)

        self.episode_number = 0
        self.save_video = save_video
        if save_video:
            # self.mj_models[0].opt.sitegroup[0] = False
            scene_option = mujoco.MjvOption()
            scene_option.sitegroup[0] = False
            # scene_option.geomgroup[2] = False
            # scene_option.geomgroup[3] = True
            self.renderer = mujoco.Renderer(self.mj_models[0], 480, 640)
            if self.camera_kwargs == None:
                self.mj_camera = Camera("evaluation_camera", self.mj_models[0], self.mj_data, self.renderer,
                                        options=scene_option,
                                        lookat=np.array([0, 0, 0]),
                                        distance=1.4,
                                        azimuth=90.0,
                                        elevation=0.0)
            else:
                self.mj_camera = Camera("evaluation_camera", self.mj_models[0], self.mj_data, self.renderer,
                        options=scene_option,
                        lookat=np.array([0, 0, 0]),
                        **self.camera_kwargs)


    def init_mj_step_data(self, n_steps):
        state_step = np.zeros(
            (self.num_envs, n_steps, mujoco.mj_stateSize(self.mj_models[0], mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
        )
        sensor_step = np.zeros(
            (self.num_envs, n_steps, self.mj_models[0].nsensordata)
        )
        ctrl_step = np.zeros(
            (self.num_envs, n_steps, self.mj_models[0].nu)
        )
        return state_step, sensor_step, ctrl_step
        
    def thread_initializer(self):
        """Initialize thread-local storage for MuJoCo data."""
        self.thread_local.data = mujoco.MjData(self.mj_models[0])

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)

    def begin_collection(self):
        self.reset_robot()

        if self.headless == False:
            self.viewer = mujoco.viewer.launch_passive(self.mj_models[0], self.mj_data)
            
            self.viewer.opt.sitegroup[0] = False
    
    def end_collection(self):
        if self.headless == False:
            self.viewer.close()

    def run_episode(self, logger=None, init_buffer=False):
        self.DIVERGED = False
        outs = self._run_episode(logger, init_buffer)
        while self.DIVERGED:
            print("Diverged, running episode again")
            self.DIVERGED = False
            outs = self._run_episode(logger, init_buffer)
        self.episode_number += 1
        return outs

    def _run_episode(self, logger=None, init_buffer=False):
        self.set_weights()
        # rand = np.random.rand()
        # if rand < 0.5:
        #     with_state_estimation = True
        # else:
        #     with_state_estimation = False
        with_state_estimation = True
        self.begin_collection()

        if self.headless == False:
            if not self.viewer.is_running():
                return
        #import time
        #t0 = time.time()
        #self.reset_robot()
        self.cost = 0


        self.mj_dataset = jax.tree.map(
            lambda x: np.zeros((self.num_envs, self.steps_per_episode, *x.shape[2:]), dtype=x.dtype),
            self.mj_dataset
        )

        if init_buffer == True:
            # n_knots = int((self.steps_per_episode / 20) * 5)
            n_knots = 50
            self.rng_key, key = jax.random.split(self.rng_key, 2)
            self.action = cubic_spline_actions(
                key, 
                self.num_envs, 
                self.n_actions, 
                n_knots, 
                self.steps_per_episode, 
                self.act_min, 
                self.act_max, 
                self.nominal_ctrl
            )

        if self.randomize_motor_offset:
            self.motor_offset = np.random.uniform(-0.01, 0.01, (self.num_envs, self.n_actions))
        else:
            self.motor_offset = np.zeros((self.num_envs, self.n_actions))

        if self.randomize_motor_strength:
            self.motor_strength = np.random.uniform(0.95, 1.05, (self.num_envs, self.n_actions))
        else:
            self.motor_strength = np.ones((self.num_envs, self.n_actions))

        for i in range(self.steps_per_episode-1):
            self.pre_step_callback()
            self._step_main_simulation(i, init_buffer=init_buffer, with_state_estimation=with_state_estimation)
            self.post_step_callback()
            if self.headless == False:
                self.viewer.sync()

            if self.DIVERGED:
                return None

            if i % 50 == 0:
                print(f"Episode step {i+1} out of {self.steps_per_episode}")
        #tf = time.time()
        #print("Time taken for episode: ", tf - t0)
        self.end_collection()
        if logger is not None:
            logger.log_dict["cost"] = self.cost
            return self.mj_dataset, logger
        else:
            return self.mj_dataset


    def _step_main_simulation(self, i, init_buffer=False, with_state_estimation=False):
        if i == 0:
            # qpos = self.mj_data.qpos
            # qvel = self.mj_data.qvel

            # initial_conditions = np.concatenate([[0], np.concatenate([qpos, qvel])])[None, ...]
            # initial_conditions = np.repeat(initial_conditions, self.num_envs, axis=0)
            #self.act = np.repeat(self.mj_data.act[None, ...], self.num_envs, axis=0)
            states = np.concatenate([self.qpos, self.qvel], axis=1)
            initial_conditions = np.concatenate([np.zeros((self.num_envs, 1)), states], axis=1)
            #print(f"Initial conditions shape: {initial_conditions.shape}")
            self.current_envs_state_data = initial_conditions

            self.step_envs_ctrl_data = np.repeat(np.repeat(self.nominal_ctrl[None, ...], self.num_envs, axis=0)[:, None, :], 2, axis=1)

            self.action_lag_buffer = np.repeat(self.nominal_ctrl[None, None, :], self.num_envs, axis=0)
            self.action_lag_buffer = np.repeat(self.action_lag_buffer, self.n_lags, axis=1)

            self.position_error_buffer = np.zeros((self.num_envs, self.decimation*2+1, self.n_actions))
            self.velocity_buffer = np.zeros((self.num_envs, self.decimation*2+1, self.n_actions))

            if init_buffer == False:
                self.action = np.repeat(self.nominal_ctrl[None, None, :], self.num_envs, axis=0)


        skip_steps = self.n_history_steps + self.n_history_steps_observer
        if i >= skip_steps:
                if init_buffer == False:
                    if i == skip_steps:
                        self.rng_key, key, key2 = jax.random.split(self.rng_key, 3)
                        self.action, _, self.commands = self.agent.act(
                                    self.agent.data_wrangler,
                                    self.mj_dataset,
                                    self.commands,
                                    self.cost_weights,
                                    key,
                                    i,
                                    initial_solve=True, 
                                    with_state_estimation=with_state_estimation)
                        
                    else:
                        self.rng_key, key, key2 = jax.random.split(self.rng_key, 3)
                        self.action, _, self.commands = self.agent.act(
                                    self.agent.data_wrangler,
                                    self.mj_dataset,
                                    self.commands,
                                    self.cost_weights,
                                    key,
                                    i, 
                                    initial_solve=False, 
                                    with_state_estimation=with_state_estimation)
                        
                

                    self.post_agent_update_callback()
                    # see if the action has NaN values
                    if jnp.isnan(self.action).any():
                        n_nans = jnp.sum(jnp.isnan(self.action))
                        print(f"NaN values in action at step {i}: {n_nans}")
                        self.action = jnp.nan_to_num(self.action, nan=self.nominal_ctrl + jax.random.uniform(key2, self.action.shape, minval=-0.5, maxval=0.5))

                    if i < self.steps_per_episode - self.horizon - self.n_history_steps:
                        step_cost = self.agent.evaluate_current_state(self.agent.data_wrangler, self.mj_dataset, self.commands, self.cost_weights, i, debug=False)
                        self.cost += step_cost.mean()
                    
                    if i == self.steps_per_episode // 3:
                        self.reset_commands()

        if init_buffer == True:
            self.applied_action = self.action[:, i, :]
        else:
            self.applied_action = self.action[:, 0, :]
        values = [3, 4]
        probs = [0.55, 0.45]
        self.n_lags = np.random.choice(values, p=probs)

        if self.action_lag_buffer.shape[1] > self.n_lags:
            self.action_lag_buffer = self.action_lag_buffer[:, :self.n_lags, :]

        elif self.action_lag_buffer.shape[1] < self.n_lags:
            n_to_add = self.n_lags - self.action_lag_buffer.shape[1]
            to_add = self.action_lag_buffer[:, -1:, :]
            to_add = np.repeat(to_add, n_to_add, axis=1)
            self.action_lag_buffer = np.concatenate([to_add, self.action_lag_buffer], axis=1)
        

        for d in range(self.decimation):
            if (i > 0) or (d > 0):      
                self.current_envs_state_data = self.step_envs_state_data[:, 0, :]

            self.action_lag_buffer = np.roll(self.action_lag_buffer, shift=-1, axis=1)
            self.action_lag_buffer[:, -1, :] = self.applied_action

            joint_positions = self.current_envs_state_data[:, 1:self.mj_models[0].nq+1][:, 7:]
            joint_velocities = self.current_envs_state_data[:, self.mj_models[0].nq+1:self.mj_models[0].nq + self.mj_models[0].nv+1][:, 6:]
            joint_pos_error = joint_positions - self.action_lag_buffer[:, 0, :] + self.motor_offset

            self.position_error_buffer = np.roll(self.position_error_buffer, shift=-1, axis=1)
            self.position_error_buffer[:, -1, :] = joint_pos_error
            self.velocity_buffer = np.roll(self.velocity_buffer, shift=-1, axis=1)
            self.velocity_buffer[:, -1, :] = joint_velocities

            last_joint_pos_error = self.position_error_buffer[:, -2, :]
            last_joint_velocities = self.velocity_buffer[:, -2, :]
            last_last_joint_pos_error = self.position_error_buffer[:, -3, :]
            last_last_joint_velocities = self.velocity_buffer[:, -3, :]
            last_last_last_joint_pos_error = self.position_error_buffer[:, -4, :]
            last_last_last_joint_velocities = self.velocity_buffer[:, -4, :]
            last_last_last_last_joint_pos_error = self.position_error_buffer[:, -5, :]
            last_last_last_last_joint_velocities = self.velocity_buffer[:, -5, :]
            last_last_last_last_last_joint_pos_error = self.position_error_buffer[:, -6, :]
            last_last_last_last_last_joint_velocities = self.velocity_buffer[:, -6, :]
            last_6_joint_pos_error = self.position_error_buffer[:, -7, :]
            last_6_joint_velocities = self.velocity_buffer[:, -7, :]
            last_7_joint_pos_error = self.position_error_buffer[:, -8, :]
            last_7_joint_velocities = self.velocity_buffer[:, -8, :]
            last_8_joint_pos_error = self.position_error_buffer[:, -9, :]
            last_8_joint_velocities = self.velocity_buffer[:, -9, :]

            joint_torques = to_torque_200hz(
                self.actuator_net, 
                joint_pos_error,
                last_joint_pos_error,
                last_last_joint_pos_error,
                last_last_last_joint_pos_error,
                last_last_last_last_joint_pos_error,
                last_last_last_last_last_joint_pos_error,
                last_6_joint_pos_error,
                last_7_joint_pos_error,
                last_8_joint_pos_error,
                joint_velocities,
                last_joint_velocities,
                last_last_joint_velocities,
                last_last_last_joint_velocities,
                last_last_last_last_joint_velocities,
                last_last_last_last_last_joint_velocities,
                last_6_joint_velocities,
                last_7_joint_velocities,
                last_8_joint_velocities
            )
            self.step_envs_ctrl_data[:, 0, :] = joint_torques.squeeze(-1) * self.motor_strength
            self.step_envs_ctrl_data[:, 1, :] = self.step_envs_ctrl_data[:, 0, :]
            #t0 = time.time()
            self.threaded_mj_step(self.step_envs_state_data, self.step_envs_ctrl_data, self.current_envs_state_data, self.step_envs_sensor_data)
            # t1 = time.time()
            # print(f"Threaded MJ Step took {t1 - t0:.4f} seconds")

        #print(f"Applied action dtype: {self.applied_action.dtype}")
        # print(f"Mj Dataset action trajectory dtype: {self.mj_dataset.action_trajectory.array.dtype}")
        # print(f"Step envs sensor data dtype: {self.step_envs_sensor_data.dtype}")
        # print(f"Mj Dataset sensor trajectory dtype: {self.mj_dataset.sensor_trajectory.array.dtype}")
              
        self.mj_dataset = update_mj_dataset_actions(self.mj_dataset, self.applied_action, i)

        if i == 0:
        # sensor data is lagged by one step, so we use the second step which is the first step after the initial state
            self.mj_dataset = update_mj_dataset_sensors(self.mj_dataset, self.step_envs_sensor_data[:, 0, :], 0)
        self.mj_dataset = update_mj_dataset_sensors(self.mj_dataset, self.step_envs_sensor_data[:, 1, :], i+1)

    def evaluate(self, logger=None, seed=None, debug=False, with_state_estimation=False):
        self.set_evaluation_commands()
        self.set_evaluation_weights()
        self.set_evaluation_mj_model()
        self.set_evaluation_task_names()
        # self.begin_collection()

        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[0] = False
        self.renderer = mujoco.Renderer(self.mj_models[0], 480, 640)
        self.mj_data = mujoco.MjData(self.mj_models[0])

        if self.camera_kwargs == None:
            self.mj_camera = Camera("evaluation_camera", self.mj_models[0], self.mj_data, self.renderer,
                                    options=scene_option,
                                    lookat=np.array([0, 0, 0]),
                                    distance=1.4,
                                    azimuth=90.0,
                                    elevation=0)
        else:
            self.mj_camera = Camera("evaluation_camera", self.mj_models[0], self.mj_data, self.renderer,
                    options=scene_option,
                    lookat=np.array([0, 0, 0]),
                    **self.camera_kwargs)

        #self.mj_data = mujoco.MjData(self.mj_models[0])

        n_tasks = len(self.evaluation_commands)

        self.inference_times = []
        if debug:
            n_tasks = 1  # For debugging, only run one task
        for task_i in range(n_tasks):
            if with_state_estimation:
                print(f"Running evaluation for task {self.evaluation_task_names[task_i]} with state estimation")
            else:
                print(f"Running evaluation for task {self.evaluation_task_names[task_i]} without state estimation")
            #mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
            self.set_evaluation_state()
            self.eval_agent.reset()
            self.first_step = True
            if seed is not None:
                self.eval_key = jax.random.PRNGKey(seed)
            else:
                self.eval_key = jax.random.PRNGKey(25)
            if self.save_video:
                self.frames = []
            self.commands = self.evaluation_commands[task_i]
            self.cost_weights = self.evaluation_cost_weights[task_i]
            self.task_name = self.evaluation_task_names[task_i]

            tracking_constraint_relaxation_weights = jax.tree.map(lambda x: jnp.ones_like(x), self.cost_weights.constraint_relaxation)
            tracking_constraint_weights = jax.tree.map(lambda x: jnp.zeros_like(x), self.cost_weights.constraint_weights)
            tracking_cost_weights = jax.tree.map(lambda x: jnp.zeros_like(x), self.cost_weights.cost_weights)
            tracking_cost_weights.v_lin += 1.0
            self.tracking_cost_weights = jax.tree.map(lambda x: x, self.cost_weights)
            self.tracking_cost_weights.constraint_relaxation = tracking_constraint_relaxation_weights
            self.tracking_cost_weights.constraint_weights = tracking_constraint_weights
            self.tracking_cost_weights.cost_weights = tracking_cost_weights


            self.commands = jax.tree.map(
                lambda x: x[None, ...],
                self.commands
            )

            self.cost = 0.0  # reset task cost
            self.total_cost = 0.0  # reset total cost
            self.total_violation = 0.0  # reset total violation
            self.violation_margin = 0.0
            self.task_success = 1.0  # reset task success

            # Allocate fresh buffers for this task
            self.step_eval_state_data, self.step_eval_sensor_data, self.step_eval_ctrl_data = self.init_mj_step_data(2)
            self.current_eval_state_data, self.current_eval_sensor_data, self.current_eval_ctrl_data = self.init_mj_step_data(1)
            self.step_eval_state_data = self.step_eval_state_data[0, :, :][None, ...]
            self.step_eval_sensor_data = self.step_eval_sensor_data[0, :, :][None, ...]
            self.step_eval_ctrl_data = self.step_eval_ctrl_data[0, :, :][None, ...]
            self.current_eval_state_data = self.current_eval_state_data[:, 0, :]
            self.current_eval_sensor_data = self.current_eval_sensor_data[:, 0, :]
            self.current_eval_ctrl_data = self.current_eval_ctrl_data[:, 0, :]

            #n_knots = int((self.steps_per_episode / 20) * 5)
            n_knots = 70
            self.rng_key, key = jax.random.split(self.rng_key, 2)
            self.spline_action = cubic_spline_actions(
                key, 
                1, 
                self.n_actions, 
                n_knots, 
                self.steps_per_episode, 
                self.act_min, 
                self.act_max, 
                self.nominal_ctrl
            )

            # Reset dataset
            # self.mj_dataset.sensor_trajectory.array = self.mj_dataset.sensor_trajectory.array.at[:].set(0)
            # self.mj_dataset.action_trajectory.array = self.mj_dataset.action_trajectory.array.at[:].set(0)
            # self.mj_dataset.sensor_rollouts.array = self.mj_dataset.sensor_rollouts.array.at[:].set(0)
            # self.mj_dataset.action_rollouts.array = self.mj_dataset.action_rollouts.array.at[:].set(0)
            def array_zeros(x):
                return np.zeros((1, self.steps_per_episode, *x.shape[2:]), dtype=x.dtype) 
            self.mj_dataset = jax.tree.map(
                array_zeros,
                self.mj_dataset
            )

            for step in range(self.steps_per_episode - 1):
                self.pre_step_callback()
                self._step_main_simulation_eval(step, debug, with_state_estimation=with_state_estimation)
                self.post_step_callback()
                if self.headless == False:
                    self.viewer.sync()

            if with_state_estimation:
                suffix = "with_state_estimation"
            else:
                suffix = "without_state_estimation"
            logger.log_dict[f"{self.task_name}_cost_{suffix}"] = self.cost
            logger.log_dict[f"{self.task_name}_success_{suffix}"] = self.task_success
            logger.log_dict[f"{self.task_name}_tracking_cost_{suffix}"] = np.array(self.tracking_error).mean()
            self.tracking_error = []
            if self.save_video:
                logger.save_video(
                    self.frames,
                    f"{self.task_name}_evaluation_{suffix}")
                # print(f" blender data pos len: {len(self.blender_data_pos)}")
                # print(f" blender data pos first 10: {self.blender_data_pos[:10][1:5]}")
                logger.save_blender_data(
                    np.array(self.blender_data_pos), 
                    np.array(self.blender_data_quat), 
                    f"{self.task_name}_evaluation_{suffix}")
                self.blender_data_pos = []
                self.blender_data_quat = []
                logger.save_mj_dataset(
                    jax.tree.map(lambda x: x, self.mj_dataset),
                    f"{self.task_name}_evaluation_{suffix}")
                # logger.save_tracking_error(
                #     np.array(self.tracking_error), 
                #     f"{self.task_name}_evaluation_{suffix}")


        inference_times = np.array(self.inference_times)
        mean_inference_time = inference_times.mean()
        logger.log_dict[f"mean_policy_time_{suffix}"] = mean_inference_time

        #self.end_collection()

        return logger



    def _step_main_simulation_eval(self, i, debug=False, with_state_estimation=False):

        if i == 0:
            qpos = self.mj_data.qpos
            qvel = self.mj_data.qvel
            #act = self.mj_data.act
            #print(f"Actuator state: {self.mj_data.act}")

            initial_conditions = np.concatenate([[0], np.concatenate([qpos, qvel])])[None, ...]
            initial_conditions = np.repeat(initial_conditions, 1, axis=0)
            self.current_eval_state_data = initial_conditions

            init_ctrl = np.repeat(self.nominal_ctrl[None, :], 1, axis=0)
            self.step_eval_ctrl_data = np.repeat(init_ctrl[:, None, :], 2, axis=1)

            self.eval_action_lag_buffer = np.repeat(self.nominal_ctrl[None, None, :], 1, axis=0)
            self.eval_action_lag_buffer = np.repeat(self.eval_action_lag_buffer, self.n_lags, axis=1)

            self.eval_position_error_buffer = np.zeros((1, self.decimation*2+1, self.n_actions))
            self.eval_velocity_buffer = np.zeros((1, self.decimation*2+1, self.n_actions))

            self.action = np.repeat(self.nominal_ctrl[None, None, :], 1, axis=0)
            self.applied_action = self.action[:, 0, :]

            self.mj_dataset = jax.tree.map(lambda x: jnp.asarray(x), self.mj_dataset)



        skip_steps = self.n_history_steps + self.n_history_steps_observer 

        if i >= skip_steps:
            self.eval_key, key = jax.random.split(self.eval_key, 2)
            self.callback_before_eval_agent(i)
            if i == skip_steps:
                self.action, costs_est, self.commands = jax.block_until_ready(self.eval_agent.act(
                    self.eval_agent.data_wrangler,
                    self.mj_dataset,
                    self.commands,
                    self.cost_weights,
                    key,
                    i,
                    initial_solve=True, 
                    with_state_estimation=with_state_estimation))
            else:
                t0 = time.time()

                self.action, costs_est, self.commands = jax.block_until_ready(self.eval_agent.act(
                    self.eval_agent.data_wrangler,
                    self.mj_dataset,
                    self.commands,
                    self.cost_weights,
                    key,
                    i, 
                    initial_solve=False, 
                    with_state_estimation=with_state_estimation))
                t1 = time.time()
            if i > skip_steps + 1:
                self.inference_times.append(t1 - t0)
            if debug:
                print(f"Step: {i}   Minimum costs: {costs_est[0][1]}")
            self.eval_key, key = jax.random.split(self.eval_key, 2)
            if jnp.isnan(self.action).any():
                print(f"NaN values in action at step {i}")
                # Replace NaN values with nominal control plus some noise
            self.action = jnp.nan_to_num(self.action, nan=self.nominal_ctrl + jax.random.uniform(key, self.action.shape, minval=-0.5, maxval=0.5))

        self.applied_action = self.action[:, 0, :] #* 0.9 + self.applied_action * 0.1  # Smooth the action with the previous action
        #self.applied_action = self.applied_action[:, 0, 2::3]
        #self.applied_action = self.spline_action[:, i, :]

        #print(f"Step {i}, Applied action: {self.applied_action[0]}")

        values = [3, 4]
        probs = [0.55, 0.45]
        self.n_lags = np.random.choice(values, p=probs)
        #self.n_lags = 3

        if self.eval_action_lag_buffer.shape[1] > self.n_lags:
            self.eval_action_lag_buffer = self.eval_action_lag_buffer[:, :self.n_lags, :]

        elif self.eval_action_lag_buffer.shape[1] < self.n_lags:
            n_to_add = self.n_lags - self.eval_action_lag_buffer.shape[1]
            to_add = self.eval_action_lag_buffer[:, -1:, :]
            to_add = np.repeat(to_add, n_to_add, axis=1)
            self.eval_action_lag_buffer = np.concatenate([to_add, self.eval_action_lag_buffer], axis=1)


        for d in range(self.decimation):
            if (i > 0) or (d > 0):
                self.current_eval_state_data = self.step_eval_state_data[:, 0, :]

            self.eval_action_lag_buffer = np.roll(self.eval_action_lag_buffer, shift=-1, axis=1)
            self.eval_action_lag_buffer[:, -1, :] = self.applied_action

            joint_positions = self.current_eval_state_data[:, 1:self.mj_models[0].nq+1][:, 7:]
            joint_velocities = self.current_eval_state_data[:, self.mj_models[0].nq+1:self.mj_models[0].nq + self.mj_models[0].nv+1][:, 6:]
            joint_pos_error = joint_positions - self.eval_action_lag_buffer[:, 0, :]
            self.eval_position_error_buffer = np.roll(self.eval_position_error_buffer, shift=-1, axis=1)
            self.eval_position_error_buffer[:, -1, :] = joint_pos_error
            self.eval_velocity_buffer = np.roll(self.eval_velocity_buffer, shift=-1, axis=1)
            self.eval_velocity_buffer[:, -1, :] = joint_velocities
            last_joint_pos_error = self.eval_position_error_buffer[:, -2, :]
            last_joint_velocities = self.eval_velocity_buffer[:, -2, :]
            last_last_joint_pos_error = self.eval_position_error_buffer[:, -3, :]
            last_last_joint_velocities = self.eval_velocity_buffer[:, -3, :]
            last_last_last_joint_pos_error = self.eval_position_error_buffer[:, -4, :]
            last_last_last_joint_velocities = self.eval_velocity_buffer[:, -4, :]
            last_last_last_last_joint_pos_error = self.eval_position_error_buffer[:, -5, :]
            last_last_last_last_joint_velocities = self.eval_velocity_buffer[:, -5, :]
            last_last_last_last_last_joint_pos_error = self.eval_position_error_buffer[:, -6, :]
            last_last_last_last_last_joint_velocities = self.eval_velocity_buffer[:, -6, :]
            last_6_joint_pos_error = self.eval_position_error_buffer[:, -7, :]
            last_6_joint_velocities = self.eval_velocity_buffer[:, -7, :]
            last_7_joint_pos_error = self.eval_position_error_buffer[:, -8, :]
            last_7_joint_velocities = self.eval_velocity_buffer[:, -8, :]
            last_8_joint_pos_error = self.eval_position_error_buffer[:, -9, :]
            last_8_joint_velocities = self.eval_velocity_buffer[:, -9, :]

            #accel = (joint_velocities - 2 * last_joint_velocities + last_last_joint_velocities) / (0.02 ** 2)

            # print(f"Step {i}, Joint pos error: {joint_pos_error[0]}")
            # print(f"Step {i}, Joint vel: {joint_velocities[0]}")
            # print(f"P desired: {self.eval_action_lag_buffer[:, 0, :]}")
            # print(f"P actual: {joint_positions[0]}")
            #print(f"decimation step {d}, applied action: {self.applied_action[0, 0]}, n_lags: {self.n_lags}, lagged action: {self.eval_action_lag_buffer[0, 0, 0]}")


            #print(f"Step {i}, Joint pos error: {joint_pos_error.shape}, Joint vel: {joint_velocities.shape}, Last joint pos error: {last_joint_pos_error.shape}, Last joint vel: {last_joint_velocities.shape}, Last last joint pos error: {last_last_joint_pos_error.shape}, Last last joint vel: {last_last_joint_velocities.shape}")
            joint_torques = to_torque_200hz(
                self.actuator_net, 
                joint_pos_error,
                last_joint_pos_error,
                last_last_joint_pos_error,
                last_last_last_joint_pos_error,
                last_last_last_last_joint_pos_error,
                last_last_last_last_last_joint_pos_error,
                last_6_joint_pos_error,
                last_7_joint_pos_error,
                last_8_joint_pos_error,
                joint_velocities,
                last_joint_velocities,
                last_last_joint_velocities,
                last_last_last_joint_velocities,
                last_last_last_last_joint_velocities,
                last_last_last_last_last_joint_velocities,
                last_6_joint_velocities,
                last_7_joint_velocities,
                last_8_joint_velocities
            )

            
            self.step_eval_ctrl_data[:, 0, :] = joint_torques.squeeze(-1) #+ 0.3 * joint_velocities  #+ 0.000012 * accel
            self.step_eval_ctrl_data[:, 1, :] = self.step_eval_ctrl_data[:, 0, :]

            self.mj_data.qpos = self.current_eval_state_data[:, 1:self.mj_models[0].nq+1]
            self.mj_data.qvel = self.current_eval_state_data[:, self.mj_models[0].nq+1:self.mj_models[0].nq + self.mj_models[0].nv+1]
            self.mj_data.ctrl = self.step_eval_ctrl_data[:, 0, :]
            mujoco.mj_step(self.mj_models[0], self.mj_data)
            
            self.threaded_mj_step(self.step_eval_state_data, self.step_eval_ctrl_data, self.current_eval_state_data, self.step_eval_sensor_data, n_envs=1)

        #print(f"Step {i}, Applied action: {self.applied_action}")
        idx = int(i) 
        jax_action = jnp.asarray(self.applied_action)
        #print(f"type of mj_dataset.action_trajectory: {type(self.mj_dataset.action_trajectory)}, type of jax_action: {type(jax_action)}, shape of jax_action: {jax_action.shape}, idx: {idx}")
        self.mj_dataset = update_mj_dataset_actions(self.mj_dataset, jax_action, idx)

        if i == 0:
            jax_sensor = jnp.asarray(self.step_eval_sensor_data[:, 0, :])
            self.mj_dataset = update_mj_dataset_sensors(self.mj_dataset, jax_sensor, 0)

        jax_sensor = jnp.asarray(self.step_eval_sensor_data[:, 1, :])
        self.mj_dataset = update_mj_dataset_sensors(self.mj_dataset, jax_sensor, idx + 1)

        # check for success or failure
        if i < self.steps_per_episode - self.horizon - skip_steps:
            quat = self.mj_data.qpos[3:7]
            quat = np.roll(quat, -1)  # mujoco to scipy
            rot = R.from_quat(quat)
            roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
            if (np.abs(roll) > np.pi / 2) or (np.abs(pitch) > np.pi / 2):
                print(f"Robot has fallen at step {i}, roll: {roll}, pitch: {pitch}")
                self.task_success = 0.0

    
        if i > skip_steps + 2:
            if i < self.steps_per_episode - self.horizon - skip_steps:   
                if debug:
                    step_cost, costs_violation = self.eval_agent.evaluate_current_state(self.eval_agent.data_wrangler, self.mj_dataset, self.commands, self.cost_weights, i, debug=True)
                    costs = jax.tree.map(lambda x: x.mean(), costs_violation[0])
                    print(f"Step {i}, Costs: {jax.tree.map(lambda x: x.item(), costs)}")
                    # print(f"Step {i}, Total cost: {step_cost.sum().item()}")
                    # print(f"Step {i}, Inference time: {self.inference_times[-1] if self.inference_times else 0:.4f} seconds")
                    # print(f"Actions: {self.applied_action[0, :]}")
                    # print(f"Torques: {self.step_eval_ctrl_data[0, 0, :]}")
                    print(f"Step {i}, Violations: {costs_violation[1]}")
                    
                else:
                    step_cost, costs_violation = self.eval_agent.evaluate_current_state(self.eval_agent.data_wrangler, self.mj_dataset, self.commands, self.cost_weights, i, debug=True)

                tracking_step_cost, _ = self.eval_agent.evaluate_current_state(self.eval_agent.data_wrangler, self.mj_dataset, self.commands, self.tracking_cost_weights, i, debug=True)
                self.tracking_error.append(tracking_step_cost.item())

                violations, _ = jax.tree.flatten(costs_violation[1])
                #n_violations = jax.tree.map(lambda x: jnp.where(x > 1e-2, 1, 0).sum(), violations)
                n_violations = jax.tree.map(lambda x: jnp.where(x > 0.01, 1, 0).sum(), violations)
                n_violations = jnp.array(n_violations)
                n_violations = n_violations.sum()
                self.total_violation += n_violations
                if step_cost.mean() < 1e3: # random outliers from the orientation cost when it is exactly pi radians from the reference
                    self.cost += step_cost.mean()
                    self.total_cost += step_cost.sum().item()
                    self.est_mse = costs_est[1].item()
                print(f"Step {i}, cost: {self.cost}")


        if self.save_video:
            if i < self.steps_per_episode - self.horizon - skip_steps:
                # save blender data
                self.blender_data_pos.append(self.mj_data.xpos.copy())
                self.blender_data_quat.append(self.mj_data.xquat.copy())

                # if i > skip_steps + 2:
                #     tracking_step_cost, _ = self.eval_agent.evaluate_current_state(self.eval_agent.data_wrangler, self.mj_dataset, self.commands, self.tracking_cost_weights, i, debug=True)
                #     self.tracking_error.append(tracking_step_cost.item())

                #mujoco.mj_forward(self.mj_models[0], self.mj_data)
                # print the xpose and xquat over the last 10 steps
                if self.camera_kwargs == None:
                    self.mj_camera.update_camera(
                        lookat=self.mj_data.qpos[:3],
                        distance=1.4,
                        azimuth=90.0, 
                        elevation=0.0
                    )
                else:
                    self.mj_camera.update_camera(
                        lookat=self.mj_data.qpos[:3],
                            **self.camera_kwargs)

                frame = self.mj_camera.get_rgb_image()
                if i > skip_steps + 4:
                    if self.video_text:
                        inference_time_ms = self.inference_times[-1] * 1000
                        total_violation = self.total_violation
                        est_ = self.est_mse
                        text = f"Cumulative cost: {self.total_cost:.2f} | Compute time: {inference_time_ms:.2f} ms"
                        #text_line_1 = f"Cumulative cost: {self.total_cost:.2f} | Constraint violations > 0.01: {total_violation:.0f}"
                        #text_line_2 = f"State-estimation error {est_:.4f} | Compute time: {inference_time_ms:.2f} ms"
                        image_pil = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image_pil)
                        font_path = OPEN_SANS_REGULAR_TTF
                        font = ImageFont.truetype(font_path, 20)
                        x = frame.shape[1] -  475
                        y = frame.shape[0] -  35
                        draw.text((x, y), text, fill=(255, 255, 255), font=font)
                        # x = frame.shape[1] -  550
                        # y = frame.shape[0] -  35
                        # draw.text((x, y), text_line_1, fill=(255, 255, 255), font=font)
                        # x_ = frame.shape[1] -  550
                        # y_ = frame.shape[0] - 65
                        # draw.text((x_, y_), text_line_2, fill=(255, 255, 255), font=font)
                        frame = np.array(image_pil)
                self.frames.append(frame)
                


    
    def threaded_mj_step(self, state, ctrl, initial_state, sensor_data, n_envs=None):
        """
        Perform rollouts in parallel using a thread pool.

        Args:
            state (np.ndarray): Array to store the results of the rollouts.
            ctrl (np.ndarray): Control actions for the rollouts.
            initial_state (np.ndarray): Initial states for the rollouts.
            sensor_data (np.ndarray): Array to store the sensor data for the rollouts.
            num_workers (int): Number of parallel threads to use.
            nstep (int): Number of steps in each rollout.
        """


        futures = []
        if n_envs is None:
            n_envs = self.num_envs
        for i in range(n_envs):
            futures.append(
                self.executor.submit(
                    self.call_mj_step,
                    self.mj_models[i],
                    initial_state[i][None, ...],
                    ctrl[i][None, ...],
                    state[i, :][None, ...],
                    sensor_data[i, :][None, ...]
                )
            )

        for future in concurrent.futures.as_completed(futures):
            future.result()

    def call_mj_step(self, mj_model, initial_state, ctrl, state, sensor_data):
        """
        Perform a rollout of the model given the initial state and control actions.

        Args:
            initial_state (np.ndarray): Initial state of the model.
            ctrl (np.ndarray): Control actions to apply during the rollout.
            state (np.ndarray): State array to store the results of the rollout.
        """
        rollout.rollout(mj_model, self.thread_local.data, skip_checks=True,
                        nroll=state.shape[0], nstep=state.shape[1],
                        initial_state=initial_state, control=ctrl, state=state, sensordata=sensor_data)

        mj_data = self.thread_local.data  # Assuming the data is stored in thread-local data

        # Check for mjWARN_BADQACC warning
        if mj_data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0:
            self.DIVERGED = True

    def pre_step_callback(self):
        pass

    def post_step_callback(self):
        pass

    def post_agent_update_callback(self):
        pass

        
    def reset_robot(self):

        

        mj_models_tmp = []
        spawn_pos_tmp = []
        for i in range(self.num_envs):
            mj_model_new, spawn_pos_new = self.new_mj_model()
            mj_models_tmp.append(mj_model_new)
            spawn_pos_tmp.append(spawn_pos_new)

        self.mj_models = mj_models_tmp
        self.spawn_pos = np.array(spawn_pos_tmp)

        self.qpos, self.qvel = self.reset_states()

        self.agent.reset()
        self.reset_commands()

    @abstractmethod
    def reset_states(self):
        pass

    @abstractmethod
    def reset_commands(self):
        pass

    @abstractmethod
    def new_mj_model(self):
        pass

    @abstractmethod
    def set_weights(self):
        pass

    @abstractmethod
    def set_evaluation_commands(self):
        pass

    @abstractmethod
    def set_evaluation_weights(self):
        pass

    @abstractmethod
    def set_evaluation_mj_model(self):
        pass

    @abstractmethod
    def set_evaluation_task_names(self):
        pass

    @abstractmethod
    def set_evaluation_state(self):
        """
        Set the initial state for evaluation.
        This method should be implemented to set the state of the robot for evaluation.
        """
        pass

    def callback_before_eval_agent(self, step):
        pass

        
