from Common.Go2_dog import Go2StatesStruct, Go2ActionsStruct, Go2ObservationsStruct
import jax
from jax.tree_util import register_pytree_node_class
from Training import DataWrangler

@jax.vmap
def get_states(mj_dataset):
    """
    Computes the dynamics of the Go2 robot given the MJ dataset and time.

    Args:
        mj_dataset: The MJ dataset containing the sensor data.

    Returns:
        An instance of Go2StatesStruct containing the sensor trajectory.
    """
    sensors = Go2StatesStruct(
        v_x=mj_dataset.sensor_trajectory.v_x,
        v_y=mj_dataset.sensor_trajectory.v_y,
        v_z=mj_dataset.sensor_trajectory.v_z,
        v_roll=mj_dataset.sensor_trajectory.v_roll,
        v_pitch=mj_dataset.sensor_trajectory.v_pitch,
        v_yaw=mj_dataset.sensor_trajectory.v_yaw,
        z=mj_dataset.sensor_trajectory.sdf_z,
        q_fl_hip=mj_dataset.sensor_trajectory.q_fl_hip,
        q_fl_knee=mj_dataset.sensor_trajectory.q_fl_knee,
        q_fl_ankle=mj_dataset.sensor_trajectory.q_fl_ankle,
        q_fr_hip=mj_dataset.sensor_trajectory.q_fr_hip,
        q_fr_knee=mj_dataset.sensor_trajectory.q_fr_knee,
        q_fr_ankle=mj_dataset.sensor_trajectory.q_fr_ankle,
        q_rl_hip=mj_dataset.sensor_trajectory.q_rl_hip,
        q_rl_knee=mj_dataset.sensor_trajectory.q_rl_knee,
        q_rl_ankle=mj_dataset.sensor_trajectory.q_rl_ankle,
        q_rr_hip=mj_dataset.sensor_trajectory.q_rr_hip,
        q_rr_knee=mj_dataset.sensor_trajectory.q_rr_knee,
        q_rr_ankle=mj_dataset.sensor_trajectory.q_rr_ankle,
        v_fl_hip=mj_dataset.sensor_trajectory.v_fl_hip,
        v_fl_knee=mj_dataset.sensor_trajectory.v_fl_knee,
        v_fl_ankle=mj_dataset.sensor_trajectory.v_fl_ankle,
        v_fr_hip=mj_dataset.sensor_trajectory.v_fr_hip,
        v_fr_knee=mj_dataset.sensor_trajectory.v_fr_knee,
        v_fr_ankle=mj_dataset.sensor_trajectory.v_fr_ankle,
        v_rl_hip=mj_dataset.sensor_trajectory.v_rl_hip,
        v_rl_knee=mj_dataset.sensor_trajectory.v_rl_knee,
        v_rl_ankle=mj_dataset.sensor_trajectory.v_rl_ankle,
        v_rr_hip=mj_dataset.sensor_trajectory.v_rr_hip,
        v_rr_knee=mj_dataset.sensor_trajectory.v_rr_knee,
        v_rr_ankle=mj_dataset.sensor_trajectory.v_rr_ankle,
        _quat_w=mj_dataset.sensor_trajectory.quat_w,
        _quat_x=mj_dataset.sensor_trajectory.quat_x,
        _quat_y=mj_dataset.sensor_trajectory.quat_y,
        _quat_z=mj_dataset.sensor_trajectory.quat_z,
        sdf_base_main=mj_dataset.sensor_trajectory.sdf_base_main,
        sdf_base_head_bottom=mj_dataset.sensor_trajectory.sdf_base_head_bottom,
        sdf_base_head_top=mj_dataset.sensor_trajectory.sdf_base_head_top,
        sdf_fl_foot=mj_dataset.sensor_trajectory.sdf_fl_foot,
        sdf_fl_shank_top=mj_dataset.sensor_trajectory.sdf_fl_shank_top,
        sdf_fl_shank_bottom=mj_dataset.sensor_trajectory.sdf_fl_shank_bottom,
        sdf_fl_thigh=mj_dataset.sensor_trajectory.sdf_fl_thigh,
        sdf_fl_hip=mj_dataset.sensor_trajectory.sdf_fl_hip,
        sdf_fr_foot=mj_dataset.sensor_trajectory.sdf_fr_foot,
        sdf_fr_shank_top=mj_dataset.sensor_trajectory.sdf_fr_shank_top,
        sdf_fr_shank_bottom=mj_dataset.sensor_trajectory.sdf_fr_shank_bottom,
        sdf_fr_thigh=mj_dataset.sensor_trajectory.sdf_fr_thigh,
        sdf_fr_hip=mj_dataset.sensor_trajectory.sdf_fr_hip,
        sdf_rl_foot=mj_dataset.sensor_trajectory.sdf_rl_foot,
        sdf_rl_shank_top=mj_dataset.sensor_trajectory.sdf_rl_shank_top,
        sdf_rl_shank_bottom=mj_dataset.sensor_trajectory.sdf_rl_shank_bottom,
        sdf_rl_thigh=mj_dataset.sensor_trajectory.sdf_rl_thigh,
        sdf_rl_hip=mj_dataset.sensor_trajectory.sdf_rl_hip,
        sdf_rr_foot=mj_dataset.sensor_trajectory.sdf_rr_foot,
        sdf_rr_shank_top=mj_dataset.sensor_trajectory.sdf_rr_shank_top,
        sdf_rr_shank_bottom=mj_dataset.sensor_trajectory.sdf_rr_shank_bottom,
        sdf_rr_thigh=mj_dataset.sensor_trajectory.sdf_rr_thigh,
        sdf_rr_hip=mj_dataset.sensor_trajectory.sdf_rr_hip
    )
    
    return sensors

@jax.vmap
def get_actions(mj_dataset):
    """
    Returns the action trajector from the MJ dataset.

    Args:
        mj_dataset: The MJ dataset containing the action trajectory.

    Returns:
        An instance of Go2ActionsStruct containing the action trajectory.
    """
    actions = Go2ActionsStruct(
        fl_hip=mj_dataset.action_trajectory.fl_hip,
        fl_knee=mj_dataset.action_trajectory.fl_knee,
        fl_ankle=mj_dataset.action_trajectory.fl_ankle,
        fr_hip=mj_dataset.action_trajectory.fr_hip,
        fr_knee=mj_dataset.action_trajectory.fr_knee,
        fr_ankle=mj_dataset.action_trajectory.fr_ankle,
        rl_hip=mj_dataset.action_trajectory.rl_hip,
        rl_knee=mj_dataset.action_trajectory.rl_knee,
        rl_ankle=mj_dataset.action_trajectory.rl_ankle,
        rr_hip=mj_dataset.action_trajectory.rr_hip,
        rr_knee=mj_dataset.action_trajectory.rr_knee,
        rr_ankle=mj_dataset.action_trajectory.rr_ankle)
    
    return actions

@jax.vmap
def get_observations(mj_dataset):
    """
    Computes the dynamics of the Go2 robot given the MJ dataset and time.

    Args:
        mj_dataset: The MJ dataset containing the sensor data.

    Returns:
        An instance of Go2StatesStruct containing the sensor trajectory.
    """
    obs = Go2ObservationsStruct(
        a_x=mj_dataset.sensor_trajectory.a_x,
        a_y=mj_dataset.sensor_trajectory.a_y,
        a_z=mj_dataset.sensor_trajectory.a_z,
        v_roll=mj_dataset.sensor_trajectory.v_roll,
        v_pitch=mj_dataset.sensor_trajectory.v_pitch,
        v_yaw=mj_dataset.sensor_trajectory.v_yaw,
        q_fl_hip=mj_dataset.sensor_trajectory.q_fl_hip,
        q_fl_knee=mj_dataset.sensor_trajectory.q_fl_knee,
        q_fl_ankle=mj_dataset.sensor_trajectory.q_fl_ankle,
        q_fr_hip=mj_dataset.sensor_trajectory.q_fr_hip,
        q_fr_knee=mj_dataset.sensor_trajectory.q_fr_knee,
        q_fr_ankle=mj_dataset.sensor_trajectory.q_fr_ankle,
        q_rl_hip=mj_dataset.sensor_trajectory.q_rl_hip,
        q_rl_knee=mj_dataset.sensor_trajectory.q_rl_knee,
        q_rl_ankle=mj_dataset.sensor_trajectory.q_rl_ankle,
        q_rr_hip=mj_dataset.sensor_trajectory.q_rr_hip,
        q_rr_knee=mj_dataset.sensor_trajectory.q_rr_knee,
        q_rr_ankle=mj_dataset.sensor_trajectory.q_rr_ankle,
        v_fl_hip=mj_dataset.sensor_trajectory.v_fl_hip,
        v_fl_knee=mj_dataset.sensor_trajectory.v_fl_knee,
        v_fl_ankle=mj_dataset.sensor_trajectory.v_fl_ankle,
        v_fr_hip=mj_dataset.sensor_trajectory.v_fr_hip,
        v_fr_knee=mj_dataset.sensor_trajectory.v_fr_knee,
        v_fr_ankle=mj_dataset.sensor_trajectory.v_fr_ankle,
        v_rl_hip=mj_dataset.sensor_trajectory.v_rl_hip,
        v_rl_knee=mj_dataset.sensor_trajectory.v_rl_knee,
        v_rl_ankle=mj_dataset.sensor_trajectory.v_rl_ankle,
        v_rr_hip=mj_dataset.sensor_trajectory.v_rr_hip,
        v_rr_knee=mj_dataset.sensor_trajectory.v_rr_knee,
        v_rr_ankle=mj_dataset.sensor_trajectory.v_rr_ankle,
        _quat_w=mj_dataset.sensor_trajectory.quat_w,
        _quat_x=mj_dataset.sensor_trajectory.quat_x,
        _quat_y=mj_dataset.sensor_trajectory.quat_y,
        _quat_z=mj_dataset.sensor_trajectory.quat_z, 
        qd_fl_hip=mj_dataset.action_trajectory.fl_hip,
        qd_fl_knee=mj_dataset.action_trajectory.fl_knee,
        qd_fl_ankle=mj_dataset.action_trajectory.fl_ankle,
        qd_fr_hip=mj_dataset.action_trajectory.fr_hip,
        qd_fr_knee=mj_dataset.action_trajectory.fr_knee,
        qd_fr_ankle=mj_dataset.action_trajectory.fr_ankle,
        qd_rl_hip=mj_dataset.action_trajectory.rl_hip,
        qd_rl_knee=mj_dataset.action_trajectory.rl_knee,
        qd_rl_ankle=mj_dataset.action_trajectory.rl_ankle,
        qd_rr_hip=mj_dataset.action_trajectory.rr_hip,
        qd_rr_knee=mj_dataset.action_trajectory.rr_knee,
        qd_rr_ankle=mj_dataset.action_trajectory.rr_ankle,
    )
    

    obs.qd_fl_hip = obs.qd_fl_hip.at[-1].set(0.0)
    obs.qd_fl_knee = obs.qd_fl_knee.at[-1].set(0.0)
    obs.qd_fl_ankle = obs.qd_fl_ankle.at[-1].set(0.0)
    obs.qd_fr_hip = obs.qd_fr_hip.at[-1].set(0.0)
    obs.qd_fr_knee = obs.qd_fr_knee.at[-1].set(0.0)
    obs.qd_fr_ankle = obs.qd_fr_ankle.at[-1].set(0.0)
    obs.qd_rl_hip = obs.qd_rl_hip.at[-1].set(0.0)
    obs.qd_rl_knee = obs.qd_rl_knee.at[-1].set(0.0)
    obs.qd_rl_ankle = obs.qd_rl_ankle.at[-1].set(0.0)
    obs.qd_rr_hip = obs.qd_rr_hip.at[-1].set(0.0)
    obs.qd_rr_knee = obs.qd_rr_knee.at[-1].set(0.0)
    obs.qd_rr_ankle = obs.qd_rr_ankle.at[-1].set(0.0)
    return obs


@register_pytree_node_class
class Go2DataWrangler(DataWrangler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_states(self, mj_dataset):
        return get_states(mj_dataset)
    
    def get_actions(self, mj_dataset):
        return get_actions(mj_dataset)
    
    def get_observations(self, mj_dataset):
        return get_observations(mj_dataset)