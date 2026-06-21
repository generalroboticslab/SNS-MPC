from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from Common.utils import SO3_6d_vector_from_quat


@register_dataclass
@dataclass
class Go2ObservationsStruct:

    a_x: jnp.ndarray
    a_y: jnp.ndarray
    a_z: jnp.ndarray

    v_roll: jnp.ndarray
    v_pitch: jnp.ndarray
    v_yaw: jnp.ndarray

    _quat_w: jnp.ndarray
    _quat_x: jnp.ndarray
    _quat_y: jnp.ndarray
    _quat_z: jnp.ndarray

    qd_fl_hip: jnp.ndarray
    qd_fl_knee: jnp.ndarray
    qd_fl_ankle: jnp.ndarray

    qd_fr_hip: jnp.ndarray
    qd_fr_knee: jnp.ndarray
    qd_fr_ankle: jnp.ndarray

    qd_rl_hip: jnp.ndarray
    qd_rl_knee: jnp.ndarray
    qd_rl_ankle: jnp.ndarray

    qd_rr_hip: jnp.ndarray
    qd_rr_knee: jnp.ndarray
    qd_rr_ankle: jnp.ndarray

    q_fl_hip: jnp.ndarray
    q_fl_knee: jnp.ndarray
    q_fl_ankle: jnp.ndarray

    q_fr_hip: jnp.ndarray
    q_fr_knee: jnp.ndarray
    q_fr_ankle: jnp.ndarray

    q_rl_hip: jnp.ndarray
    q_rl_knee: jnp.ndarray
    q_rl_ankle: jnp.ndarray

    q_rr_hip: jnp.ndarray
    q_rr_knee: jnp.ndarray
    q_rr_ankle: jnp.ndarray

    v_fl_hip: jnp.ndarray
    v_fl_knee: jnp.ndarray
    v_fl_ankle: jnp.ndarray

    v_fr_hip: jnp.ndarray
    v_fr_knee: jnp.ndarray
    v_fr_ankle: jnp.ndarray

    v_rl_hip: jnp.ndarray
    v_rl_knee: jnp.ndarray
    v_rl_ankle: jnp.ndarray

    v_rr_hip: jnp.ndarray
    v_rr_knee: jnp.ndarray
    v_rr_ankle: jnp.ndarray

    @property
    def _quaternion(self):
        return jnp.concatenate([self._quat_w,
                                self._quat_x,
                                self._quat_y,
                                self._quat_z], axis=-1)

    @property
    def v11(self):
        """first element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 0][..., None]
    
    @property
    def v12(self):
        """second element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 1][..., None]
    
    @property
    def v13(self):
        """third element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 2][..., None]
    
    @property
    def v21(self):
        """fourth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 3][..., None]
    
    @property
    def v22(self):
        """fifth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 4][..., None]
    
    @property
    def v23(self):
        """sixth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 5][..., None]
    

@register_dataclass
@dataclass
class Go2StatesStruct:

    z: jnp.ndarray

    v_x: jnp.ndarray
    v_y: jnp.ndarray
    v_z: jnp.ndarray

    v_roll: jnp.ndarray
    v_pitch: jnp.ndarray
    v_yaw: jnp.ndarray

    _quat_w: jnp.ndarray
    _quat_x: jnp.ndarray
    _quat_y: jnp.ndarray
    _quat_z: jnp.ndarray

    q_fl_hip: jnp.ndarray
    q_fl_knee: jnp.ndarray
    q_fl_ankle: jnp.ndarray

    q_fr_hip: jnp.ndarray
    q_fr_knee: jnp.ndarray
    q_fr_ankle: jnp.ndarray

    q_rl_hip: jnp.ndarray
    q_rl_knee: jnp.ndarray
    q_rl_ankle: jnp.ndarray

    q_rr_hip: jnp.ndarray
    q_rr_knee: jnp.ndarray
    q_rr_ankle: jnp.ndarray

    v_fl_hip: jnp.ndarray
    v_fl_knee: jnp.ndarray
    v_fl_ankle: jnp.ndarray

    v_fr_hip: jnp.ndarray
    v_fr_knee: jnp.ndarray
    v_fr_ankle: jnp.ndarray

    v_rl_hip: jnp.ndarray
    v_rl_knee: jnp.ndarray
    v_rl_ankle: jnp.ndarray

    v_rr_hip: jnp.ndarray
    v_rr_knee: jnp.ndarray
    v_rr_ankle: jnp.ndarray

    sdf_base_main: jnp.ndarray
    sdf_base_head_top: jnp.ndarray
    sdf_base_head_bottom: jnp.ndarray
    
    sdf_fl_foot: jnp.ndarray
    sdf_fl_shank_top: jnp.ndarray
    sdf_fl_shank_bottom: jnp.ndarray
    sdf_fl_thigh: jnp.ndarray
    sdf_fl_hip: jnp.ndarray

    sdf_fr_foot: jnp.ndarray
    sdf_fr_shank_top: jnp.ndarray
    sdf_fr_shank_bottom: jnp.ndarray
    sdf_fr_thigh: jnp.ndarray
    sdf_fr_hip: jnp.ndarray

    sdf_rl_foot: jnp.ndarray
    sdf_rl_shank_top: jnp.ndarray
    sdf_rl_shank_bottom: jnp.ndarray
    sdf_rl_thigh: jnp.ndarray
    sdf_rl_hip: jnp.ndarray

    sdf_rr_foot: jnp.ndarray
    sdf_rr_shank_top: jnp.ndarray
    sdf_rr_shank_bottom: jnp.ndarray
    sdf_rr_thigh: jnp.ndarray
    sdf_rr_hip: jnp.ndarray

    @property
    def _quaternion(self):
        return jnp.concatenate([self._quat_w,
                                self._quat_x,
                                self._quat_y,
                                self._quat_z], axis=-1)

    @property
    def v11(self):
        """first element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 0][..., None]
    
    @property
    def v12(self):
        """second element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 1][..., None]
    
    @property
    def v13(self):
        """third element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 2][..., None]
    
    @property
    def v21(self):
        """fourth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 3][..., None]
    
    @property
    def v22(self):
        """fifth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 4][..., None]
    
    @property
    def v23(self):
        """sixth element of 6d vector from quaternion."""
        shape = self._quaternion.shape
        quat = jnp.reshape(self._quaternion, (-1, 4))
        newshape = shape[:-1] + (6,)
        sixd = jax.vmap(SO3_6d_vector_from_quat, in_axes=(0))(quat).reshape(newshape)
        return sixd[..., 5][..., None]
    

    
@register_dataclass
@dataclass
class Go2ActionsStruct:
    """action space for the go2 robot."""

    fl_hip: jnp.ndarray
    fl_knee: jnp.ndarray
    fl_ankle: jnp.ndarray

    fr_hip: jnp.ndarray
    fr_knee: jnp.ndarray
    fr_ankle: jnp.ndarray

    rl_hip: jnp.ndarray
    rl_knee: jnp.ndarray
    rl_ankle: jnp.ndarray

    rr_hip: jnp.ndarray
    rr_knee: jnp.ndarray
    rr_ankle: jnp.ndarray

@register_dataclass
@dataclass
class Go2CommandsStruct:
    """Reference commands for the go2 robot."""

    v_x: jnp.ndarray
    v_y: jnp.ndarray

    v_yaw: jnp.ndarray

    z: jnp.ndarray

    _roll: jnp.ndarray
    _pitch: jnp.ndarray
    _yaw: jnp.ndarray

    yaw_invariant: jnp.ndarray = jnp.array(0.0)  # yaw invariant command nullifies the yaw component of the rotation tracking
    # max_z: jnp.ndarray = jnp.array(0.05)  # difference from commanded z to actual z
    # max_log_R: jnp.ndarray = jnp.array(0.2)

    pitch_err_lb: jnp.ndarray = jnp.array(0.2) 
    roll_err_lb: jnp.ndarray = jnp.array(0.2) 
    yaw_err_lb: jnp.ndarray = jnp.array(0.2)

    pitch_err_ub: jnp.ndarray = jnp.array(0.2)
    roll_err_ub: jnp.ndarray = jnp.array(0.2)
    yaw_err_ub: jnp.ndarray = jnp.array(0.2)


    z_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for z error
    z_err_ub: jnp.ndarray = jnp.array(0.05)

    v_x_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for x velocity error
    v_x_err_ub: jnp.ndarray = jnp.array(0.05)

    v_y_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for y velocity error
    v_y_err_ub: jnp.ndarray = jnp.array(0.05)

    v_z_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for z velocity error
    v_z_err_ub: jnp.ndarray = jnp.array(0.05)

    v_roll_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for roll velocity error
    v_roll_err_ub: jnp.ndarray = jnp.array(0.05)

    v_pitch_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for pitch velocity error
    v_pitch_err_ub: jnp.ndarray = jnp.array(0.05)

    v_yaw_err_lb: jnp.ndarray = jnp.array(0.05)  # lower bound for yaw velocity error
    v_yaw_err_ub: jnp.ndarray = jnp.array(0.05)

    # max_v_xyz: jnp.ndarray = jnp.array(0.4)
    # max_v_rpy: jnp.ndarray = jnp.array(0.4)
    # seconds
    global_time: jnp.ndarray = jnp.array(0.0)

    # unitless (1.0 is stand/always in contact)
    duty_ratio: jnp.ndarray =  jnp.array(0.45) #jnp.array(0.75) #jnp.array(0.45) #jnp.array(1.0) #

    # Hz
    cadence: jnp.ndarray = jnp.array(2.0) #jnp.array(1.0) #jnp.array(2.0) #jnp.array(3.5) 

    # meters
    swing_height: jnp.ndarray = jnp.array(0.06) # walk jnp.array(0.04) # stairs jnp.array(0.1) # normal jnp.array(0.06)

    # unitless (0.0 for all is stand)
    # fl_phase: jnp.ndarray = jnp.array(0.0)
    # fr_phase: jnp.ndarray = jnp.array(0.5)
    # rl_phase: jnp.ndarray = jnp.array(0.75)
    # rr_phase: jnp.ndarray = jnp.array(0.25)

    fl_phase: jnp.ndarray = jnp.array(0.0)
    fr_phase: jnp.ndarray = jnp.array(0.5)
    rl_phase: jnp.ndarray = jnp.array(0.5)
    rr_phase: jnp.ndarray = jnp.array(0.0)

    # fl_phase: jnp.ndarray = jnp.array(0.0)
    # fr_phase: jnp.ndarray = jnp.array(0.0)
    # rl_phase: jnp.ndarray = jnp.array(0.0)
    # rr_phase: jnp.ndarray = jnp.array(0.0)

    # gallop
    # fl_phase: jnp.ndarray = jnp.array(0.0)
    # fr_phase: jnp.ndarray = jnp.array(0.05)
    # rl_phase: jnp.ndarray = jnp.array(0.4)
    # rr_phase: jnp.ndarray = jnp.array(0.45)

    # fl_phase: jnp.ndarray = jnp.array(0.2)
    # fr_phase: jnp.ndarray = jnp.array(0.2)
    # rl_phase: jnp.ndarray = jnp.array(0.2)
    # rr_phase: jnp.ndarray = jnp.array(0.2)

    tripod: jnp.ndarray = jnp.array(0.0)  # 1.0 for tripod, 0.0 for quadruped

    # calculated later
    fl_height_target: jnp.ndarray = jnp.array(0.0)
    fr_height_target: jnp.ndarray = jnp.array(0.0)
    rl_height_target: jnp.ndarray = jnp.array(0.0)
    rr_height_target: jnp.ndarray = jnp.array(0.0)

    max_joint_v: jnp.ndarray = jnp.array(9.0)  # max joint velocity in rad/s

    max_foot_height: jnp.ndarray = jnp.array(0.15)  # max foot height in meters


    @property
    def quaternion(self):
        """Return quaternion (w, x, y, z) from roll, pitch, yaw."""
        roll = self._roll
        pitch = self._pitch
        yaw = self._yaw

        cr = jnp.cos(roll / 2)
        sr = jnp.sin(roll / 2)
        cp = jnp.cos(pitch / 2)
        sp = jnp.sin(pitch / 2)
        cy = jnp.cos(yaw / 2)
        sy = jnp.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        if roll.ndim == 0:
            quat = jnp.stack([w, x, y, z], axis=0)
        else:
            quat = jnp.concatenate([w, x, y, z], axis=-1)
        return quat





joint_constants = jnp.ones((12,))
joint_constants = joint_constants.at[::3].set(10)
joint_constants = joint_constants.at[1::3].set(1.0)
joint_constants = joint_constants.at[2::3].set(0.1)

@register_dataclass
@dataclass
class Go2CostResidualStruct:

    z: jnp.ndarray = jnp.ones((1,))

    log_R: jnp.ndarray = jnp.ones((3,))

    v_lin: jnp.ndarray = jnp.ones((3,))

    v_ang: jnp.ndarray = jnp.ones((3,))

    nom_joint_q: jnp.ndarray = jnp.ones((12,))

    nom_joint_v: jnp.ndarray = jnp.ones((12,))

    nom_torque: jnp.ndarray = jnp.ones((12,))

    gait: jnp.ndarray = jnp.ones((4,))

    mech_work: jnp.ndarray = jnp.ones((12,))

@register_dataclass
@dataclass
class Go2ConstraintResidualStruct:

    base_contact: jnp.ndarray = jnp.ones((3,))

    hip_contact: jnp.ndarray = jnp.ones((4,))
    thigh_contact: jnp.ndarray = jnp.ones((4,))
    shank_top_contact: jnp.ndarray = jnp.ones((4,))
    shank_bottom_contact: jnp.ndarray = jnp.ones((4,))

    hip_q_lb: jnp.ndarray = jnp.ones((4,))
    hip_q_ub: jnp.ndarray = jnp.ones((4,))
    knee_q_lb: jnp.ndarray = jnp.ones((4,))
    knee_q_ub: jnp.ndarray = jnp.ones((4,))
    ankle_q_lb: jnp.ndarray = jnp.ones((4,))
    ankle_q_ub: jnp.ndarray = jnp.ones((4,))

    torque_limits: jnp.ndarray = jnp.ones((24,))

    pd_limits: jnp.ndarray = jnp.ones((24,))
    
    stance: jnp.ndarray = jnp.ones((4,))

    swing: jnp.ndarray = jnp.ones((4,))

    max_foot_height: jnp.ndarray = jnp.ones((4,))

    hip_v: jnp.ndarray = jnp.ones((4,))
    knee_v: jnp.ndarray = jnp.ones((4,))
    ankle_v: jnp.ndarray = jnp.ones((4,))

    rpy_err_lb: jnp.ndarray = jnp.ones((3,))
    rpy_err_ub: jnp.ndarray = jnp.ones((3,))

    v_xyz_err_lb: jnp.ndarray = jnp.ones((3,))
    v_xyz_err_ub: jnp.ndarray = jnp.ones((3,))

    v_rpy_err_lb: jnp.ndarray = jnp.ones((3,))
    v_rpy_err_ub: jnp.ndarray = jnp.ones((3,))

    z_err_lb: jnp.ndarray = jnp.ones((1,))
    z_err_ub: jnp.ndarray = jnp.ones((1,))


@register_dataclass
@dataclass
class Go2GlobalCostResidualStruct:

    xy: jnp.ndarray = jnp.ones((2,))

