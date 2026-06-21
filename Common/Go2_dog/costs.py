
import jax
import jax.numpy as jnp
from Common.utils import *
from Common.Go2_dog import Go2CostResidualStruct, Go2ConstraintResidualStruct, Go2GlobalCostResidualStruct

nominal_joint_position = {"fl_hip": 0.0,
                    "fl_knee": 0.9,
                    "fl_ankle": -1.8,
                    "fr_hip": 0.0,
                    "fr_knee": 0.9,
                    "fr_ankle": -1.8,
                    "rl_hip": 0.0,
                    "rl_knee": 0.9,
                    "rl_ankle": -1.8,
                    "rr_hip": 0.0,
                    "rr_knee": 0.9,
                    "rr_ankle": -1.8}

locomotion_joint_limits_delta = {"fl_hip": (-3.14/6, 3.14/6),
                "fl_knee": (-3.14/4, 3.14/4),
                "fl_ankle": ((-2*3.14)/5, 3.14/4),
                "fr_hip": (-3.14/6, 3.14/6),
                "fr_knee": (-3.14/4, 3.14/4),
                "fr_ankle": ((-2*3.14)/5, 3.14/4),
                "rl_hip": (-3.14/6, 3.14/6),
                "rl_knee": (-3.14/4, 3.14/4),
                "rl_ankle": ((-2*3.14)/5, 3.14/4),
                "rr_hip": (-3.14/6, 3.14/6),
                "rr_knee": (-3.14/4, 3.14/4),
                "rr_ankle": ((-2*3.14)/5, 3.14/4)}



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

locomotion_joint_limits_ = {key: (val + locomotion_joint_limits_delta[key][0], val + locomotion_joint_limits_delta[key][1]) for key, val in nominal_joint_position.items()}
locomotion_joint_limits = {key: (jnp.clip(val[0], joint_limits[key][0], joint_limits[key][1]), jnp.clip(val[1], joint_limits[key][0], joint_limits[key][1])) for key, val in locomotion_joint_limits_.items()}

# buffer = 0.01
# joint_limits = {key: (val[0] + buffer, val[1] - buffer) for key, val in joint_limits_.items()}

# -24 to 24 Nm for each joint except ankle, which is -45 to 45
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

# buffer_torque = 1.0
# torque_limits = {key: (val[0] + buffer_torque, val[1] - buffer_torque) for key, val in torque_limits_.items()}

def inheritrange_bounds(joint_name, factor=1.5):
    q_min, q_max = joint_limits[joint_name]
    mid = 0.5 * (q_min + q_max)
    half_range = 0.5 * (q_max - q_min)
    relaxed_lb = mid - factor * half_range
    relaxed_ub = mid + factor * half_range
    return relaxed_lb, relaxed_ub

relaxed_bounds = {
    "fl_hip": inheritrange_bounds("fl_hip"),
    "fl_knee": inheritrange_bounds("fl_knee"),
    "fl_ankle": inheritrange_bounds("fl_ankle"),
    "fr_hip": inheritrange_bounds("fr_hip"),
    "fr_knee": inheritrange_bounds("fr_knee"),
    "fr_ankle": inheritrange_bounds("fr_ankle"),
    "rl_hip": inheritrange_bounds("rl_hip"),
    "rl_knee": inheritrange_bounds("rl_knee"),
    "rl_ankle": inheritrange_bounds("rl_ankle"),
    "rr_hip": inheritrange_bounds("rr_hip"),
    "rr_knee": inheritrange_bounds("rr_knee"),
    "rr_ankle": inheritrange_bounds("rr_ankle")
}

def hard_clipped_quadratic_left(x, threshold=0.1, scale=100.0):
    residual = jnp.maximum(threshold - x, 0.0)  # only nonzero when x < threshold
    return scale * residual ** 2

def hard_clipped_quadratic_right(x, threshold=0.1, scale=100.0):
    residual = jnp.maximum(x - threshold, 0.0)  # nonzero when x > threshold
    return scale * residual ** 2


# def residual_x_leq_ub(x, threshold=0.1):
#     """Residual is positive if x > threshold (i.e., violates x <= b)."""
#     return jnp.maximum(x - threshold, 0.0)

# def residual_x_geq_lb(x, threshold=0.1):
#     """Residual is positive if x < threshold (i.e., violates x >= b)."""
#     return jnp.maximum(threshold - x, 0.0)

def residual_x_leq_ub(x, threshold=0.1):
    """Residual is positive if x > threshold (i.e., violates x <= b)."""
    return x - threshold

def residual_x_geq_lb(x, threshold=0.1):
    """Residual is positive if x < threshold (i.e., violates x >= b)."""
    return threshold - x

# def clip_residual_left(x, threshold=0.1):
#     """Clip residual to zero if it is less than the threshold."""
#     return jnp.maximum(threshold - x, 0.0)

# def clip_residual_right(x, threshold=0.1):
#     """Clip residual to zero if it is greater than the threshold."""
#     return jnp.maximum(x - threshold, 0.0)

def torque(q, qd, v):
    """Compute the torque based on the joint position, velocity, and action."""
    return 25.0 * (q - qd) - 1.5 * v

######################################## Costs for Deployment ########################################################
def v_lin_residual(Go2States, Go2Actions, Go2Commands):
    v_x_residual = Go2States.v_x - Go2Commands.v_x
    v_y_residual = Go2States.v_y - Go2Commands.v_y
    v_z_residual = Go2States.v_z / 2
    tripod_scale = 1.0 * (1 - Go2Commands.tripod) + 5.0 * Go2Commands.tripod
    return jnp.array([v_x_residual, v_y_residual * tripod_scale, v_z_residual]).flatten()

def v_ang_residual(Go2States, Go2Actions, Go2Commands):
    v_roll_residual = Go2States.v_roll
    v_pitch_residual = Go2States.v_pitch
    v_yaw_residual = Go2States.v_yaw - Go2Commands.v_yaw
    tripod_scale = 1.0 * (1 - Go2Commands.tripod) + 5.0 * Go2Commands.tripod
    return jnp.array([v_roll_residual, v_pitch_residual, v_yaw_residual * tripod_scale]).flatten()

def z_residual(Go2States, Go2Actions, Go2Commands):
    z_residual = Go2States.z - Go2Commands.z
    return jnp.array([z_residual]).flatten()

def z_limit_residual(Go2States, Go2Actions, Go2Commands):
    z_residual = Go2States.z - Go2Commands.z
    z_residual = jnp.abs(jnp.array([z_residual]).flatten())
    return residual_x_leq_ub(z_residual, threshold=Go2Commands.max_z)

def v_lin_limit_residual(Go2States, Go2Actions, Go2Commands):
    v_x_residual = Go2States.v_x - Go2Commands.v_x
    v_y_residual = Go2States.v_y - Go2Commands.v_y
    v_z_residual = Go2States.v_z
    res = jnp.abs(jnp.array([v_x_residual, v_y_residual, v_z_residual]).flatten())
    return residual_x_leq_ub(res, threshold=Go2Commands.max_v_xyz)

def v_ang_limit_residual(Go2States, Go2Actions, Go2Commands):
    v_roll_residual = Go2States.v_roll
    v_pitch_residual = Go2States.v_pitch
    v_yaw_residual = Go2States.v_yaw - Go2Commands.v_yaw
    res = jnp.abs(jnp.array([v_roll_residual, v_pitch_residual, v_yaw_residual]).flatten())
    return residual_x_leq_ub(res, threshold=Go2Commands.max_v_rpy)

def nom_joint_q_residual(Go2States, Go2Actions, Go2Commands):
    q_fl_hip_target = -0.25 * Go2Commands.tripod
    q_fl_hip_residual = Go2States.q_fl_hip - q_fl_hip_target
    q_fr_hip_residual = Go2States.q_fr_hip
    q_rl_hip_residual = Go2States.q_rl_hip
    q_rr_hip_residual = Go2States.q_rr_hip
    q_fl_knee_residual = Go2States.q_fl_knee - 0.9
    q_fr_knee_residual = Go2States.q_fr_knee - 0.9
    q_rl_knee_residual = Go2States.q_rl_knee - 0.9
    q_rr_knee_residual = Go2States.q_rr_knee - 0.9
    q_fl_ankle_residual = Go2States.q_fl_ankle + 1.8
    q_fr_ankle_residual = Go2States.q_fr_ankle + 1.8
    q_rl_ankle_residual = Go2States.q_rl_ankle + 1.8
    q_rr_ankle_residual = Go2States.q_rr_ankle + 1.8

    return jnp.array([
        q_fl_hip_residual, q_fr_hip_residual, q_rl_hip_residual, q_rr_hip_residual,
        q_fl_knee_residual, q_fr_knee_residual, q_rl_knee_residual, q_rr_knee_residual,
        q_fl_ankle_residual, q_fr_ankle_residual, q_rl_ankle_residual, q_rr_ankle_residual
    ]).flatten()

def nom_joint_v_residual(Go2States, Go2Actions, Go2Commands):
    v_fl_hip_residual = Go2States.v_fl_hip
    v_fr_hip_residual = Go2States.v_fr_hip
    v_rl_hip_residual = Go2States.v_rl_hip
    v_rr_hip_residual = Go2States.v_rr_hip
    v_fl_knee_residual = Go2States.v_fl_knee
    v_fr_knee_residual = Go2States.v_fr_knee
    v_rl_knee_residual = Go2States.v_rl_knee
    v_rr_knee_residual = Go2States.v_rr_knee
    v_fl_ankle_residual = Go2States.v_fl_ankle
    v_fr_ankle_residual = Go2States.v_fr_ankle
    v_rl_ankle_residual = Go2States.v_rl_ankle
    v_rr_ankle_residual = Go2States.v_rr_ankle

    return jnp.array([
        v_fl_hip_residual, v_fr_hip_residual, v_rl_hip_residual, v_rr_hip_residual,
        v_fl_knee_residual, v_fr_knee_residual, v_rl_knee_residual, v_rr_knee_residual,
        v_fl_ankle_residual, v_fr_ankle_residual, v_rl_ankle_residual, v_rr_ankle_residual
    ]).flatten()

# def nom_torque_residual(Go2States, Go2Actions, Go2Commands):
#     fl_hip_residual = torque(Go2States.q_fl_hip, Go2Actions.fl_hip, Go2States.v_fl_hip)
#     fr_hip_residual = torque(Go2States.q_fr_hip, Go2Actions.fr_hip, Go2States.v_fr_hip)
#     rl_hip_residual = torque(Go2States.q_rl_hip, Go2Actions.rl_hip, Go2States.v_rl_hip)
#     rr_hip_residual = torque(Go2States.q_rr_hip, Go2Actions.rr_hip, Go2States.v_rr_hip)
#     fl_knee_residual = torque(Go2States.q_fl_knee, Go2Actions.fl_knee, Go2States.v_fl_knee)
#     fr_knee_residual = torque(Go2States.q_fr_knee, Go2Actions.fr_knee, Go2States.v_fr_knee)
#     rl_knee_residual = torque(Go2States.q_rl_knee, Go2Actions.rl_knee, Go2States.v_rl_knee)
#     rr_knee_residual = torque(Go2States.q_rr_knee, Go2Actions.rr_knee, Go2States.v_rr_knee)
#     fl_ankle_residual = torque(Go2States.q_fl_ankle, Go2Actions.fl_ankle, Go2States.v_fl_ankle)
#     fr_ankle_residual = torque(Go2States.q_fr_ankle, Go2Actions.fr_ankle, Go2States.v_fr_ankle)
#     rl_ankle_residual = torque(Go2States.q_rl_ankle, Go2Actions.rl_ankle, Go2States.v_rl_ankle)
#     rr_ankle_residual = torque(Go2States.q_rr_ankle, Go2Actions.rr_ankle, Go2States.v_rr_ankle)
#     return jnp.array([
#         fl_hip_residual, fr_hip_residual, rl_hip_residual, rr_hip_residual,
#         fl_knee_residual, fr_knee_residual, rl_knee_residual, rr_knee_residual,
#         fl_ankle_residual, fr_ankle_residual, rl_ankle_residual, rr_ankle_residual
#     ]).flatten()

def nom_torque_residual(Go2States, Go2Actions, Go2Commands):
    return jnp.array([Go2Actions.fl_hip, Go2Actions.fr_hip, Go2Actions.rl_hip, Go2Actions.rr_hip,
                      Go2Actions.fl_knee, Go2Actions.fr_knee, Go2Actions.rl_knee, Go2Actions.rr_knee,
                      Go2Actions.fl_ankle, Go2Actions.fr_ankle, Go2Actions.rl_ankle, Go2Actions.rr_ankle
    ]).flatten()

def pos_mech_work_residual(Go2States, Go2Actions, Go2Commands):
    fl_hip_residual = Go2Actions.fl_hip * Go2States.v_fl_hip
    fr_hip_residual = Go2Actions.fr_hip * Go2States.v_fr_hip
    rl_hip_residual = Go2Actions.rl_hip * Go2States.v_rl_hip
    rr_hip_residual = Go2Actions.rr_hip * Go2States.v_rr_hip
    fl_knee_residual = Go2Actions.fl_knee * Go2States.v_fl_knee
    fr_knee_residual = Go2Actions.fr_knee * Go2States.v_fr_knee
    rl_knee_residual = Go2Actions.rl_knee * Go2States.v_rl_knee
    rr_knee_residual = Go2Actions.rr_knee * Go2States.v_rr_knee
    fl_ankle_residual = Go2Actions.fl_ankle * Go2States.v_fl_ankle
    fr_ankle_residual = Go2Actions.fr_ankle * Go2States.v_fr_ankle
    rl_ankle_residual = Go2Actions.rl_ankle * Go2States.v_rl_ankle
    rr_ankle_residual = Go2Actions.rr_ankle * Go2States.v_rr_ankle

    residual = jnp.array([
        fl_hip_residual, fr_hip_residual, rl_hip_residual, rr_hip_residual,
        fl_knee_residual, fr_knee_residual, rl_knee_residual, rr_knee_residual,
        fl_ankle_residual, fr_ankle_residual, rl_ankle_residual, rr_ankle_residual
    ]).flatten()
    return jnp.clip(residual, a_min=0.0, a_max=None)  # Ensure non-negative work

def torque_limits_residual(Go2States, Go2Actions, Go2Commands):
    tau_fl_hip = torque(Go2States.q_fl_hip, Go2Actions.fl_hip, Go2States.v_fl_hip)
    tau_fr_hip = torque(Go2States.q_fr_hip, Go2Actions.fr_hip, Go2States.v_fr_hip)
    tau_rl_hip = torque(Go2States.q_rl_hip, Go2Actions.rl_hip, Go2States.v_rl_hip)
    tau_rr_hip = torque(Go2States.q_rr_hip, Go2Actions.rr_hip, Go2States.v_rr_hip)
    tau_fl_knee = torque(Go2States.q_fl_knee, Go2Actions.fl_knee, Go2States.v_fl_knee)
    tau_fr_knee = torque(Go2States.q_fr_knee, Go2Actions.fr_knee, Go2States.v_fr_knee)
    tau_rl_knee = torque(Go2States.q_rl_knee, Go2Actions.rl_knee, Go2States.v_rl_knee)
    tau_rr_knee = torque(Go2States.q_rr_knee, Go2Actions.rr_knee, Go2States.v_rr_knee)
    tau_fl_ankle = torque(Go2States.q_fl_ankle, Go2Actions.fl_ankle, Go2States.v_fl_ankle)
    tau_fr_ankle = torque(Go2States.q_fr_ankle, Go2Actions.fr_ankle, Go2States.v_fr_ankle)
    tau_rl_ankle = torque(Go2States.q_rl_ankle, Go2Actions.rl_ankle, Go2States.v_rl_ankle)
    tau_rr_ankle = torque(Go2States.q_rr_ankle, Go2Actions.rr_ankle, Go2States.v_rr_ankle)

    fl_ankle_residual_lb = residual_x_geq_lb(tau_fl_ankle, threshold=torque_limits["fl_ankle"][0])
    fl_ankle_residual_ub = residual_x_leq_ub(tau_fl_ankle, threshold=torque_limits["fl_ankle"][1])
    fl_knee_residual_lb = residual_x_geq_lb(tau_fl_knee, threshold=torque_limits["fl_knee"][0])
    fl_knee_residual_ub = residual_x_leq_ub(tau_fl_knee, threshold=torque_limits["fl_knee"][1])
    fl_hip_residual_lb = residual_x_geq_lb(tau_fl_hip, threshold=torque_limits["fl_hip"][0])
    fl_hip_residual_ub = residual_x_leq_ub(tau_fl_hip, threshold=torque_limits["fl_hip"][1])
    fr_ankle_residual_lb = residual_x_geq_lb(tau_fr_ankle, threshold=torque_limits["fr_ankle"][0])
    fr_ankle_residual_ub = residual_x_leq_ub(tau_fr_ankle, threshold=torque_limits["fr_ankle"][1])
    fr_knee_residual_lb = residual_x_geq_lb(tau_fr_knee, threshold=torque_limits["fr_knee"][0])
    fr_knee_residual_ub = residual_x_leq_ub(tau_fr_knee, threshold=torque_limits["fr_knee"][1])
    fr_hip_residual_lb = residual_x_geq_lb(tau_fr_hip, threshold=torque_limits["fr_hip"][0])
    fr_hip_residual_ub = residual_x_leq_ub(tau_fr_hip, threshold=torque_limits["fr_hip"][1])
    rl_ankle_residual_lb = residual_x_geq_lb(tau_rl_ankle, threshold=torque_limits["rl_ankle"][0])
    rl_ankle_residual_ub = residual_x_leq_ub(tau_rl_ankle, threshold=torque_limits["rl_ankle"][1])
    rl_knee_residual_lb = residual_x_geq_lb(tau_rl_knee, threshold=torque_limits["rl_knee"][0])
    rl_knee_residual_ub = residual_x_leq_ub(tau_rl_knee, threshold=torque_limits["rl_knee"][1])
    rl_hip_residual_lb = residual_x_geq_lb(tau_rl_hip, threshold=torque_limits["rl_hip"][0])
    rl_hip_residual_ub = residual_x_leq_ub(tau_rl_hip, threshold=torque_limits["rl_hip"][1])
    rr_ankle_residual_lb = residual_x_geq_lb(tau_rr_ankle, threshold=torque_limits["rr_ankle"][0])
    rr_ankle_residual_ub = residual_x_leq_ub(tau_rr_ankle, threshold=torque_limits["rr_ankle"][1])
    rr_knee_residual_lb = residual_x_geq_lb(tau_rr_knee, threshold=torque_limits["rr_knee"][0])
    rr_knee_residual_ub = residual_x_leq_ub(tau_rr_knee, threshold=torque_limits["rr_knee"][1])
    rr_hip_residual_lb = residual_x_geq_lb(tau_rr_hip, threshold=torque_limits["rr_hip"][0])
    rr_hip_residual_ub = residual_x_leq_ub(tau_rr_hip, threshold=torque_limits["rr_hip"][1])

    return jnp.array([
        fl_hip_residual_lb, fl_hip_residual_ub,
        fl_knee_residual_lb, fl_knee_residual_ub,
        fl_ankle_residual_lb, fl_ankle_residual_ub,
        fr_hip_residual_lb, fr_hip_residual_ub,
        fr_knee_residual_lb, fr_knee_residual_ub,
        fr_ankle_residual_lb, fr_ankle_residual_ub,
        rl_hip_residual_lb, rl_hip_residual_ub,
        rl_knee_residual_lb, rl_knee_residual_ub,
        rl_ankle_residual_lb, rl_ankle_residual_ub,
        rr_hip_residual_lb, rr_hip_residual_ub,
        rr_knee_residual_lb, rr_knee_residual_ub,
        rr_ankle_residual_lb, rr_ankle_residual_ub
    ]).flatten()

def joint_limits_residual(Go2States, Go2Actions, Go2Commands):
    fl_hip_residual_lb = residual_x_geq_lb(Go2States.q_fl_hip, threshold=joint_limits["fl_hip"][0])
    fl_hip_residual_ub = residual_x_leq_ub(Go2States.q_fl_hip, threshold=joint_limits["fl_hip"][1])
    fl_knee_residual_lb = residual_x_geq_lb(Go2States.q_fl_knee, threshold=joint_limits["fl_knee"][0])
    fl_knee_residual_ub = residual_x_leq_ub(Go2States.q_fl_knee, threshold=joint_limits["fl_knee"][1])
    fl_ankle_residual_lb = residual_x_geq_lb(Go2States.q_fl_ankle, threshold=joint_limits["fl_ankle"][0])
    fl_ankle_residual_ub = residual_x_leq_ub(Go2States.q_fl_ankle, threshold=joint_limits["fl_ankle"][1])
    fr_hip_residual_lb = residual_x_geq_lb(Go2States.q_fr_hip, threshold=joint_limits["fr_hip"][0])
    fr_hip_residual_ub = residual_x_leq_ub(Go2States.q_fr_hip, threshold=joint_limits["fr_hip"][1])
    fr_knee_residual_lb = residual_x_geq_lb(Go2States.q_fr_knee, threshold=joint_limits["fr_knee"][0])
    fr_knee_residual_ub = residual_x_leq_ub(Go2States.q_fr_knee, threshold=joint_limits["fr_knee"][1])
    fr_ankle_residual_lb = residual_x_geq_lb(Go2States.q_fr_ankle, threshold=joint_limits["fr_ankle"][0])
    fr_ankle_residual_ub = residual_x_leq_ub(Go2States.q_fr_ankle, threshold=joint_limits["fr_ankle"][1])
    rl_hip_residual_lb = residual_x_geq_lb(Go2States.q_rl_hip, threshold=joint_limits["rl_hip"][0])
    rl_hip_residual_ub = residual_x_leq_ub(Go2States.q_rl_hip, threshold=joint_limits["rl_hip"][1])
    rl_knee_residual_lb = residual_x_geq_lb(Go2States.q_rl_knee, threshold=joint_limits["rl_knee"][0])
    rl_knee_residual_ub = residual_x_leq_ub(Go2States.q_rl_knee, threshold=joint_limits["rl_knee"][1])
    rl_ankle_residual_lb = residual_x_geq_lb(Go2States.q_rl_ankle, threshold=joint_limits["rl_ankle"][0])
    rl_ankle_residual_ub = residual_x_leq_ub(Go2States.q_rl_ankle, threshold=joint_limits["rl_ankle"][1])
    rr_hip_residual_lb = residual_x_geq_lb(Go2States.q_rr_hip, threshold=joint_limits["rr_hip"][0])
    rr_hip_residual_ub = residual_x_leq_ub(Go2States.q_rr_hip, threshold=joint_limits["rr_hip"][1])
    rr_knee_residual_lb = residual_x_geq_lb(Go2States.q_rr_knee, threshold=joint_limits["rr_knee"][0])
    rr_knee_residual_ub = residual_x_leq_ub(Go2States.q_rr_knee, threshold=joint_limits["rr_knee"][1])
    rr_ankle_residual_lb = residual_x_geq_lb(Go2States.q_rr_ankle, threshold=joint_limits["rr_ankle"][0])
    rr_ankle_residual_ub = residual_x_leq_ub(Go2States.q_rr_ankle, threshold=joint_limits["rr_ankle"][1])

    return jnp.array([
        fl_hip_residual_lb, fl_hip_residual_ub,
        fl_knee_residual_lb, fl_knee_residual_ub,
        fl_ankle_residual_lb, fl_ankle_residual_ub,
        fr_hip_residual_lb, fr_hip_residual_ub,
        fr_knee_residual_lb, fr_knee_residual_ub,
        fr_ankle_residual_lb, fr_ankle_residual_ub,
        rl_hip_residual_lb, rl_hip_residual_ub,
        rl_knee_residual_lb, rl_knee_residual_ub,
        rl_ankle_residual_lb, rl_ankle_residual_ub,
        rr_hip_residual_lb, rr_hip_residual_ub,
        rr_knee_residual_lb, rr_knee_residual_ub,
        rr_ankle_residual_lb, rr_ankle_residual_ub
    ]).flatten()


def locomotion_hip_q_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion hip joint position lower bounds."""
    fl_hip_residual = residual_x_geq_lb(Go2States.q_fl_hip, threshold=locomotion_joint_limits["fl_hip"][0])
    fr_hip_residual = residual_x_geq_lb(Go2States.q_fr_hip, threshold=locomotion_joint_limits["fr_hip"][0])
    rl_hip_residual = residual_x_geq_lb(Go2States.q_rl_hip, threshold=locomotion_joint_limits["rl_hip"][0])
    rr_hip_residual = residual_x_geq_lb(Go2States.q_rr_hip, threshold=locomotion_joint_limits["rr_hip"][0])
    return jnp.array([fl_hip_residual, fr_hip_residual, rl_hip_residual, rr_hip_residual]).flatten()

def locomotion_hip_q_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion hip joint position upper bounds."""
    tripod_fctr = 1 - Go2Commands.tripod
    fl_hip_threshold = locomotion_joint_limits["fl_hip"][1] * tripod_fctr + locomotion_joint_limits["fl_hip"][1] * 0.25 * Go2Commands.tripod
    fl_hip_residual = residual_x_leq_ub(Go2States.q_fl_hip, threshold=fl_hip_threshold)
    fr_hip_residual = residual_x_leq_ub(Go2States.q_fr_hip, threshold=locomotion_joint_limits["fr_hip"][1])
    rl_hip_residual = residual_x_leq_ub(Go2States.q_rl_hip, threshold=locomotion_joint_limits["rl_hip"][1])
    rr_hip_residual = residual_x_leq_ub(Go2States.q_rr_hip, threshold=locomotion_joint_limits["rr_hip"][1])
    return jnp.array([fl_hip_residual, fr_hip_residual, rl_hip_residual, rr_hip_residual]).flatten()

def locomotion_knee_q_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion knee joint position lower bounds."""
    fl_knee_residual = residual_x_geq_lb(Go2States.q_fl_knee, threshold=locomotion_joint_limits["fl_knee"][0])
    fr_knee_residual = residual_x_geq_lb(Go2States.q_fr_knee, threshold=locomotion_joint_limits["fr_knee"][0])
    rl_knee_residual = residual_x_geq_lb(Go2States.q_rl_knee, threshold=locomotion_joint_limits["rl_knee"][0])
    rr_knee_residual = residual_x_geq_lb(Go2States.q_rr_knee, threshold=locomotion_joint_limits["rr_knee"][0])
    return jnp.array([fl_knee_residual, fr_knee_residual, rl_knee_residual, rr_knee_residual]).flatten()

def locomotion_knee_q_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion knee joint position upper bounds."""
    fl_knee_residual = residual_x_leq_ub(Go2States.q_fl_knee, threshold=locomotion_joint_limits["fl_knee"][1])
    fr_knee_residual = residual_x_leq_ub(Go2States.q_fr_knee, threshold=locomotion_joint_limits["fr_knee"][1])
    rl_knee_residual = residual_x_leq_ub(Go2States.q_rl_knee, threshold=locomotion_joint_limits["rl_knee"][1])
    rr_knee_residual = residual_x_leq_ub(Go2States.q_rr_knee, threshold=locomotion_joint_limits["rr_knee"][1])
    return jnp.array([fl_knee_residual, fr_knee_residual, rl_knee_residual, rr_knee_residual]).flatten()

def locomotion_ankle_q_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion ankle joint position lower bounds."""
    fl_ankle_residual = residual_x_geq_lb(Go2States.q_fl_ankle, threshold=locomotion_joint_limits["fl_ankle"][0])
    fr_ankle_residual = residual_x_geq_lb(Go2States.q_fr_ankle, threshold=locomotion_joint_limits["fr_ankle"][0])
    rl_ankle_residual = residual_x_geq_lb(Go2States.q_rl_ankle, threshold=locomotion_joint_limits["rl_ankle"][0])
    rr_ankle_residual = residual_x_geq_lb(Go2States.q_rr_ankle, threshold=locomotion_joint_limits["rr_ankle"][0])
    return jnp.array([fl_ankle_residual, fr_ankle_residual, rl_ankle_residual, rr_ankle_residual]).flatten()

def locomotion_ankle_q_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the locomotion ankle joint position upper bounds."""
    fl_ankle_residual = residual_x_leq_ub(Go2States.q_fl_ankle, threshold=locomotion_joint_limits["fl_ankle"][1])
    fr_ankle_residual = residual_x_leq_ub(Go2States.q_fr_ankle, threshold=locomotion_joint_limits["fr_ankle"][1])
    rl_ankle_residual = residual_x_leq_ub(Go2States.q_rl_ankle, threshold=locomotion_joint_limits["rl_ankle"][1])
    rr_ankle_residual = residual_x_leq_ub(Go2States.q_rr_ankle, threshold=locomotion_joint_limits["rr_ankle"][1])
    return jnp.array([fl_ankle_residual, fr_ankle_residual, rl_ankle_residual, rr_ankle_residual]).flatten()

def pd_limits_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for the PD limits."""
    fl_hip_residual_lb = residual_x_geq_lb(Go2Actions.fl_hip, threshold=relaxed_bounds["fl_hip"][0])
    fl_hip_residual_ub = residual_x_leq_ub(Go2Actions.fl_hip, threshold=relaxed_bounds["fl_hip"][1])
    fl_knee_residual_lb = residual_x_geq_lb(Go2Actions.fl_knee, threshold=relaxed_bounds["fl_knee"][0])
    fl_knee_residual_ub = residual_x_leq_ub(Go2Actions.fl_knee, threshold=relaxed_bounds["fl_knee"][1])
    fl_ankle_residual_lb = residual_x_geq_lb(Go2Actions.fl_ankle, threshold=relaxed_bounds["fl_ankle"][0])
    fl_ankle_residual_ub = residual_x_leq_ub(Go2Actions.fl_ankle, threshold=relaxed_bounds["fl_ankle"][1])
    fr_hip_residual_lb = residual_x_geq_lb(Go2Actions.fr_hip, threshold=relaxed_bounds["fr_hip"][0])
    fr_hip_residual_ub = residual_x_leq_ub(Go2Actions.fr_hip, threshold=relaxed_bounds["fr_hip"][1])
    fr_knee_residual_lb = residual_x_geq_lb(Go2Actions.fr_knee, threshold=relaxed_bounds["fr_knee"][0])
    fr_knee_residual_ub = residual_x_leq_ub(Go2Actions.fr_knee, threshold=relaxed_bounds["fr_knee"][1])
    fr_ankle_residual_lb = residual_x_geq_lb(Go2Actions.fr_ankle, threshold=relaxed_bounds["fr_ankle"][0])
    fr_ankle_residual_ub = residual_x_leq_ub(Go2Actions.fr_ankle, threshold=relaxed_bounds["fr_ankle"][1])
    rl_hip_residual_lb = residual_x_geq_lb(Go2Actions.rl_hip, threshold=relaxed_bounds["rl_hip"][0])
    rl_hip_residual_ub = residual_x_leq_ub(Go2Actions.rl_hip, threshold=relaxed_bounds["rl_hip"][1])
    rl_knee_residual_lb = residual_x_geq_lb(Go2Actions.rl_knee, threshold=relaxed_bounds["rl_knee"][0])
    rl_knee_residual_ub = residual_x_leq_ub(Go2Actions.rl_knee, threshold=relaxed_bounds["rl_knee"][1])
    rl_ankle_residual_lb = residual_x_geq_lb(Go2Actions.rl_ankle, threshold=relaxed_bounds["rl_ankle"][0])
    rl_ankle_residual_ub = residual_x_leq_ub(Go2Actions.rl_ankle, threshold=relaxed_bounds["rl_ankle"][1])
    rr_hip_residual_lb = residual_x_geq_lb(Go2Actions.rr_hip, threshold=relaxed_bounds["rr_hip"][0])
    rr_hip_residual_ub = residual_x_leq_ub(Go2Actions.rr_hip, threshold=relaxed_bounds["rr_hip"][1])
    rr_knee_residual_lb = residual_x_geq_lb(Go2Actions.rr_knee, threshold=relaxed_bounds["rr_knee"][0])
    rr_knee_residual_ub = residual_x_leq_ub(Go2Actions.rr_knee, threshold=relaxed_bounds["rr_knee"][1])
    rr_ankle_residual_lb = residual_x_geq_lb(Go2Actions.rr_ankle, threshold=relaxed_bounds["rr_ankle"][0])
    rr_ankle_residual_ub = residual_x_leq_ub(Go2Actions.rr_ankle, threshold=relaxed_bounds["rr_ankle"][1])

    return jnp.array([
        fl_hip_residual_lb, fl_hip_residual_ub,
        fl_knee_residual_lb, fl_knee_residual_ub,
        fl_ankle_residual_lb, fl_ankle_residual_ub,
        fr_hip_residual_lb, fr_hip_residual_ub,
        fr_knee_residual_lb, fr_knee_residual_ub,
        fr_ankle_residual_lb, fr_ankle_residual_ub,
        rl_hip_residual_lb, rl_hip_residual_ub,
        rl_knee_residual_lb, rl_knee_residual_ub,
        rl_ankle_residual_lb, rl_ankle_residual_ub,
        rr_hip_residual_lb, rr_hip_residual_ub,
        rr_knee_residual_lb, rr_knee_residual_ub,
        rr_ankle_residual_lb, rr_ankle_residual_ub
    ]).flatten()

def joint_velocity_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for joint velocities."""
    bound = Go2Commands.max_joint_v
    v_fl_hip_lb_residual = residual_x_geq_lb(Go2States.v_fl_hip, threshold=-bound)
    v_fl_hip_ub_residual = residual_x_leq_ub(Go2States.v_fl_hip, threshold=bound)
    v_fr_hip_lb_residual = residual_x_geq_lb(Go2States.v_fr_hip, threshold=-bound)
    v_fr_hip_ub_residual = residual_x_leq_ub(Go2States.v_fr_hip, threshold=bound)
    v_rl_hip_lb_residual = residual_x_geq_lb(Go2States.v_rl_hip, threshold=-bound)
    v_rl_hip_ub_residual = residual_x_leq_ub(Go2States.v_rl_hip, threshold=bound)
    v_rr_hip_lb_residual = residual_x_geq_lb(Go2States.v_rr_hip, threshold=-bound)
    v_rr_hip_ub_residual = residual_x_leq_ub(Go2States.v_rr_hip, threshold=bound)
    v_fl_knee_lb_residual = residual_x_geq_lb(Go2States.v_fl_knee, threshold=-bound)
    v_fl_knee_ub_residual = residual_x_leq_ub(Go2States.v_fl_knee, threshold=bound)
    v_fr_knee_lb_residual = residual_x_geq_lb(Go2States.v_fr_knee, threshold=-bound)
    v_fr_knee_ub_residual = residual_x_leq_ub(Go2States.v_fr_knee, threshold=bound)
    v_rl_knee_lb_residual = residual_x_geq_lb(Go2States.v_rl_knee, threshold=-bound)
    v_rl_knee_ub_residual = residual_x_leq_ub(Go2States.v_rl_knee, threshold=bound)
    v_rr_knee_lb_residual = residual_x_geq_lb(Go2States.v_rr_knee, threshold=-bound)
    v_rr_knee_ub_residual = residual_x_leq_ub(Go2States.v_rr_knee, threshold=bound)
    v_fl_ankle_lb_residual = residual_x_geq_lb(Go2States.v_fl_ankle, threshold=-bound)
    v_fl_ankle_ub_residual = residual_x_leq_ub(Go2States.v_fl_ankle, threshold=bound)
    v_fr_ankle_lb_residual = residual_x_geq_lb(Go2States.v_fr_ankle, threshold=-bound)
    v_fr_ankle_ub_residual = residual_x_leq_ub(Go2States.v_fr_ankle, threshold=bound)
    v_rl_ankle_lb_residual = residual_x_geq_lb(Go2States.v_rl_ankle, threshold=-bound)
    v_rl_ankle_ub_residual = residual_x_leq_ub(Go2States.v_rl_ankle, threshold=bound)
    v_rr_ankle_lb_residual = residual_x_geq_lb(Go2States.v_rr_ankle, threshold=-bound)
    v_rr_ankle_ub_residual = residual_x_leq_ub(Go2States.v_rr_ankle, threshold=bound)

    return jnp.array([
        v_fl_hip_lb_residual, v_fl_hip_ub_residual,
        v_fr_hip_lb_residual, v_fr_hip_ub_residual,
        v_rl_hip_lb_residual, v_rl_hip_ub_residual,
        v_rr_hip_lb_residual, v_rr_hip_ub_residual,
        v_fl_knee_lb_residual, v_fl_knee_ub_residual,
        v_fr_knee_lb_residual, v_fr_knee_ub_residual,
        v_rl_knee_lb_residual, v_rl_knee_ub_residual,
        v_rr_knee_lb_residual, v_rr_knee_ub_residual,
        v_fl_ankle_lb_residual, v_fl_ankle_ub_residual,
        v_fr_ankle_lb_residual, v_fr_ankle_ub_residual,
        v_rl_ankle_lb_residual, v_rl_ankle_ub_residual,
        v_rr_ankle_lb_residual, v_rr_ankle_ub_residual,
    ]).flatten()

def hip_v_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for hip velocities."""
    bound = Go2Commands.max_joint_v
    v_fl_hip_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fl_hip), threshold=bound)
    v_fr_hip_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fr_hip), threshold=bound)
    v_rl_hip_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rl_hip), threshold=bound)
    v_rr_hip_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rr_hip), threshold=bound)

    return jnp.array([
        v_fl_hip_residual, v_fr_hip_residual,
        v_rl_hip_residual, v_rr_hip_residual
    ]).flatten()

def knee_v_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for knee velocities."""
    bound = Go2Commands.max_joint_v
    v_fl_knee_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fl_knee), threshold=bound)
    v_fr_knee_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fr_knee), threshold=bound)
    v_rl_knee_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rl_knee), threshold=bound)
    v_rr_knee_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rr_knee), threshold=bound)

    return jnp.array([
        v_fl_knee_residual, v_fr_knee_residual,
        v_rl_knee_residual, v_rr_knee_residual
    ]).flatten()

def ankle_v_residual(Go2States, Go2Actions, Go2Commands):
    """Compute the residuals for ankle velocities."""
    bound = Go2Commands.max_joint_v
    v_fl_ankle_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fl_ankle), threshold=bound)
    v_fr_ankle_residual = residual_x_leq_ub(jnp.abs(Go2States.v_fr_ankle), threshold=bound)
    v_rl_ankle_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rl_ankle), threshold=bound)
    v_rr_ankle_residual = residual_x_leq_ub(jnp.abs(Go2States.v_rr_ankle), threshold=bound)

    return jnp.array([
        v_fl_ankle_residual, v_fr_ankle_residual,
        v_rl_ankle_residual, v_rr_ankle_residual
    ]).flatten()

def base_contact_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the base contact."""
    base_main_residual = residual_x_geq_lb(Go2States.sdf_base_main, threshold=0.03)
    base_head_top_residual = residual_x_geq_lb(Go2States.sdf_base_head_top, threshold=0.03)
    base_head_bottom_residual = residual_x_geq_lb(Go2States.sdf_base_head_bottom, threshold=0.03)

    return jnp.array([
        base_main_residual, base_head_top_residual, base_head_bottom_residual
    ]).flatten()

def hip_contact_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for hip contact."""
    sdf_fl_hip_residual = residual_x_geq_lb(Go2States.sdf_fl_hip, threshold=0.03)
    sdf_fr_hip_residual = residual_x_geq_lb(Go2States.sdf_fr_hip, threshold=0.03)
    sdf_rl_hip_residual = residual_x_geq_lb(Go2States.sdf_rl_hip, threshold=0.03)
    sdf_rr_hip_residual = residual_x_geq_lb(Go2States.sdf_rr_hip, threshold=0.03)

    return jnp.array([
        sdf_fl_hip_residual, sdf_fr_hip_residual,
        sdf_rl_hip_residual, sdf_rr_hip_residual
    ]).flatten()

def thigh_contact_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for thigh contact."""
    sdf_fl_thigh_residual = residual_x_geq_lb(Go2States.sdf_fl_thigh, threshold=0.02)
    sdf_fr_thigh_residual = residual_x_geq_lb(Go2States.sdf_fr_thigh, threshold=0.02)
    sdf_rl_thigh_residual = residual_x_geq_lb(Go2States.sdf_rl_thigh, threshold=0.02)
    sdf_rr_thigh_residual = residual_x_geq_lb(Go2States.sdf_rr_thigh, threshold=0.02)

    return jnp.array([
        sdf_fl_thigh_residual, sdf_fr_thigh_residual,
        sdf_rl_thigh_residual, sdf_rr_thigh_residual
    ]).flatten()

def shank_top_contact_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for shank top contact."""
    sdf_fl_shank_top_residual = residual_x_geq_lb(Go2States.sdf_fl_shank_top, threshold=0.02)
    sdf_fr_shank_top_residual = residual_x_geq_lb(Go2States.sdf_fr_shank_top, threshold=0.02)
    sdf_rl_shank_top_residual = residual_x_geq_lb(Go2States.sdf_rl_shank_top, threshold=0.02)
    sdf_rr_shank_top_residual = residual_x_geq_lb(Go2States.sdf_rr_shank_top, threshold=0.02)

    return jnp.array([
        sdf_fl_shank_top_residual, sdf_fr_shank_top_residual,
        sdf_rl_shank_top_residual, sdf_rr_shank_top_residual
    ]).flatten()

def shank_bottom_contact_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for shank bottom contact."""
    sdf_fl_shank_bottom_residual = residual_x_geq_lb(Go2States.sdf_fl_shank_bottom, threshold=0.0)
    sdf_fr_shank_bottom_residual = residual_x_geq_lb(Go2States.sdf_fr_shank_bottom, threshold=0.0)
    sdf_rl_shank_bottom_residual = residual_x_geq_lb(Go2States.sdf_rl_shank_bottom, threshold=0.0)
    sdf_rr_shank_bottom_residual = residual_x_geq_lb(Go2States.sdf_rr_shank_bottom, threshold=0.0)

    return jnp.array([
        sdf_fl_shank_bottom_residual, sdf_fr_shank_bottom_residual,
        sdf_rl_shank_bottom_residual, sdf_rr_shank_bottom_residual
    ]).flatten()
   
def log_R_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the cost based on the logarithmic map of the orientation."""
    R_cmd = rotation_matrix_from_quat(Go2Commands.quaternion)
    SO3_6d = jnp.concatenate([Go2States.v11, Go2States.v12, Go2States.v13,
                            Go2States.v21, Go2States.v22, Go2States.v23], axis=-1)
    
    R = R_from_SO3_6d_vector(SO3_6d)
    R_rel = R_cmd.T @ R
    log_R_rel = log_so3(R_rel)
    residual = log_R_rel.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])

    # tripod_fctr = 1 * (1 - Go2Commands.tripod[0]) + 0.5 * Go2Commands.tripod[0]
    # residual = residual.flatten().at[2].multiply(tripod_fctr)
    return residual

def log_R_limit_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the cost based on the logarithmic map of the orientation."""
    R_cmd = rotation_matrix_from_quat(Go2Commands.quaternion)
    SO3_6d = jnp.concatenate([Go2States.v11, Go2States.v12, Go2States.v13,
                            Go2States.v21, Go2States.v22, Go2States.v23], axis=-1)
    
    R = R_from_SO3_6d_vector(SO3_6d)
    R_rel = R_cmd.T @ R
    log_R_rel = log_so3(R_rel)
    residual = log_R_rel.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])
    norm =  jnp.linalg.norm(residual)
    return residual_x_leq_ub(norm, threshold=Go2Commands.max_log_R).flatten()


def rpy_err_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the cost based on the logarithmic map of the orientation."""
    R_cmd = rotation_matrix_from_quat(Go2Commands.quaternion)
    SO3_6d = jnp.concatenate([Go2States.v11, Go2States.v12, Go2States.v13,
                            Go2States.v21, Go2States.v22, Go2States.v23], axis=-1)
    
    R = R_from_SO3_6d_vector(SO3_6d)
    R_rel = R_cmd.T @ R
    log_R_rel = log_so3(R_rel)
    residual = log_R_rel.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])
    roll_lb = residual_x_geq_lb(residual[0], threshold=Go2Commands.roll_err_lb)
    pitch_lb = residual_x_geq_lb(residual[1], threshold=Go2Commands.pitch_err_lb)
    yaw_lb = residual_x_geq_lb(residual[2], threshold=Go2Commands.yaw_err_lb)
    r_lb = jnp.array([roll_lb, pitch_lb, yaw_lb])
    return r_lb.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])

def rpy_err_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the cost based on the logarithmic map of the orientation."""
    R_cmd = rotation_matrix_from_quat(Go2Commands.quaternion)
    SO3_6d = jnp.concatenate([Go2States.v11, Go2States.v12, Go2States.v13,
                            Go2States.v21, Go2States.v22, Go2States.v23], axis=-1)
    
    R = R_from_SO3_6d_vector(SO3_6d)
    R_rel = R_cmd.T @ R
    log_R_rel = log_so3(R_rel)
    residual = log_R_rel.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])
    roll_ub = residual_x_leq_ub(residual[0], threshold=Go2Commands.roll_err_ub)
    pitch_ub = residual_x_leq_ub(residual[1], threshold=Go2Commands.pitch_err_ub)
    yaw_ub = residual_x_leq_ub(residual[2], threshold=Go2Commands.yaw_err_ub)
    r_ub = jnp.array([roll_ub, pitch_ub, yaw_ub])
    return r_ub.flatten().at[2].multiply(1 - Go2Commands.yaw_invariant[0])

def v_xyz_err_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the velocity in the x, y, and z directions."""
    v_x_res = Go2States.v_x - Go2Commands.v_x
    v_y_res = Go2States.v_y - Go2Commands.v_y
    v_z_res = Go2States.v_z 

    v_x_lb = residual_x_geq_lb(v_x_res, threshold=Go2Commands.v_x_err_lb)
    v_y_lb = residual_x_geq_lb(v_y_res, threshold=Go2Commands.v_y_err_lb)
    v_z_lb = residual_x_geq_lb(v_z_res, threshold=Go2Commands.v_z_err_lb)
    return jnp.array([v_x_lb, v_y_lb, v_z_lb]).flatten()

def v_xyz_err_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the base linear velocity upper bounds."""
    v_x_res = Go2States.v_x - Go2Commands.v_x
    v_y_res = Go2States.v_y - Go2Commands.v_y
    v_z_res = Go2States.v_z

    v_x_ub = residual_x_leq_ub(v_x_res, threshold=Go2Commands.v_x_err_ub)
    v_y_ub = residual_x_leq_ub(v_y_res, threshold=Go2Commands.v_y_err_ub)
    v_z_ub = residual_x_leq_ub(v_z_res, threshold=Go2Commands.v_z_err_ub)

    return jnp.array([v_x_ub, v_y_ub, v_z_ub]).flatten()


def v_rpy_err_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the base angular velocity lower bounds."""
    v_r_res = Go2States.v_roll
    v_p_res = Go2States.v_pitch
    v_y_res = Go2States.v_yaw - Go2Commands.v_yaw

    v_r_lb = residual_x_geq_lb(v_r_res, threshold=Go2Commands.v_roll_err_lb)
    v_p_lb = residual_x_geq_lb(v_p_res, threshold=Go2Commands.v_pitch_err_lb)
    v_y_lb = residual_x_geq_lb(v_y_res, threshold=Go2Commands.v_yaw_err_lb)

    return jnp.array([v_r_lb, v_p_lb, v_y_lb]).flatten()

def v_rpy_err_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the base angular velocity upper bounds."""
    v_r_res = Go2States.v_roll
    v_p_res = Go2States.v_pitch
    v_y_res = Go2States.v_yaw - Go2Commands.v_yaw

    v_r_ub = residual_x_leq_ub(v_r_res, threshold=Go2Commands.v_roll_err_ub)
    v_p_ub = residual_x_leq_ub(v_p_res, threshold=Go2Commands.v_pitch_err_ub)
    v_y_ub = residual_x_leq_ub(v_y_res, threshold=Go2Commands.v_yaw_err_ub)

    return jnp.array([v_r_ub, v_p_ub, v_y_ub]).flatten()

def z_err_lb_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the z position lower bounds."""
    z_res = Go2States.z - Go2Commands.z
    z_lb = residual_x_geq_lb(z_res, threshold=Go2Commands.z_err_lb)
    return z_lb.flatten()

def z_err_ub_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals for the z position upper bounds."""
    z_res = Go2States.z - Go2Commands.z
    z_ub = residual_x_leq_ub(z_res, threshold=Go2Commands.z_err_ub)
    return z_ub.flatten()


def gait_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals based on the gait of the robot."""


    
    
    tripod_fctr = 1 - Go2Commands.tripod
    fr_height_target = Go2Commands.fr_height_target * tripod_fctr + 0.1 * Go2Commands.tripod
    rr_foot_height_target = Go2Commands.rr_height_target * tripod_fctr + Go2Commands.rr_height_target * Go2Commands.tripod
    rl_foot_height_target = Go2Commands.rl_height_target * tripod_fctr + Go2Commands.rl_height_target * Go2Commands.tripod

    fl_foot_height_residual = Go2States.sdf_fl_foot - Go2Commands.fl_height_target
    fr_foot_height_residual = Go2States.sdf_fr_foot - fr_height_target
    rl_foot_height_residual = Go2States.sdf_rl_foot - rl_foot_height_target
    rr_foot_height_residual = Go2States.sdf_rr_foot - rr_foot_height_target

    fl_foot_height_residual = jnp.where(Go2Commands.fl_height_target > 0.0,
                                        fl_foot_height_residual,
                                        residual_x_leq_ub(fl_foot_height_residual, threshold=0.0))
    
    fr_foot_height_residual = jnp.where(Go2Commands.fr_height_target > 0.0,
                                        fr_foot_height_residual,
                                        residual_x_leq_ub(fr_foot_height_residual, threshold=0.0))
    
    rl_foot_height_residual = jnp.where(Go2Commands.rl_height_target > 0.0,
                                        rl_foot_height_residual,
                                        residual_x_leq_ub(rl_foot_height_residual, threshold=0.0))
    
    rr_foot_height_residual = jnp.where(Go2Commands.rr_height_target > 0.0,
                                        rr_foot_height_residual,
                                        residual_x_leq_ub(rr_foot_height_residual, threshold=0.0))


    tripod_scale = 1 * tripod_fctr + 5 * Go2Commands.tripod

    return jnp.array([
        fl_foot_height_residual, tripod_scale * fr_foot_height_residual,
        rl_foot_height_residual, rr_foot_height_residual
    ]).flatten()



def swing_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals based on the gait of the robot."""

    tripod_fctr = 1 - Go2Commands.tripod

    eps = 1e-3
    apex_fl = Go2Commands.fl_height_target - Go2Commands.swing_height
    apex_fr = Go2Commands.fr_height_target - Go2Commands.swing_height
    apex_rl = Go2Commands.rl_height_target - Go2Commands.swing_height
    apex_rr = Go2Commands.rr_height_target - Go2Commands.swing_height

    threshold_fl = jnp.where(jnp.abs(apex_fl) < eps, Go2Commands.swing_height, 0.0)
    threshold_fr = jnp.where(jnp.abs(apex_fr) < eps, Go2Commands.swing_height, 0.0)
    threshold_rl = jnp.where(jnp.abs(apex_rl) < eps, Go2Commands.swing_height, 0.0)
    threshold_rr = jnp.where(jnp.abs(apex_rr) < eps, Go2Commands.swing_height, 0.0)

    threshold_fr = threshold_fr * tripod_fctr + 0.08 * Go2Commands.tripod
    fl_foot_height_residual = Go2States.sdf_fl_foot - threshold_fl
    fr_foot_height_residual = Go2States.sdf_fr_foot - threshold_fr
    rl_foot_height_residual = Go2States.sdf_rl_foot - threshold_rl
    rr_foot_height_residual = Go2States.sdf_rr_foot - threshold_rr


    fl_foot_height_residual = jnp.where(threshold_fl > 0.0,
                                        residual_x_geq_lb(fl_foot_height_residual, threshold=0.0),
                                        0.0)
    fr_foot_height_residual = jnp.where(threshold_fr > 0.0,
                                        residual_x_geq_lb(fr_foot_height_residual, threshold=0.0),
                                        0.0)
    rl_foot_height_residual = jnp.where(threshold_rl > 0.0, 
                                        residual_x_geq_lb(rl_foot_height_residual, threshold=0.0),
                                        0.0)
    rr_foot_height_residual = jnp.where(threshold_rr > 0.0,
                                        residual_x_geq_lb(rr_foot_height_residual, threshold=0.0),
                                        0.0)
    return 10*jnp.array([
        fl_foot_height_residual, fr_foot_height_residual,
        rl_foot_height_residual, rr_foot_height_residual
    ]).flatten()


def stance_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals based on the gait of the robot."""
    fl_foot_height_residual = Go2States.sdf_fl_foot 
    fr_foot_height_residual = Go2States.sdf_fr_foot
    rl_foot_height_residual = Go2States.sdf_rl_foot
    rr_foot_height_residual = Go2States.sdf_rr_foot
    

    fl_foot_height_residual = jnp.where(Go2Commands.fl_height_target > 0.001,
                                        0.0,
                                        residual_x_leq_ub(fl_foot_height_residual, threshold=0.0))
    
    fr_foot_height_residual = jnp.where(Go2Commands.fr_height_target > 0.001,
                                        0.0,
                                        residual_x_leq_ub(fr_foot_height_residual, threshold=0.0))

    rl_foot_height_residual = jnp.where(Go2Commands.rl_height_target > 0.001,
                                        0.0,
                                        residual_x_leq_ub(rl_foot_height_residual, threshold=0.0))

    rr_foot_height_residual = jnp.where(Go2Commands.rr_height_target > 0.001,
                                        0.0,
                                        residual_x_leq_ub(rr_foot_height_residual, threshold=0.0))
    tripod_fctr = 1 - Go2Commands.tripod
    rl_foot_height_fctr = 1 * tripod_fctr + 10 * Go2Commands.tripod
    return jnp.array([
        fl_foot_height_residual, tripod_fctr*fr_foot_height_residual,
        rl_foot_height_residual * rl_foot_height_fctr, rr_foot_height_residual
    ]).flatten()


def max_foot_height_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the residuals based on the gait of the robot."""
    fl_foot_height_residual = Go2States.sdf_fl_foot - Go2Commands.max_foot_height
    fr_foot_height_residual = Go2States.sdf_fr_foot - Go2Commands.max_foot_height
    rl_foot_height_residual = Go2States.sdf_rl_foot - Go2Commands.max_foot_height
    rr_foot_height_residual = Go2States.sdf_rr_foot - Go2Commands.max_foot_height

    # fl_foot_height_residual = jnp.where(Go2Commands.fl_height_target > 0.0,
    #                                     0.0,
    #                                     residual_x_leq_ub(fl_foot_height_residual, threshold=0.0))
    
    # fr_foot_height_residual = jnp.where(Go2Commands.fr_height_target > 0.0,
    #                                     0.0,
    #                                     residual_x_leq_ub(fr_foot_height_residual, threshold=0.0))

    # rl_foot_height_residual = jnp.where(Go2Commands.rl_height_target > 0.0,
    #                                     0.0,
    #                                     residual_x_leq_ub(rl_foot_height_residual, threshold=0.0))

    # rr_foot_height_residual = jnp.where(Go2Commands.rr_height_target > 0.0,
    #                                     0.0,
    #                                     residual_x_leq_ub(rr_foot_height_residual, threshold=0.0))
    fl_foot_height_residual = residual_x_leq_ub(fl_foot_height_residual, threshold=0.0)
    fr_foot_height_residual = residual_x_leq_ub(fr_foot_height_residual, threshold=0.0)
    rl_foot_height_residual = residual_x_leq_ub(rl_foot_height_residual, threshold=0.0)
    rr_foot_height_residual = residual_x_leq_ub(rr_foot_height_residual, threshold=0.0)
    return jnp.array([
        fl_foot_height_residual, fr_foot_height_residual,
        rl_foot_height_residual, rr_foot_height_residual
    ]).flatten()


def gait_constraint_residual(Go2States, Go2Actions, Go2Commands):

    swing = swing_residual(Go2States, Go2Actions, Go2Commands)
    stance = stance_residual(Go2States, Go2Actions, Go2Commands)
    foot_height = max_foot_height_residual(Go2States, Go2Actions, Go2Commands)

    fl_residual = 10*swing[0] + stance[0] + foot_height[0]
    fr_residual = 10*swing[1] + stance[1] + foot_height[1]
    rl_residual = 10*swing[2] + stance[2] + foot_height[2]
    rr_residual = 10*swing[3] + stance[3] + foot_height[3]
    return jnp.array([
        fl_residual, fr_residual, rl_residual, rr_residual
    ]).flatten()

def xy_global_cost_residual(Go2States, Go2Actions, Go2Commands):
    """Computes the cost based on the global position error."""
    dt = 0.02
    x_pos_cmd = jnp.cumsum(Go2Commands.v_x * dt)
    y_pos_cmd = jnp.cumsum(Go2Commands.v_y * dt)

    x_pos = jnp.cumsum(Go2States.v_x * dt)
    y_pos = jnp.cumsum(Go2States.v_y * dt)
    x_err = x_pos - x_pos_cmd
    y_err = y_pos - y_pos_cmd
    return jnp.array([x_err, y_err]).flatten()


def relaxed_log_barrier(residual, delta, weight):
    """Standard relaxed log barrier for inequality constraints g(x) <= 0."""
    return jnp.where(
        residual < -delta,
        -jnp.log(-residual),
        -jnp.log(delta) + 0.5 * ((residual + 2 * delta) / delta) ** 2 - 0.5
    ) * weight


def step_cost_from_residuals(States, Actions, Commands, weights, debug=False):

    cost_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), cost_residual_tree)
    constraint_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), constraint_residual_tree)

    costs = jax.tree.map(lambda r, w: r * w, cost_residuals, weights.cost_weights)
    costs = jax.tree.map(lambda r, wr: jnp.dot(r, wr), costs, cost_residuals)

    violation = jax.tree.map(lambda r: jnp.maximum(r, 0.0), constraint_residuals)
    constraints = jax.tree.map(lambda r, w, relax: relaxed_log_barrier(r, relax, w), constraint_residuals, weights.constraint_weights, weights.constraint_relaxation)

    costs_flat, _ = jax.tree.flatten(costs)
    constraints_flat, _ = jax.tree.flatten(constraints)
    costs_flat = jnp.stack(costs_flat, axis=0)  

    constraints_flat = jnp.concatenate(constraints_flat)
    lagragian = jnp.sum(costs_flat) + jnp.sum(constraints_flat)

    if debug:
        return lagragian, (costs, violation)
    return lagragian

def residuals(States, Actions, Commands):
    """Compute the residuals."""
    cost_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), cost_residual_tree)
    constraint_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), constraint_residual_tree)
    
    cost_residuals, _ = jax.tree.flatten(cost_residuals)
    constraint_residuals, _ = jax.tree.flatten(constraint_residuals)
    cost_residuals = jnp.concatenate(cost_residuals)
    constraint_residuals = jnp.concatenate(constraint_residuals)
    return cost_residuals, constraint_residuals

def residuals_cauchy(States, Actions, Commands):
    """Compute the residuals."""
    cost_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), cost_residual_tree)
    constraint_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), constraint_residual_tree)
    cauchy_cost_residuals = jax.tree.map(lambda x: x, cost_residuals)

    cost_residuals_mask = jax.tree.map(lambda x: jnp.zeros_like(x), cost_residuals)
    cost_residuals_mask.nom_torque = jnp.ones_like(cost_residuals.nom_torque)
    cost_residuals_mask.mech_work = jnp.ones_like(cost_residuals.mech_work)

    cauchy_cost_residuals_mask = jax.tree.map(lambda x: jnp.ones_like(x), cost_residuals)
    cauchy_cost_residuals_mask.nom_torque = jnp.zeros_like(cost_residuals.nom_torque)
    cauchy_cost_residuals_mask.mech_work = jnp.zeros_like(cost_residuals.mech_work)

    cost_residuals, _ = jax.tree.flatten(cost_residuals)
    constraint_residuals, _ = jax.tree.flatten(constraint_residuals)
    cost_residuals = jnp.concatenate(cost_residuals)
    constraint_residuals = jnp.concatenate(constraint_residuals)

    cauchy_cost_residuals, _ = jax.tree.flatten(cauchy_cost_residuals)
    cauchy_cost_residuals = jnp.concatenate(cauchy_cost_residuals)

    return cost_residuals, constraint_residuals, cauchy_cost_residuals

def global_residuals(States, Actions, Commands):
    """Compute the residuals."""
    cost_residuals = jax.tree.map(lambda fn: fn(States, Actions, Commands), global_cost_residual_tree)
    
    cost_residuals, _ = jax.tree.flatten(cost_residuals)
    cost_residuals = jnp.concatenate(cost_residuals)
    return cost_residuals

def setup_weights():
    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 2e-3
    cost_weights.gait = cost_weights.gait * 0
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    # constraint_weights = Go2ConstraintResidualStruct()
    # constraint_weights.body_contact = constraint_weights.body_contact * 0
    # constraint_weights.pd_limits = constraint_weights.pd_limits * 0
    # constraint_weights.joint_limits = constraint_weights.joint_limits * 0
    # constraint_weights.torque_limits = constraint_weights.torque_limits * 0
    # constraint_weights.stance = constraint_weights.stance * 0
    # constraint_weights.swing = constraint_weights.swing * 0
    # constraint_weights.joint_velocity = constraint_weights.joint_velocity * 0


    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    # terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    # terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0
    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights

default_cost_weights = setup_weights()

def evaluation_weights():
    cost_weights = Go2CostResidualStruct()

    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 1.0 
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
    cost_weights.gait = cost_weights.gait * 2e3
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    #constraint_weights.torque_limits = constraint_weights.torque_limits + 1e-9 #1e-3

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 1e-2

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def trotting_evaluation_weights():
    cost_weights = Go2CostResidualStruct()

    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 1.0 
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
    cost_weights.gait = cost_weights.gait * 2e3
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    #constraint_weights.torque_limits = constraint_weights.torque_limits + 1e-9 #1e-3

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 1e-2

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


# def galloping_evaluation_weights():
#     cost_weights = Go2CostResidualStruct()

#     cost_weights.z = cost_weights.z * 5e2
#     cost_weights.log_R = cost_weights.log_R * 1e2
#     cost_weights.v_lin = cost_weights.v_lin * 5e1
#     cost_weights.v_ang = cost_weights.v_ang * 0.5 
#     cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0.0
#     cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
#     cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
#     cost_weights.gait = cost_weights.gait * 0.0 #2e3
#     cost_weights.mech_work = cost_weights.mech_work * 1e-3

#     constraint_weights = Go2ConstraintResidualStruct()
#     constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
#     #constraint_weights.torque_limits = constraint_weights.torque_limits + 1e-9 #1e-3
#     constraint_weights.base_contact = constraint_weights.base_contact + 1.0
#     constraint_weights.hip_contact = constraint_weights.hip_contact + 1.0
#     constraint_weights.thigh_contact = constraint_weights.thigh_contact + 1.0
#     constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 1.0
#     #constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact + 1e-6

#     constraint_weights.stance = constraint_weights.stance + 1.0
#     constraint_weights.swing = constraint_weights.swing + 1.0

#     # constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb + 1.0
#     # constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub + 1.0

#     constraint_weights.hip_q_lb = constraint_weights.hip_q_lb + 1.0
#     constraint_weights.hip_q_ub = constraint_weights.hip_q_ub + 1.0
#     constraint_weights.knee_q_lb = constraint_weights.knee_q_lb + 1.0
#     constraint_weights.knee_q_ub = constraint_weights.knee_q_ub + 1.0
#     constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb + 1.0
#     constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub + 1.0

#     constraint_relaxation = Go2ConstraintResidualStruct()
#     constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
#     #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01
#     constraint_relaxation.base_contact = constraint_relaxation.base_contact * 5e-4
#     constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 5e-4
#     constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 5e-4
#     constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 5e-4
#     # constraint_relaxation.shank_bottom_contact = constraint_relaxation.shank_bottom_contact * 1e-2


#     # constraint_relaxation.rpy_err_lb = constraint_relaxation.rpy_err_lb * 0.05
#     # constraint_relaxation.rpy_err_ub = constraint_relaxation.rpy_err_ub * 0.05

#     constraint_relaxation.stance = constraint_relaxation.stance * 0.05
#     constraint_relaxation.swing = constraint_relaxation.swing * 0.05

#     constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 0.08
#     constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 0.08
#     constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 0.08
#     constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 0.08
#     constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 0.08
#     constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 0.08

#     terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
#     terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

#     terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

#     terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

#     global_cost_weights = Go2GlobalCostResidualStruct()
#     global_cost_weights.xy = global_cost_weights.xy * 1e-2

#     cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
#     terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
#     constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
#     terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

#     return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def galloping_evaluation_weights():
    cost_weights = Go2CostResidualStruct()

    cost_weights.z = cost_weights.z * 5e2
    cost_weights.log_R = cost_weights.log_R * 5e2
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 0.5 
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
    cost_weights.gait = cost_weights.gait * 3e3
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    # constraint_weights.base_contact = constraint_weights.base_contact + 1.0
    # constraint_weights.hip_contact = constraint_weights.hip_contact + 1.0
    constraint_weights.thigh_contact = constraint_weights.thigh_contact + 1.0
    constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 1.0

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # constraint_relaxation.base_contact = constraint_relaxation.base_contact * 5e-4
    # constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 5e-4
    constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 5e-4
    constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 5e-4
    #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 1e-2

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def bounding_evaluation_weights():
    cost_weights = Go2CostResidualStruct()

    cost_weights.z = cost_weights.z * 1e3
    cost_weights.log_R = cost_weights.log_R * 5e2
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 0.5 
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 5e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
    cost_weights.gait = cost_weights.gait * 5e3 #5e3
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    # constraint_weights.base_contact = constraint_weights.base_contact + 1.0
    # constraint_weights.hip_contact = constraint_weights.hip_contact + 1.0
    constraint_weights.thigh_contact = constraint_weights.thigh_contact + 1.0
    constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 1.0

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # constraint_relaxation.base_contact = constraint_relaxation.base_contact * 5e-4
    # constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 5e-4
    constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 5e-4
    constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 5e-4
    #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 1e-2

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def tripod_evaluation_weights():
    cost_weights = Go2CostResidualStruct()

    cost_weights.z = cost_weights.z * 5e2
    cost_weights.log_R = cost_weights.log_R * 7e2
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 0.5 
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 5e-3
    cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
    cost_weights.gait = cost_weights.gait * 7e3
    cost_weights.mech_work = cost_weights.mech_work * 1e-3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    # constraint_weights.base_contact = constraint_weights.base_contact + 1.0
    # constraint_weights.hip_contact = constraint_weights.hip_contact + 1.0
    # constraint_weights.thigh_contact = constraint_weights.thigh_contact + 1.0
    # constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 1.0
    constraint_weights.z_err_ub = constraint_weights.z_err_ub + 1.0
    constraint_weights.z_err_lb = constraint_weights.z_err_lb + 1.0

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # constraint_relaxation.base_contact = constraint_relaxation.base_contact * 5e-4
    # # constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 5e-4
    # constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 5e-4
    # constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 5e-4
    #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01
    constraint_relaxation.z_err_ub = constraint_relaxation.z_err_ub * 0.01
    constraint_relaxation.z_err_lb = constraint_relaxation.z_err_lb * 0.01


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 1e-2

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights



# def tripod_evaluation_weights():
#     cost_weights = Go2CostResidualStruct()

#     cost_weights.z = cost_weights.z * 5e2
#     cost_weights.log_R = cost_weights.log_R * 5e2
#     cost_weights.v_lin = cost_weights.v_lin * 5e1
#     cost_weights.v_ang = cost_weights.v_ang * 0.5 
#     cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0.0
#     cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
#     cost_weights.nom_torque = cost_weights.nom_torque * 4e-3
#     cost_weights.gait = cost_weights.gait * 0.0
#     cost_weights.mech_work = cost_weights.mech_work * 1e-3

#     constraint_weights = Go2ConstraintResidualStruct()
#     constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
#     constraint_weights.base_contact = constraint_weights.base_contact + 100.0
#     constraint_weights.hip_contact = constraint_weights.hip_contact + 100.0
#     constraint_weights.thigh_contact = constraint_weights.thigh_contact + 10.0
#     constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 10.0

#     constraint_weights.stance = constraint_weights.stance + 1.0
#     constraint_weights.swing = constraint_weights.swing + 1.0

#     # joint_limits
#     constraint_weights.hip_q_lb = constraint_weights.hip_q_lb + 10.0
#     constraint_weights.hip_q_ub = constraint_weights.hip_q_ub + 10.0
#     constraint_weights.knee_q_lb = constraint_weights.knee_q_lb + 10.0
#     constraint_weights.knee_q_ub = constraint_weights.knee_q_ub + 10.0
#     constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb + 10.0
#     constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub + 10.0

#     constraint_relaxation = Go2ConstraintResidualStruct()
#     constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
#     constraint_relaxation.base_contact = constraint_relaxation.base_contact * 5e-4
#     constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 5e-4
#     constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 5e-4
#     constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 5e-4

#     constraint_relaxation.stance = constraint_relaxation.stance * 0.01
#     constraint_relaxation.swing = constraint_relaxation.swing * 0.01

#     constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 1e-2
#     constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 1e-2
#     constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 1e-2
#     constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 1e-2
#     constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 1e-2
#     constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 1e-2
#     #constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01

#     terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
#     terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

#     terminal_constraints_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

#     terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

#     global_cost_weights = Go2GlobalCostResidualStruct()
#     global_cost_weights.xy = global_cost_weights.xy * 1e-2

#     cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
#     terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
#     constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
#     terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

#     return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


# def testing_weights():
#     cost_weights = Go2CostResidualStruct()
#     # cost_weights.z = cost_weights.z * 5e3
#     # cost_weights.log_R = cost_weights.log_R * 2e1
#     # cost_weights.v_lin = cost_weights.v_lin * 3e1
#     # cost_weights.v_ang = cost_weights.v_ang * 1
#     # cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
#     # cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
#     # cost_weights.nom_torque = cost_weights.nom_torque * 5e-4
#     # cost_weights.gait = cost_weights.gait * 2e3

#     cost_weights.z = cost_weights.z * 5e3
#     cost_weights.log_R = cost_weights.log_R * 1e3
#     cost_weights.v_lin = cost_weights.v_lin * 3.0e1 #3.8e1
#     cost_weights.v_ang = cost_weights.v_ang * 1 #2
#     cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
#     cost_weights.nom_joint_v = cost_weights.nom_joint_v * 2.4e-5
#     cost_weights.nom_torque = cost_weights.nom_torque * 8e-3 #8e-3 #1e-2
#     cost_weights.gait = cost_weights.gait * 2e3 #6e3

#     constraint_weights = Go2ConstraintResidualStruct()
#     constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

#     constraint_relaxation = Go2ConstraintResidualStruct()
#     constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

#     terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
#     terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

#     terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)

#     terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
#     # terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
#     # terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
#     # terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

#     global_cost_weights = Go2GlobalCostResidualStruct()
#     global_cost_weights.xy = global_cost_weights.xy * 1

#     cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
#     terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
#     constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
#     terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

#     return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def testing_weights():
    cost_weights = Go2CostResidualStruct()
    # cost_weights.z = cost_weights.z * 5e3
    # cost_weights.log_R = cost_weights.log_R * 2e1
    # cost_weights.v_lin = cost_weights.v_lin * 3e1
    # cost_weights.v_ang = cost_weights.v_ang * 1
    # cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    # cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    # cost_weights.nom_torque = cost_weights.nom_torque * 5e-4
    # cost_weights.gait = cost_weights.gait * 2e3

    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 3e1 #3.8e1
    cost_weights.v_ang = cost_weights.v_ang * 1 #2
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5 #10
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-2 #8e-3 #1e-2
    cost_weights.gait = cost_weights.gait * 2e3 #6e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)


    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    # terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    # terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def gallop_weights():
    cost_weights = Go2CostResidualStruct()
    # cost_weights.z = cost_weights.z * 5e3
    # cost_weights.log_R = cost_weights.log_R * 2e1
    # cost_weights.v_lin = cost_weights.v_lin * 3e1
    # cost_weights.v_ang = cost_weights.v_ang * 1
    # cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    # cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    # cost_weights.nom_torque = cost_weights.nom_torque * 5e-4
    # cost_weights.gait = cost_weights.gait * 2e3

    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e2
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 0.5
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 2e-3
    cost_weights.gait = cost_weights.gait * 2e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    # terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    # terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation


def constraint_test_weights():
    cost_weights = Go2CostResidualStruct()
    # cost_weights.z = cost_weights.z * 5e3
    # cost_weights.log_R = cost_weights.log_R * 2e1
    # cost_weights.v_lin = cost_weights.v_lin * 3e1
    # cost_weights.v_ang = cost_weights.v_ang * 1
    # cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    # cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    # cost_weights.nom_torque = cost_weights.nom_torque * 5e-4
    # cost_weights.gait = cost_weights.gait * 2e3

    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 5e1
    cost_weights.v_ang = cost_weights.v_ang * 1.0
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 2e-2
    cost_weights.gait = cost_weights.gait * 0 #2e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    #constraint_weights.base_contact = constraint_weights.base_contact + 1e-3 #5e-3
    #constraint_weights.hip_contact = constraint_weights.hip_contact + 1e-3 #5e-3
    # constraint_weights.thigh_contact = constraint_weights.thigh_contact + 1e-4
    # constraint_weights.shank_top_contact = constraint_weights.shank_top_contact + 1e-5
    #constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact + 1e-6

    # constraint_weights.torque_limits = constraint_weights.torque_limits + 1e-11 #1e-3
    # #constraint_weights.pd_limits = constraint_weights.pd_limits * 0
    constraint_weights.stance = constraint_weights.stance + 1e-4
    constraint_weights.swing = constraint_weights.swing + 1e-4
    constraint_weights.max_foot_height = constraint_weights.max_foot_height + 5e-5


    constraint_weights.hip_q_lb = constraint_weights.hip_q_lb + 1e-3
    constraint_weights.hip_q_ub = constraint_weights.hip_q_ub + 1e-3
    constraint_weights.knee_q_lb = constraint_weights.knee_q_lb + 2e-4
    constraint_weights.knee_q_ub = constraint_weights.knee_q_ub + 2e-4
    constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb + 1e-3
    constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub + 2e-3
    constraint_weights.hip_v = constraint_weights.hip_v + 1e-5
    constraint_weights.knee_v = constraint_weights.knee_v + 1e-5
    constraint_weights.ankle_v = constraint_weights.ankle_v + 1e-5

    # constraint_weights.z_err_ub = constraint_weights.z_err_ub + 1e-2
    # constraint_weights.z_err_lb = constraint_weights.z_err_lb + 1e-2

    # constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb + 1e-2 
    # constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub + 1e-2

    # constraint_weights.v_xyz_err_lb = constraint_weights.v_xyz_err_lb + 1e-2 
    # constraint_weights.v_xyz_err_ub = constraint_weights.v_xyz_err_ub + 1e-2
    # constraint_weights.v_rpy_err_lb = constraint_weights.v_rpy_err_lb + 1e-2
    # constraint_weights.v_rpy_err_ub = constraint_weights.v_rpy_err_ub + 1e-2

    constraint_relaxation = Go2ConstraintResidualStruct()
    constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), constraint_relaxation)
    # constraint_relaxation.base_contact = constraint_relaxation.base_contact * 1e-2
    # constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 1e-2
    # constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 1e-2
    # constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 1e-2
    # constraint_relaxation.shank_bottom_contact = constraint_relaxation.shank_bottom_contact * 1e-2

    # constraint_relaxation.z_err_ub = constraint_relaxation.z_err_ub * 1e-2
    # constraint_relaxation.z_err_lb = constraint_relaxation.z_err_lb * 1e-2


    # constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.01
    constraint_relaxation.stance = constraint_relaxation.stance * 0.01 #0.03
    constraint_relaxation.swing = constraint_relaxation.swing * 0.01 #0.03
    constraint_relaxation.max_foot_height = constraint_relaxation.max_foot_height * 0.05

    # constraint_relaxation.rpy_err_lb = constraint_relaxation.rpy_err_lb * 0.05
    # constraint_relaxation.rpy_err_ub = constraint_relaxation.rpy_err_ub * 0.05

    # constraint_relaxation.v_xyz_err_lb = constraint_relaxation.v_xyz_err_lb * 0.2
    # constraint_relaxation.v_xyz_err_ub = constraint_relaxation.v_xyz_err_ub * 0.2
    # constraint_relaxation.v_rpy_err_lb = constraint_relaxation.v_rpy_err_lb * 0.2
    # constraint_relaxation.v_rpy_err_ub = constraint_relaxation.v_rpy_err_ub * 0.2

    constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 0.08
    constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 0.08
    constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 0.08
    constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 0.08
    constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 0.08
    constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 0.08
    constraint_relaxation.hip_v = constraint_relaxation.hip_v * 1
    constraint_relaxation.knee_v = constraint_relaxation.knee_v * 1
    constraint_relaxation.ankle_v = constraint_relaxation.ankle_v * 2



    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)

    terminal_constraint_relaxation = jax.tree.map(lambda x: x, constraint_relaxation)

    global_cost_weights = Go2GlobalCostResidualStruct()
    global_cost_weights.xy = global_cost_weights.xy * 0
    
    # terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    # terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    # terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights, constraint_relaxation, terminal_constraint_relaxation, global_cost_weights


def galloping_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 1e3
    cost_weights.log_R = cost_weights.log_R * 1e2
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-4
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-3
    cost_weights.gait = cost_weights.gait * 0.0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
    constraint_weights.joint_limits = constraint_weights.joint_limits * 1e1
    constraint_weights.torque_limits = constraint_weights.torque_limits * 1
    constraint_weights.stance = constraint_weights.stance * 2e5
    constraint_weights.swing = constraint_weights.swing * 2e5
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 1


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e6, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e6, terminal_constraints_weights)
    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights


def rough_terrain_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 0
    cost_weights.log_R = cost_weights.log_R * 0
    cost_weights.v_lin = cost_weights.v_lin * 0 
    cost_weights.v_ang = cost_weights.v_ang * 0
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0 
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 0
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-6
    cost_weights.gait = cost_weights.gait * 0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.base_contact = constraint_weights.base_contact * 1e-2 #5e-3
    constraint_weights.hip_contact = constraint_weights.hip_contact * 1e-2 #5e-3
    constraint_weights.thigh_contact = constraint_weights.thigh_contact * 1e-3
    constraint_weights.shank_top_contact = constraint_weights.shank_top_contact * 1e-4
    constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact * 1e-4

    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-6 #1e-3
    constraint_weights.pd_limits = constraint_weights.pd_limits * 0
    constraint_weights.stance = constraint_weights.stance * 1e-4
    constraint_weights.swing = constraint_weights.swing * 1e-4
    constraint_weights.max_foot_height = constraint_weights.max_foot_height * 1e-4

    constraint_weights.hip_q_lb = constraint_weights.hip_q_lb * 8e-3
    constraint_weights.hip_q_ub = constraint_weights.hip_q_ub * 8e-3
    constraint_weights.knee_q_lb = constraint_weights.knee_q_lb * 8e-3
    constraint_weights.knee_q_ub = constraint_weights.knee_q_ub * 8e-3
    constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb * 8e-3
    constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub * 8e-3
    constraint_weights.hip_v = constraint_weights.hip_v * 1e-3
    constraint_weights.knee_v = constraint_weights.knee_v * 1e-3
    constraint_weights.ankle_v = constraint_weights.ankle_v * 1e-3

    constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb * 1e-2 
    constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub * 1e-2

    constraint_weights.v_xyz_err_lb = constraint_weights.v_xyz_err_lb * 1e-2 
    constraint_weights.v_xyz_err_ub = constraint_weights.v_xyz_err_ub * 1e-2
    constraint_weights.v_rpy_err_lb = constraint_weights.v_rpy_err_lb * 1e-2
    constraint_weights.v_rpy_err_ub = constraint_weights.v_rpy_err_ub * 1e-2

    constraint_weights.z_err_lb = constraint_weights.z_err_lb * 1e-1
    constraint_weights.z_err_ub = constraint_weights.z_err_ub * 1e-1

    
    constraint_relaxation = Go2ConstraintResidualStruct()

    constraint_relaxation.base_contact = constraint_relaxation.base_contact * 0.01
    constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 0.01
    constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 0.01
    constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 0.01 
    constraint_relaxation.shank_bottom_contact = constraint_relaxation.shank_bottom_contact * 0.01

    constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 2.0
    constraint_relaxation.pd_limits = constraint_relaxation.pd_limits 
    constraint_relaxation.stance = constraint_relaxation.stance * 0.01
    constraint_relaxation.swing = constraint_relaxation.swing * 0.01
    constraint_relaxation.max_foot_height = constraint_relaxation.max_foot_height * 0.05

    constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 0.08
    constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 0.08
    constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 0.08
    constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 0.08
    constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 0.08
    constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 0.08
    constraint_relaxation.hip_v = constraint_relaxation.hip_v * 2
    constraint_relaxation.knee_v = constraint_relaxation.knee_v * 2
    constraint_relaxation.ankle_v = constraint_relaxation.ankle_v * 5

    constraint_relaxation.rpy_err_lb = constraint_relaxation.rpy_err_lb * 0.05
    constraint_relaxation.rpy_err_ub = constraint_relaxation.rpy_err_ub * 0.05

    constraint_relaxation.v_xyz_err_lb = constraint_relaxation.v_xyz_err_lb * 0.1
    constraint_relaxation.v_xyz_err_ub = constraint_relaxation.v_xyz_err_ub * 0.1
    constraint_relaxation.v_rpy_err_lb = constraint_relaxation.v_rpy_err_lb * 0.1
    constraint_relaxation.v_rpy_err_ub = constraint_relaxation.v_rpy_err_ub * 0.1

    constraint_relaxation.z_err_lb = constraint_relaxation.z_err_lb * 0.01
    constraint_relaxation.z_err_ub = constraint_relaxation.z_err_ub * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0
    terminal_constraint_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraint_relaxation = jax.tree.map(lambda x: x, constraint_relaxation)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraint_weights, constraint_relaxation, terminal_constraint_relaxation


def deploy_locomotion_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 0
    cost_weights.log_R = cost_weights.log_R * 0
    cost_weights.v_lin = cost_weights.v_lin * 0 
    cost_weights.v_ang = cost_weights.v_ang * 0
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0 
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 0
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-5
    cost_weights.gait = cost_weights.gait * 0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.base_contact = constraint_weights.base_contact * 1e-2 #5e-3
    constraint_weights.hip_contact = constraint_weights.hip_contact * 1e-2 #5e-3
    constraint_weights.thigh_contact = constraint_weights.thigh_contact * 1e-3
    constraint_weights.shank_top_contact = constraint_weights.shank_top_contact * 1e-4
    constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact * 1e-4

    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-3
    constraint_weights.pd_limits = constraint_weights.pd_limits * 0
    constraint_weights.stance = constraint_weights.stance * 1e-4
    constraint_weights.swing = constraint_weights.swing * 1e-4
    constraint_weights.max_foot_height = constraint_weights.max_foot_height * 1e-4

    constraint_weights.hip_q_lb = constraint_weights.hip_q_lb * 8e-3
    constraint_weights.hip_q_ub = constraint_weights.hip_q_ub * 8e-3
    constraint_weights.knee_q_lb = constraint_weights.knee_q_lb * 8e-3
    constraint_weights.knee_q_ub = constraint_weights.knee_q_ub * 8e-3
    constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb * 8e-3
    constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub * 8e-3
    constraint_weights.hip_v = constraint_weights.hip_v * 1e-3
    constraint_weights.knee_v = constraint_weights.knee_v * 1e-3
    constraint_weights.ankle_v = constraint_weights.ankle_v * 1e-3

    constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb * 1e-2 
    constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub * 1e-2

    constraint_weights.v_xyz_err_lb = constraint_weights.v_xyz_err_lb * 1e-2 
    constraint_weights.v_xyz_err_ub = constraint_weights.v_xyz_err_ub * 1e-2
    constraint_weights.v_rpy_err_lb = constraint_weights.v_rpy_err_lb * 1e-2
    constraint_weights.v_rpy_err_ub = constraint_weights.v_rpy_err_ub * 1e-2

    constraint_weights.z_err_lb = constraint_weights.z_err_lb * 1e-2
    constraint_weights.z_err_ub = constraint_weights.z_err_ub * 1e-2

    
    constraint_relaxation = Go2ConstraintResidualStruct()

    constraint_relaxation.base_contact = constraint_relaxation.base_contact * 0.01
    constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 0.01
    constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 0.01
    constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 0.01 
    constraint_relaxation.shank_bottom_contact = constraint_relaxation.shank_bottom_contact * 0.01

    constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.5
    constraint_relaxation.pd_limits = constraint_relaxation.pd_limits 
    constraint_relaxation.stance = constraint_relaxation.stance * 0.01
    constraint_relaxation.swing = constraint_relaxation.swing * 0.01
    constraint_relaxation.max_foot_height = constraint_relaxation.max_foot_height * 0.05

    constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 0.08
    constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 0.08
    constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 0.08
    constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 0.08
    constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 0.08
    constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 0.08
    constraint_relaxation.hip_v = constraint_relaxation.hip_v * 2
    constraint_relaxation.knee_v = constraint_relaxation.knee_v * 2
    constraint_relaxation.ankle_v = constraint_relaxation.ankle_v * 5

    constraint_relaxation.rpy_err_lb = constraint_relaxation.rpy_err_lb * 0.05
    constraint_relaxation.rpy_err_ub = constraint_relaxation.rpy_err_ub * 0.05

    constraint_relaxation.v_xyz_err_lb = constraint_relaxation.v_xyz_err_lb * 0.1
    constraint_relaxation.v_xyz_err_ub = constraint_relaxation.v_xyz_err_ub * 0.1
    constraint_relaxation.v_rpy_err_lb = constraint_relaxation.v_rpy_err_lb * 0.1
    constraint_relaxation.v_rpy_err_ub = constraint_relaxation.v_rpy_err_ub * 0.1

    constraint_relaxation.z_err_lb = constraint_relaxation.z_err_lb * 0.01
    constraint_relaxation.z_err_ub = constraint_relaxation.z_err_ub * 0.01

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0
    terminal_constraint_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraint_relaxation = jax.tree.map(lambda x: x, constraint_relaxation)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraint_weights, constraint_relaxation, terminal_constraint_relaxation



def dial_vs_spline_shooter_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5
    cost_weights.log_R = cost_weights.log_R * 2
    cost_weights.v_lin = cost_weights.v_lin * 0.1
    cost_weights.v_ang = cost_weights.v_ang * 0.01
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0.01
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-7
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-5
    cost_weights.gait = cost_weights.gait * 2

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights = jax.tree.map(lambda x: x * 0, constraint_weights)
    constraint_relaxation = Go2ConstraintResidualStruct()

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0
    terminal_constraint_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraint_relaxation = jax.tree.map(lambda x: x, constraint_relaxation)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraint_weights, constraint_relaxation, terminal_constraint_relaxation

def trotting_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 0
    cost_weights.log_R = cost_weights.log_R * 0
    cost_weights.v_lin = cost_weights.v_lin * 0 
    cost_weights.v_ang = cost_weights.v_ang * 0
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0 
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 0
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-5
    cost_weights.gait = cost_weights.gait * 0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.base_contact = constraint_weights.base_contact * 1e-3
    constraint_weights.hip_contact = constraint_weights.hip_contact * 1e-3
    constraint_weights.thigh_contact = constraint_weights.thigh_contact * 1e-4
    constraint_weights.shank_top_contact = constraint_weights.shank_top_contact * 1e-4
    constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact * 0.0

    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-2
    constraint_weights.pd_limits = constraint_weights.pd_limits * 0
    constraint_weights.stance = constraint_weights.stance * 5e-3
    constraint_weights.swing = constraint_weights.swing * 3e-3
    constraint_weights.max_foot_height = constraint_weights.max_foot_height * 1e-3

    constraint_weights.hip_q_lb = constraint_weights.hip_q_lb * 1e-2
    constraint_weights.hip_q_ub = constraint_weights.hip_q_ub * 1e-2
    constraint_weights.knee_q_lb = constraint_weights.knee_q_lb * 1e-2
    constraint_weights.knee_q_ub = constraint_weights.knee_q_ub * 1e-2
    constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb * 1e-2
    constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub * 1e-2
    constraint_weights.hip_v = constraint_weights.hip_v * 1e-3
    constraint_weights.knee_v = constraint_weights.knee_v * 1e-3
    constraint_weights.ankle_v = constraint_weights.ankle_v * 1e-3

    constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb * 1e-2 
    constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub * 1e-2

    constraint_weights.v_xyz_err_lb = constraint_weights.v_xyz_err_lb * 1e-2 
    constraint_weights.v_xyz_err_ub = constraint_weights.v_xyz_err_ub * 1e-2
    constraint_weights.v_rpy_err_lb = constraint_weights.v_rpy_err_lb * 1e-2
    constraint_weights.v_rpy_err_ub = constraint_weights.v_rpy_err_ub * 1e-2

    constraint_weights.z_err_lb = constraint_weights.z_err_lb * 0.0
    constraint_weights.z_err_ub = constraint_weights.z_err_ub * 0.0

    
    constraint_relaxation = Go2ConstraintResidualStruct()

    constraint_relaxation.base_contact = constraint_relaxation.base_contact * 0.01
    constraint_relaxation.hip_contact = constraint_relaxation.hip_contact * 0.01
    constraint_relaxation.thigh_contact = constraint_relaxation.thigh_contact * 0.01
    constraint_relaxation.shank_top_contact = constraint_relaxation.shank_top_contact * 0.01 
    constraint_relaxation.shank_bottom_contact = constraint_relaxation.shank_bottom_contact * 0.01

    constraint_relaxation.torque_limits = constraint_relaxation.torque_limits * 0.5
    constraint_relaxation.pd_limits = constraint_relaxation.pd_limits 
    constraint_relaxation.stance = constraint_relaxation.stance * 0.01
    constraint_relaxation.swing = constraint_relaxation.swing * 0.01
    constraint_relaxation.max_foot_height = constraint_relaxation.max_foot_height * 0.05

    constraint_relaxation.hip_q_lb = constraint_relaxation.hip_q_lb * 0.08
    constraint_relaxation.hip_q_ub = constraint_relaxation.hip_q_ub * 0.08
    constraint_relaxation.knee_q_lb = constraint_relaxation.knee_q_lb * 0.08
    constraint_relaxation.knee_q_ub = constraint_relaxation.knee_q_ub * 0.08
    constraint_relaxation.ankle_q_lb = constraint_relaxation.ankle_q_lb * 0.08
    constraint_relaxation.ankle_q_ub = constraint_relaxation.ankle_q_ub * 0.08
    constraint_relaxation.hip_v = constraint_relaxation.hip_v * 1
    constraint_relaxation.knee_v = constraint_relaxation.knee_v * 1
    constraint_relaxation.ankle_v = constraint_relaxation.ankle_v * 5

    constraint_relaxation.rpy_err_lb = constraint_relaxation.rpy_err_lb * 0.05
    constraint_relaxation.rpy_err_ub = constraint_relaxation.rpy_err_ub * 0.05

    constraint_relaxation.v_xyz_err_lb = constraint_relaxation.v_xyz_err_lb * 0.1
    constraint_relaxation.v_xyz_err_ub = constraint_relaxation.v_xyz_err_ub * 0.1
    constraint_relaxation.v_rpy_err_lb = constraint_relaxation.v_rpy_err_lb * 0.1
    constraint_relaxation.v_rpy_err_ub = constraint_relaxation.v_rpy_err_ub * 0.1

    constraint_relaxation.z_err_lb = constraint_relaxation.z_err_lb 
    constraint_relaxation.z_err_ub = constraint_relaxation.z_err_ub

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0
    terminal_constraint_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraint_relaxation = jax.tree.map(lambda x: x, constraint_relaxation)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraint_weights, constraint_relaxation, terminal_constraint_relaxation


# def trotting_weights():

#     # cost_weights = Go2CostResidualStruct()
#     # cost_weights.z = cost_weights.z * 5e3
#     # cost_weights.log_R = cost_weights.log_R * 1e3
#     # cost_weights.v_lin = cost_weights.v_lin * 3e1
#     # cost_weights.v_ang = cost_weights.v_ang * 1
#     # cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
#     # cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-4
#     # cost_weights.nom_torque = cost_weights.nom_torque * 5e-3
#     # cost_weights.gait = cost_weights.gait * 0.0

#     # constraint_weights = Go2ConstraintResidualStruct()
#     # constraint_weights.body_contact = constraint_weights.body_contact * 1e5
#     # constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
#     # constraint_weights.joint_limits = constraint_weights.joint_limits * 2e1
#     # constraint_weights.torque_limits = constraint_weights.torque_limits * 1
#     # constraint_weights.stance = constraint_weights.stance * 4e5
#     # constraint_weights.swing = constraint_weights.swing * 1e6
#     # constraint_weights.joint_velocity = constraint_weights.joint_velocity * 1
#     # constraint_weights.max_foot_height = constraint_weights.max_foot_height * 1e6


#     cost_weights = Go2CostResidualStruct()
#     cost_weights.z = cost_weights.z * 5
#     cost_weights.log_R = cost_weights.log_R * 1
#     cost_weights.v_lin = cost_weights.v_lin * 0.1 # 0.01
#     cost_weights.v_ang = cost_weights.v_ang * 0.01 #0.001
#     cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0 # 0.01
#     #cost_weights.nom_joint_q = cost_weights.nom_joint_q.at[2::3].set(0.0) # Set roll and pitch joint velocities to 0
#     cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-6
#     cost_weights.nom_torque = cost_weights.nom_torque * 1e-5
#     cost_weights.gait = cost_weights.gait * 2

#     # constraint_weights = Go2ConstraintResidualStruct()
#     # constraint_weights.body_contact = constraint_weights.body_contact * 1e5
#     # constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
#     # constraint_weights.joint_limits = constraint_weights.joint_limits * 1e4
#     # constraint_weights.torque_limits = constraint_weights.torque_limits * 1
#     # constraint_weights.stance = constraint_weights.stance * 4e5
#     # constraint_weights.swing = constraint_weights.swing * 4e5
#     # constraint_weights.joint_velocity = constraint_weights.joint_velocity * 2
#     # constraint_weights.max_foot_height = constraint_weights.max_foot_height * 4e5
#     # constraint_weights.max_log_R = constraint_weights.max_log_R * 1e4
#     # constraint_weights.max_z = constraint_weights.max_z * 1e5
#     # constraint_weights.max_v_xyz = constraint_weights.max_v_xyz * 5e4
#     # constraint_weights.max_v_rpy = constraint_weights.max_v_rpy * 5e4

#     # constraint_weights = Go2ConstraintResidualStruct()
#     # constraint_weights.body_contact = constraint_weights.body_contact * 5e3
#     # constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
#     # constraint_weights.joint_limits = constraint_weights.joint_limits * 1e4
#     # constraint_weights.torque_limits = constraint_weights.torque_limits * 1e1
#     # constraint_weights.stance = constraint_weights.stance * 2e4
#     # constraint_weights.swing = constraint_weights.swing * 2e4
#     # constraint_weights.joint_velocity = constraint_weights.joint_velocity * 3e1
#     # constraint_weights.max_foot_height = constraint_weights.max_foot_height * 2e4 # 4e5
#     # constraint_weights.max_log_R = constraint_weights.max_log_R * 1e4
#     # constraint_weights.max_z = constraint_weights.max_z * 1e5
#     # constraint_weights.max_v_xyz = constraint_weights.max_v_xyz * 3e4
#     # constraint_weights.max_v_rpy = constraint_weights.max_v_rpy * 3e4

#     constraint_weights = Go2ConstraintResidualStruct()
#     constraint_weights.base_contact = constraint_weights.base_contact * 0
#     constraint_weights.hip_contact = constraint_weights.hip_contact * 0
#     constraint_weights.thigh_contact = constraint_weights.thigh_contact * 0
#     constraint_weights.shank_top_contact = constraint_weights.shank_top_contact * 0
#     constraint_weights.shank_bottom_contact = constraint_weights.shank_bottom_contact * 0

#     constraint_weights.torque_limits = constraint_weights.torque_limits * 0
#     constraint_weights.pd_limits = constraint_weights.pd_limits * 0
#     constraint_weights.stance = constraint_weights.stance * 0.08
#     constraint_weights.swing = constraint_weights.swing * 0.08
#     constraint_weights.max_foot_height = constraint_weights.max_foot_height * 0.08

#     constraint_weights.hip_q_lb = constraint_weights.hip_q_lb * 0.08
#     constraint_weights.hip_q_ub = constraint_weights.hip_q_ub * 0.08
#     constraint_weights.knee_q_lb = constraint_weights.knee_q_lb * 0.08
#     constraint_weights.knee_q_ub = constraint_weights.knee_q_ub * 0.08
#     constraint_weights.ankle_q_lb = constraint_weights.ankle_q_lb * 0.08
#     constraint_weights.ankle_q_ub = constraint_weights.ankle_q_ub * 0.08
#     constraint_weights.hip_v = constraint_weights.hip_v * 0.0
#     constraint_weights.knee_v = constraint_weights.knee_v * 0.0
#     constraint_weights.ankle_v = constraint_weights.ankle_v * 0.0

#     constraint_weights.rpy_err_lb = constraint_weights.rpy_err_lb * 0 #5e4
#     constraint_weights.rpy_err_ub = constraint_weights.rpy_err_ub * 0 #5e4

#     constraint_weights.v_xyz_err_lb = constraint_weights.v_xyz_err_lb * 0 #2e3
#     constraint_weights.v_xyz_err_ub = constraint_weights.v_xyz_err_ub * 0 #2e3 #3e4
#     constraint_weights.v_rpy_err_lb = constraint_weights.v_rpy_err_lb * 0 #2e3 #3e4
#     constraint_weights.v_rpy_err_ub = constraint_weights.v_rpy_err_ub * 0 #2e3 #3e4

#     constraint_weights.z_err_lb = constraint_weights.z_err_lb * 0.0 #1e4
#     constraint_weights.z_err_ub = constraint_weights.z_err_ub * 0.0 #1e4

#     terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
#     terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

#     terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
#     terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
#     terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

#     # cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
#     # terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
#     # constraint_weights = jax.tree.map(lambda x: x / 1e4, constraint_weights)
#     # terminal_constraints_weights = jax.tree.map(lambda x: x / 1e4, terminal_constraints_weights)
#     # constraint_weights = jax.tree.map(lambda x: x / 1e6, constraint_weights)
#     # terminal_constraints_weights = jax.tree.map(lambda x: x / 1e6, terminal_constraints_weights)
#     return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights

# def trotting_weights():

#     cost_weights = Go2CostResidualStruct()
#     cost_weights.z = cost_weights.z * 0.0 # 5e3
#     cost_weights.log_R = cost_weights.log_R * 0.0 #1e3
#     cost_weights.v_lin = cost_weights.v_lin * 0.0 #3e1
#     cost_weights.v_ang = cost_weights.v_ang * 0.0 #1
#     cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0.0 #1e1
#     cost_weights.nom_joint_v = cost_weights.nom_joint_v * 0.0 #1e-4
#     cost_weights.nom_torque = cost_weights.nom_torque * 1e-2 #1e-2
#     cost_weights.gait = cost_weights.gait * 0.0

#     # constraint_weights = Go2ConstraintResidualStruct()
#     # constraint_weights.body_contact = constraint_weights.body_contact * 1e4
#     # constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
#     # constraint_weights.joint_limits = constraint_weights.joint_limits * 1e4
#     # constraint_weights.torque_limits = constraint_weights.torque_limits * 1
#     # constraint_weights.stance = constraint_weights.stance * 4e5
#     # constraint_weights.swing = constraint_weights.swing * 4e5
#     # constraint_weights.joint_velocity = constraint_weights.joint_velocity * 2
#     # constraint_weights.max_foot_height = constraint_weights.max_foot_height * 4e5
#     # constraint_weights.max_log_R = constraint_weights.max_log_R * 1e4
#     # constraint_weights.max_z = constraint_weights.max_z * 3e4
#     # constraint_weights.max_v_xyz = constraint_weights.max_v_xyz * 5e4
#     # constraint_weights.max_v_rpy = constraint_weights.max_v_rpy * 5e4

#     constraint_weights = Go2ConstraintResidualStruct()
#     constraint_weights.body_contact = constraint_weights.body_contact * 1e3
#     constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
#     constraint_weights.joint_limits = constraint_weights.joint_limits * 1e4
#     constraint_weights.torque_limits = constraint_weights.torque_limits * 1
#     constraint_weights.stance = constraint_weights.stance * 4e5
#     constraint_weights.swing = constraint_weights.swing * 4e5
#     constraint_weights.joint_velocity = constraint_weights.joint_velocity * 2
#     constraint_weights.max_foot_height = constraint_weights.max_foot_height * 4e5
#     constraint_weights.max_log_R = constraint_weights.max_log_R * 1e4
#     constraint_weights.max_z = constraint_weights.max_z * 5e4
#     constraint_weights.max_v_xyz = constraint_weights.max_v_xyz * 5e4
#     constraint_weights.max_v_rpy = constraint_weights.max_v_rpy * 5e4

#     terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
#     terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

#     terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
#     terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
#     terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
#     terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

#     cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
#     terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
#     constraint_weights = jax.tree.map(lambda x: x / 1e5, constraint_weights)
#     terminal_constraints_weights = jax.tree.map(lambda x: x / 1e5, terminal_constraints_weights)
#     # constraint_weights = jax.tree.map(lambda x: x / 1e6, constraint_weights)
#     # terminal_constraints_weights = jax.tree.map(lambda x: x / 1e6, terminal_constraints_weights)
#     return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights

def trotting_weights_original():

    # no constraints

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 1e3
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 2e-3
    cost_weights.gait = cost_weights.gait * 2e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
    constraint_weights.joint_limits = constraint_weights.joint_limits * 1e-1
    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-2
    constraint_weights.stance = constraint_weights.stance * 0.0
    constraint_weights.swing = constraint_weights.swing * 0.0
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 0.0


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)
    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights

def rearing_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 3e1
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 5.0
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 0.0 #1e-4
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-4 #1e-2
    cost_weights.gait = cost_weights.gait * 0.0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
    constraint_weights.joint_limits = constraint_weights.joint_limits * 1e1
    constraint_weights.torque_limits = constraint_weights.torque_limits * 1
    constraint_weights.stance = constraint_weights.stance * 0.0
    constraint_weights.swing = constraint_weights.swing * 0.0
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 5 #1e1
    constraint_weights.max_foot_height = constraint_weights.max_foot_height * 0.0
    constraint_weights.max_log_R = constraint_weights.max_log_R * 0.0 #2e4
    constraint_weights.max_z = constraint_weights.max_z * 0.0 #1e5
    constraint_weights.max_v_xyz = constraint_weights.max_v_xyz * 0.0 #1e4
    constraint_weights.max_v_rpy = constraint_weights.max_v_rpy * 0.0 #1e4

    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e6, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e6, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights

def biped_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 2e1
    cost_weights.v_lin = cost_weights.v_lin * 1e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 0.1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-4
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-4
    cost_weights.gait = cost_weights.gait * 2e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 0.0
    constraint_weights.joint_limits = constraint_weights.joint_limits * 0.0
    constraint_weights.torque_limits = constraint_weights.torque_limits * 0.0
    constraint_weights.stance = constraint_weights.stance * 0.0
    constraint_weights.swing = constraint_weights.swing * 0.0
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 0.0


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights


def locomotion_weights():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 2e1
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-4
    cost_weights.gait = cost_weights.gait * 2e3

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
    constraint_weights.joint_limits = constraint_weights.joint_limits * 1e-1
    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-2
    constraint_weights.stance = constraint_weights.stance * 0.0
    constraint_weights.swing = constraint_weights.swing * 0.0
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 0.0


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights

def locomotion_weights_no_gait():

    cost_weights = Go2CostResidualStruct()
    cost_weights.z = cost_weights.z * 5e3
    cost_weights.log_R = cost_weights.log_R * 2e1
    cost_weights.v_lin = cost_weights.v_lin * 3e1
    cost_weights.v_ang = cost_weights.v_ang * 1
    cost_weights.nom_joint_q = cost_weights.nom_joint_q * 1e1
    cost_weights.nom_joint_v = cost_weights.nom_joint_v * 1e-5
    cost_weights.nom_torque = cost_weights.nom_torque * 1e-4
    cost_weights.gait = cost_weights.gait * 0.0

    constraint_weights = Go2ConstraintResidualStruct()
    constraint_weights.body_contact = constraint_weights.body_contact * 1e4
    constraint_weights.pd_limits = constraint_weights.pd_limits * 1e-1
    constraint_weights.joint_limits = constraint_weights.joint_limits * 1e-1
    constraint_weights.torque_limits = constraint_weights.torque_limits * 1e-2
    constraint_weights.stance = constraint_weights.stance * 0.0
    constraint_weights.swing = constraint_weights.swing * 0.0
    constraint_weights.joint_velocity = constraint_weights.joint_velocity * 0.0


    terminal_cost_weights = jax.tree.map(lambda x: x, cost_weights)
    terminal_cost_weights.nom_torque = terminal_cost_weights.nom_torque * 0.0

    terminal_constraints_weights = jax.tree.map(lambda x: x, constraint_weights)
    terminal_constraints_weights.pd_limits = terminal_constraints_weights.pd_limits * 0.0
    terminal_constraints_weights.joint_limits = terminal_constraints_weights.joint_limits * 0.0
    terminal_constraints_weights.torque_limits = terminal_constraints_weights.torque_limits * 0.0

    cost_weights = jax.tree.map(lambda x: x / 1e3, cost_weights)
    terminal_cost_weights = jax.tree.map(lambda x: x / 1e3, terminal_cost_weights)
    constraint_weights = jax.tree.map(lambda x: x / 1e3, constraint_weights)
    terminal_constraints_weights = jax.tree.map(lambda x: x / 1e3, terminal_constraints_weights)

    return cost_weights, terminal_cost_weights, constraint_weights, terminal_constraints_weights


# ----------------- Residual Tree ----------------

cost_residual_tree = Go2CostResidualStruct(
    z=z_residual, 
    log_R=log_R_residual,
    v_lin=v_lin_residual,
    v_ang=v_ang_residual,
    nom_joint_q=nom_joint_q_residual,
    nom_joint_v=nom_joint_v_residual,
    nom_torque=nom_torque_residual,
    gait=gait_residual, 
    mech_work=pos_mech_work_residual
)

global_cost_residual_tree = Go2GlobalCostResidualStruct(
    xy=xy_global_cost_residual)

constraint_residual_tree = Go2ConstraintResidualStruct(
    base_contact=base_contact_residual,
    hip_contact=hip_contact_residual,
    thigh_contact=thigh_contact_residual,
    shank_top_contact=shank_top_contact_residual,
    shank_bottom_contact=shank_bottom_contact_residual,
    pd_limits=pd_limits_residual,
    hip_q_lb=locomotion_hip_q_lb_residual,
    hip_q_ub=locomotion_hip_q_ub_residual,
    knee_q_lb=locomotion_knee_q_lb_residual,
    knee_q_ub=locomotion_knee_q_ub_residual,
    ankle_q_lb=locomotion_ankle_q_lb_residual,
    ankle_q_ub=locomotion_ankle_q_ub_residual,
    torque_limits=torque_limits_residual,
    stance=stance_residual,
    swing=swing_residual,
    max_foot_height=max_foot_height_residual,
    hip_v=hip_v_residual,
    knee_v=knee_v_residual,
    ankle_v=ankle_v_residual,
    rpy_err_lb=rpy_err_lb_residual,
    rpy_err_ub=rpy_err_ub_residual,
    v_xyz_err_lb=v_xyz_err_lb_residual,
    v_xyz_err_ub=v_xyz_err_ub_residual,
    v_rpy_err_lb=v_rpy_err_lb_residual,
    v_rpy_err_ub=v_rpy_err_ub_residual,
    z_err_lb=z_err_lb_residual,
    z_err_ub=z_err_ub_residual,
)