import numpy as np
import mujoco
from Common.Go2_dog import *
from Training import *
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def smooth_update(old, new, alpha):
    return alpha * new + (1 - alpha) * old


def _geom_fade_weight(w_prev, w_tgt, alpha):
    eps = 1e-8
    log_prev = jnp.log(jnp.clip(w_prev, eps))
    log_tgt  = jnp.log(jnp.clip(w_tgt,  eps))
    return jnp.exp((1.0 - alpha) * log_prev + alpha * log_tgt)

def _std_fade_weight(w_prev, w_tgt, alpha):
    return (1.0 - alpha) * w_prev + alpha * w_tgt

def gait_weights_false(cost_weights, cost_weights_target, alpha_new_gait):
    cost_weights_new_gait = jax.tree.map(
        lambda w_prev, w_tgt: _geom_fade_weight(w_prev, w_tgt, alpha_new_gait),
        cost_weights.cost_weights.gait,
        cost_weights_target.cost_weights.gait
    )
    return cost_weights_new_gait

def gait_weights_true(cost_weights, cost_weights_target, alpha_new_gait):
    return jnp.zeros_like(cost_weights.cost_weights.gait)


def rough_terrain_branch(_):
    c2 = jax.tree.map(lambda x: x, commands_new)
    w2 = jax.tree.map(lambda x: x, weights_new)

    c2.tripod = c2.tripod.at[:, -1].set(0.0)
    c2.swing_height = c2.swing_height.at[:, -1].set(0.08)
    c2.z = c2.z.at[:, -1].set(0.27)
    c2.fl_phase = c2.fl_phase.at[:, -1].set(0.5)
    c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
    c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
    c2.rl_phase = c2.rl_phase.at[:, -1].set(0.0)
    c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.45)
    c2.cadence = c2.cadence.at[:, -1].set(2.0)
    c2.pitch_err_lb = c2.pitch_err_lb.at[:, -1].set(-0.6)
    c2.pitch_err_ub = c2.pitch_err_ub.at[:, -1].set(0.6)
    c2.roll_err_lb = c2.roll_err_lb.at[:, -1].set(-0.3)
    c2.roll_err_ub = c2.roll_err_ub.at[:, -1].set(0.3)
    c2.yaw_err_lb = c2.yaw_err_lb.at[:, -1].set(-1.0)
    c2.yaw_err_ub = c2.yaw_err_ub.at[:, -1].set(1.0)
    c2.z_err_lb = c2.z_err_lb.at[:, -1].set(-0.02)
    c2.z_err_ub = c2.z_err_ub.at[:, -1].set(0.08)

    w2.cost_weights.nom_torque = w2.cost_weights.nom_torque.at[:, -1].set(4e-6)
    w2.cost_weights.v_lin = w2.cost_weights.v_lin.at[:, -1].set(5e-2)


    w2.cost_weights.gait = w2.cost_weights.gait.at[:, -1].set(2.0)
    w2.cost_weights.nom_joint_q = w2.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
    w2.cost_weights.nom_joint_v = w2.cost_weights.nom_joint_v.at[:, -1].set(1e-8)

    # shared weights for walk-like
    w2.cost_weights.z = w2.cost_weights.z.at[:, -1].set(5.0)
    w2.cost_weights.log_R = w2.cost_weights.log_R.at[:, -1].set(0.75)
    w2.cost_weights.v_ang = w2.cost_weights.v_ang.at[:, -1].set(1e-3)
    w2.constraint_weights.thigh_contact = w2.constraint_weights.thigh_contact.at[:, -1].set(0.0)
    w2.constraint_weights.shank_top_contact = w2.constraint_weights.shank_top_contact.at[:, -1].set(0.0)
    w2.constraint_relaxation.thigh_contact = w2.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
    w2.constraint_relaxation.shank_top_contact = w2.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)
    w2.constraint_weights.z_err_lb = w2.constraint_weights.z_err_lb.at[:, -1].set(0.0)
    w2.constraint_weights.z_err_ub = w2.constraint_weights.z_err_ub.at[:, -1].set(0.0)
    w2.constraint_relaxation.z_err_lb = w2.constraint_relaxation.z_err_lb.at[:, -1].set(1.0)
    w2.constraint_relaxation.z_err_ub = w2.constraint_relaxation.z_err_ub.at[:, -1].set(1.0)

    w2.constraint_weights.z_err_lb = w2.constraint_weights.z_err_lb.at[:, -1].set(0.1)
    w2.constraint_weights.z_err_ub = w2.constraint_weights.z_err_ub.at[:, -1].set(0.1)
    w2.constraint_relaxation.z_err_lb = w2.constraint_relaxation.z_err_lb.at[:, -1].set(0.01)
    w2.constraint_relaxation.z_err_ub = w2.constraint_relaxation.z_err_ub.at[:, -1].set(0.01)

    w2.constraint_weights.rpy_err_lb = w2.constraint_weights.rpy_err_lb.at[:, -1].set(0.1)
    w2.constraint_weights.rpy_err_ub = w2.constraint_weights.rpy_err_ub.at[:, -1].set(0.1)
    w2.constraint_relaxation.rpy_err_lb = w2.constraint_relaxation.rpy_err_lb.at[:, -1].set(0.01)
    w2.constraint_relaxation.rpy_err_ub = w2.constraint_relaxation.rpy_err_ub.at[:, -1].set(0.01)

    return c2, w2
    
from jax import lax
from functools import partial
gpus = jax.devices("gpu")
@partial(jax.jit, device=gpus[-1])
def gait_commands_weights(commands, weights,
                          trot, rotary_gallop, transverse_gallop,
                          bound, pace, stand, transition, tripod=0.0, rough_terrain=0.0):
    """JIT-safe rewrite using lax.cond for all logical branches."""

    # roll inputs forward
    global_time = jax.tree.map(lambda x: x, commands.global_time)
    commands_new = jax.tree.map(lambda x: jnp.roll(x, shift=-1), commands)
    weights_new = jax.tree.map(lambda x: jnp.roll(x, shift=-1), weights)
    commands_new.global_time = global_time


    # weights_new.cost_weights.nom_torque = weights_new.cost_weights.nom_torque.at[:, -1].set(4e-6)
    # weights_new.cost_weights.v_lin = weights_new.cost_weights.v_lin.at[:, -1].set(5e-2)

    # --- transition adjustments ------------------------------------------------
    def do_transition(_):
        w = jax.tree.map(lambda x: x, weights_new)
        w.cost_weights.gait = w.cost_weights.gait.at[:, -1].set(0.0)
        w.cost_weights.nom_joint_v = w.cost_weights.nom_joint_v.at[:, -1].set(1e-6)
        w.cost_weights.nom_joint_q = w.cost_weights.nom_joint_q.at[:, -1].set(5e-4)
        return w

    weights_new = lax.cond(transition > 0.5, do_transition, lambda _: weights_new, operand=None)

    # --- WALK / STAND branch ---------------------------------------------------
    def walk_like_branch(_):
        c, w = commands_new, weights_new
        c = jax.tree.map(lambda x: x, c)
        w2 = jax.tree.map(lambda x: x, w)

        c.tripod = c.tripod.at[:, -1].set(0.0)
        w2.cost_weights.nom_torque = w2.cost_weights.nom_torque.at[:, -1].set(4e-6)
        w2.cost_weights.v_lin = w2.cost_weights.v_lin.at[:, -1].set(5e-2)

        # set trot / pace / stand subtypes
        def trot_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.swing_height = c2.swing_height.at[:, -1].set(0.08)
            c2.z = c2.z.at[:, -1].set(0.27)
            c2.fl_phase = c2.fl_phase.at[:, -1].set(0.5)
            c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
            c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
            c2.rl_phase = c2.rl_phase.at[:, -1].set(0.0)
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.45)
            c2.cadence = c2.cadence.at[:, -1].set(2.0)
            return c2, w2

        def pace_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.swing_height = c2.swing_height.at[:, -1].set(0.08)
            c2.z = c2.z.at[:, -1].set(0.27)
            c2.fl_phase = c2.fl_phase.at[:, -1].set(0.5)
            c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
            c2.rr_phase = c2.rr_phase.at[:, -1].set(0.0)
            c2.rl_phase = c2.rl_phase.at[:, -1].set(0.5)
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.4)
            c2.cadence = c2.cadence.at[:, -1].set(2.25)
            return c2, w2

        def stand_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.swing_height = c2.swing_height.at[:, -1].set(0.0)
            c2.z = c2.z.at[:, -1].set(0.27)
            for name in ["fr_phase", "fl_phase", "rr_phase", "rl_phase"]:
                setattr(c2, name, getattr(c2, name).at[:, -1].set(0.0))
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(1.0)
            c2.cadence = c2.cadence.at[:, -1].set(0.0)
            c2.v_x = c2.v_x.at[:, -1].set(0.0)
            c2.v_y = c2.v_y.at[:, -1].set(0.0)
            c2.v_yaw = c2.v_yaw.at[:, -1].set(0.0)
            return c2, w2

        c, w = lax.cond(
            trot > 0.5,
            trot_branch,
            lambda _: lax.cond(pace > 0.5, pace_branch, stand_branch, operand=None),
            operand=None,
        )

        # adjust weights if not transitioning
        def not_transition_block(_):
            w2 = jax.tree.map(lambda x: x, w)
            gait_w = lax.cond(
                (trot + stand) > 0.5,
                lambda _: 2.0,
                lambda _: 4.0,
                operand=None,
            )
            w2.cost_weights.gait = w2.cost_weights.gait.at[:, -1].set(gait_w)
            w2.cost_weights.nom_joint_q = w2.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
            w2.cost_weights.nom_joint_v = w2.cost_weights.nom_joint_v.at[:, -1].set(1e-8)
            return w2

        w = lax.cond(transition < 0.5, not_transition_block, lambda _: w, operand=None)

        # shared weights for walk-like
        w.cost_weights.z = w.cost_weights.z.at[:, -1].set(5.0)
        w.cost_weights.log_R = w.cost_weights.log_R.at[:, -1].set(1.0)
        w.cost_weights.v_ang = w.cost_weights.v_ang.at[:, -1].set(1e-3)
        w.constraint_weights.thigh_contact = w.constraint_weights.thigh_contact.at[:, -1].set(0.0)
        w.constraint_weights.shank_top_contact = w.constraint_weights.shank_top_contact.at[:, -1].set(0.0)
        w.constraint_relaxation.thigh_contact = w.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
        w.constraint_relaxation.shank_top_contact = w.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)
        w.constraint_weights.z_err_lb = w.constraint_weights.z_err_lb.at[:, -1].set(0.0)
        w.constraint_weights.z_err_ub = w.constraint_weights.z_err_ub.at[:, -1].set(0.0)
        w.constraint_relaxation.z_err_lb = w.constraint_relaxation.z_err_lb.at[:, -1].set(1.0)
        w.constraint_relaxation.z_err_ub = w.constraint_relaxation.z_err_ub.at[:, -1].set(1.0)

        return c, w

    # --- GALLOP / BOUND branch -------------------------------------------------
    def gallop_like_branch(_):
        c, w = commands_new, weights_new
        c = jax.tree.map(lambda x: x, c)
        w2 = jax.tree.map(lambda x: x, w)
        c.z = c.z.at[:, -1].set(0.32)
        c.swing_height = c.swing_height.at[:, -1].set(0.08)
        c.tripod = c.tripod.at[:, -1].set(0.0)

        w2.cost_weights.nom_torque = w2.cost_weights.nom_torque.at[:, -1].set(4e-6)
        w2.cost_weights.v_lin = w2.cost_weights.v_lin.at[:, -1].set(5e-2)

        def bound_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.fl_phase = c2.fl_phase.at[:, -1].set(0.0)
            c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
            c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
            c2.rl_phase = c2.rl_phase.at[:, -1].set(0.5)
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.35)
            c2.cadence = c2.cadence.at[:, -1].set(2.25)
            return c2, w2

        def rotary_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.fl_phase = c2.fl_phase.at[:, -1].set(0.2)
            c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
            c2.rr_phase = c2.rr_phase.at[:, -1].set(0.7)
            c2.rl_phase = c2.rl_phase.at[:, -1].set(0.5)
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.3)
            c2.cadence = c2.cadence.at[:, -1].set(3.0)
            return c2, w2

        def transverse_branch(_):
            c2 = jax.tree.map(lambda x: x, c)
            c2.fl_phase = c2.fl_phase.at[:, -1].set(0.2)
            c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
            c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
            c2.rl_phase = c2.rl_phase.at[:, -1].set(0.7)
            c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.35)
            c2.cadence = c2.cadence.at[:, -1].set(2.75)
            return c2, w2

        # nested selection
        c, w = lax.cond(
            bound > 0.5,
            bound_branch,
            lambda _: lax.cond(rotary_gallop > 0.5,
                               rotary_branch,
                               transverse_branch,
                               operand=None),
            operand=None,
        )

        # weights
        def not_transition_block(_):
            w2 = jax.tree.map(lambda x: x, w)
            w2.cost_weights.gait = w2.cost_weights.gait.at[:, -1].set(7.0)
            w2.cost_weights.nom_joint_q = w2.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
            w2.cost_weights.nom_joint_v = w2.cost_weights.nom_joint_v.at[:, -1].set(5e-8)
            return w2

        w = lax.cond(transition < 0.5, not_transition_block, lambda _: w, operand=None)

        w.cost_weights.z = w.cost_weights.z.at[:, -1].set(1.0)
        w.cost_weights.log_R = w.cost_weights.log_R.at[:, -1].set(5e-1)
        w.cost_weights.v_ang = w.cost_weights.v_ang.at[:, -1].set(5e-4)
        w.constraint_weights.thigh_contact = w.constraint_weights.thigh_contact.at[:, -1].set(1e-3)
        w.constraint_weights.shank_top_contact = w.constraint_weights.shank_top_contact.at[:, -1].set(1e-3)
        w.constraint_relaxation.thigh_contact = w.constraint_relaxation.thigh_contact.at[:, -1].set(5e-4)
        w.constraint_relaxation.shank_top_contact = w.constraint_relaxation.shank_top_contact.at[:, -1].set(5e-4)
        w.constraint_weights.z_err_lb = w.constraint_weights.z_err_lb.at[:, -1].set(0.0)
        w.constraint_weights.z_err_ub = w.constraint_weights.z_err_ub.at[:, -1].set(0.0)
        w.constraint_relaxation.z_err_lb = w.constraint_relaxation.z_err_lb.at[:, -1].set(1.0)
        w.constraint_relaxation.z_err_ub = w.constraint_relaxation.z_err_ub.at[:, -1].set(1.0)

        return c, w

    def tripod_branch(_):
        c, w = commands_new, weights_new
        c2 = jax.tree.map(lambda x: x, c)
        w2 = jax.tree.map(lambda x: x, w)
        # --- command parameters ---
        c2.swing_height = c2.swing_height.at[:, -1].set(0.08)
        c2.z = c2.z.at[:, -1].set(0.27)
        c2.fl_phase = c2.fl_phase.at[:, -1].set(0.0)
        c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
        c2.rl_phase = c2.rl_phase.at[:, -1].set(0.5)
        c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
        c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.6)
        c2.cadence = c2.cadence.at[:, -1].set(2.25)
        c2.tripod = c2.tripod.at[:, -1].set(1.0)
        #        z_err_lb=-0.02,
        #z_err_ub=0.08,
        c2.z_err_lb = c2.z_err_lb.at[:, -1].set(-0.02)
        c2.z_err_ub = c2.z_err_ub.at[:, -1].set(0.08)

        # --- constraint weights (tripod-specific) ---
        # keep only z_err bounds
        w2.constraint_weights.thigh_contact = w2.constraint_weights.thigh_contact.at[:, -1].set(0.0)
        w2.constraint_weights.shank_top_contact = w2.constraint_weights.shank_top_contact.at[:, -1].set(0.0)
        w2.constraint_relaxation.thigh_contact = w2.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
        w2.constraint_relaxation.shank_top_contact = w2.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)
        w2.constraint_weights.z_err_lb = w2.constraint_weights.z_err_lb.at[:, -1].set(1.0)
        w2.constraint_weights.z_err_ub = w2.constraint_weights.z_err_ub.at[:, -1].set(1.0)
        w2.constraint_relaxation.z_err_lb = w2.constraint_relaxation.z_err_lb.at[:, -1].set(0.01)
        w2.constraint_relaxation.z_err_ub = w2.constraint_relaxation.z_err_ub.at[:, -1].set(0.01)

        # --- cost weights (tripod-specific emphasis) ---
        w2.cost_weights.z = w2.cost_weights.z.at[:, -1].set(5e2 / 1e3)          # from evaluation weights
        w2.cost_weights.log_R = w2.cost_weights.log_R.at[:, -1].set(7e2 / 1e3)
        w2.cost_weights.v_lin = w2.cost_weights.v_lin.at[:, -1].set(5e1 / 1e3)
        w2.cost_weights.v_ang = w2.cost_weights.v_ang.at[:, -1].set(0.5 / 1e3)
        w2.cost_weights.nom_joint_q = w2.cost_weights.nom_joint_q.at[:, -1].set(1e1 / 1e3)
        w2.cost_weights.nom_joint_v = w2.cost_weights.nom_joint_v.at[:, -1].set(5e-3 / 1e3)
        w2.cost_weights.nom_torque = w2.cost_weights.nom_torque.at[:, -1].set(4e-3 / 1e3)
        w2.cost_weights.gait = w2.cost_weights.gait.at[:, -1].set(7e3 / 1e3)
        w2.cost_weights.mech_work = w2.cost_weights.mech_work.at[:, -1].set(1e-3 / 1e3)

        return c2, w2


    
    def rough_terrain_branch(_):
        c2 = jax.tree.map(lambda x: x, commands_new)
        w2 = jax.tree.map(lambda x: x, weights_new)

        c2.tripod = c2.tripod.at[:, -1].set(0.0)
        c2.swing_height = c2.swing_height.at[:, -1].set(0.08)
        c2.z = c2.z.at[:, -1].set(0.3)
        c2.fl_phase = c2.fl_phase.at[:, -1].set(0.5)
        c2.fr_phase = c2.fr_phase.at[:, -1].set(0.0)
        c2.rr_phase = c2.rr_phase.at[:, -1].set(0.5)
        c2.rl_phase = c2.rl_phase.at[:, -1].set(0.0)
        c2.duty_ratio = c2.duty_ratio.at[:, -1].set(0.45)
        c2.cadence = c2.cadence.at[:, -1].set(2.5)
        c2.pitch_err_lb = c2.pitch_err_lb.at[:, -1].set(-0.3)
        c2.pitch_err_ub = c2.pitch_err_ub.at[:, -1].set(0.6)
        c2.roll_err_lb = c2.roll_err_lb.at[:, -1].set(-0.2)
        c2.roll_err_ub = c2.roll_err_ub.at[:, -1].set(0.2)
        c2.yaw_err_lb = c2.yaw_err_lb.at[:, -1].set(-1.0)
        c2.yaw_err_ub = c2.yaw_err_ub.at[:, -1].set(1.0)
        c2.z_err_lb = c2.z_err_lb.at[:, -1].set(-0.02)
        c2.z_err_ub = c2.z_err_ub.at[:, -1].set(0.08)
        c2.max_joint_v = c2.max_joint_v.at[:, -1].set(12.0)

        w2.cost_weights.nom_torque = w2.cost_weights.nom_torque.at[:, -1].set(4e-6)
        w2.cost_weights.v_lin = w2.cost_weights.v_lin.at[:, -1].set(5e-2)


        w2.cost_weights.gait = w2.cost_weights.gait.at[:, -1].set(2.0)
        w2.cost_weights.nom_joint_q = w2.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
        w2.cost_weights.nom_joint_v = w2.cost_weights.nom_joint_v.at[:, -1].set(1e-8)


        #         w.cost_weights.z = w.cost_weights.z.at[:, -1].set(5.0)
        # w.cost_weights.log_R = w.cost_weights.log_R.at[:, -1].set(1.0)
        # w.cost_weights.v_ang = w.cost_weights.v_ang.at[:, -1].set(1e-3)
        # w.constraint_weights.thigh_contact = w.constraint_weights.thigh_contact.at[:, -1].set(0.0)
        # w.constraint_weights.shank_top_contact = w.constraint_weights.shank_top_contact.at[:, -1].set(0.0)
        # w.constraint_relaxation.thigh_contact = w.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
        # w.constraint_relaxation.shank_top_contact = w.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)
        # w.constraint_weights.z_err_lb = w.constraint_weights.z_err_lb.at[:, -1].set(0.0)
        # w.constraint_weights.z_err_ub = w.constraint_weights.z_err_ub.at[:, -1].set(0.0)
        # w.constraint_relaxation.z_err_lb = w.constraint_relaxation.z_err_lb.at[:, -1].set(1.0)
        # w.constraint_relaxation.z_err_ub = w.constraint_relaxation.z_err_ub.at[:, -1].set(1.0)

        # shared weights for walk-like
        w2.cost_weights.z = w2.cost_weights.z.at[:, -1].set(5.0)
        w2.cost_weights.log_R = w2.cost_weights.log_R.at[:, -1].set(0.05)
        w2.cost_weights.v_ang = w2.cost_weights.v_ang.at[:, -1].set(1e-3)
        w2.constraint_weights.thigh_contact = w2.constraint_weights.thigh_contact.at[:, -1].set(0.001)
        w2.constraint_weights.shank_top_contact = w2.constraint_weights.shank_top_contact.at[:, -1].set(0.001)
        w2.constraint_relaxation.thigh_contact = w2.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
        w2.constraint_relaxation.shank_top_contact = w2.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)
        w2.constraint_weights.z_err_lb = w2.constraint_weights.z_err_lb.at[:, -1].set(0.0)
        w2.constraint_weights.z_err_ub = w2.constraint_weights.z_err_ub.at[:, -1].set(0.0)
        w2.constraint_relaxation.z_err_lb = w2.constraint_relaxation.z_err_lb.at[:, -1].set(1.0)
        w2.constraint_relaxation.z_err_ub = w2.constraint_relaxation.z_err_ub.at[:, -1].set(1.0)

        # w2.constraint_weights.z_err_lb = w2.constraint_weights.z_err_lb.at[:, -1].set(0.1)
        # w2.constraint_weights.z_err_ub = w2.constraint_weights.z_err_ub.at[:, -1].set(0.1)
        # w2.constraint_relaxation.z_err_lb = w2.constraint_relaxation.z_err_lb.at[:, -1].set(0.01)
        # w2.constraint_relaxation.z_err_ub = w2.constraint_relaxation.z_err_ub.at[:, -1].set(0.01)

        w2.constraint_weights.rpy_err_lb = w2.constraint_weights.rpy_err_lb.at[:, -1].set(0.1)
        w2.constraint_weights.rpy_err_ub = w2.constraint_weights.rpy_err_ub.at[:, -1].set(0.1)
        w2.constraint_relaxation.rpy_err_lb = w2.constraint_relaxation.rpy_err_lb.at[:, -1].set(0.01)
        w2.constraint_relaxation.rpy_err_ub = w2.constraint_relaxation.rpy_err_ub.at[:, -1].set(0.01)

        # w2.constraint_weights.hip_v = w2.constraint_weights.hip_v.at[:, -1].set(0.001)
        # w2.constraint_weights.knee_v = w2.constraint_weights.knee_v.at[:, -1].set(0.001)
        # w2.constraint_weights.ankle_v = w2.constraint_weights.ankle_v.at[:, -1].set(0.001)

        # w2.constraint_weights.hip_q_lb = w2.constraint_weights.hip_q_lb.at[:, -1].set(0.01)
        # w2.constraint_weights.hip_q_ub = w2.constraint_weights.hip_q_ub.at[:, -1].set(0.01)
        # w2.constraint_weights.knee_q_lb = w2.constraint_weights.knee_q_lb.at[:, -1].set(0.01)
        # w2.constraint_weights.knee_q_ub = w2.constraint_weights.knee_q_ub.at[:, -1].set(0.01)
        # w2.constraint_weights.ankle_q_lb = w2.constraint_weights.ankle_q_lb.at[:, -1].set(0.01)
        # w2.constraint_weights.ankle_q_ub = w2.constraint_weights.ankle_q_ub.at[:, -1].set(0.01)


        # w2.constraint_relaxation.hip_v = w2.constraint_relaxation.hip_v.at[:, -1].set(1.0)
        # w2.constraint_relaxation.knee_v = w2.constraint_relaxation.knee_v.at[:, -1].set(1.0)
        # w2.constraint_relaxation.ankle_v = w2.constraint_relaxation.ankle_v.at[:, -1].set(1.0)

        # w2.constraint_relaxation.hip_q_lb = w2.constraint_relaxation.hip_q_lb.at[:, -1].set(0.1)
        # w2.constraint_relaxation.hip_q_ub = w2.constraint_relaxation.hip_q_ub.at[:, -1].set(0.1)
        # w2.constraint_relaxation.knee_q_lb = w2.constraint_relaxation.knee_q_lb.at[:, -1].set(0.1)
        # w2.constraint_relaxation.knee_q_ub = w2.constraint_relaxation.knee_q_ub.at[:, -1].set(0.1)
        # w2.constraint_relaxation.ankle_q_lb = w2.constraint_relaxation.ankle_q_lb.at[:, -1].set(0.5)
        # w2.constraint_relaxation.ankle_q_ub = w2.constraint_relaxation.ankle_q_ub.at[:, -1].set(0.5)

        # w2.constraint_weights.torque_limits = w2.constraint_weights.torque_limits.at[:, -1].set(0.0001)
        # w2.constraint_relaxation.torque_limits = w2.constraint_relaxation.torque_limits.at[:, -1].set(1.0)


        # w2.constraint_weights.swing = w2.constraint_weights.swing.at[:, -1].set(0.005)
        # w2.constraint_relaxation.swing = w2.constraint_relaxation.swing.at[:, -1].set(0.1)

        return c2, w2



    # choose between walk-like vs gallop-like groups
    cond_walk_like = (trot + pace + stand) > 0.5
    tripod_cond = tripod > 0.5
    rough_terrain_cond = rough_terrain > 0.5
    commands_new, weights_new = lax.cond(
        cond_walk_like,
        walk_like_branch,
        lambda _: lax.cond(
            tripod_cond,
            tripod_branch,
            lambda _: lax.cond(rough_terrain_cond,
                               rough_terrain_branch,
                               gallop_like_branch,
        operand=None),
        operand=None),
        operand=None,
    )

    return commands_new, weights_new





def smooth(cost_weights, commands, cost_weights_target, commands_target):


    alpha_geom = 0.5
    alpha_std = 0.5

    cost_weights_new = jax.tree.map(
        lambda w_prev, w_tgt: _std_fade_weight(w_prev, w_tgt, alpha_geom),
        cost_weights,
        cost_weights_target
    )



    commands_new = jax.tree.map(
        lambda c_prev, c_tgt: _std_fade_weight(c_prev, c_tgt, alpha_std),
        commands,
        commands_target,
    )

    swing_height = commands_target.swing_height
    phase_fl = commands_target.fl_phase
    phase_fr = commands_target.fr_phase
    phase_rr = commands_target.rr_phase
    phase_rl = commands_target.rl_phase

    cadence = commands_target.cadence

    z_weight = cost_weights_target.cost_weights.z
    log_r_weight = cost_weights_target.cost_weights.log_R

    commands_new.swing_height = swing_height
    commands_new.fl_phase = phase_fl
    commands_new.fr_phase = phase_fr
    commands_new.rr_phase = phase_rr
    commands_new.rl_phase = phase_rl

    commands_new.cadence = cadence

    cost_weights_new.cost_weights.z = z_weight
    cost_weights_new.cost_weights.log_R = log_r_weight

    return commands_new, cost_weights_new









class ThreadedGo2Deploy(ThreadedOnlineDataCollectionModule):
    def __init__(self, *args, **kwargs):
        self.go2_xml_path = "./Mj_models/Go2_dog/go2_torque_test.xml"
        self.scene_xml_path = "./Mj_models/Go2_dog/scene_torque_test.xml"

        super().__init__(*args, **kwargs)

        num_envs = self.num_envs
        num_envs_random = num_envs // 3
        num_envs_locomotion = num_envs // 3
        num_envs_random_pose = num_envs - num_envs_random - num_envs_locomotion
        self.num_envs_random = num_envs_random
        self.num_envs_locomotion = num_envs_locomotion
        self.num_envs_random_pose = num_envs_random_pose
        self.get_new_commands = jax.jit(lambda key: get_new_commands_(self.horizon, 
                                                                    num_envs_random, 
                                                                    num_envs_locomotion, 
                                                                    num_envs_random_pose, 
                                                                    key))
        self.commands = self.reset_commands()


        # mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
        mujoco.mj_fwdPosition(self.mj_models[0], self.mj_data)

        self.n_eval_tasks = 4
    
    def eval_commands(self):
        #cmd = trotting_commands(self.horizon)
        # cmd = rough_terrain_commands(self.horizon)
        
        # names = cmd[1]
        # cmd = init_deploy_commands_trot(self.horizon)
        # cmd = rearing_commands(self.horizon)
        # cmd = galloping_commands(self.horizon)
        # cmd = walking_commands(self.horizon)
        #cmd = dial_vs_spline_shooter_commands(self.horizon)
        #cmd = evaluation_commands(self.horizon)
        outs = trotting_evaluation(self.horizon)
        #outs = galloping_evaluation(self.horizon)
        #outs = tripod_evaluation(self.horizon)
        #outs = bounding_evaluation(self.horizon)
        return outs
        #return [[cmd], [names[0]]]

    def eval_weights(self):
        #weights = rough_terrain_weights()
        #weights = constraint_test_weights()
        #weights = testing_weights()
        #weights = evaluation_weights()
        weights = trotting_evaluation_weights()
        #weights = galloping_evaluation_weights()
        #weights = tripod_evaluation_weights()
        #weights  = setup_weights()
        #weights = gallop_weights()
        # weights = rearing_weights()
        # weights = galloping_weights()
        #weights = dial_vs_spline_shooter_weights()
        #weights = trotting_weights()
        #weights = bounding_evaluation_weights()
        return weights
    
    def eval_env(self):
        
        xml, spawn_position = get_base_xml_spawn_position(self)
        #xml, spawn_position = get_base_xml_terrain_spawn_position(self, terrain_type="flat")
        #xml, spawn_position = get_base_xml_terrain_spawn_position(self, terrain_type="stairs")
        #xml, spawn_position = get_base_xml_terrain_spawn_position(self, terrain_type="perlin")
        #xml, spawn_position = get_base_xml_terrain_spawn_position(self, terrain_type="rough")
        return xml, spawn_position
    
    def new_mj_model(self):
        new_xml, spawn_position = self.eval_env()
        model =  mujoco.MjModel.from_xml_string(new_xml)
        model.opt.enableflags = 1
        #damp_ratio = np.random.uniform(0.9, 1.1)
        damp_ratio = 1.0
        model.opt.o_solref = np.array([0.01, damp_ratio])
        model.opt.integrator = 3
        model.opt.timestep = 0.005
        return model, spawn_position

    
    def reset_states(self):
        return reset_states_(self, self.spawn_pos)
    

    def set_weights(self):
        weights = self.eval_weights()

        cost_weights, constraint_weights, constraint_relaxation, global_cost_weights = repeat_weights(*weights, self.horizon)

        self.cost_weights = CostWeights(cost_weights=cost_weights,
                            constraint_weights=constraint_weights, 
                            constraint_relaxation=constraint_relaxation, 
                            global_cost_weights=global_cost_weights)



    def reset_commands(self):
        self.rng_key, key = jax.random.split(self.rng_key)
        self.commands = self.eval_commands()[0]
        self.commands = jax.tree.map(lambda x: x.reshape(self.num_envs, self.horizon, 1), self.commands)


    def set_evaluation_commands(self):
        self.evaluation_commands = self.eval_commands()[0]
        self.n_eval_tasks = len(self.evaluation_commands)

    def set_evaluation_weights(self):
        weights = self.eval_weights()
        cost_weight, constraint_weights, constraint_relaxation, global_cost_weights = repeat_weights(*weights, self.horizon)
        cost_weight = jax.tree.map(lambda x: x[None, :], cost_weight)
        constraint_weights = jax.tree.map(lambda x: x[None, :], constraint_weights)
        constraint_relaxation = jax.tree.map(lambda x: x[None, :], constraint_relaxation)
        global_cost_weights = jax.tree.map(lambda x: x[None, :], global_cost_weights)
        weights = CostWeights(cost_weights=cost_weight,
                    constraint_weights=constraint_weights, 
                    constraint_relaxation=constraint_relaxation,
                    global_cost_weights=global_cost_weights)
        
        self.evaluation_cost_weights = []
        for i in range(self.n_eval_tasks):
            self.evaluation_cost_weights.append(weights)

    def set_evaluation_state(self):
        mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
        rand_z = np.random.uniform(0.0, 0.01)
        print(f"Random z offset for evaluation: {rand_z}")
        self.mj_data.qpos[2] += rand_z

        ## for stairs
        
        # self.mj_data.qpos[0] -= 2.0
        # self.mj_data.qpos[2] += 0.2

        # for perlin
        # self.mj_data.qpos[:3] = self.eval_spawn_pos #np.array([-5.0, 0.0, 0.5]) # # #np.array([0.0, 0.0, 2.5]) #np.array([0.0, 0.0, 1.6]) #
        #self.mj_data.qpos[2] += 0.4
        #self.mj_data.qpos[2] += 0.02
        # # self.mj_data.qpos[1] += 3.5
        # self.mj_data.qpos[0] += 2.5
        
        # yaw_noise_random_envs = -3.14/2 + 0.1 #np.random.uniform(-np.pi, np.pi, (1,))
        # yaw_quats = R.from_euler('z', yaw_noise_random_envs).as_quat()  # shape: (N, 4), [x, y, z, w]

        # # # # [w, x, y, z] order for mujoco
        # yaw_quats = np.roll(yaw_quats, 1) 
        # self.mj_data.qpos[3:7] = yaw_quats  # Set the quaternion orientation

    def set_evaluation_mj_model(self):
        xml, spawn_position = self.eval_env()
        self.eval_spawn_pos = spawn_position
        self.mj_models[0] = mujoco.MjModel.from_xml_string(xml)

        self.mj_models[0].opt.enableflags = 1
        self.mj_models[0].opt.o_solref = np.array([0.01, 1.0])
        self.mj_models[0].opt.integrator = 3
        self.mj_models[0].opt.timestep = 0.005


    def set_evaluation_task_names(self):
        self.evaluation_task_names = self.eval_commands()[1]


    # def callback_before_eval_agent(self, step):
    #     self.tripod_gait_transition(step)

    def tripod_gait_transition(self, step):
        if step < 150:
            # self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
            #                                       trot=False,
            #                                       rotary_gallop=False,
            #                                       transverse_gallop=False,
            #                                       bound=False,
            #                                       pace=False,
            #                                       stand=True,
            #                                       transition=False)

            self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                  trot=0.0,
                                                  rotary_gallop=0.0,
                                                  transverse_gallop=0.0,
                                                  bound=0.0,
                                                  pace=0.0,
                                                  stand=1.0,
                                                  tripod=0.0,
                                                  transition=0.0, 
                                                  rough_terrain=0.0)
        elif step < 500:
            self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                  trot=0.0,
                                                  rotary_gallop=0.0,
                                                  transverse_gallop=0.0,
                                                  bound=0.0,
                                                  pace=0.0,
                                                  stand=0.0,
                                                  tripod=0.0,
                                                  transition=0.0, 
                                                rough_terrain=1.0)
            self.commands.v_x = self.commands.v_x.at[:, -1].set(-1.0)
        else:
            if step == 500:
                self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                      trot=0.0,
                                                      rotary_gallop=1.0,
                                                      transverse_gallop=0.0,
                                                      bound=0.0,
                                                      pace=0.0,
                                                      stand=0.0,
                                                      tripod=0.0,
                                                      transition=1.0, 
                                                        rough_terrain=0.0)
                self.commands.v_x = self.commands.v_x.at[:, -1].set(1.5)
            else:
                self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                        trot=0.0,
                                                        rotary_gallop=1.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        tripod=0.0,
                                                        transition=0.0, 
                                                        rough_terrain=0.0)
                self.commands.v_x = self.commands.v_x.at[:, -1].set(1.5)
            

    # def callback_before_eval_agent(self, step):
    #     self.gait_transition(step)

        
    def gait_transition(self, step):

        # def gait_commands_weights(commands, weights, trot, rotary_gallop, transverse_gallop, bound, pace, stand, transition):

        #     global_time = jax.tree.map(lambda x: x, commands.global_time)
        #     commands_new = jax.tree.map(lambda x: jnp.roll(x, shift=-1), commands)
        #     weights_new = jax.tree.map(lambda x: jnp.roll(x, shift=-1), weights)
        #     commands_new.global_time = global_time

        #     weights_new.cost_weights.nom_torque = weights_new.cost_weights.nom_torque.at[:, -1].set(4e-6)
        #     weights_new.cost_weights.v_lin = weights_new.cost_weights.v_lin.at[:, -1].set(5e-2)

        #     if transition:
        #         weights_new.cost_weights.gait = weights_new.cost_weights.gait.at[:, -1].set(0.0)
        #         weights_new.cost_weights.nom_joint_v = weights_new.cost_weights.nom_joint_v.at[:, -1].set(1e-6)
        #         weights_new.cost_weights.nom_joint_q = weights_new.cost_weights.nom_joint_q.at[:, -1].set(5e-4)

        #     if trot or pace or stand:
        #         if trot or pace:
        #             commands_new.swing_height = commands_new.swing_height.at[:, -1].set(0.08)
        #             commands_new.z = commands_new.z.at[:, -1].set(0.27)
        #             if trot:
        #                 commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.5)
        #                 commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #                 commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.5)
        #                 commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.0)
        #                 commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(0.45)
        #                 commands_new.cadence = commands_new.cadence.at[:, -1].set(2.0)

        #             elif pace:
        #                 commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.5)
        #                 commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #                 commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.0)
        #                 commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.5)
        #                 commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(0.4)
        #                 commands_new.cadence = commands_new.cadence.at[:, -1].set(2.25)
        #         else:
        #             commands_new.swing_height = commands_new.swing_height.at[:, -1].set(0.0)
        #             commands_new.z = commands_new.z.at[:, -1].set(0.27)
        #             commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #             commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.0)
        #             commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.0)
        #             commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.0)
        #             commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(1.0)
        #             commands_new.cadence = commands_new.cadence.at[:, -1].set(0.0)
        #             commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
        #             commands_new.v_y = commands_new.v_y.at[:, -1].set(0.0)
        #             commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)

        #         if not transition:
        #             if trot or stand:
        #                 weights_new.cost_weights.gait = weights_new.cost_weights.gait.at[:, -1].set(2.0)
        #             elif pace:
        #                 weights_new.cost_weights.gait = weights_new.cost_weights.gait.at[:, -1].set(4.0)
        #             weights_new.cost_weights.nom_joint_q = weights_new.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
        #             weights_new.cost_weights.nom_joint_v = weights_new.cost_weights.nom_joint_v.at[:, -1].set(1e-8)

        #         weights_new.cost_weights.z = weights_new.cost_weights.z.at[:, -1].set(5.0)
        #         weights_new.cost_weights.log_R = weights_new.cost_weights.log_R.at[:, -1].set(1.0)
        #         weights_new.cost_weights.v_ang = weights_new.cost_weights.v_ang.at[:, -1].set(1e-3)

        #         weights_new.constraint_weights.thigh_contact = weights_new.constraint_weights.thigh_contact.at[:, -1].set(0.0)
        #         weights_new.constraint_weights.shank_top_contact = weights_new.constraint_weights.shank_top_contact.at[:, -1].set(0.0)
        #         weights_new.constraint_relaxation.thigh_contact = weights_new.constraint_relaxation.thigh_contact.at[:, -1].set(1.0)
        #         weights_new.constraint_relaxation.shank_top_contact = weights_new.constraint_relaxation.shank_top_contact.at[:, -1].set(1.0)


        #     elif bound or transverse_gallop or rotary_gallop:
        #         commands_new.z = commands_new.z.at[:, -1].set(0.32)
        #         commands_new.swing_height = commands_new.swing_height.at[:, -1].set(0.08)
        #         if bound:
        #             commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.0)
        #             commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #             commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.5)
        #             commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.5)
        #             commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(0.35)
        #             commands_new.cadence = commands_new.cadence.at[:, -1].set(2.25)
        #         elif rotary_gallop:
        #             commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.2)
        #             commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #             commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.7)
        #             commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.5)
        #             commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(0.3)
        #             commands_new.cadence = commands_new.cadence.at[:, -1].set(3.0)
                
        #         elif transverse_gallop:
        #             commands_new.fl_phase = commands_new.fl_phase.at[:, -1].set(0.2)
        #             commands_new.fr_phase = commands_new.fr_phase.at[:, -1].set(0.0)
        #             commands_new.rr_phase = commands_new.rr_phase.at[:, -1].set(0.5)
        #             commands_new.rl_phase = commands_new.rl_phase.at[:, -1].set(0.7)
        #             commands_new.duty_ratio = commands_new.duty_ratio.at[:, -1].set(0.35)
        #             commands_new.cadence = commands_new.cadence.at[:, -1].set(2.75)


        #         if not transition:
        #             weights_new.cost_weights.gait = weights_new.cost_weights.gait.at[:, -1].set(7.0)
        #             weights_new.cost_weights.nom_joint_q = weights_new.cost_weights.nom_joint_q.at[:, -1].set(1e-2)
        #             weights_new.cost_weights.nom_joint_v = weights_new.cost_weights.nom_joint_v.at[:, -1].set(5e-8)

        #         weights_new.cost_weights.z = weights_new.cost_weights.z.at[:, -1].set(1.0)
        #         weights_new.cost_weights.log_R = weights_new.cost_weights.log_R.at[:, -1].set(5e-1)
        #         weights_new.cost_weights.v_ang = weights_new.cost_weights.v_ang.at[:, -1].set(5e-4)

        #         weights_new.constraint_weights.thigh_contact = weights_new.constraint_weights.thigh_contact.at[:, -1].set(1e-3)
        #         weights_new.constraint_weights.shank_top_contact = weights_new.constraint_weights.shank_top_contact.at[:, -1].set(1e-3)
        #         weights_new.constraint_relaxation.thigh_contact = weights_new.constraint_relaxation.thigh_contact.at[:, -1].set(5e-4)
        #         weights_new.constraint_relaxation.shank_top_contact = weights_new.constraint_relaxation.shank_top_contact.at[:, -1].set(5e-4)

        #     return commands_new, weights_new
                


        if step < 150:
            # self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
            #                                       trot=False,
            #                                       rotary_gallop=False,
            #                                       transverse_gallop=False,
            #                                       bound=False,
            #                                       pace=False,
            #                                       stand=True,
            #                                       transition=False)

            self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                  trot=0.0,
                                                  rotary_gallop=0.0,
                                                  transverse_gallop=0.0,
                                                  bound=0.0,
                                                  pace=0.0,
                                                  stand=1.0,
                                                  transition=0.0)

        elif step <= 250:
            # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
            #                                         trot=True,
            #                                         rotary_gallop=False,
            #                                         transverse_gallop=False,
            #                                         bound=False,
            #                                         pace=False,
            #                                         stand=False,
            #                                         transition=False)
            # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.5)
            # self.commands, self.cost_weights = commands_new, cost_weights_new
            self.commands, self.cost_weights = gait_commands_weights(self.commands, self.cost_weights,
                                                  trot=1.0,
                                                  rotary_gallop=0.0,
                                                  transverse_gallop=0.0,
                                                  bound=0.0,
                                                  pace=0.0,
                                                  stand=0.0,
                                                  transition=0.0)
            self.commands.v_x = self.commands.v_x.at[:, -1].set(0.5)
            

        elif step <= 450:

            if step == 251:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=False,
                #                                         rotary_gallop=False,
                #                                         transverse_gallop=False,
                #                                         bound=True,
                #                                         pace=False,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                      trot=0.0,
                                                      rotary_gallop=0.0,
                                                      transverse_gallop=0.0,
                                                      bound=1.0,
                                                      pace=0.0,
                                                      stand=0.0,
                                                      transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=False,
                #                                       rotary_gallop=False,
                #                                       transverse_gallop=False,
                #                                       bound=True,
                #                                       pace=False,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                      trot=0.0,
                                                      rotary_gallop=0.0,
                                                      transverse_gallop=0.0,
                                                      bound=1.0,
                                                      pace=0.0,
                                                      stand=0.0,
                                                      transition=0.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

   
        elif step <= 800:

            if step == 551:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=False,
                #                                         rotary_gallop=True,
                #                                         transverse_gallop=False,
                #                                         bound=False,
                #                                         pace=False,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.5)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                        trot=0.0,
                                                        rotary_gallop=1.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(1.5)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=False,
                #                                       rotary_gallop=True,
                #                                       transverse_gallop=False,
                #                                       bound=False,
                #                                       pace=False,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.5)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                      trot=0.0,
                                                      rotary_gallop=1.0,
                                                      transverse_gallop=0.0,
                                                      bound=0.0,
                                                      pace=0.0,
                                                      stand=0.0,
                                                      transition=0.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(1.5)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

        elif step <= 1050:

            if step == 801:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=False,
                #                                         rotary_gallop=False,
                #                                         transverse_gallop=True,
                #                                         bound=False,
                #                                         pace=False,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.4)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
                commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                                                        self.cost_weights,
                                                        trot=0.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=1.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.4)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=False,
                #                                       rotary_gallop=False,
                #                                       transverse_gallop=True,
                #                                       bound=False,
                #                                       pace=False,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.4)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                      trot=0.0,
                                                      rotary_gallop=0.0,
                                                      transverse_gallop=1.0,
                                                      bound=0.0,
                                                      pace=0.0,
                                                      stand=0.0,
                                                      transition=0.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(1.0)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.4)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

        elif step <= 1250:
            if step == 1051:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=False,
                #                                         rotary_gallop=False,
                #                                         transverse_gallop=False,
                #                                         bound=False,
                #                                         pace=True,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.75)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                                                        self.cost_weights,
                                                        trot=0.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=1.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.75)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=False,
                #                                       rotary_gallop=False,
                #                                       transverse_gallop=False,
                #                                       bound=False,
                #                                       pace=True,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.75)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                        trot=0.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=1.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.75)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

        elif step <= 1400:

            if step == 1251:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=True,
                #                                         rotary_gallop=False,
                #                                         transverse_gallop=False,
                #                                         bound=False,
                #                                         pace=False,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                # commands_new.v_y = commands_new.v_y.at[:, -1].set(0.5)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(-1.5)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
                commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                                                        self.cost_weights,
                                                        trot=1.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                commands_new.v_y = commands_new.v_y.at[:, -1].set(0.5)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(-1.5)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=True,
                #                                       rotary_gallop=False,
                #                                       transverse_gallop=False,
                #                                       bound=False,
                #                                       pace=False,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                # commands_new.v_y = commands_new.v_y.at[:, -1].set(0.5)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(-1.5)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                        trot=1.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=0.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                commands_new.v_y = commands_new.v_y.at[:, -1].set(0.5)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(-1.5)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

        else:
            if step == 1401:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                #                                         self.cost_weights,
                #                                         trot=True,
                #                                         rotary_gallop=False,
                #                                         transverse_gallop=False,
                #                                         bound=False,
                #                                         pace=False,
                #                                         stand=False,
                #                                         transition=True)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                # commands_new.v_y = commands_new.v_y.at[:, -1].set(1.0)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

                commands_new, cost_weights_new = gait_commands_weights(self.commands, 
                                                        self.cost_weights,
                                                        trot=1.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=1.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                commands_new.v_y = commands_new.v_y.at[:, -1].set(1.0)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)

            else:
                # commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                #                                       trot=True,
                #                                       rotary_gallop=False,
                #                                       transverse_gallop=False,
                #                                       bound=False,
                #                                       pace=False,
                #                                       stand=False,
                #                                       transition=False)
                # commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                # commands_new.v_y = commands_new.v_y.at[:, -1].set(1.0)
                # commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                # self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)
                commands_new, cost_weights_new = gait_commands_weights(self.commands, self.cost_weights,
                                                        trot=1.0,
                                                        rotary_gallop=0.0,
                                                        transverse_gallop=0.0,
                                                        bound=0.0,
                                                        pace=0.0,
                                                        stand=0.0,
                                                        transition=0.0)
                commands_new.v_x = commands_new.v_x.at[:, -1].set(0.0)
                commands_new.v_y = commands_new.v_y.at[:, -1].set(1.0)
                commands_new.v_yaw = commands_new.v_yaw.at[:, -1].set(0.0)
                self.commands, self.cost_weights = smooth(self.cost_weights, self.commands, cost_weights_new, commands_new)