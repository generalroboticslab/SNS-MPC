import jax
import jax.numpy as jnp
import numpy as np
import optax
from abc import ABC
from Training import JaxDataLoader, OnlineTorchDataset, DataWrangler
from tqdm import tqdm
from Common import *
from Common.Go2_dog import Go2StatesStruct, Go2ActionsStruct, Go2ObservationsStruct
from .base_data_collector import *
from jax.tree_util import register_pytree_node_class
import torch
import glob


def init_optimizer(lr, weight_decay=0.0, b1=0.9, b2=0.99, eps=1e-8, eps_root=0, clip=False, clip_thresh=5e-1, clip_eps=1e-3, dtype=jnp.float32):
    lr = jnp.array(lr, dtype=dtype)
    b1 = jnp.array(b1, dtype=dtype)
    b2 = jnp.array(b2, dtype=dtype)
    eps = jnp.array(eps, dtype=dtype)
    eps_root = jnp.array(eps_root, dtype=dtype)
    clip_thresh = jnp.array(clip_thresh, dtype=dtype)
    clip_eps = jnp.array(clip_eps, dtype=dtype)
    optim = optax.inject_hyperparams(optax.lion)(learning_rate=lr, b1=b1, b2=b2, weight_decay=0.0) #eps=eps, eps_root=eps_root
    if clip:
        clipper = optax.adaptive_grad_clip(clip_thresh, clip_eps)
        optim = optax.chain(clipper, optim)
    return optim



@jax.jit
def update_mj_dataset(mj_dataset, batch):
    mj_dataset.sensor_trajectory.array = batch.sensor_trajectory
    mj_dataset.action_trajectory.array = batch.action_trajectory
    return mj_dataset

# def _take_step(optimizer, loss, opt_state, _model, x):
#     values, grads = jax.value_and_grad(loss, has_aux=True)(_model, x)
#     updates, opt_state = optimizer.update(grads, opt_state, _model)
#     _model = optax.apply_updates(_model, updates)

#     return _model, opt_state, values[1]


def _take_step_dynamics(optimizer, loss, opt_state, dynamics_model, observer_model, x, discount, alpha_lipschitz_dynamics):
    values, grads = jax.value_and_grad(loss, has_aux=True)(dynamics_model, observer_model, x, discount, alpha_lipschitz_dynamics)
    updates, opt_state = optimizer.update(grads, opt_state, dynamics_model)
    dynamics_model = optax.apply_updates(dynamics_model, updates)
    return dynamics_model, opt_state, values[1]
    

def _take_step_observer(optimizer, loss, opt_state, observer_model, dynamics_model, x):
    values, grads = jax.value_and_grad(loss, has_aux=True)(observer_model, dynamics_model, x)
    updates, opt_state = optimizer.update(grads, opt_state, observer_model)
    observer_model = optax.apply_updates(observer_model, updates)

    return observer_model, opt_state, values[1]
    
@jax.jit
def process_data(_data_wrangler, _mj_dataset):
    return _data_wrangler.process_data_training(_mj_dataset)

def innovation_data(past_states_tm1_est, past_actions_tm1, future_actions_tm1, past_observations_t, dynamics_model):


    ps_tm1_norm = past_states_tm1_est.normalized()
    pa_tm1_norm = past_actions_tm1.normalized()
    fa_tm1_norm = future_actions_tm1.normalized()
    po_t_norm = past_observations_t.normalized()

    def innovation_r2_func(x, y):
        def innovation_r2_func_(ps_tm1_norm, pa_tm1_norm, fa_tm1_norm, po_t_norm):
            r, states_t_pred = innovation_residual(ps_tm1_norm, pa_tm1_norm, fa_tm1_norm, po_t_norm, dynamics_model)
            r2 = r @ r
            return r2, (r, states_t_pred)
        return innovation_r2_func_(x[0], x[1], x[2], y)
    
    grad_fn = jax.grad(innovation_r2_func, has_aux=True)
    dr2dps_tm1_norm, (r, states_t_pred) = jax.vmap(grad_fn)([ps_tm1_norm, pa_tm1_norm, fa_tm1_norm], po_t_norm)
    #dr2dps_tm1_norm = jax.tree.map(lambda x: x*10, dr2dps_tm1_norm)
    # print(f"max: {jnp.max(jnp.abs(dr2dps_tm1_norm[0].array))}")
    #jax.debug.print("max: {x}", x=jnp.abs(dr2dps_tm1_norm[0].array))
    return dr2dps_tm1_norm, r, states_t_pred

def corrupt_observations(past_observations_t, key):

        key, subkey = jax.random.split(key)
        past_observations_t.a_x = past_observations_t.a_x + jax.random.uniform(subkey, past_observations_t.a_x.shape, minval=-0.08, maxval=0.08)
        subkey, key = jax.random.split(subkey)
        past_observations_t.a_y = past_observations_t.a_y + jax.random.uniform(subkey, past_observations_t.a_y.shape, minval=-0.08, maxval=0.08)
        subkey, key = jax.random.split(subkey)
        past_observations_t.a_z = past_observations_t.a_z + jax.random.uniform(subkey, past_observations_t.a_z.shape, minval=-0.08, maxval=0.08)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_roll = past_observations_t.v_roll + jax.random.uniform(subkey, past_observations_t.v_roll.shape, minval=-0.025, maxval=0.025)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_pitch = past_observations_t.v_pitch + jax.random.uniform(subkey, past_observations_t.v_pitch.shape, minval=-0.025, maxval=0.025)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_yaw = past_observations_t.v_yaw + jax.random.uniform(subkey, past_observations_t.v_yaw.shape, minval=-0.025, maxval=0.025)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fl_hip = past_observations_t.q_fl_hip + jax.random.uniform(subkey, past_observations_t.q_fl_hip.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fl_knee = past_observations_t.q_fl_knee + jax.random.uniform(subkey, past_observations_t.q_fl_knee.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fl_ankle = past_observations_t.q_fl_ankle + jax.random.uniform(subkey, past_observations_t.q_fl_ankle.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fr_hip = past_observations_t.q_fr_hip + jax.random.uniform(subkey, past_observations_t.q_fr_hip.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fr_knee = past_observations_t.q_fr_knee + jax.random.uniform(subkey, past_observations_t.q_fr_knee.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_fr_ankle = past_observations_t.q_fr_ankle + jax.random.uniform(subkey, past_observations_t.q_fr_ankle.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rl_hip = past_observations_t.q_rl_hip + jax.random.uniform(subkey, past_observations_t.q_rl_hip.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rl_knee = past_observations_t.q_rl_knee + jax.random.uniform(subkey, past_observations_t.q_rl_knee.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rl_ankle = past_observations_t.q_rl_ankle + jax.random.uniform(subkey, past_observations_t.q_rl_ankle.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rr_hip = past_observations_t.q_rr_hip + jax.random.uniform(subkey, past_observations_t.q_rr_hip.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rr_knee = past_observations_t.q_rr_knee + jax.random.uniform(subkey, past_observations_t.q_rr_knee.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.q_rr_ankle = past_observations_t.q_rr_ankle + jax.random.uniform(subkey, past_observations_t.q_rr_ankle.shape, minval=-1e-2, maxval=1e-2)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fl_hip = past_observations_t.v_fl_hip + jax.random.uniform(subkey, past_observations_t.v_fl_hip.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fl_knee = past_observations_t.v_fl_knee + jax.random.uniform(subkey, past_observations_t.v_fl_knee.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fl_ankle = past_observations_t.v_fl_ankle + jax.random.uniform(subkey, past_observations_t.v_fl_ankle.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fr_hip = past_observations_t.v_fr_hip + jax.random.uniform(subkey, past_observations_t.v_fr_hip.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fr_knee = past_observations_t.v_fr_knee + jax.random.uniform(subkey, past_observations_t.v_fr_knee.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_fr_ankle = past_observations_t.v_fr_ankle + jax.random.uniform(subkey, past_observations_t.v_fr_ankle.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rl_hip = past_observations_t.v_rl_hip + jax.random.uniform(subkey, past_observations_t.v_rl_hip.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rl_knee = past_observations_t.v_rl_knee + jax.random.uniform(subkey, past_observations_t.v_rl_knee.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rl_ankle = past_observations_t.v_rl_ankle + jax.random.uniform(subkey, past_observations_t.v_rl_ankle.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rr_hip = past_observations_t.v_rr_hip + jax.random.uniform(subkey, past_observations_t.v_rr_hip.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rr_knee = past_observations_t.v_rr_knee + jax.random.uniform(subkey, past_observations_t.v_rr_knee.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v_rr_ankle = past_observations_t.v_rr_ankle + jax.random.uniform(subkey, past_observations_t.v_rr_ankle.shape, minval=-0.1, maxval=0.1)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v11 = past_observations_t.v11 + jax.random.uniform(subkey, past_observations_t.v11.shape, minval=-0.001, maxval=0.001)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v12 = past_observations_t.v12 + jax.random.uniform(subkey, past_observations_t.v12.shape, minval=-0.001, maxval=0.001)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v13 = past_observations_t.v13 + jax.random.uniform(subkey, past_observations_t.v13.shape, minval=-0.001, maxval=0.001)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v21 = past_observations_t.v21 + jax.random.uniform(subkey, past_observations_t.v21.shape, minval=-0.001, maxval=0.001)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v22 = past_observations_t.v22 + jax.random.uniform(subkey, past_observations_t.v22.shape, minval=-0.001, maxval=0.001)
        subkey, key = jax.random.split(subkey)
        past_observations_t.v23 = past_observations_t.v23 + jax.random.uniform(subkey, past_observations_t.v23.shape, minval=-0.001, maxval=0.001)

        return past_observations_t, key
    

def innovation_residual(past_states_tm1_est_normalized, past_actions_tm1_normalized, future_actions_tm1_normalized, observations_t_normalized, dynamics_model):

    observations_t_normalized = jax.tree.map(lambda x: x[-1], observations_t_normalized)
    past_states_tm1_est = past_states_tm1_est_normalized.denormalized()
    past_actions_tm1 = past_actions_tm1_normalized.denormalized()
    future_actions_tm1 = future_actions_tm1_normalized.denormalized()

    lifted_states = LiftedStates(past_states_tm1_est, past_actions_tm1, T=dynamics_model.T, H=dynamics_model.H)

    states_t_forecast = dynamics_model.F_x(lifted_states, future_actions_tm1) # this is the same as states_t_forecast at the input but we are doing it functionally to get the gradients of the resisdual wrt the previous states and actions

    states_tm1 = jax.tree.map(lambda x: x[-1], past_states_tm1_est)
    #dxdt = dynamics_model.dxdt(lifted_states, future_actions_tm1)
    #dxdt = jax.tree.map(lambda x: x/0.02, dxdt)
    dxdt = jax.tree.map(lambda xt, xtm1: (xt - xtm1)/0.02, states_t_forecast, states_tm1)

    states_t_forecast_normalized = states_t_forecast.normalized()
    
    dummy_obs = jax.tree.map(lambda x: x, observations_t_normalized.denormalized())
    dummy_obs.a_x = dxdt.v_x
    dummy_obs.a_y = dxdt.v_y
    dummy_obs.a_z = dxdt.v_z
    dummy_obs_normalized = dummy_obs.normalized()

    
    q_fl_hip = states_t_forecast_normalized.q_fl_hip - observations_t_normalized.q_fl_hip
    v_fl_hip = states_t_forecast_normalized.v_fl_hip - observations_t_normalized.v_fl_hip
    q_fl_knee = states_t_forecast_normalized.q_fl_knee - observations_t_normalized.q_fl_knee
    v_fl_knee = states_t_forecast_normalized.v_fl_knee - observations_t_normalized.v_fl_knee
    q_fl_ankle = states_t_forecast_normalized.q_fl_ankle - observations_t_normalized.q_fl_ankle
    v_fl_ankle = states_t_forecast_normalized.v_fl_ankle - observations_t_normalized.v_fl_ankle
    q_fr_hip = states_t_forecast_normalized.q_fr_hip - observations_t_normalized.q_fr_hip
    v_fr_hip = states_t_forecast_normalized.v_fr_hip - observations_t_normalized.v_fr_hip
    q_fr_knee = states_t_forecast_normalized.q_fr_knee - observations_t_normalized.q_fr_knee
    v_fr_knee = states_t_forecast_normalized.v_fr_knee - observations_t_normalized.v_fr_knee
    q_fr_ankle = states_t_forecast_normalized.q_fr_ankle - observations_t_normalized.q_fr_ankle
    v_fr_ankle = states_t_forecast_normalized.v_fr_ankle - observations_t_normalized.v_fr_ankle
    q_rl_hip = states_t_forecast_normalized.q_rl_hip - observations_t_normalized.q_rl_hip
    v_rl_hip = states_t_forecast_normalized.v_rl_hip - observations_t_normalized.v_rl_hip
    q_rl_knee = states_t_forecast_normalized.q_rl_knee - observations_t_normalized.q_rl_knee
    v_rl_knee = states_t_forecast_normalized.v_rl_knee - observations_t_normalized.v_rl_knee
    q_rl_ankle = states_t_forecast_normalized.q_rl_ankle - observations_t_normalized.q_rl_ankle
    v_rl_ankle = states_t_forecast_normalized.v_rl_ankle - observations_t_normalized.v_rl_ankle
    q_rr_hip = states_t_forecast_normalized.q_rr_hip - observations_t_normalized.q_rr_hip
    v_rr_hip = states_t_forecast_normalized.v_rr_hip - observations_t_normalized.v_rr_hip
    q_rr_knee = states_t_forecast_normalized.q_rr_knee - observations_t_normalized.q_rr_knee
    v_rr_knee = states_t_forecast_normalized.v_rr_knee - observations_t_normalized.v_rr_knee
    q_rr_ankle = states_t_forecast_normalized.q_rr_ankle - observations_t_normalized.q_rr_ankle
    v_rr_ankle = states_t_forecast_normalized.v_rr_ankle - observations_t_normalized.v_rr_ankle

    v11 = states_t_forecast_normalized.v11 - observations_t_normalized.v11 # angle representation using gram-schmidt
    v12 = states_t_forecast_normalized.v12 - observations_t_normalized.v12  
    v13 = states_t_forecast_normalized.v13 - observations_t_normalized.v13
    v21 = states_t_forecast_normalized.v21 - observations_t_normalized.v21
    v22 = states_t_forecast_normalized.v22 - observations_t_normalized.v22
    v23 = states_t_forecast_normalized.v23 - observations_t_normalized.v23

    v_roll = states_t_forecast_normalized.v_roll - observations_t_normalized.v_roll
    v_pitch = states_t_forecast_normalized.v_pitch - observations_t_normalized.v_pitch
    v_yaw = states_t_forecast_normalized.v_yaw - observations_t_normalized.v_yaw

    a_x = dummy_obs_normalized.a_x - observations_t_normalized.a_x
    a_y = dummy_obs_normalized.a_y - observations_t_normalized.a_y
    a_z = dummy_obs_normalized.a_z - observations_t_normalized.a_z


    r = jnp.stack([q_fl_hip, v_fl_hip, q_fl_knee, v_fl_knee, q_fl_ankle, v_fl_ankle,
                    q_fr_hip, v_fr_hip, q_fr_knee, v_fr_knee, q_fr_ankle, v_fr_ankle,
                    q_rl_hip, v_rl_hip, q_rl_knee, v_rl_knee, q_rl_ankle, v_rl_ankle,
                    q_rr_hip, v_rr_hip, q_rr_knee, v_rr_knee, q_rr_ankle, v_rr_ankle,
                    v11, v12, v13, v21, v22, v23,
                    v_roll, v_pitch, v_yaw, a_x, a_y, a_z], axis=-1).flatten()
    
    return r, states_t_forecast

def corrupt_past_states_tm1(past_observations_t, past_states_tm1, key):
    H = 8
    noise = jax.tree.map(lambda x: jax.random.normal(key, x.shape) * past_states_tm1.scales.array * 0.5, past_states_tm1)
    past_states_copy = jax.tree.map(lambda x: x, past_states_tm1)

    past_states_copy.sdf_base_main = past_states_copy.sdf_base_main + noise.sdf_base_main
    past_states_copy.sdf_base_head_top = past_states_copy.sdf_base_head_top + noise.sdf_base_head_top
    past_states_copy.sdf_base_head_bottom = past_states_copy.sdf_base_head_bottom + noise.sdf_base_head_bottom

    past_states_copy.sdf_fl_foot = past_states_copy.sdf_fl_foot + noise.sdf_fl_foot
    past_states_copy.sdf_fr_foot = past_states_copy.sdf_fr_foot + noise.sdf_fr_foot
    past_states_copy.sdf_rl_foot = past_states_copy.sdf_rl_foot + noise.sdf_rl_foot
    past_states_copy.sdf_rr_foot = past_states_copy.sdf_rr_foot + noise.sdf_rr_foot

    past_states_copy.sdf_fl_shank_top = past_states_copy.sdf_fl_shank_top + noise.sdf_fl_shank_top
    past_states_copy.sdf_fl_shank_bottom = past_states_copy.sdf_fl_shank_bottom + noise.sdf_fl_shank_bottom
    past_states_copy.sdf_fr_shank_top = past_states_copy.sdf_fr_shank_top + noise.sdf_fr_shank_top
    past_states_copy.sdf_fr_shank_bottom = past_states_copy.sdf_fr_shank_bottom + noise.sdf_fr_shank_bottom
    past_states_copy.sdf_rl_shank_top = past_states_copy.sdf_rl_shank_top + noise.sdf_rl_shank_top
    past_states_copy.sdf_rl_shank_bottom = past_states_copy.sdf_rl_shank_bottom + noise.sdf_rl_shank_bottom
    past_states_copy.sdf_rr_shank_top = past_states_copy.sdf_rr_shank_top + noise.sdf_rr_shank_top
    past_states_copy.sdf_rr_shank_bottom = past_states_copy.sdf_rr_shank_bottom + noise.sdf_rr_shank_bottom

    past_states_copy.sdf_fl_thigh = past_states_copy.sdf_fl_thigh + noise.sdf_fl_thigh
    past_states_copy.sdf_fr_thigh = past_states_copy.sdf_fr_thigh + noise.sdf_fr_thigh
    past_states_copy.sdf_rl_thigh = past_states_copy.sdf_rl_thigh + noise.sdf_rl_thigh
    past_states_copy.sdf_rr_thigh = past_states_copy.sdf_rr_thigh + noise.sdf_rr_thigh

    past_states_copy.sdf_fl_hip = past_states_copy.sdf_fl_hip + noise.sdf_fl_hip
    past_states_copy.sdf_fr_hip = past_states_copy.sdf_fr_hip + noise.sdf_fr_hip
    past_states_copy.sdf_rl_hip = past_states_copy.sdf_rl_hip + noise.sdf_rl_hip
    past_states_copy.sdf_rr_hip = past_states_copy.sdf_rr_hip + noise.sdf_rr_hip

    past_states_copy.z = past_states_copy.z + noise.z
    past_states_copy.v_x = past_states_copy.v_x + noise.v_x
    past_states_copy.v_y = past_states_copy.v_y + noise.v_y
    past_states_copy.v_z = past_states_copy.v_z + noise.v_z

    past_states_copy.v_roll = past_observations_t.v_roll[:, -H-1:-1]
    past_states_copy.v_pitch = past_observations_t.v_pitch[:, -H-1:-1]
    past_states_copy.v_yaw = past_observations_t.v_yaw[:, -H-1:-1]

    past_states_copy.q_fl_hip = past_observations_t.q_fl_hip[:, -H-1:-1]
    past_states_copy.q_fr_hip = past_observations_t.q_fr_hip[:, -H-1:-1]
    past_states_copy.q_rl_hip = past_observations_t.q_rl_hip[:, -H-1:-1]
    past_states_copy.q_rr_hip = past_observations_t.q_rr_hip[:, -H-1:-1]
    past_states_copy.q_fl_knee = past_observations_t.q_fl_knee[:, -H-1:-1]
    past_states_copy.q_fr_knee = past_observations_t.q_fr_knee[:, -H-1:-1]
    past_states_copy.q_rl_knee = past_observations_t.q_rl_knee[:, -H-1:-1]
    past_states_copy.q_rr_knee = past_observations_t.q_rr_knee[:, -H-1:-1]
    past_states_copy.q_fl_ankle = past_observations_t.q_fl_ankle[:, -H-1:-1]
    past_states_copy.q_fr_ankle = past_observations_t.q_fr_ankle[:, -H-1:-1]
    past_states_copy.q_rl_ankle = past_observations_t.q_rl_ankle[:, -H-1:-1]
    past_states_copy.q_rr_ankle = past_observations_t.q_rr_ankle[:, -H-1:-1]

    past_states_copy.v_fl_hip = past_observations_t.v_fl_hip[:, -H-1:-1]
    past_states_copy.v_fr_hip = past_observations_t.v_fr_hip[:, -H-1:-1]
    past_states_copy.v_rl_hip = past_observations_t.v_rl_hip[:, -H-1:-1]
    past_states_copy.v_rr_hip = past_observations_t.v_rr_hip[:, -H-1:-1]
    past_states_copy.v_fl_knee = past_observations_t.v_fl_knee[:, -H-1:-1]
    past_states_copy.v_fr_knee = past_observations_t.v_fr_knee[:, -H-1:-1]
    past_states_copy.v_rl_knee = past_observations_t.v_rl_knee[:, -H-1:-1]
    past_states_copy.v_rr_knee = past_observations_t.v_rr_knee[:, -H-1:-1]
    past_states_copy.v_fl_ankle = past_observations_t.v_fl_ankle[:, -H-1:-1]
    past_states_copy.v_fr_ankle = past_observations_t.v_fr_ankle[:, -H-1:-1]
    past_states_copy.v_rl_ankle = past_observations_t.v_rl_ankle[:, -H-1:-1]
    past_states_copy.v_rr_ankle = past_observations_t.v_rr_ankle[:, -H-1:-1]

    past_states_copy.v11 = past_observations_t.v11[:, -H-1:-1]
    past_states_copy.v12 = past_observations_t.v12[:, -H-1:-1]
    past_states_copy.v13 = past_observations_t.v13[:, -H-1:-1]
    past_states_copy.v21 = past_observations_t.v21[:, -H-1:-1]
    past_states_copy.v22 = past_observations_t.v22[:, -H-1:-1]
    past_states_copy.v23 = past_observations_t.v23[:, -H-1:-1]

    return past_states_copy



def loss_observer_filter(observer_model, dynamics_model, nn_data, cauchy=False, lipschitz=False, alpha_lipschitz_dynamics=0.0, alpha_lipschitz_observer=0.0, smooth_term=True):

    
    dynamics_model = jax.tree.map(lambda x: x, dynamics_model)
    params = dynamics_model.params
    dynamics_model.set_inference_mode(True, params)
    dynamics_model = jax.lax.stop_gradient(dynamics_model)

    n_blocks = nn_data.n_blocks
    sigma_obs = 0.03
    key, subkey = jax.random.split(nn_data.key)


    for i in range(1, n_blocks):

    
        past_states_t = getattr(nn_data, f"past_states_block{i}")
        past_actions_t = getattr(nn_data, f"past_actions_block{i}")
        future_actions_t = getattr(nn_data, f"future_actions_block{i}")
        past_observations_t = getattr(nn_data, f"past_obs_block{i}")
        future_states_t = getattr(nn_data, f"future_states_block{i}")
        target_states_normalized = future_states_t.normalized()


        past_observations_t, key = corrupt_observations(past_observations_t, key)

        if i == 1:
            past_states_tm1 = getattr(nn_data, f"past_states_block{0}")
            past_actions_tm1 = getattr(nn_data, f"past_actions_block{0}")
            future_actions_tm1 = getattr(nn_data, f"future_actions_block{0}")
            rollout_true = jax.tree.map(lambda x: x[0, -1][None, ...], past_states_tm1)
            #past_states_tm1 = jax.tree.map(lambda x: x + jax.random.normal(subkey, x.shape) * past_states_tm1.scales.array, past_states_tm1)
            past_states_tm1 = corrupt_past_states_tm1(past_observations_t, past_states_tm1, subkey)
            rollout_preds = jax.tree.map(lambda x: x[0, -1][None, ...], past_states_tm1)
            past_states_tm1_sg = jax.lax.stop_gradient(past_states_tm1)

        # if i == 8:
        #     key, subkey = jax.random.split(key)
        #     past_states_tm1_sg = corrupt_past_states_tm1(past_observations_t, past_states_tm1_sg, subkey)


        # innovation_data
        dinnovation, innovation, states_t_pred = innovation_data(past_states_tm1_sg, past_actions_tm1, future_actions_tm1, past_observations_t, dynamics_model)
        dinnovation_sg, innovation_sg, states_t_pred_sg = jax.lax.stop_gradient((dinnovation, innovation, states_t_pred))
        past_states_t_in_sg = jax.tree.map(lambda x, y: jnp.concatenate([x[:, 1:], y], axis=1), past_states_tm1_sg, states_t_pred_sg)
        past_states_t_hat_sg = jax.vmap(observer_model.F_x)(past_states_t_in_sg, past_observations_t, past_states_tm1_sg, innovation_sg, dinnovation_sg)

        lifted_states_estimated = LiftedStates(past_states_t_hat_sg, past_actions_t, T=nn_data.T, H=nn_data.H)
        predicted_states_from_estimated = jax.vmap(dynamics_model.F_x, in_axes=(0, 0))(lifted_states_estimated, future_actions_t)
        predicted_states_from_estimated_normalized = predicted_states_from_estimated.normalized()

        r2_past_states_block_sg = jax.tree.map(lambda gt, pred: (gt - pred)**2, past_states_t_hat_sg.normalized(), past_states_t.normalized())
        #r2_from_estimated_block = jax.tree.map(lambda gt, pred: (gt - pred)**2, target_states_normalized, predicted_states_from_estimated_normalized)

        r_unnormalized = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), past_states_t_hat_sg, past_states_t)

        if i == 1:
            r2_past_states_rollout_sg = r2_past_states_block_sg
            r_unnormalized_rollout = r_unnormalized
            #r2_from_estimated = r2_from_estimated_block
        else:
            r2_past_states_rollout_sg = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_past_states_rollout_sg, r2_past_states_block_sg)
            #r2_from_estimated = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_from_estimated, r2_from_estimated_block)
            r_unnormalized_rollout = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r_unnormalized_rollout, r_unnormalized)

        rollout_true_block = jax.tree.map(lambda x: x[0, -1][None, ...], past_states_t)
        rollout_preds_block = jax.tree.map(lambda x: x[0, -1][None, ...], past_states_t_hat_sg)
        rollout_true = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), rollout_true, rollout_true_block)
        rollout_preds = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=0), rollout_preds, rollout_preds_block)


        past_states_tm1_sg = jax.lax.stop_gradient(past_states_t_hat_sg)
        past_actions_tm1 = past_actions_t
        future_actions_tm1 = future_actions_t

    rollout_error_sg = r2_past_states_rollout_sg.array
    rollout_error_unnormalized = r_unnormalized_rollout.array
    #dynamics_error_sg = r2_from_estimated.array


    if cauchy:

        n_states = rollout_error_sg.shape[-1]
        fctr = n_states + 1

        def cauchy_transform(x):
            sum_sq = jnp.sum(x, axis=-1, keepdims=True)
            return fctr * jnp.log(1 + sum_sq)

        rollout_error = cauchy_transform(rollout_error_sg)
        #rollout_error = jnp.log(1 + rollout_error_sg)
        #dynamics_error = jnp.log(1 + dynamics_error_sg)
    else:
        rollout_error = rollout_error_sg
        #dynamics_error = dynamics_error_sg

    loss_rollout = jnp.mean(rollout_error)
    #loss_rollout_orig = jnp.mean(rollout_error_sg)
    #loss_dynamics = jnp.mean(dynamics_error)
    #loss_dynamics_orig = jnp.mean(dynamics_error_sg)

    loss_unnormalized = jnp.mean(rollout_error_unnormalized)

    # if lipschitz:
    #     c_o = observer_model.Lipschitz_constants
    # else:
    #     c_o = observer_model.calc_Lipschitz_constants()

    # c_o, _ = jax.tree_util.tree_flatten(c_o)
    # c_o = jax.tree_map(lambda x: x.reshape(1, -1), c_o)
    # c_o = jnp.concatenate(c_o, axis=-1).flatten()
    # c_o = jax.nn.softplus(c_o)
    # lipschitz_loss_observer = jnp.prod(c_o)

    lipschitz_ub_model, J_lipschitz_ub, lipschitz_residual, J_lipschitz_residual = observer_model.calc_residual()
    if lipschitz:

        if smooth_term:
            loss = lipschitz_residual*alpha_lipschitz_observer + loss_rollout + J_lipschitz_residual*alpha_lipschitz_observer
        else:
            loss = loss_rollout
    else:
        # c_o = observer_model.calc_Lipschitz_constants()
        # c_o, _ = jax.tree_util.tree_flatten(c_o)
        # c_o = jax.tree_map(lambda x: x.reshape(1, -1), c_o)
        # c_o = jnp.concatenate(c_o, axis=-1).flatten()
        # c_o = jax.nn.softplus(c_o)
        # lipschitz_ub_model = jnp.prod(c_o)

        loss = loss_rollout

    log = {"loss_observer": loss, "loss_rollout_observer": loss_unnormalized, "rollout_preds_observer": rollout_preds, "rollout_true_observer": rollout_true,
           "lipschitz_loss_observer": lipschitz_ub_model, "J_lipschitz_loss_observer": J_lipschitz_ub}
    
    return loss, log


def loss_dynamics(dynamics_model, observer_model, nn_data, discount, alpha_lipschitz_dynamics, cauchy=False, lipschitz=False, alpha_lipschitz_observer=0.0, smooth_term=True):
    
    observer_model = jax.tree.map(lambda x: x, observer_model)
    params = observer_model.params
    observer_model.set_inference_mode(True, params)
    observer_model = jax.lax.stop_gradient(observer_model)

    dynamics_model_for_estimation = jax.tree.map(lambda x: x, dynamics_model)
    params = dynamics_model_for_estimation.params
    dynamics_model_for_estimation.set_inference_mode(True, params)
    dynamics_model_for_estimation = jax.lax.stop_gradient(dynamics_model_for_estimation)

    n_blocks = nn_data.n_blocks
    key, subkey = jax.random.split(nn_data.key)
    sigma_obs = 0.03
    
    for i in range(n_blocks):
        past_states = getattr(nn_data, f"past_states_block{i}")
        past_actions = getattr(nn_data, f"past_actions_block{i}")
        future_states = getattr(nn_data, f"future_states_block{i}")
        future_actions = getattr(nn_data, f"future_actions_block{i}")
        past_observations_t = getattr(nn_data, f"past_obs_block{i}")
        lifted_states = LiftedStates(past_states, past_actions, T=nn_data.T, H=nn_data.H)
        lifted_actions = future_actions

        predicted_states = jax.vmap(dynamics_model.F_x, in_axes=(0, 0))(lifted_states, lifted_actions)
        predicted_states_normalized = predicted_states.normalized()
        target_states_normalized = future_states.normalized()

        if i == 0:
            r2_teacher =  jax.tree.map(lambda gt, pred: (gt - pred)**2, target_states_normalized, predicted_states_normalized)
            r_unnormalized = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), future_states, predicted_states)
            lifted_states_block0 = lifted_states
            stacked_future_states = jax.tree.map(lambda x: x[None, ...], future_states)
            stacked_future_actions = jax.tree.map(lambda x: x[None, ...], future_actions)

            past_states_tm1 = jax.tree.map(lambda x: x, past_states)
            past_actions_tm1 = jax.tree.map(lambda x: x, past_actions)
            future_actions_tm1 = jax.tree.map(lambda x: x, future_actions)



        else:
            # past_states_tm1 = jax.tree.map(lambda x: x + jax.random.normal(subkey, x.shape) * past_states_tm1.scales.array, past_states_tm1)
            past_states_tm1 = jax.lax.stop_gradient(past_states_tm1)
            # past_observations_t = jax.tree.map(lambda x: x + jax.random.normal(subkey, x.shape) * past_observations_t.scales.array * sigma_obs, past_observations_t)
            # past_observations_t = jax.tree.map(lambda x: x[:, -1], past_observations_t)
            past_observations_t, subkey = corrupt_observations(past_observations_t, subkey)
            past_states_tm1 = corrupt_past_states_tm1(past_observations_t, past_states_tm1, subkey)
            dinnovation, innovation, states_t_pred = innovation_data(past_states_tm1, past_actions_tm1, future_actions_tm1, past_observations_t, dynamics_model_for_estimation)
            dinnovation, innovation, states_t_pred = jax.lax.stop_gradient((dinnovation, innovation, states_t_pred))
            past_states_t_in = jax.tree.map(lambda x, y: jnp.concatenate([x[:, 1:], y], axis=1), past_states_tm1, states_t_pred)
            past_states_t_hat = jax.vmap(observer_model.F_x)(past_states_t_in, past_observations_t, past_states_tm1, innovation, dinnovation)

            lifted_states_estimated = LiftedStates(past_states_t_hat, past_actions, T=nn_data.T, H=nn_data.H)
            predicted_states_from_estimated = jax.vmap(dynamics_model.F_x, in_axes=(0, 0))(lifted_states_estimated, lifted_actions)
            predicted_states_from_estimated_normalized = predicted_states_from_estimated.normalized()

            if i == 1:
                r2_from_estimated = jax.tree.map(lambda gt, pred: (gt - pred)**2, target_states_normalized, predicted_states_from_estimated_normalized)
                r2_observer = jax.tree.map(lambda gt, pred: (gt - pred)**2, past_states_t_hat.normalized(), past_states.normalized())
                r2_from_estimated_unnormalized = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), future_states, predicted_states_from_estimated)

            else:
                r2_from_estimated_block = jax.tree.map(lambda gt, pred: (gt - pred)**2, target_states_normalized, predicted_states_from_estimated_normalized)
                r2_from_estimated = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_from_estimated, r2_from_estimated_block)

                r2_observer_block = jax.tree.map(lambda gt, pred: (gt - pred)**2, past_states_t_hat.normalized(), past_states.normalized())
                r2_observer = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_observer, r2_observer_block)

                r2_from_estimated_unnormalized_block = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), future_states, predicted_states_from_estimated)
                r2_from_estimated_unnormalized = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_from_estimated_unnormalized, r2_from_estimated_unnormalized_block)


            r_unnormalized_block = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), future_states, predicted_states)
            r2_block = jax.tree.map(lambda gt, pred: (gt - pred)**2, target_states_normalized, predicted_states_normalized)

            stacked_future_actions = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0), stacked_future_actions, future_actions)
            stacked_future_states = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, ...]], axis=0), stacked_future_states, future_states)

            r2_teacher = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r2_teacher, r2_block)
            r_unnormalized = jax.tree.map(lambda x, y: jnp.concatenate([x, y], axis=1), r_unnormalized, r_unnormalized_block)

            past_states_tm1 = jax.lax.stop_gradient(past_states_t_hat)
            past_actions_tm1 = jax.tree.map(lambda x: x, past_actions)
            future_actions_tm1 = jax.tree.map(lambda x: x, future_actions)



    rollout_preds, _ = jax.vmap(dynamics_model.rollout, in_axes=(0, 1))(lifted_states_block0, stacked_future_actions)
    stacked_future_states = jax.tree.map(lambda x: x.transpose(1, 0, 2, 3), stacked_future_states)
    rollout_true = jax.tree.map(lambda x: x.reshape(x.shape[0], -1, x.shape[-1]), stacked_future_states)

    rollout_preds_log = jax.tree.map(lambda x: x[0], rollout_preds)
    rollout_true_log = jax.tree.map(lambda x: x[0], rollout_true)

    rollout_preds_ = rollout_preds.normalized()
    rollout_true_ = rollout_true.normalized()

    r2_rollout = jax.tree.map(lambda gt, pred: (gt - pred)**2, rollout_true_, rollout_preds_)

    r_rollout_unnormalized = jax.tree.map(lambda gt, pred: jnp.abs(gt - pred), rollout_true, rollout_preds)

    
    powers = jnp.arange(0, rollout_preds.array.shape[1], dtype=rollout_preds.array.dtype)
    #discount = jnp.where(powers <= 3, 1.0, 0.0)[None, ..., None]
    discount = jnp.power(discount, powers)[None, ..., None]

    r2_rollout_dsct = jax.tree.map(lambda x: x * discount, r2_rollout)

    if cauchy:
        # r2_teacher_transformed = jax.tree.map(lambda x: jnp.log(1 + x), r2_teacher)
        # r2_rollout_transformed = jax.tree.map(lambda x: jnp.log(1 + x), r2_rollout_dsct)
        # r2_from_estimated_transformed = jax.tree.map(lambda x: jnp.log(1 + x), r2_from_estimated)
        # r2_observer_transformed = jax.tree.map(lambda x: jnp.log(1 + x), r2_observer)
        # r2_mini_rollout_transformed = jax.tree.map(lambda x: jnp.log(1 + x), r2_mini_rollout)

        n_states = r2_teacher.array.shape[-1]
        fctr = n_states + 1

        def cauchy_transform(x):
            sum_sq = jnp.sum(x, axis=-1, keepdims=True)
            return fctr * jnp.log(1 + sum_sq)
        
        r2_teacher_transformed = jax.tree.map(lambda x: cauchy_transform(x), r2_teacher)
        r2_rollout_transformed = jax.tree.map(lambda x: cauchy_transform(x), r2_rollout_dsct)
        r2_from_estimated_transformed = jax.tree.map(lambda x: cauchy_transform(x), r2_from_estimated)
        r2_observer_transformed = jax.tree.map(lambda x: cauchy_transform(x), r2_observer)

    else:
        r2_teacher_transformed = jax.tree.map(lambda x: x, r2_teacher)
        r2_rollout_transformed = jax.tree.map(lambda x: x, r2_rollout_dsct)
        r2_from_estimated_transformed = jax.tree.map(lambda x: x, r2_from_estimated)
        r2_observer_transformed = jax.tree.map(lambda x: x, r2_observer)


    loss_teacher_forcing = jnp.mean(r2_teacher_transformed.array)
    loss_teacher_forcing_orig = jnp.mean(r2_teacher.array)

    loss_rollout = jnp.mean(r2_rollout_transformed.array)
    loss_rollout_orig = jnp.mean(r2_rollout.array)

    loss_from_estimated = jnp.mean(r2_from_estimated_transformed.array)

    loss_observer = jnp.mean(r2_observer_transformed.array)

    loss_unnormalized = jnp.mean(r_unnormalized.array)

    loss_from_estimated_unnormalized = jnp.mean(r2_from_estimated_unnormalized.array)

    loss_rollout_unnormalized = jnp.mean(r_rollout_unnormalized.array)

    lipschitz_ub_model, J_lipschitz_ub, lipschitz_residual, J_lipschitz_residual = dynamics_model.calc_residual()
    if lipschitz:
        #loss = (1/3)*loss_teacher_forcing + (1/3)*loss_rollout  + lipschitz_residual*alpha_lipschitz_dynamics + 0.05*loss_from_estimated + (1/3)*loss_mini_rollout
        #loss = (1/2)*loss_mini_rollout + (1/2)*loss_rollout  + lipschitz_residual*alpha_lipschitz_dynamics + 0.05*loss_from_estimated
        #loss = loss_mini_rollout + lipschitz_residual*alpha_lipschitz_dynamics + 0.05*loss_from_estimated
        if smooth_term:
            loss = (1/2)*loss_rollout  + lipschitz_residual*alpha_lipschitz_dynamics + (1/2)*loss_teacher_forcing + 0.05*loss_from_estimated + J_lipschitz_residual*alpha_lipschitz_dynamics
        else:
            loss = (1/2)*loss_rollout + (1/2)*loss_teacher_forcing + 0.05*loss_from_estimated
    else:
        # c = dynamics_model.calc_Lipschitz_constants()
        # c, _ = jax.tree_util.tree_flatten(c)
        # c = jax.tree_map(lambda x: x.reshape(1, -1), c)
        # c = jnp.concatenate(c, axis=-1).flatten()
        # c = jax.nn.softplus(c)
        # lipschitz_ub_model = jnp.prod(c)
        loss = (1/2)*loss_teacher_forcing + (1/2)*loss_rollout + 0.05*loss_from_estimated

    log = {"loss": loss, "loss_teacher_forcing": loss_unnormalized, "loss_rollout": loss_rollout_unnormalized, "loss_from_estimated": loss_from_estimated_unnormalized,
           "lipschitz_loss_dynamics": lipschitz_ub_model, "rollout_preds": rollout_preds_log, "rollout_true": rollout_true_log, "J_lipschitz_loss_dynamics": J_lipschitz_ub}
    
    return loss, log

class OnlineNNDynamicsTrainer(ABC):
    def __init__(self, nn_model, nn_policy, data_wrangler, 
                 data_collection_module, logger, weight_decay, lrs, 
                 n_steps, accum_steps=1, seed=0, dtype=jnp.float32):

        self.logger = logger
        self.lrs = lrs
        self.n_steps = n_steps
        self.seed = seed
        self.rng_key = jax.random.PRNGKey(seed)
        self.data_collector = data_collection_module
        self.data_wrangler = data_wrangler
        self.nn_model = nn_model
        self.nn_policy = nn_policy
        self.val_key = jax.random.PRNGKey(91996)
        self.dtype = dtype

        # Initialize the optimizer
        assert lrs[0].shape[0] == n_steps, f"Learning rates shape {lrs.shape} does not match number of steps {n_steps}"
        self.dynamics_optimizer = init_optimizer(lrs[0][0], weight_decay=weight_decay[0], dtype=dtype)
        self.dynamics_optimizer = optax.MultiSteps(self.dynamics_optimizer, accum_steps)

        self.observer_optimizer = init_optimizer(lrs[1][0], weight_decay=weight_decay[1], dtype=dtype)
        self.observer_optimizer = optax.MultiSteps(self.observer_optimizer, accum_steps)

        self.dynamics_optimizer_state = self.dynamics_optimizer.init(self.nn_model.dynamics)
        self.observer_optimizer_state = self.observer_optimizer.init(self.nn_model.observer)


        if nn_model.dynamics.Lipschitz:
            name = "Lipschitz"
        else:
            name = "Vanilla"
        log_name = f"{name}_{nn_policy.__class__.__name__}"
        self.logger._create_log_file(log_name)

    def train(self, loss_kwargs, buffer_size, batch_size, num_workers, jit=False, buffer_path=None, buffer_save=False, eval_metric=None):
        
        dynamics_params_training = jax.tree.map(lambda x: x, self.nn_model.dynamics.params)
        observer_params_training = jax.tree.map(lambda x: x, self.nn_model.observer.params)
        self.nn_model.dynamics.set_inference_mode(True, params=dynamics_params_training)
        self.nn_model.observer.set_inference_mode(True, params=observer_params_training)
        self.nn_policy.update_model(self.nn_model)
        self.data_collector.agent.update_nn_policy(self.nn_policy)
    
        #self.data_collector.begin_collection()
        mj_dataset, self.logger = self.data_collector.run_episode(self.logger, init_buffer=True)
        #shapes = jax.tree.map(lambda x: x.shape, mj_dataset)
        #print(f"Shapes of the MJ dataset: {shapes.sensor_trajectory.array}, {shapes.action_trajectory.array}")


        if buffer_save == True:
            assert buffer_path is not None, "If buffer_save is True, buffer_path must be provided"
            os.makedirs(os.path.dirname(buffer_path), exist_ok=True)

        if buffer_path == None or buffer_save == True:

            pbar = tqdm(total=buffer_size // 8, desc="Filling up 1/8th of the buffer")
            for i in range(int(np.ceil((buffer_size // 8) / self.data_collector.num_envs))):
                if i == 0:
                    dataset = OnlineTorchDataset(mj_dataset, buffer_size, H=self.nn_model.dynamics.H, horizon=self.nn_model.dynamics.horizon, H_obs=self.nn_model.observer.H_Obs, dtype=self.dtype)
                else:
                    mj_dataset = self.data_collector.run_episode(init_buffer=True)
                    dataset.update(mj_dataset, mode="standard")

                pbar.update(self.data_collector.num_envs)
            pbar.close()


            if buffer_save == True:
                dataset.save(buffer_path)
                print(f"Buffer saved at {buffer_path}")

        else:
            dataset = OnlineTorchDataset(H=self.nn_model.dynamics.H, horizon=self.nn_model.dynamics.horizon, H_obs=self.nn_model.observer.H_Obs, path=buffer_path, dtype=self.dtype)
            print(f"Current dataset size: {dataset.__len__()}")
            print(f"Buffer loaded from {buffer_path}")

        torch.manual_seed(self.seed)

        #self.data_collector.end_collection()
        # num_data = len(dataset)
        # num_train = int(0.9 * num_data)
        # num_test = num_data - num_train
        # train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_test])
        dl_dtype = np.float32 if self.dtype == jnp.float32 else np.float64
        init_batch_size = max(1, min(8192, len(dataset)))
        init_dl = JaxDataLoader(dataset, batch_size=init_batch_size, shuffle=True, num_workers=num_workers, drop_last=True, dtype=dl_dtype)
        #init_dl = JaxDataLoader(dataset, batch_size=256, shuffle=True, num_workers=num_workers, drop_last=True)

        mj_dataset = update_mj_dataset(mj_dataset, next(iter(init_dl)))
        #print(f"mj_dataset.sensor_trajectory.array.dtype{mj_dataset.sensor_trajectory.array.dtype}")
        # print(f"states: {self.data_wrangler.states.scales.array}, actions: {self.data_wrangler.actions.scales.array}, observations: {self.data_wrangler.observations.scales.array}")
        # print("Initializing data wrangler...")
        robust = loss_kwargs["cauchy"]
        self.data_wrangler.initialize(mj_dataset, robust=robust)
        # self.data_wrangler.states.scales.array = jnp.ones_like(self.data_wrangler.states.scales.array)
        # self.data_wrangler.actions.scales.array = jnp.ones_like(self.data_wrangler.actions.scales.array)
        # self.data_wrangler.observations.scales.array = jnp.ones_like(self.data_wrangler.observations.scales.array)

        self.data_wrangler.states.save(self.logger.ckpt_filename + "_states.dill")
        self.data_wrangler.actions.save(self.logger.ckpt_filename + "_actions.dill")
        self.data_wrangler.observations.save(self.logger.ckpt_filename + "_observations.dill")

        self.data_collector.agent.update_data_wrangler(self.data_wrangler)
        self.data_collector.eval_agent.update_data_wrangler(self.data_wrangler)

        self.logger.epoch = 0
        self.logger = self.data_collector.evaluate(self.logger, seed=self.seed+96, with_state_estimation=True)
        self.logger = self.data_collector.evaluate(self.logger, seed=self.seed+96, with_state_estimation=False)
        best_eval_metric = np.inf
        # if eval_metric is not None:
        #     best_eval_metric = #self.logger.log_dict[eval_metric]
        # self.logger.log_dict["eval_metric"] = best_eval_metric

        # eval_metrics = ["stand_still_cost_with_state_estimation", "forward_cost_with_state_estimation", "backward_cost_with_state_estimation",
        #                 "left_cost_with_state_estimation", "right_cost_with_state_estimation"]

        discounts = jnp.ones(self.n_steps, dtype=self.dtype)*0.85
        dl = JaxDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, dtype=dl_dtype)
        #val_dl = JaxDataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

        alpha_lipschitz_dynamics = loss_kwargs.pop("alpha_lipschitz_dynamics")
        # ub = self.nn_model.dynamics.Lipschitz_ub
        # alpha_max = 1e6 / ub #increase alpha more if ub is small (tighter constraints)
        # alphas_lipschitz_dynamics = jnp.linspace(1, alpha_max, self.n_steps, dtype=self.dtype)*alpha_lipschitz_dynamics
        # alphas_lipschitz_dynamics = jnp.ones(15000, dtype=self.dtype)*alpha_lipschitz_dynamics

        # remaining_steps = self.n_steps - 15000
        # if remaining_steps > 0:
        #     alphas_lipschitz_dynamics = jnp.concatenate([alphas_lipschitz_dynamics, jnp.linspace(1, 10.0, remaining_steps, dtype=self.dtype)*alpha_lipschitz_dynamics], axis=0)
        # else:
        #     alphas_lipschitz_dynamics = jnp.ones(self.n_steps, dtype=self.dtype)*alpha_lipschitz_dynamics

        alphas_lipschitz_dynamics = jnp.ones(self.n_steps, dtype=self.dtype)*alpha_lipschitz_dynamics

        _loss_dynamics = lambda dyn_model, obs_model, x, discount, alpha_lipschitz_dynamics: loss_dynamics(dyn_model, obs_model, x, discount, alpha_lipschitz_dynamics=alpha_lipschitz_dynamics, **loss_kwargs)
        _loss_observer = lambda obs_model, dyn_model, x: loss_observer_filter(obs_model, dyn_model, x, alpha_lipschitz_dynamics=alpha_lipschitz_dynamics, **loss_kwargs)

        if jit == True:
            take_step_dynamics = jax.jit(lambda opt_state, dynamics_model, observer_model, x, discount, alpha_lipschitz_dynamics:
                                         _take_step_dynamics(self.dynamics_optimizer, _loss_dynamics, opt_state, dynamics_model, observer_model, x, discount, alpha_lipschitz_dynamics=alpha_lipschitz_dynamics))
            take_step_observer = jax.jit(lambda opt_state, observer_model, dynamics_model, x:
                                         _take_step_observer(self.observer_optimizer, _loss_observer, opt_state, observer_model, dynamics_model, x))


        else:
            take_step_dynamics = lambda opt_state, dynamics_model, observer_model, x: _take_step_dynamics(self.dynamics_optimizer, _loss_dynamics, opt_state, dynamics_model, observer_model, x)
            take_step_observer = lambda opt_state, observer_model, dynamics_model, x: _take_step_observer(self.observer_optimizer, _loss_observer, opt_state, observer_model, dynamics_model, x)


        step = 1
        epoch = 0
        steps_per_epoch = 1000 #* (512 // batch_size)
        
        pretrain_steps = 0
        collect_data = True
        while step < self.n_steps:
            
            self.dynamics_optimizer_state = optax.tree_utils.tree_set(self.dynamics_optimizer_state, {"learning_rate": self.lrs[0][step]})
            self.observer_optimizer_state = optax.tree_utils.tree_set(self.observer_optimizer_state, {"learning_rate": self.lrs[1][step]})

            # self.logger.log_dict["step"] = step
            self.logger.log_dict["learning_rate_dynamics"] = self.lrs[0][step-1].item()
            self.logger.log_dict["learning_rate_observer"] = self.lrs[1][step-1].item()              

            batch_iter = iter(dl)

            self.nn_model.dynamics.set_inference_mode(False, params=dynamics_params_training)
            self.nn_model.observer.set_inference_mode(False, params=observer_params_training)
            self.nn_policy.update_model(self.nn_model)
            self.data_collector.agent.update_nn_policy(self.nn_policy)
            self.data_collector.eval_agent.update_nn_policy(self.nn_policy)

            pbar = tqdm(total=self.n_steps, desc="Training steps", initial=step)

            if step >= pretrain_steps:
                steps_per_epoch = 500

            for _ in range(steps_per_epoch):

                try:
                    batch = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(dl)
                    batch = next(batch_iter)

                discount = discounts[step-1]
                mj_dataset = update_mj_dataset(mj_dataset, batch)
                nn_data = process_data(self.data_wrangler, mj_dataset)
                #print(f"mj_dataset.sensor_trajectory.array.dtype{mj_dataset.sensor_trajectory.array.dtype}")
                nn_data.key = jax.random.split(nn_data.key, 1)[0]
                # self.nn_model.dynamics = update_lipschitz_constants(self.nn_model.dynamics)
                # self.nn_model.observer = update_lipschitz_constants(self.nn_model.observer)
                alpha_lipschitz_dynamics = alphas_lipschitz_dynamics[step-1]
                dynamics_model, self.dynamics_optimizer_state, dynamics_losses = take_step_dynamics(self.dynamics_optimizer_state, self.nn_model.dynamics, self.nn_model.observer, nn_data, discount, alpha_lipschitz_dynamics)
                #print(f"Dynamics model params dtype: {dynamics_model.params[0][0].dtype}")
                observer_model, self.observer_optimizer_state, observer_losses = take_step_observer(self.observer_optimizer_state, self.nn_model.observer, self.nn_model.dynamics, nn_data)
                self.nn_model.observer = observer_model
                rollout_preds_observer = observer_losses.pop("rollout_preds_observer")
                rollout_true_observer = observer_losses.pop("rollout_true_observer")
                self.nn_model.dynamics = dynamics_model
                rollout_preds_dynamics = dynamics_losses.pop("rollout_preds")
                rollout_true_dynamics = dynamics_losses.pop("rollout_true")
                losses = {**dynamics_losses, **observer_losses}
                logs = self.logger.process_loss(losses)
                logs = jax.tree.map(lambda x: x.item(), logs)
                self.logger.log_dict.update(logs)

                if step % 20 == 0 or step == 1:
                    self.logger._log_step(step-1)

                # if step % 1000 == 0 or step == 1:
                #     self.nn_model.save(self.logger.ckpt_filename)

                if step % 500 == 0 or step == 1:
                    full_ckpt_filename = self.logger.ckpt_filename
                    dir = os.path.dirname(full_ckpt_filename)
                    filename = f"step_{step}"
                    ckpt_filename = os.path.join(dir, filename)
                    self.nn_model.save(ckpt_filename)
                
                if step <= 20000:
                    if step % 500 == 0 or step == 1:
                        self.logger.plot_logs()
                        self.logger.plot_rollouts(rollout_preds_dynamics, rollout_true_dynamics, mode="DYNAMICS")
                        self.logger.plot_rollouts(rollout_preds_observer, rollout_true_observer, mode="OBSERVER")
                elif step < 100000:
                    if step % 10000 == 0 or step == 1:
                        self.logger.plot_logs()
                        self.logger.plot_rollouts(rollout_preds_dynamics, rollout_true_dynamics, mode="DYNAMICS")
                        self.logger.plot_rollouts(rollout_preds_observer, rollout_true_observer, mode="OBSERVER")
                else:
                    if step % 20000 == 0 or step == 1:
                        self.logger.plot_logs()
                        self.logger.plot_rollouts(rollout_preds_dynamics, rollout_true_dynamics, mode="DYNAMICS")
                        self.logger.plot_rollouts(rollout_preds_observer, rollout_true_observer, mode="OBSERVER")


                # if step == 1:
                #     self.logger.plot_rollouts(rollout_preds_dynamics, rollout_true_dynamics, mode="DYNAMICS")
                #     self.logger.plot_rollouts(rollout_preds_observer, rollout_true_observer, mode="OBSERVER")

                # if collect_data:
                #     if step % 500 == 0 and step >= pretrain_steps:
                        # self.logger.plot_rollouts(rollout_preds_dynamics, rollout_true_dynamics, mode="DYNAMICS")
                        # self.logger.plot_rollouts(rollout_preds_observer, rollout_true_observer, mode="OBSERVER")
                        # dynamics_params_training = jax.tree.map(lambda x: x, self.nn_model.dynamics.params)
                        # observer_params_training = jax.tree.map(lambda x: x, self.nn_model.observer.params)
                        # self.nn_model.dynamics.set_inference_mode(True, params=dynamics_params_training)
                        # self.nn_model.observer.set_inference_mode(True, params=observer_params_training)
                        # self.nn_policy.update_model(self.nn_model)

                        # self.data_collector.agent.update_nn_policy(self.nn_policy)
                        # self.data_collector.eval_agent.update_nn_policy(self.nn_policy)
                        # self.logger = self.data_collector.evaluate(self.logger, seed=self.seed+96, with_state_estimation=True)
                        # self.logger = self.data_collector.evaluate(self.logger, seed=self.seed+96, with_state_estimation=False)

                        # eval_metric_val = 0.0
                        # for eval_metric_name in eval_metrics:
                        #     eval_metric_val_i = self.logger.log_dict[eval_metric_name]
                        #     eval_metric_val += eval_metric_val_i
                        # self.logger.log_dict["eval_metric"] = eval_metric_val

                        # self.nn_model.dynamics.set_inference_mode(False, params=dynamics_params_training)
                        # self.nn_model.observer.set_inference_mode(False, params=observer_params_training)
                        # self.nn_policy.update_model(self.nn_model)
                        # self.data_collector.agent.update_nn_policy(self.nn_policy)
                        # self.data_collector.eval_agent.update_nn_policy(self.nn_policy)

                        # if eval_metric is not None:
                        #     eval_metric_val = self.logger.log_dict["eval_metric"]
                        #     if eval_metric_val < best_eval_metric:
                        #         best_eval_metric = eval_metric_val
                        #         ckpt_filename = self.logger.ckpt_filename + f"_step_{step}_best_{best_eval_metric:.3f}.ckpt"
                        #         prev_best = glob.glob(self.logger.ckpt_filename + f"_step_{step}_best_*.ckpt")
                        #         for f in prev_best:
                        #             if f != ckpt_filename:
                        #                 os.remove(f)
                        #         self.nn_model.save(ckpt_filename)

                pbar.update(1)
                step += 1

                if step >= self.n_steps:
                    break
            
            epoch += 1
            pbar.close()
            self.logger.epoch = epoch

            if step >= pretrain_steps:
                collect_data = True
            else:
                collect_data = False

            dynamics_params_training = jax.tree.map(lambda x: x, self.nn_model.dynamics.params)
            observer_params_training = jax.tree.map(lambda x: x, self.nn_model.observer.params)
            self.nn_model.dynamics.set_inference_mode(True, params=dynamics_params_training)
            self.nn_model.observer.set_inference_mode(True, params=observer_params_training)
            self.nn_policy.update_model(self.nn_model)


            self.data_collector.agent.update_nn_policy(self.nn_policy)
            self.data_collector.eval_agent.update_nn_policy(self.nn_policy)
            #self.logger = self.data_collector.evaluate(self.logger, seed=self.seed+96, with_state_estimation=True)

            if collect_data:
                pbar = tqdm(total=int(self.data_collector.num_envs), desc=f"Adding {self.data_collector.num_envs} trajectories to the buffer")
                mj_dataset, self.logger = self.data_collector.run_episode(self.logger)
                dataset.update(mj_dataset, mode="standard")
                pbar.update(self.data_collector.num_envs)
                pbar.close()
                # else:
                #     print(f"Adding {self.data_collector.num_envs} samples to the buffer")
                #     mj_dataset, self.logger = self.data_collector.run_episode(self.logger)
                #     dataset.update(mj_dataset, mode="standard")
                #     pbar.update(self.data_collector.num_envs)
                print(f"Current dataset size: {dataset.__len__()}")



                dl = JaxDataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, dtype=dl_dtype)





