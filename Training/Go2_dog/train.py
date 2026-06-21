import os
import jax
import jax.numpy as jnp
from Common import *
from Common.Go2_dog import *
from Common.runtime_paths import (
    ACTUATOR_NETWORK,
    EXAMPLE_ACTIONS_PATH,
    EXAMPLE_OBSERVATIONS_PATH,
    EXAMPLE_STATES_PATH,
    SCENE_TORQUE_XML,
)
from .data_collector import ThreadedGo2DataCollector
from Policies import *
from Training import *
from .nn_data import Go2DataWrangler


def main(trial_name="", nn_policy_type="mppi", buffer_size=3200, 
         n_training_steps=150000, lr_dynamics=1e-5, lr_observer=1e-5,
         weight_decay_dynamics=0.0, weight_decay_observer=0.0, batch_size=256, 
         n_workers=4, n_blocks=1, seed=0, n_envs=8, dtype=jnp.float32,
         n_env_workers=8, accum_steps=1, jit=True, headless=False, save_video=False,
         loss_kwargs=None, nn_dynamics_kwargs=None, nn_policy_kwargs=None, nn_observer_kwargs=None,
         buffer_path=None, buffer_save=False, data_collection_kwargs=None):

    np.random.seed(seed)
    n_states, n_actions, n_obs = state_and_action_size()
    print(f"n_states: {n_states}, n_actions: {n_actions}, n_obs: {n_obs}")

    if nn_dynamics_kwargs is None:
        raise ValueError("nn_dynamics_kwargs cannot be None")
    if nn_observer_kwargs is None:
        raise ValueError("nn_observer_kwargs cannot be None")
    if nn_policy_kwargs is None:
        raise ValueError("nn_policy_kwargs cannot be None")
    if data_collection_kwargs is None:
        raise ValueError("data_collection_kwargs cannot be None")
    if loss_kwargs is None:
        raise ValueError("loss_kwargs cannot be None")

    mj_model_path = SCENE_TORQUE_XML

    logger = get_logger(trial_name, nn_dynamics_kwargs, "Training")

    # valid_T_and_n_blocks = [[1, 18], [2, 9], [3, 6], [6, 3], [9, 2], [18, 1]]
    # T_and_n_blocks = [nn_dynamics_kwargs["T"], n_blocks]
    # if T_and_n_blocks not in valid_T_and_n_blocks:
    #     raise ValueError(f"Invalid T and n_blocks combination: {T_and_n_blocks}. Valid combinations are: {valid_T_and_n_blocks}")
    

    if nn_dynamics_kwargs["Lipschitz"] == True or nn_observer_kwargs["Lipschitz"] == True:
        nn_dynamics_kwargs["Lipschitz"] = True
        nn_observer_kwargs["Lipschitz"] = True
    else:
        nn_dynamics_kwargs["Lipschitz"] = False
        nn_observer_kwargs["Lipschitz"] = False

    n_history_steps = nn_dynamics_kwargs["H"] + 1
    n_history_steps_observer = nn_observer_kwargs["H_Obs"] + 1
    horizon = n_blocks

    actuator_net = MLPBase.load(ACTUATOR_NETWORK)
    actuator_net.Lipschitz_ub = 1e12
    params = actuator_net.params
    actuator_net.set_inference_mode(True, params)
    
    nn_dynamics_kwargs["seed"] = seed
    nn_observer_kwargs["seed"] = seed
    nn_dynamics_kwargs["dtype"] = dtype
    nn_observer_kwargs["dtype"] = dtype
    nn_dynamics = MLPDynamics(n_states=n_states, n_actions=n_actions, actuator_net=actuator_net, **nn_dynamics_kwargs)
    #nn_dynamics = QuadrupedDynamics(n_states=n_states, n_actions=n_actions, actuator_net=actuator_net, **nn_dynamics_kwargs)
    nn_observer = MLPObserver(n_states=n_states, n_actions=n_actions, n_obs=n_obs, **nn_observer_kwargs)
    nn_model = NNModel(dynamics=nn_dynamics, observer=nn_observer)

    print(f"nn_dynamics: {nn_dynamics.input_dim}, nn_observer: {nn_observer.input_dim}")
    print(f"nn_dynamics: {nn_dynamics.output_dim}, nn_observer: {nn_observer.output_dim}")

    import mujoco
    mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
    action_bounds = mj_model.actuator_ctrlrange
    action_bounds = jnp.asarray(action_bounds)

    init_sigma, init_guess = init_sigma_and_guess()

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
    
    def inheritrange_bounds(joint_name, factor=1.5):
        q_min, q_max = joint_limits[joint_name]
        mid = 0.5 * (q_min + q_max)
        half_range = 0.5 * (q_max - q_min)
        relaxed_lb = mid - factor * half_range
        relaxed_ub = mid + factor * half_range
        return relaxed_lb, relaxed_ub
    
    action_limits = {k: inheritrange_bounds(k) for k in joint_limits.keys()}

    action_bounds_tree = jax.tree_map(lambda x: x, init_sigma)
    action_bounds_tree.fl_hip = jnp.array(action_limits["fl_hip"])
    action_bounds_tree.fl_knee = jnp.array(action_limits["fl_knee"])
    action_bounds_tree.fl_ankle = jnp.array(action_limits["fl_ankle"])
    action_bounds_tree.fr_hip = jnp.array(action_limits["fr_hip"])
    action_bounds_tree.fr_knee = jnp.array(action_limits["fr_knee"])
    action_bounds_tree.fr_ankle = jnp.array(action_limits["fr_ankle"])
    action_bounds_tree.rl_hip = jnp.array(action_limits["rl_hip"])
    action_bounds_tree.rl_knee = jnp.array(action_limits["rl_knee"])
    action_bounds_tree.rl_ankle = jnp.array(action_limits["rl_ankle"])
    action_bounds_tree.rr_hip = jnp.array(action_limits["rr_hip"])
    action_bounds_tree.rr_knee = jnp.array(action_limits["rr_knee"])
    action_bounds_tree.rr_ankle = jnp.array(action_limits["rr_ankle"])


    #nn_policy_kwargs["horizon"] = horizon
    if nn_policy_type == "mppi":
        nn_policy_kwargs["seed"] = seed + 1
        nn_policy = MPPI_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)
    elif nn_policy_type == "spline_shooter":
        nn_policy = Spline_Shooter_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, residuals=residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)
    else:
        raise ValueError(f"Unknown nn_policy_type: {nn_policy_type}")
    
    states = States.load(EXAMPLE_STATES_PATH)
    actions = Actions.load(EXAMPLE_ACTIONS_PATH)
    observations = Observations.load(EXAMPLE_OBSERVATIONS_PATH)


    data_wrangler = Go2DataWrangler(horizon=horizon, T=nn_dynamics_kwargs["T"], 
                                    H=nn_dynamics_kwargs["H"], H_Obs=nn_observer_kwargs["H_Obs"], 
                                    n_blocks=n_blocks, mj_dataset_cls=mj_dataset_cls, 
                                    states=states, actions=actions, observations=observations)


    init_sigma = jax.tree.flatten(init_sigma)[0]
    init_sigma = jnp.stack(init_sigma)
    init_guess = jax.tree.flatten(init_guess)[0]
    init_guess = jnp.stack(init_guess)
    data_collection_agent = NNDataCollection_Policy(
                    nn_policy=nn_policy,
                    data_wrangler=data_wrangler,
                    jit=jit)
    
    eval_agent = NNDataCollection_Policy(
                nn_policy=nn_policy,
                data_wrangler=data_wrangler,
                jit=jit)
    
    
    data_collection_module = ThreadedGo2DataCollector(data_collection_agent=data_collection_agent,
        eval_agent=eval_agent,
        mj_dataset_cls=mj_dataset_cls,
        mj_model_path=mj_model_path,
        n_history_steps=n_history_steps,
        n_history_steps_observer=n_history_steps_observer,
        horizon=horizon,
        n_envs=n_envs,
        n_workers= n_env_workers,
        save_video=save_video,
        video_text=True,
        seed=seed,
        headless=headless,
        dtype=dtype,
        **data_collection_kwargs)
    
    # data_collection_module.begin_collection()
    # dataset = data_collection_module.run_episode()
    # data_collection_module.end_collection()

    # lrs = jnp.ones(((n_training_steps)//2,)) * jnp.array(lr)
    # remaining_steps = n_training_steps - (n_training_steps)//2
    # lrs = jnp.concatenate((lrs, jnp.linspace(lr, lr/30, remaining_steps)))

    lrs_dyn = jnp.ones((n_training_steps,)) * jnp.array(lr_dynamics)
    lrs_obs = jnp.ones((n_training_steps,)) * jnp.array(lr_observer)

    # lrs0_dyn = jnp.ones((150000,)) * jnp.array(lr_dynamics)
    # lrs0_obs = jnp.ones((150000,)) * jnp.array(lr_observer)
    # remaining_steps = n_training_steps - 150000
    # half = remaining_steps // 2
    # lrs_dyn = jnp.concatenate((lrs0_dyn, jnp.linspace(lr_dynamics, lr_dynamics*1.5, half)))
    # lrs_obs = jnp.concatenate((lrs0_obs, jnp.linspace(lr_observer, lr_observer*1.5, half)))
    # lrs_dyn = jnp.concatenate((lrs_dyn, jnp.ones((half,)) * jnp.array(lr_dynamics*1.5)))
    # lrs_obs = jnp.concatenate((lrs_obs, jnp.ones((half,)) * jnp.array(lr_observer*1.5)))
    lrs = [lrs_dyn, lrs_obs]
    weight_decay = [weight_decay_dynamics, weight_decay_observer]

    trainer = OnlineNNDynamicsTrainer(nn_model=nn_model,
                                nn_policy=nn_policy,
                                data_collection_module=data_collection_module,
                                data_wrangler=data_wrangler,
                                logger=logger,
                                lrs=lrs,
                                weight_decay=weight_decay,
                                n_steps=n_training_steps,
                                accum_steps=accum_steps,
                                seed=seed, 
                                dtype=dtype)

    loss_kwargs["lipschitz"] = nn_dynamics_kwargs["Lipschitz"]
    eval_metric = "stand_still_cost_with_state_estimation"
    trainer.train(loss_kwargs=loss_kwargs,
        buffer_size=buffer_size,
        batch_size=batch_size,
        num_workers=n_workers, 
        buffer_path=buffer_path,
        buffer_save=buffer_save,
        eval_metric=eval_metric,
        jit=jit)
