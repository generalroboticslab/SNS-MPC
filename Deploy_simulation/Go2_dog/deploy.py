import os
import jax
import jax.numpy as jnp
import mujoco
import xml.etree.ElementTree as ET
from Common import *
from Common.Go2_dog import *
from Common.runtime_paths import (
    ACTUATOR_NETWORK,
    GO2_ASSETS_DIR,
    GO2_TORQUE_XML,
    SCENE_TORQUE_XML,
    resolve_repo_path,
)
from .deployment_collector import ThreadedGo2Deploy
from datetime import datetime
from Policies import *
from Training import *
from Training.Go2_dog import Go2DataWrangler



def main(trial_name="", nn_policy_type="mppi", seed=0, jit=True, steps_per_episode=100, debug=True,
         headless=False, save_video=False, video_text=True, nn_dynamics_path=None, nn_observer_path=None, nn_policy_kwargs=None, camera_kwargs=None):
   
    np.random.seed(seed)

    #jax.config.update("jax_check_tracer_leaks", True)


    if nn_policy_kwargs is None:
        raise ValueError("nn_policy_kwargs cannot be None")

    mj_model_path = SCENE_TORQUE_XML

    actuator_net = MLPBase.load(ACTUATOR_NETWORK)
    actuator_net.Lipschitz_ub = 1e12
    params = actuator_net.params
    actuator_net.set_inference_mode(True, params)

    nn_dynamics_path = resolve_repo_path(nn_dynamics_path)
    nn_observer_path = resolve_repo_path(nn_observer_path)
    nn_dynamics = MLPDynamics(path=nn_dynamics_path, actuator_net=actuator_net, SNS=True)
    nn_observer = MLPObserver(path=nn_observer_path, SNS=True)
    nn_model = NNModel(dynamics=nn_dynamics, observer=nn_observer)
    #nn_model = MLP(path=nn_model_path)

    model_kwargs = {"Lipschitz": nn_dynamics.Lipschitz}
    logger = get_logger(trial_name, model_kwargs, "Deployment")

    T = nn_dynamics.T
    H = nn_dynamics.H
    H_Obs = nn_observer.H_Obs



    # valid_T_and_n_blocks = [[1, 18], [2, 9], [3, 6], [6, 3], [9, 2], [18, 1]]

    # for valid in valid_T_and_n_blocks:
    #     if valid[0] == T:
    #         n_blocks = valid[1]
    #         break
    # else:
    #     raise ValueError(f"Invalid T: {T}. Valid T values are: {[v[0] for v in valid_T_and_n_blocks]}")

        # print(f"T: {T}, H: {H}, H_Obs_100hz: {H_Obs_100hz}, n_blocks: {n_blocks}")

    n_history_steps = H
    n_history_steps_observer = H_Obs
    horizon = 19
    print(f"horizon: {horizon}")
    
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
    nn_policy_kwargs["jit"] = jit
    if nn_policy_type == "mppi":
        nn_policy_kwargs["seed"] = seed + 1

        #nn_policy = MPPI_Policy(nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, **nn_policy_kwargs)
        nn_policy = MPPI_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)
    elif nn_policy_type == "ddp":
        nn_policy = Block_DDP_Policy(nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)
    elif nn_policy_type == "oracle":
        pass
    elif nn_policy_type == "spline_shooter":
        nn_policy = Spline_Shooter_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, residuals=residuals, global_residuals=global_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)

    else:
        raise ValueError(f"Unknown nn_policy_type: {nn_policy_type}")
    
    # actions_path = "./Training/Go2_dog/actions.dill"
    # states_path = "./Training/Go2_dog/states.dill"
    # states = States.load(states_path)
    # actions = Actions.load(actions_path)

    # split nn_model_path at _params.npz to get the first part of the state and actions path
    # beginning = nn_dynamics_path.split("_dynamics_params.npz")[0]
    # actions_path = f"{beginning}_actions.dill"
    # states_path = f"{beginning}_states.dill"
    # observations_path = f"{beginning}_observations.dill"

    import glob

    # Step 1: Split to get directory
    directory = os.path.dirname(nn_dynamics_path)

    actions_files = glob.glob(os.path.join(directory, "*_actions.dill"))
    states_files = glob.glob(os.path.join(directory, "*_states.dill"))
    observations_files = glob.glob(os.path.join(directory, "*_observations.dill"))
    if len(actions_files) == 0 or len(states_files) == 0 or len(observations_files) == 0:
        raise FileNotFoundError("Could not find actions, states, or observations files in the directory.")
    actions_path = actions_files[0]
    states_path = states_files[0]
    observations_path = observations_files[0]
    states = States.load(states_path)
    actions = Actions.load(actions_path)
    observations = Observations.load(observations_path)
    print(f"states scales: {states.scales.array}, actions scales: {actions.scales.array}, observations scales: {observations.scales.array}")

    data_wrangler = Go2DataWrangler(horizon=horizon, T=T, H=H, H_Obs=H_Obs, n_blocks=horizon, 
                                    mj_dataset_cls=mj_dataset_cls, states=states, actions=actions, observations=observations)


    init_sigma = jax.tree.flatten(init_sigma)[0]
    init_sigma = jnp.stack(init_sigma)
    init_guess = jax.tree.flatten(init_guess)[0]
    init_guess = jnp.stack(init_guess)
    if nn_policy_type != "oracle":
        data_collection_agent = NNDataCollection_Policy(
                        nn_policy=nn_policy,
                        data_wrangler=data_wrangler,
                        jit=jit)
        eval_agent = NNDataCollection_Policy(
                nn_policy=nn_policy,
                data_wrangler=data_wrangler,
                jit=jit)
    else:

        scene_xml_path = SCENE_TORQUE_XML
        go2_xml_path = GO2_TORQUE_XML

        scene_tree = ET.parse(scene_xml_path)
        scene_root = scene_tree.getroot()

        robot_tree = ET.parse(go2_xml_path)
        robot_root = robot_tree.getroot()

        # Replace <include> with inlined robot XML
        for include in scene_root.findall("include"):
            if "file" in include.attrib:
                scene_root.remove(include)
                for child in list(robot_root):
                    scene_root.append(child)
                break
    
        compiler = scene_root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            scene_root.insert(0, compiler)
        compiler.attrib["meshdir"] = GO2_ASSETS_DIR

        xml_string = ET.tostring(scene_root, encoding="unicode")

        data_collection_agent = OracleDataCollection_Policy(
                        xml_string=xml_string,
                        step_cost=step_cost_from_residuals,
                        data_wrangler=data_wrangler,
                        commands_callback=update_gait_commands,
                        init_sigma=init_sigma,
                        **nn_policy_kwargs)
    

    data_collection_module = ThreadedGo2Deploy(data_collection_agent=data_collection_agent,
        eval_agent=eval_agent,
        mj_dataset_cls=mj_dataset_cls,
        mj_model_path=mj_model_path,
        n_history_steps=n_history_steps,
        n_history_steps_observer=n_history_steps_observer,
        horizon=horizon,
        n_envs=1,
        n_workers=1,
        save_video=save_video,
        video_text=video_text,
        seed=seed,
        headless=headless,
        steps_per_episode=steps_per_episode, 
        camera_kwargs=camera_kwargs)
    
    
    if nn_policy_type != "oracle":
        dynamics_params_training = jax.tree.map(lambda x: x, nn_model.dynamics.params)
        observer_params_training = jax.tree.map(lambda x: x, nn_model.observer.params)
        nn_model.dynamics.set_inference_mode(True, params=dynamics_params_training)
        nn_model.observer.set_inference_mode(True, params=observer_params_training)
        nn_policy.update_model(nn_model)
        data_collection_module.agent.update_nn_policy(nn_policy)
        data_collection_module.agent.update_data_wrangler(data_wrangler)
        #print(f"Params: {dynamics_params_training}, {observer_params_training}")

    logger.epoch = 0

    #seeds = [seed + 96, seed + 97, seed + 98, seed + 99, seed + 100, seed + 101, seed + 102, seed + 103, seed + 104, seed + 105]
    seeds = [seed + 96]
    costs = []
    logger._create_log_file("eval")
    for i, seed in enumerate(seeds):
        logger = data_collection_module.evaluate(logger, seed=seed, debug=debug, with_state_estimation=True)
        print(f"logs: {logger.log_dict}")
        # if has cost in key
        for key in logger.log_dict.keys():
            if "cost" in key:
                costs.append(logger.log_dict[key].item())
        logger._log_step(i)

    print(f"Costs: {costs}, Mean: {np.mean(costs)}, Std: {np.std(costs)}")






# #########

# import os
# import jax
# import jax.numpy as jnp
# from Common import *
# from Common.Go2_dog import *
# from .data_collector import ThreadedGo2DataCollector
# from Policies import *
# from Training import *
# from .nn_data import Go2DataWrangler


# def main(trial_name="", nn_policy_type="mppi", seed=0, jit=True, steps_per_episode=100, debug=True,
#          headless=False, save_video=False, nn_dynamics_path=None, nn_observer_path=None, nn_policy_kwargs=None, camera_kwargs=None):
   
#     np.random.seed(seed)
#     n_states, n_actions, n_obs = state_and_action_size()
#     print(f"n_states: {n_states}, n_actions: {n_actions}, n_obs: {n_obs}")

#     mj_model_path = "./Mj_models/Go2_dog/scene_pd.xml"

#     logger = get_logger(trial_name, nn_dynamics_kwargs, "Training")

#     valid_T_and_n_blocks = [[1, 18], [2, 9], [3, 6], [6, 3], [9, 2], [18, 1]]
#     T_and_n_blocks = [nn_dynamics_kwargs["T"], n_blocks]
#     if T_and_n_blocks not in valid_T_and_n_blocks:
#         raise ValueError(f"Invalid T and n_blocks combination: {T_and_n_blocks}. Valid combinations are: {valid_T_and_n_blocks}")


#     n_history_steps = nn_dynamics_kwargs["H"] * 2 + 2
#     n_history_steps_observer = nn_observer_kwargs["H_Obs_100hz"] + 2
#     horizon = 37

#     nn_dynamics_kwargs["seed"] = seed
#     nn_dynamics = MLPDynamics(n_states=n_states, n_actions=n_actions, **nn_dynamics_kwargs)
#     nn_observer = MLPObserver(n_states=n_states, n_actions=n_actions, n_obs=n_obs, **nn_observer_kwargs)
#     nn_model = NNModel(dynamics=nn_dynamics, observer=nn_observer)


    
#     import mujoco
#     mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
#     action_bounds = mj_model.actuator_ctrlrange
#     action_bounds = jnp.asarray(action_bounds)

#     init_sigma, init_guess = init_sigma_and_guess()

#     action_bounds_tree = jax.tree_map(lambda x: x, init_sigma)
#     action_bounds_tree.fl_hip = action_bounds[0]
#     action_bounds_tree.fl_knee = action_bounds[1]
#     action_bounds_tree.fl_ankle = action_bounds[2]
#     action_bounds_tree.fr_hip = action_bounds[3]
#     action_bounds_tree.fr_knee = action_bounds[4]
#     action_bounds_tree.fr_ankle = action_bounds[5]
#     action_bounds_tree.rl_hip = action_bounds[6]
#     action_bounds_tree.rl_knee = action_bounds[7]
#     action_bounds_tree.rl_ankle = action_bounds[8]
#     action_bounds_tree.rr_hip = action_bounds[9]
#     action_bounds_tree.rr_knee = action_bounds[10]
#     action_bounds_tree.rr_ankle = action_bounds[11]


#     #nn_policy_kwargs["horizon"] = horizon
#     if nn_policy_type == "mppi":
#         nn_policy_kwargs["seed"] = seed + 1
#         nn_policy = MPPI_Policy(nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, **nn_policy_kwargs)
#     else:
#         raise ValueError(f"Unknown nn_policy_type: {nn_policy_type}")
    
#     # actions_path = "./Training/Go2_dog/actions.dill"
#     # states_path = "./Training/Go2_dog/states.dill"
#     # states = States.load(states_path)
#     # actions = Actions.load(actions_path)


#     data_wrangler = Go2DataWrangler(horizon=horizon, T=nn_dynamics_kwargs["T"], H=nn_dynamics_kwargs["H"], H_Obs_100hz=nn_observer_kwargs["H_Obs_100hz"], n_blocks=n_blocks, mj_dataset_cls=mj_dataset_cls)


#     init_sigma = jax.tree.flatten(init_sigma)[0]
#     init_sigma = jnp.stack(init_sigma)
#     init_guess = jax.tree.flatten(init_guess)[0]
#     init_guess = jnp.stack(init_guess)
#     data_collection_agent = NNDataCollection_Policy(
#                     nn_policy=nn_policy,
#                     data_wrangler=data_wrangler,
#                     jit=jit)
    
    
#     data_collection_module = ThreadedGo2DataCollector(data_collection_agent=data_collection_agent,
#         mj_dataset_cls=mj_dataset_cls,
#         mj_model_path=mj_model_path,
#         n_history_steps=n_history_steps,
#         n_history_steps_observer=n_history_steps_observer,
#         horizon=horizon,
#         n_envs=1,
#         n_workers=1,
#         save_video=save_video,
#         seed=seed,
#         headless=headless,
#         steps_per_episode=1)
    


#     logger.epoch = 0
#     logger = data_collection_module.evaluate(logger, seed=seed+96, debug=debug, with_state_estimation=False)
