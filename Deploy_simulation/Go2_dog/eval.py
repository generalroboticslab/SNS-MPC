import os
import jax
import jax.numpy as jnp
import mujoco
from Common import *
from Common.Go2_dog import *
from Common.runtime_paths import ACTUATOR_NETWORK, SCENE_TORQUE_XML, resolve_repo_path
from .deployment_collector import ThreadedGo2Deploy
from datetime import datetime
from Policies import *
from Training import *
from Training.Go2_dog import Go2DataWrangler
import glob
import re



def _checkpoint_pairs(ckpt_dir, requested_steps):
    observer_ckpts = sorted(
        f for f in os.listdir(ckpt_dir) if f.endswith("observer_params.npz")
    )
    pairs = []
    for observer_ckpt in observer_ckpts:
        dynamics_ckpt = observer_ckpt.replace("observer_params.npz", "dynamics_params.npz")
        dynamics_path = os.path.join(ckpt_dir, dynamics_ckpt)
        observer_path = os.path.join(ckpt_dir, observer_ckpt)
        if not os.path.exists(dynamics_path):
            continue

        match = re.search(r"step_(\d+)_observer_params\.npz$", observer_ckpt)
        step = int(match.group(1)) if match else None
        pairs.append((step, dynamics_path, observer_path))

    if not pairs:
        raise FileNotFoundError(f"No checkpoint pairs found in {ckpt_dir}")

    if requested_steps is not None:
        requested_steps = list(requested_steps)
        step_pairs = [pair for pair in pairs if pair[0] in requested_steps]
        if step_pairs:
            return sorted(step_pairs, key=lambda pair: pair[0])

        static_pairs = [pair for pair in pairs if pair[0] is None]
        if len(static_pairs) == 1 and requested_steps:
            _, dynamics_path, observer_path = static_pairs[0]
            return [(requested_steps[0], dynamics_path, observer_path)]

        return []

    if all(step is None for step, _, _ in pairs):
        return [(0, dynamics_path, observer_path) for _, dynamics_path, observer_path in pairs]

    return sorted(
        pairs,
        key=lambda pair: (pair[0] is None, pair[0] if pair[0] is not None else 0),
    )


def main(trial_name="", nn_policy_type="mppi", seed=0, jit=True, steps_per_episode=100, debug=True, n_seeds=10, with_state_estimation=True,
         headless=False, save_video=False, video_text=True, ckpt_dirs=None, ckpt_names=None, nn_policy_kwargs=None, camera_kwargs=None, steps=None):



    #jax.config.update("jax_check_tracer_leaks", True)


    if nn_policy_kwargs is None:
        raise ValueError("nn_policy_kwargs cannot be None")

    mj_model_path = SCENE_TORQUE_XML

    actuator_net = MLPBase.load(ACTUATOR_NETWORK)
    actuator_net.Lipschitz_ub = 1e12
    params = actuator_net.params
    actuator_net.set_inference_mode(True, params)


    log_paths = []

    assert ckpt_dirs is not None, "ckpt_dirs cannot be None"
    # ends with step_*_dynamics_params.npz or step_*_observer_params.npz
    print(f"Evaluating ckpt_dirs: {ckpt_dirs}")
    for ckpt_dir in ckpt_dirs:
        ckpt_dir = resolve_repo_path(ckpt_dir)
        if not os.path.exists(ckpt_dir):
            raise ValueError(f"ckpt_dir {ckpt_dir} does not exist")
        checkpoint_pairs = _checkpoint_pairs(ckpt_dir, steps)
        if not checkpoint_pairs:
            raise ValueError(f"No checkpoints selected for evaluation in {ckpt_dir}")

        model_kwargs = {"Lipschitz": True}
        logger = get_logger(trial_name, model_kwargs, "Deployment")
        logger._create_log_file("eval")
        total_steps = 0
        for j, (step, nn_dynamics_path, nn_observer_path) in enumerate(checkpoint_pairs):
            print(
                f"Evaluating step {step} with dynamics checkpoint {os.path.basename(nn_dynamics_path)} "
                f"and observer checkpoint {os.path.basename(nn_observer_path)}"
            )


            nn_dynamics = MLPDynamics(path=nn_dynamics_path, actuator_net=actuator_net, SNS=True)
            #nn_dynamics = QuadrupedDynamics(path=nn_dynamics_path, actuator_net=actuator_net, T=1, H=8, horizon=25, n_states=60, n_actions=12)
            nn_observer = MLPObserver(path=nn_observer_path, SNS=True)
            nn_model = NNModel(dynamics=nn_dynamics, observer=nn_observer)
            #nn_model = MLP(path=nn_model_path)

            dynamics_params_training = jax.tree.map(lambda x: x, nn_model.dynamics.params)
            observer_params_training = jax.tree.map(lambda x: x, nn_model.observer.params)
            nn_model.dynamics.set_inference_mode(True, params=dynamics_params_training)
            nn_model.observer.set_inference_mode(True, params=observer_params_training)


            if j == 0:
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

                directory = ckpt_dir   

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
                # state_rollouts = States.load(states_path)
                # action_rollouts = Actions.load(actions_path)

                # print(f"states scales: {states.scales.array}, actions scales: {actions.scales.array}, observations scales: {observations.scales.array}")
                # assert False

                T = nn_dynamics.T
                H = nn_dynamics.H
                H_Obs = nn_observer.H_Obs

                n_history_steps = H
                n_history_steps_observer = H_Obs
                horizon = 25
                print(f"horizon: {horizon}")
                


                #nn_policy_kwargs["horizon"] = horizon
                nn_policy_kwargs["jit"] = jit
                if nn_policy_type == "mppi":
                    nn_policy_kwargs["seed"] = seed + 1
                    nn_policy = MPPI_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)
                elif nn_policy_type == "spline_shooter":
                    nn_policy = Spline_Shooter_Policy(horizon=horizon, nn_model=nn_model, step_cost=step_cost_from_residuals, residuals=residuals, global_residuals=global_residuals, initial_solution=init_guess, initial_sigma=init_sigma, action_bounds=action_bounds_tree, commands_callback=update_gait_commands, **nn_policy_kwargs)

                else:
                    raise ValueError(f"Unknown nn_policy_type: {nn_policy_type}")

                # print(f"states scales: {states.scales.array}, actions scales: {actions.scales.array}, observations scales: {observations.scales.array}")

                data_wrangler = Go2DataWrangler(horizon=horizon, T=T, H=H, H_Obs=H_Obs, n_blocks=horizon, 
                                                mj_dataset_cls=mj_dataset_cls, states=states, actions=actions, observations=observations)


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
                
            

            nn_policy.update_model(nn_model)
            data_collection_module.agent.update_nn_policy(nn_policy)
            data_collection_module.agent.update_data_wrangler(data_wrangler)
            data_collection_module.eval_agent.update_nn_policy(nn_policy)
            data_collection_module.eval_agent.update_data_wrangler(data_wrangler)
                #print(f"Params: {dynamics_params_training}, {observer_params_training}")

            logger.epoch = step

            # #seeds = [seed + 96, seed + 97, seed + 98, seed + 99, seed + 100, seed + 101, seed + 102, seed + 103, seed + 104, seed + 105]
            # seeds = [seed + 96]
            seeds = [seed + i for i in range(n_seeds)]
            costs = []

            for i, seed in enumerate(seeds):
                logger = data_collection_module.evaluate(logger, seed=seed, debug=debug, with_state_estimation=with_state_estimation)
                print(f"logs: {logger.log_dict}")
                # if has cost in key
                logger.log_dict["epoch"] = step
                for key in logger.log_dict.keys():
                    if "cost" in key:
                        costs.append(logger.log_dict[key].item())
                logger._log_step(total_steps)
                if total_steps == 0:
                    log_paths.append(logger.log_filename)
                total_steps += 1
                print(f"Total steps: {total_steps}")

    logger.plot_evaluation(log_paths, labels=ckpt_names, fields=None)
