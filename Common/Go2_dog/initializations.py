import jax
import jax.numpy as jnp
from Common.Go2_dog import Go2StatesStruct, Go2ActionsStruct, Go2ObservationsStruct
from Common import Logger
from datetime import datetime

def get_logger(trial_name, model_kwargs, train_or_deploy="Training"):
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if model_kwargs["Lipschitz"] == True:
        name = "Lipschitz"

    elif model_kwargs["Lipschitz"] == False:
        name = "Vanilla"

    else:
        raise ValueError(f"Invalid Lipshitz value: {model_kwargs['Lipshitz']}")

    if trial_name == "":
        log_dir = f"Logs/{train_or_deploy}/Go2_dog/Metrics/{name}_{dt}"
        ckpt_dir = f"Logs/{train_or_deploy}/Go2_dog/Ckpts/{name}_{dt}"
        plot_dir = f"Logs/{train_or_deploy}/Go2_dog/Plots/{name}_{dt}"
    else:
        log_dir = f"Logs/{train_or_deploy}/Go2_dog/Metrics/{trial_name}/{name}_{dt}"
        ckpt_dir = f"Logs/{train_or_deploy}/Go2_dog/Ckpts/{trial_name}/{name}_{dt}"
        plot_dir = f"Logs/{train_or_deploy}/Go2_dog/Plots/{trial_name}/{name}_{dt}"


    logger = Logger(ckpt_dir=ckpt_dir,
                    log_dir=log_dir,
                    plot_dir=plot_dir)

    return logger

def state_and_action_size():
    n_actions = 12
    from dataclasses import fields
    states = fields(Go2StatesStruct)
    n_states = len(states) + 2
    obs = fields(Go2ObservationsStruct)
    n_obs = len(obs) + 2
    return n_states, n_actions, n_obs


def init_sigma_and_guess(action_noise_scale=1.0):

    action_noise_std = Go2ActionsStruct(fl_hip=0.125, 
                                        fl_knee=0.25, 
                                        fl_ankle=0.25,
                                        fr_hip=0.125,
                                        fr_knee=0.25,
                                        fr_ankle=0.25,
                                        rl_hip=0.125,
                                        rl_knee=0.25,
                                        rl_ankle=0.25,
                                        rr_hip=0.125,
                                        rr_knee=0.25,
                                        rr_ankle=0.25)
    
    # action_noise_std = Go2ActionsStruct(fl_hip=0.06, 
    #                                 fl_knee=0.2, 
    #                                 fl_ankle=0.2,
    #                                 fr_hip=0.06,
    #                                 fr_knee=0.2,
    #                                 fr_ankle=0.2,
    #                                 rl_hip=0.06,
    #                                 rl_knee=0.2,
    #                                 rl_ankle=0.2,
    #                                 rr_hip=0.06,
    #                                 rr_knee=0.2,
    #                                 rr_ankle=0.2)
    
    action_noise_std = jax.tree.map(lambda x: jnp.array(x)*action_noise_scale, action_noise_std)

    solution = Go2ActionsStruct(fl_hip=0.0, 
                            fl_knee=0.9, 
                            fl_ankle=-1.8,
                            fr_hip=0.0,
                            fr_knee=0.9,
                            fr_ankle=-1.8,
                            rl_hip=0.0,
                            rl_knee=0.9,
                            rl_ankle=-1.8,
                            rr_hip=0.0,
                            rr_knee=0.9,
                            rr_ankle=-1.8)
    solution = jax.tree.map(lambda x: jnp.array(x), solution)

    return action_noise_std, solution