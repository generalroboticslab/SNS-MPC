import os
import argparse


# Minimal parse to extract early environment settings
early_parser = argparse.ArgumentParser()
early_parser.add_argument("--gpu", type=str, help="CUDA_VISIBLE_DEVICES override", default="0")
args, _ = early_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
GPU = args.gpu


import yaml
import json
from datetime import datetime
import subprocess
import socket

CONFIG_DIR = "Logs"
LOG_DIR = os.path.join(CONFIG_DIR, "Deployment")

def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_config(base, override):
    config = base.copy()
    for key, val in override.items():
        if val is not None:
            config[key] = val
    return config

def log_config(config, trial_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    sub_dir = f"Go2_dog"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(LOG_DIR, sub_dir, f"Configs/{trial_name}/config_{timestamp}.json")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Config logged at {path}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def dispatch(config):
    from Deploy_simulation.Go2_dog import deploy
    deploy.main(**config)


def main():
    train_parser = argparse.ArgumentParser()
    train_parser.add_argument("--config", type=str)

    train_parser.add_argument("--debug", type=str2bool)
    train_parser.add_argument("--steps_per_episode", type=int)
    train_parser.add_argument("--nn_policy_type", type=str)
    train_parser.add_argument("--nn_dynamics_path", type=json.loads)
    train_parser.add_argument("--nn_observer_path", type=json.loads)
    train_parser.add_argument("--nn_policy_kwargs", type=json.loads)
    train_parser.add_argument("--camera_kwargs", type=json.loads)
    train_parser.add_argument("--trial_name", type=str)
    train_parser.add_argument("--save_video", type=str2bool)
    train_parser.add_argument("--video_text", type=str2bool)

    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--jit", type=str2bool)
    train_parser.add_argument("--headless", type=str2bool)

    train_parser.add_argument("--gpu", type=str, default=GPU)


    args = train_parser.parse_args()

    # Convert Namespace → dict and handle dtype conversion
    raw_args = vars(args)
    config_path = raw_args.get("config")

    base_config = load_yaml_config(config_path)
    full_config = merge_config(base_config, raw_args)

    # Add metadata
    full_config["hostname"] = socket.gethostname()
    try:
        full_config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        full_config["git_commit"] = "unknown"

    log_config(full_config, full_config["trial_name"])
    import jax
    
    if "dtype" in full_config and isinstance(full_config["dtype"], str):
        if full_config["dtype"] == "float64":
            jax.config.update("jax_enable_x64", True)

        full_config["dtype"] = getattr(jax.numpy, full_config["dtype"])

    full_config.pop("config")
    full_config.pop("hostname")
    full_config.pop("git_commit")
    full_config.pop("gpu")

    dispatch(full_config)

if __name__ == "__main__":
    main()
