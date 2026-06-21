import os
import pickle as pkl
from matplotlib import pyplot as plt
import time
import imageio
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam



import jax 
import jax.numpy as jnp
from Common.nn_model import MLPBase
from Common.runtime_paths import ACTUATOR_DATA_200HZ
import optax

from Common import Observations, States


def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

def cauchy_loss(pred, target):
    diff = pred - target
    return jnp.mean(jnp.log1p(diff ** 2))

# Training step function for JAX

def jax_collate(batch):
  """
  Collate function specifies how to combine a list of data samples into a batch.
  default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
  """
  return jax.tree.map(jnp.asarray, default_collate(batch))

class ActuatorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['joint_states'])

    def __getitem__(self, idx):
        return {k: v[idx] for k,v in self.data.items()}


def train_actuator_network_jax(xs, ys, actuator_network_path, my_data=False):
    print("Training actuator network...")

    num_data = xs.shape[0]
    num_train = num_data // 5 * 4
    num_test = num_data - num_train

    dataset = ActuatorDataset({"joint_states": xs, "tau_ests": ys})
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_test])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, collate_fn=jax_collate)
    test_loader = DataLoader(val_set, batch_size=128, shuffle=True, collate_fn=jax_collate)

    # Model and optimizer setup
    key = jax.random.PRNGKey(0)
    if not my_data:
        model = MLPBase(key=key, input_dim=6, output_dim=1, hidden_scale=[6, 6], Lipschitz=True)
    else:
        model = MLPBase(key=key, input_dim=18, output_dim=1, hidden_scale=[5, 5], Lipschitz=True)
    print(f"Model hidden dimensions: {model.hidden_dims}")
    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(model)

    @jax.jit
    def loss_fn(model, data, labels, alpha=1e-4):
        print("data shape:", data.shape)
        preds = jax.vmap(model.forward)(data)
        loss = cauchy_loss(preds, labels)
        mse = mse_loss(preds, labels)
        c_o = model.Lipschitz_constants
        c_o, _ = jax.tree_util.tree_flatten(c_o)
        c_o = jax.tree.map(lambda x: x.reshape(1, -1), c_o)
        c_o = jnp.concatenate(c_o, axis=-1).flatten()
        c_o = jax.nn.softplus(c_o)
        lipschitz_loss = jnp.prod(c_o)
        return loss + alpha * lipschitz_loss, {'mse': mse, 'lipschitz_loss': lipschitz_loss}

    @jax.jit
    def train_step(model, opt_state, batch, alpha=1e-4):
        data = batch['joint_states']
        labels = batch['tau_ests']

        values, grads = jax.value_and_grad(loss_fn, has_aux=True)(model, data, labels, alpha)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, values[1]

    # Training loop
    epochs = 300
    alpha_max = 5e-5
    #power = 4.0 # 4.0
    #alphas = np.linspace(0, 1, epochs) * alpha
    alphas = np.ones(epochs) * alpha_max
    # steps = np.arange(epochs//2)
    # raw = (steps / (epochs - 1)) ** power
    # alphas = alpha_max * raw / raw[-1]  # normalize to hit alpha_max exactly

    # cycles = 10
    # pretrain_epochs = 0
    # cycle_length = (epochs-pretrain_epochs) / cycles
    # alphas = np.zeros(epochs)
    # for i in range(pretrain_epochs, epochs):
    #     cycle_pos = (i % cycle_length) / (cycle_length - 1)
    #     raw = (cycle_pos) ** power
    #     alphas[i] = alpha_max * raw


    for epoch in range(epochs):
        epoch_loss = 0
        epoch_smooth = 0
        for batch in train_loader:
            model, opt_state, loss = train_step(model, opt_state, batch, alphas[epoch] if epoch < len(alphas) else alphas[-1])
            epoch_loss += loss['mse']
            epoch_smooth += loss['lipschitz_loss']
        epoch_loss /= len(train_loader)
        epoch_smooth /= len(train_loader)

        #print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Smooth Loss: {epoch_smooth}")

        # Validation step
        val_loss = 0
        val_smooth = 0
        for batch in test_loader:
            data = batch['joint_states']
            labels = batch['tau_ests']
            _, loss = loss_fn(model, data, labels, alphas[epoch] if epoch < len(alphas) else alphas[-1])
            val_loss += loss['mse']
            val_smooth += loss['lipschitz_loss']
        val_loss /= len(test_loader)
        val_smooth /= len(test_loader)
        #print(f"Validation Loss: {val_loss}, Validation Smooth Loss: {val_smooth}")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Val Loss: {val_loss}, Smooth Loss: {epoch_smooth}, Val Smooth Loss: {val_smooth}, Alpha: {alphas[epoch] if epoch < len(alphas) else alphas[-1]}")

        # Save the model
        if epoch % 10 == 0:
            model.save(actuator_network_path)

    return model


def train_actuator_network_and_plot_predictions_jax(log_dir, actuator_network_path, load_pretrained_model=False, my_data=False):

    log_path = log_dir + "log.pkl"
    with open(log_path, 'rb') as file:
        data = pkl.load(file)

    datas = data['hardware_closed_loop'][1]

    print(f"Cfg: {data['hardware_closed_loop'][0]}")

    print(f"Loaded {len(datas)} data points from {log_path}")
    if len(datas) < 1:
        return

    tau_ests = np.zeros((len(datas), 12))
    torques = np.zeros((len(datas), 12))
    joint_positions = np.zeros((len(datas), 12))
    joint_position_targets = np.zeros((len(datas), 12))
    joint_velocities = np.zeros((len(datas), 12))
    actions = np.zeros((len(datas), 12))
    times = np.zeros((len(datas), 1))

    print("key names in data:", datas[0].keys())

    if "tau_est" not in datas[0].keys():
        return

    for i in range(len(datas)):
        tau_ests[i, :] = datas[i]["tau_est"]
        torques[i, :] = datas[i]["torques"]
        joint_positions[i, :] = datas[i]["joint_pos"]
        joint_position_targets[i, :] = datas[i]["joint_pos_target"]
        joint_velocities[i, :] = datas[i]["joint_vel"]
        actions[i, :] = datas[i]["action"]
        times[i, 0] = datas[i]["time"]

    print("times:", times[:20, 0])

    timesteps = np.array(range(len(datas))) / 50.0

    import matplotlib.pyplot as plt

    joint_position_errors = joint_positions - joint_position_targets
    joint_velocities = joint_velocities

    joint_position_errors = torch.tensor(joint_position_errors, dtype=torch.float)
    joint_velocities = torch.tensor(joint_velocities, dtype=torch.float)
    tau_ests = torch.tensor(tau_ests, dtype=torch.float)

    xs = []
    ys = []
    step = 2
    # all joints are equal
    for i in range(12):
        xs_joint = [joint_position_errors[2:-step+1, i:i+1],
                joint_position_errors[1:-step, i:i+1],
                joint_position_errors[:-step-1, i:i+1],
                joint_velocities[2:-step+1, i:i+1],
                joint_velocities[1:-step, i:i+1],
                joint_velocities[:-step-1, i:i+1]]
        tau_ests_joint = [tau_ests[step:-1, i:i+1]]

        xs_joint = torch.cat(xs_joint, dim=1)
        xs += [xs_joint]
        ys += tau_ests_joint

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)
    xs = xs.numpy()
    ys = ys.numpy()

    print("xs shape:", xs.shape, "ys shape:", ys.shape)


    start = 8000
    stop = 21000
    if my_data:
        data = np.load(ACTUATOR_DATA_200HZ)
        obs = data['obs'][start:stop]
        torques = data['torque'][start:stop]
        action_times = data['action_time'][start:stop]
    else:
        torques = np.load("./torque_buffer.npz")["torque"]
        obs = Observations.load("./obs_buffer.dill")
       

    for i in range(12):
        if my_data:
           joint_data_, tau_ = get_joint_data_200hz(action_times, torques, obs, i)
        else:
            # obs = jax.tree.map(lambda x: x[:, 2:-2], obs)
            # taus = torques['torque'][:, 2:-2]
            joint_data_, tau_ = get_joint_data_preload(torques, obs, i)
        if i == 0:
           joint_data = joint_data_
           tau = tau_
        else:
           joint_data = jnp.concatenate([joint_data, joint_data_], axis=0)
           tau = jnp.concatenate([tau, tau_], axis=0)

        print(f"Joint data for joint {i}: {joint_data.shape}")
        print(f"Torque for joint {i}: {tau.shape}")

    # assert False, "This code is not fully implemented yet."
    xs = np.asarray(joint_data, dtype=np.float32)
    ys = np.asarray(tau, dtype=np.float32)
    # reflect and double the data
    xs = np.concatenate([xs, -1*xs], axis=0)
    ys = np.concatenate([ys, -1*ys], axis=0)

    if load_pretrained_model:
           model = MLPBase.load(actuator_network_path)
    else:
        model = train_actuator_network_jax(xs, ys, actuator_network_path, my_data=my_data)

    print("Model trained, making predictions...")

    tau_true = ys.reshape(12, -1).T
    tau_preds = jax.vmap(model.forward)(xs).reshape(12, -1).T

    # Plot predictions
    if my_data:
       div = 200
    else:
       div = 50
    timesteps = np.arange(len(xs)) / div
    starting_point = 3000
    plot_length = 800

    timesteps_ = timesteps[starting_point:starting_point+plot_length]
    tau_ = tau_true[starting_point:starting_point+plot_length]
    tau_preds_ = tau_preds[starting_point:starting_point+plot_length]

    fig, axs = plt.subplots(6, 2, figsize=(14, 6))
    axs = np.array(axs).flatten()
    for i in range(12):
        axs[i].plot(timesteps_, tau_[:, i], label="true torque", color='tab:orange')
        axs[i].plot(timesteps_, tau_preds_[:, i], linestyle='--', label="actuator model", color='tab:green')
        #axs[i].plot(timesteps, pos_targets[:, i] * 10 - np.mean(pos_targets[:, i] * 10), label="joint position target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{log_dir}/actuator_model_predictions_jax.png", dpi=300)
    plt.close()
    plt.clf()


def get_joint_data_preload(torque, past_observations_t, idx):
    step = 2
    q_pos = past_observations_t.array[0, 2:-step+1, past_observations_t.index_map['q_fl_hip'][0]+idx:past_observations_t.index_map['q_fl_hip'][1]+idx]
    last_q_pos = past_observations_t.array[0, 1:-step, past_observations_t.index_map['q_fl_hip'][0]+idx:past_observations_t.index_map['q_fl_hip'][1]+idx]
    last_last_q_pos = past_observations_t.array[0, :-step-1, past_observations_t.index_map['q_fl_hip'][0]+idx:past_observations_t.index_map['q_fl_hip'][1]+idx]

    q_pos_target = past_observations_t.array[0, 2:-step+1, past_observations_t.index_map['qd_fl_hip'][0]+idx:past_observations_t.index_map['qd_fl_hip'][1]+idx]
    last_q_pos_target = past_observations_t.array[0, 1:-step, past_observations_t.index_map['qd_fl_hip'][0]+idx:past_observations_t.index_map['qd_fl_hip'][1]+idx]
    last_last_q_pos_target = past_observations_t.array[0, :-step-1, past_observations_t.index_map['qd_fl_hip'][0]+idx:past_observations_t.index_map['qd_fl_hip'][1]+idx]

    pos_error = q_pos - q_pos_target
    last_pos_error = last_q_pos - last_q_pos_target
    last_last_pos_error = last_last_q_pos - last_last_q_pos_target

    velocity = past_observations_t.array[0, 2:-step+1, past_observations_t.index_map['v_fl_hip'][0]+idx:past_observations_t.index_map['v_fl_hip'][1]+idx]
    last_velocity = past_observations_t.array[0, 1:-step, past_observations_t.index_map['v_fl_hip'][0]+idx:past_observations_t.index_map['v_fl_hip'][1]+idx]
    last_last_velocity = past_observations_t.array[0, :-step-1, past_observations_t.index_map['v_fl_hip'][0]+idx:past_observations_t.index_map['v_fl_hip'][1]+idx]

    joint_data = jnp.concatenate([pos_error, last_pos_error, last_last_pos_error, velocity, last_velocity, last_last_velocity], axis=-1)

    torque = torque[0, 2:-step+1, idx:idx+1]

    return joint_data, torque

def get_joint_data_200hz(action_time_array, torque_array, obs_array, idx):
    step = 8
    q_pos = obs_array[step:-step+1, idx:idx+1]
    last_q_pos = obs_array[step-1:-step, idx:idx+1]
    last_last_q_pos = obs_array[step-2:-step-1, idx:idx+1]
    last_last_last_q_pos = obs_array[step-3:-step-2, idx:idx+1]
    last_last_last_last_q_pos = obs_array[step-4:-step-3, idx:idx+1]
    last_last_last_last_last_q_pos = obs_array[step-5:-step-4, idx:idx+1]
    last_6_q_pos = obs_array[step-6:-step-5, idx:idx+1]
    last_7_q_pos = obs_array[step-7:-step-6, idx:idx+1]
    last_8_q_pos = obs_array[step-8:-step-7, idx:idx+1]

    q_pos_target = obs_array[step:-step+1, idx+12:idx+13]
    last_q_pos_target = obs_array[step-1:-step, idx+12:idx+13]
    last_last_q_pos_target = obs_array[step-2:-step-1, idx+12:idx+13]
    last_last_last_q_pos_target = obs_array[step-3:-step-2, idx+12:idx+13]
    last_last_last_last_q_pos_target = obs_array[step-4:-step-3, idx+12:idx+13]
    last_last_last_last_last_q_pos_target = obs_array[step-5:-step-4, idx+12:idx+13]
    last_6_q_pos_target = obs_array[step-6:-step-5, idx+12:idx+13]
    last_7_q_pos_target = obs_array[step-7:-step-6, idx+12:idx+13]
    last_8_q_pos_target = obs_array[step-8:-step-7, idx+12:idx+13]

    pos_error = q_pos - q_pos_target
    last_pos_error = last_q_pos - last_q_pos_target
    last_last_pos_error = last_last_q_pos - last_last_q_pos_target
    last_last_last_pos_error = last_last_last_q_pos - last_last_last_q_pos_target
    last_last_last_last_pos_error = last_last_last_last_q_pos - last_last_last_last_q_pos_target
    last_last_last_last_last_pos_error = last_last_last_last_last_q_pos - last_last_last_last_last_q_pos_target
    last_6_pos_error = last_6_q_pos - last_6_q_pos_target
    last_7_pos_error = last_7_q_pos - last_7_q_pos_target
    last_8_pos_error = last_8_q_pos - last_8_q_pos_target

    velocity = obs_array[step:-step+1, idx+24:idx+25] 
    last_velocity = obs_array[step-1:-step, idx+24:idx+25]
    last_last_velocity = obs_array[step-2:-step-1, idx+24:idx+25]
    last_last_last_velocity = obs_array[step-3:-step-2, idx+24:idx+25]
    last_last_last_last_velocity = obs_array[step-4:-step-3, idx+24:idx+25]
    last_last_last_last_last_velocity = obs_array[step-5:-step-4, idx+24:idx+25]
    last_6_velocity = obs_array[step-6:-step-5, idx+24:idx+25]
    last_7_velocity = obs_array[step-7:-step-6, idx+24:idx+25]
    last_8_velocity = obs_array[step-8:-step-7, idx+24:idx+25]


    joint_data = np.concatenate([3*pos_error, 3*last_pos_error, 3*last_last_pos_error, 3*last_last_last_pos_error, 3*last_last_last_last_pos_error, 3*last_last_last_last_last_pos_error, 3*last_6_pos_error, 3*last_7_pos_error, 3*last_8_pos_error,
                                 velocity, last_velocity, last_last_velocity, last_last_last_velocity, last_last_last_last_velocity, last_last_last_last_last_velocity, last_6_velocity, last_7_velocity, last_8_velocity], axis=-1)

    torque = torque_array[step:-step+1, idx:idx+1]

    return joint_data, torque

def get_joint_data(past_observations_t):

    q_pos = past_observations_t.array[..., past_observations_t.index_map['q_fl_hip'][0]:past_observations_t.index_map['q_rr_ankle'][1]]
    q_pos_target = past_observations_t.array[-2, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]
    last_q_pos_target = past_observations_t.array[-3, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]
    last_last_q_pos_target = past_observations_t.array[-4, past_observations_t.index_map['qd_fl_hip'][0]:past_observations_t.index_map['qd_rr_ankle'][1]]

    pos_error = q_pos[-1] - q_pos_target
    last_pos_error = q_pos[-2] - last_q_pos_target
    last_last_pos_error = q_pos[-3] - last_last_q_pos_target

    velocity = past_observations_t.array[..., past_observations_t.index_map['v_fl_hip'][0]:past_observations_t.index_map['v_rr_ankle'][1]]

    joint_data = jnp.stack([
        pos_error, last_pos_error, last_last_pos_error,
        velocity[-1], velocity[-2], velocity[-3],
    ], axis=-1)
    return joint_data


if __name__ == "__main__":

    # log_dir = "./actuator_logs/"
    # actuator_network_path = log_dir + "actuator_network_new"
    # train_actuator_network_and_plot_predictions_jax(log_dir, actuator_network_path, load_pretrained_model=False, my_data=True)

    log_dir = "./actuator_logs/"
    actuator_network_path = log_dir + "actuator_network_200hz"
    train_actuator_network_and_plot_predictions_jax(log_dir, actuator_network_path, load_pretrained_model=True, my_data=True)
