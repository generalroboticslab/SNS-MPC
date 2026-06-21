import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
import ctypes
from Common import MJDataset
import dill

class OnlineTorchDataset(Dataset):
    def __init__(self, mj_dataset=None, buffer_size=None, H=5, H_obs=20, horizon=18, path=None, dtype=jnp.float32):
        super().__init__()

        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.n_traj = 0


        self.H = H
        self.horizon = horizon
        self.H_obs = H_obs
        self.dtype = dtype


        if path is not None:
            self.load(path)
        elif mj_dataset is not None:
            self.traj_length = mj_dataset.sensor_trajectory.array.shape[1]
            self.pts_per_traj = self.traj_length - self.H - self.horizon - 2 - self.H_obs
            self.data_buffer = MJDataset(sensor_trajectory=mj_dataset.sensor_trajectory.array,
                                            action_trajectory=mj_dataset.action_trajectory.array)
            self.data_buffer = jax.tree.map(lambda data: self.allocate_data(data, self.buffer_size, dtype=self.dtype), self.data_buffer)
            self.init_buffer_size = buffer_size // 4
            self.n_overwrites = 0
            self.update(mj_dataset, mode='standard')

    def __len__(self):
        return self.n_traj * self.pts_per_traj

    def __getitem__(self, idx):
        traj_idx = idx // self.pts_per_traj
        slice_idx = idx % self.pts_per_traj
        start_idx = slice_idx
        end_idx = start_idx + self.horizon + self.H + 2 + self.H_obs
        return jax.tree.map(lambda x: x[traj_idx, start_idx:end_idx], self.data_buffer)
        

    def update(self, mj_dataset_arrays, mode="standard"):
        """
        Update the dataset buffer with new data.

        Modes:
        - "overwrite": Overwrite the beginning of the buffer sequentially until self.init_buffer_size is reached.
        - "standard": FIFO logic with wraparound when buffer is full.
        """
        new_traj = mj_dataset_arrays.sensor_trajectory.array
        new_action = mj_dataset_arrays.action_trajectory.array
        n_traj = new_traj.shape[0]
        assert self.dtype == new_traj.dtype, f"Data type mismatch: buffer dtype {self.dtype}, new data dtype {new_traj.dtype}"

        if mode == "overwrite" and self.n_overwrites < self.init_buffer_size:
            # Overwrite the initial buffer linearly
            insert_idx = self.n_overwrites
            end_idx = insert_idx + n_traj

            if end_idx <= self.init_buffer_size:
                self.data_buffer.sensor_trajectory[insert_idx:end_idx] = new_traj
                self.data_buffer.action_trajectory[insert_idx:end_idx] = new_action
                self.n_overwrites += n_traj
            else:
                # Partial overwrite to fill initial buffer, then switch to standard for the rest
                first_slice = self.init_buffer_size - insert_idx
                second_slice = n_traj - first_slice

                self.data_buffer.sensor_trajectory[insert_idx:self.init_buffer_size] = new_traj[:first_slice]
                self.data_buffer.sensor_trajectory[:second_slice] = new_traj[first_slice:]

                self.data_buffer.action_trajectory[insert_idx:self.init_buffer_size] = new_action[:first_slice]
                self.data_buffer.action_trajectory[:second_slice] = new_action[first_slice:]

                self.n_overwrites = self.init_buffer_size  # overwrite phase is done

                # Update standard counters after wrap
                self.n_traj = max(self.n_traj, self.init_buffer_size + second_slice)
                self.buffer_idx = second_slice % self.buffer_size

        else:
            # Standard FIFO insert
            insert_idx = self.n_traj if self.n_traj < self.buffer_size else self.buffer_idx
            end_idx = insert_idx + n_traj

            if end_idx <= self.buffer_size:
                # No wraparound
                self.data_buffer.sensor_trajectory[insert_idx:end_idx] = new_traj
                self.data_buffer.action_trajectory[insert_idx:end_idx] = new_action
            else:
                # Wraparound
                first_slice = self.buffer_size - insert_idx
                second_slice = n_traj - first_slice

                self.data_buffer.sensor_trajectory[insert_idx:] = new_traj[:first_slice]
                self.data_buffer.sensor_trajectory[:second_slice] = new_traj[first_slice:]

                self.data_buffer.action_trajectory[insert_idx:] = new_action[:first_slice]
                self.data_buffer.action_trajectory[:second_slice] = new_action[first_slice:]

            # Update standard counters
            if self.n_traj < self.buffer_size:
                self.n_traj = min(self.buffer_size, self.n_traj + n_traj)
            else:
                self.buffer_idx = (self.buffer_idx + n_traj) % self.buffer_size


    
    def allocate_data(self, data, n_data_pts, dtype=jnp.float32):
            shape = (n_data_pts, ) + data.shape[1:]
            data_all = self.allocate_array(shape, dtype=dtype)
            return data_all

    def allocate_array(self, shape, dtype=jnp.float32):
        """
        Allocates a contiguous block of memory for a NumPy array with the given shape and dtype .
        """
        # Compute the total number of elements in the array
        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if dtype == jnp.float32:
            ArrayType = ctypes.c_float * total_elements
            data = ArrayType()
            # Create a NumPy array from the allocated memory and view it as float64
            size_in_gb = total_elements * np.dtype(np.float32).itemsize / (1024 ** 3)
            print(f"Allocated {size_in_gb:.2f} GB of memory for the array.")
            np_array = np.frombuffer(data, dtype=np.float32).reshape(shape)

        elif dtype == jnp.float64:
            ArrayType = ctypes.c_double * total_elements
            data = ArrayType()

            # Create a NumPy array from the allocated memory and view it as float64
            size_in_gb = total_elements * np.dtype(np.float64).itemsize / (1024 ** 3)
            print(f"Allocated {size_in_gb:.2f} GB of memory for the array.")
            np_array = np.frombuffer(data, dtype=np.float64).reshape(shape)

        return np_array
    

    def save(self, path):
        """
        Save the dataset to a file.
        """
        data = {
            'sensor_trajectory': self.data_buffer.sensor_trajectory,
            'action_trajectory': self.data_buffer.action_trajectory,
            "buffer_size": self.buffer_size,
            "H": self.H,
            "horizon": self.horizon,
            "n_traj": self.n_traj,
            "traj_length": self.traj_length,
            "H_obs": self.H_obs,
            "pts_per_traj": self.pts_per_traj,
            "buffer_idx": self.buffer_idx
        }

        with open(path, 'wb') as f:
            dill.dump(data, f)

    def load(self, path):
        """
        Load the dataset from a file.
        """
        with open(path, 'rb') as f:
            data = dill.load(f)
        
        self.data_buffer = MJDataset(sensor_trajectory=data['sensor_trajectory'],
                                    action_trajectory=data['action_trajectory'])

        self.buffer_size = data['buffer_size']
        self.H = data['H']
        self.horizon = data['horizon']
        self.n_traj = data['n_traj']
        self.traj_length = data['traj_length']
        self.H_obs = data['H_obs']
        self.pts_per_traj = data['pts_per_traj']
        self.buffer_idx = data['buffer_idx']
        self.init_buffer_size = self.buffer_size // 2
        self.n_overwrites = 0



def jax_collate(batch, dtype=np.float32):
  batch = jax.tree.map(lambda *arrays: np.stack([*arrays], axis=0), *batch)
  batch = jax.tree_map(lambda x: x.astype(dtype), batch)
  return batch

class JaxDataLoader(DataLoader):
    
    def __init__(self, dataset, batch_size=1, shuffle=False, dtype=np.float32,
                 collate_fn=jax_collate,
                 **kwargs):
            
        super().__init__(dataset, batch_size, shuffle=shuffle,
                         collate_fn=lambda x: collate_fn(x, dtype=dtype),
                         **kwargs)
