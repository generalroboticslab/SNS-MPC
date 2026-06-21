
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from Common import Contiguous, States, Actions, Observations, NNData
from abc import ABC, abstractmethod

@register_pytree_node_class
class DataWrangler(ABC):
    """
    A class for processing and managing data for the training and deployment pipeline.
    """
    def __init__(self, horizon, T, H, H_Obs, n_blocks, mj_dataset_cls, states=None, actions=None, observations=None):

        self.horizon = horizon
        self.T = T
        self.H = H
        self.H_Obs = H_Obs
        self.n_blocks = n_blocks
        self.mj_dataset_cls = mj_dataset_cls

        if states is not None and actions is not None:
            self.states = states
            self.actions = actions
            self.observations = observations
            self.nn_data = NNData(horizon=self.horizon, T=self.T, H=self.H, H_Obs=self.H_Obs, 
                                    state_trajectory=states, action_trajectory=actions, obs_trajectory=observations)
        
    def initialize(self, mj_dataset, robust=True):
        """
        Initialize the DataWrangler from a MJDataset object.
        """

        full_states_struct, full_actions_struct, full_observations_struct = self._process_data_training(mj_dataset)

        full_states_contiguous = Contiguous.from_struct(full_states_struct)
        full_actions_contiguous = Contiguous.from_struct(full_actions_struct)
        full_observations_contiguous = Contiguous.from_struct(full_observations_struct)

        # full_states = States.from_data(full_states_contiguous)
        # full_actions = Actions.from_data(full_actions_contiguous)
        # full_observations = Observations.from_data(full_observations_contiguous)

        if robust:
            full_states = States.from_data_robust(full_states_contiguous)
            full_actions = Actions.from_data_robust(full_actions_contiguous)
            full_observations = Observations.from_data_robust(full_observations_contiguous)
        else:
            full_states = States.from_data(full_states_contiguous)
            full_actions = Actions.from_data(full_actions_contiguous)
            full_observations = Observations.from_data(full_observations_contiguous)
        #print(f"States : {self.states.scales.array}, Actions : {self.actions.scales.array}, Observations : {self.observations.scales.array}")
        # print(f"States : {full_states.centers.array}, Actions : {full_actions.centers.array}, Observations : {full_observations.centers.array}")
        print(f"State scales: {full_states.scales.array}, Action scales: {full_actions.scales.array}, Observation scales: {full_observations.scales.array}")

        self.states = jax.tree.map(lambda x: x, full_states)
        self.actions = jax.tree.map(lambda x: x, full_actions)
        self.observations = jax.tree.map(lambda x: x, full_observations)


        self.nn_data = NNData(horizon=self.horizon, T=self.T, H=self.H, H_Obs=self.H_Obs,
                               state_trajectory=full_states, action_trajectory=full_actions, obs_trajectory=full_observations)

    def process_data_training(self, mj_dataset):

        full_states_struct, full_actions_struct, full_observations_struct = self._process_data_training(mj_dataset)

        self.nn_data.state_trajectory.update_array_from_struct(full_states_struct)
        self.nn_data.action_trajectory.update_array_from_struct(full_actions_struct)
        self.nn_data.obs_trajectory.update_array_from_struct(full_observations_struct)

        return self.nn_data
    

    def _process_data_training(self, mj_dataset):

        full_states = self.get_states(mj_dataset)
        full_actions = self.get_actions(mj_dataset)
        full_observations = self.get_observations(mj_dataset)
        return full_states, full_actions, full_observations
    

    
    def tree_flatten(self):
        """Flatten the object for JAX."""
        dynamic_children = (self.nn_data, )
        static_aux = (self.T, self.H, self.H_Obs, self.n_blocks, self.states, self.actions, self.observations, self.horizon, self.mj_dataset_cls)
        return dynamic_children, static_aux

    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        """Unflatten the object for JAX."""

        # print(f"dynamic_children: {dynamic_children}")
        # print(f"****************************************************************")
        # print(f"static_aux_data: {static_aux_data}")
        nn_data =  dynamic_children[0]
        T, H, H_Obs, n_blocks, states, actions, observations, horizon, mj_dataset_cls = static_aux_data

        obj = cls.__new__(cls)

        object.__setattr__(obj, "nn_data", nn_data)
        object.__setattr__(obj, "T", T)
        object.__setattr__(obj, "H", H)
        object.__setattr__(obj, "H_Obs", H_Obs)
        object.__setattr__(obj, "n_blocks", n_blocks)
        object.__setattr__(obj, "states", states)
        object.__setattr__(obj, "actions", actions)
        object.__setattr__(obj, "observations", observations)
        object.__setattr__(obj, "horizon", horizon)
        object.__setattr__(obj, "mj_dataset_cls", mj_dataset_cls)

        return obj
    
    @abstractmethod
    def get_states(self, mj_dataset):
        pass

    @abstractmethod
    def get_actions(self, mj_dataset):
        pass

    @abstractmethod
    def get_observations(self, mj_dataset):
        pass

