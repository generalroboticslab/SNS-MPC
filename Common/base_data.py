import jax.numpy as jnp
import jax
import dill
from jax.tree_util import register_pytree_node_class, register_dataclass
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import fields

@register_dataclass
@dataclass
class CostWeights:
    cost_weights: jnp.ndarray
    constraint_weights: jnp.ndarray
    constraint_relaxation: jnp.ndarray
    global_cost_weights: jnp.ndarray

def flatten_struct(struct):
    flat_dict = {
        f.name: getattr(struct, f.name)
        for f in fields(struct)
        if not f.name.startswith("_")
    }

    # Include additional properties or dynamically added attributes
    for name in dir(struct):
        if name.startswith("_") or name in flat_dict:
            continue
        try:
            attr = getattr(struct, name)
        except AttributeError:
            continue
        if isinstance(attr, jnp.ndarray):
            flat_dict[name] = attr

    return flat_dict


@register_pytree_node_class
class Contiguous(ABC):
    def __init__(self, array: jnp.ndarray, index_map: Dict[str, Tuple[int, int]], name: str = None):
        self.array = array
        self.index_map = index_map
        self.name = name if name else self.__class__.__name__

    def update_array_from_struct(self, struct):
        """Update the array with the dataclass instance."""
        dict_repr = flatten_struct(struct)
        all_vals = list(dict_repr.values()) 
        new_array = jnp.concatenate(all_vals, axis=-1)
        self.array = new_array

    @classmethod
    def from_struct(cls, struct):
        """Create a Contiguous instance from a dataclass instance."""
        dict_repr = flatten_struct(struct)
        name = struct.__class__.__name__
        all_keys = list(dict_repr.keys())
        all_vals = list(dict_repr.values()) 

        starts = jnp.cumsum(jnp.array([0] + [v.shape[-1] for v in all_vals[:-1]]))

        index_map = {
            k: (s, s + v.shape[-1])  # don't wrap in int()
            for k, v, s in zip(all_keys, all_vals, starts)
        }


        array = jnp.concatenate(all_vals, axis=-1)
        return cls(array, index_map, name)

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map})"
    
    def __getattr__(self, name):
        if name in self.index_map:
            i, j = self.index_map[name]
            return self.array[..., i:j]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in ("array", "index_map"):
            object.__setattr__(self, name, value)
        elif name in self.index_map:
            i, j = self.index_map[name]
            self.array = self.array.at[..., i:j].set(value)
        else:
            object.__setattr__(self, name, value)
    

    def flatten_dynamic_aux(self) -> Tuple[Any, ...]:
        """Return additional dynamic attributes to be flattened."""
        return ()

    def flatten_static_aux(self) -> Tuple[Any, ...]:
        """Return additional static attributes to be flattened."""
        return ()

    def unflatten_dynamic_aux(self, aux: Tuple[Any, ...]):
        """Unflatten additional dynamic attributes."""
        pass

    def unflatten_static_aux(self, aux: Tuple[Any, ...]):
        """Unflatten additional static attributes."""
        pass

    
    def tree_flatten(self):
        """Flatten the object for JAX."""
        dynamic_children = (self.array,) + tuple(self.flatten_dynamic_aux())
        static_aux = (self.index_map,) + tuple(self.flatten_static_aux())
        return dynamic_children, static_aux

    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        """Unflatten the object for JAX."""
        index_map, *static_extra = static_aux_data
        array, *dynamic_extra = dynamic_children

        obj = cls.__new__(cls)

        object.__setattr__(obj, "array", array)
        object.__setattr__(obj, "index_map", index_map)
        object.__setattr__(obj, "name", cls.__name__)

        # Let subclass fill in the rest
        obj.unflatten_dynamic_aux(tuple(dynamic_extra))
        obj.unflatten_static_aux(tuple(static_extra))
        return obj
    


@register_pytree_node_class
class MJData(ABC):
    def __init__(self, array: jnp.ndarray, index_map: Dict[str, Tuple[int, int]], name: str = None):
        self.index_map = index_map
        self.array = array
        self.name = name if name else self.__class__.__name__

    def __getattr__(self, name):
        if name in self.index_map:
            i, j = self.index_map[name]
            return self.array[..., i:j]
        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in ("array", "index_map"):
            object.__setattr__(self, name, value)
        elif "index_map" in self.__dict__ and name in self.index_map:
            i, j = self.index_map[name]
            self.array = self.array.at[..., i:j].set(value)
        else:
            object.__setattr__(self, name, value)

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map})"
    
    def flatten_dynamic_aux(self) -> Tuple[Any, ...]:
        """Return additional dynamic attributes to be flattened."""
        return ()
    
    def flatten_static_aux(self) -> Tuple[Any, ...]:
        """Return additional static attributes to be flattened."""
        return ()
    
    def unflatten_dynamic_aux(self, aux: Tuple[Any, ...]):
        """Unflatten additional dynamic attributes."""
        pass

    def unflatten_static_aux(self, aux: Tuple[Any, ...]):
        """Unflatten additional static attributes."""
        pass


    def tree_flatten(self):
        """Flatten the object for JAX."""
        dynamic_children = (self.array,) + tuple(self.flatten_dynamic_aux())
        static_aux = (self.index_map,) + tuple(self.flatten_static_aux())
        return dynamic_children, static_aux

    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        """Unflatten the object for JAX."""
        index_map, *static_extra = static_aux_data
        array, *dynamic_extra = dynamic_children

        obj = cls.__new__(cls)

        object.__setattr__(obj, "array", array)
        object.__setattr__(obj, "index_map", index_map)
        object.__setattr__(obj, "name", cls.__name__)

        # Let subclass fill in the rest
        obj.unflatten_dynamic_aux(tuple(dynamic_extra))
        obj.unflatten_static_aux(tuple(static_extra))
        return obj
    
    @classmethod
    @abstractmethod
    def from_mjmodel(cls):
        """Create an instance from a Mujoco model."""
        pass


@register_dataclass
@dataclass
class MJDataset:
    action_trajectory: MJData
    sensor_trajectory: MJData

def median_absolute_deviation(x, axis=None, keepdims=False):
    median = jnp.median(x, axis=axis, keepdims=True)
    abs_dev = jnp.abs(x - median)
    mad = jnp.median(abs_dev, axis=axis, keepdims=keepdims)
    return mad

@register_pytree_node_class
class NormalizedContiguous(Contiguous):
    def __init__(self, array: jnp.ndarray, index_map: Dict[str, Tuple[int, int]], name: str = None):
        super().__init__(array, index_map, name)

    @abstractmethod
    def normalized(self):
        pass

    @abstractmethod
    def denormalized(self):
        pass

def get_states_centers_and_scales(states: Contiguous):
    shape = states.array.shape
    states = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), states)   
    centers = jax.tree.map(lambda x: jnp.median(x, axis=0, keepdims=True), states)

    centers_q = centers.array[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]]
    centers_q = centers_q.at[..., ::3].set(0.0)
    centers_q = centers_q.at[..., 1::3].set(0.9)
    centers_q = centers_q.at[..., 2::3].set(-1.8)
    centers.array = centers.array.at[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v11'][0]:states.index_map['v23'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]].set(0.1)
    centers.array = centers.array.at[..., states.index_map['z'][0]:states.index_map['z'][1]].set(0.1)

    #print(f"States centers after: {centers.array}")

    scales = jax.tree.map(lambda x: median_absolute_deviation(x, axis=0, keepdims=True), states)
    scales_q = jnp.max(scales.array[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]])
    scales_v = jnp.max(scales.array[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]])
    scales_v_base = jnp.max(scales.array[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]])
    scales_ang = jnp.max(scales.array[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]])
    scales_6D = jnp.max(scales.array[..., states.index_map['v11'][0]:states.index_map['v23'][1]])
    scales_sdf = jnp.max(scales.array[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]])
    scales_z = jnp.max(scales.array[..., states.index_map['z'][0]:states.index_map['z'][1]])
    scales.array = scales.array.at[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]].set(scales_v)
    scales.array = scales.array.at[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]].set(scales_v_base)
    scales.array = scales.array.at[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]].set(scales_ang)
    scales.array = scales.array.at[..., states.index_map['v11'][0]:states.index_map['v23'][1]].set(scales_6D)
    scales.array = scales.array.at[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]].set(scales_sdf)
    scales.array = scales.array.at[..., states.index_map['z'][0]:states.index_map['z'][1]].set(scales_z)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
    states = jax.tree.map(lambda x: x.reshape(shape), states)
    return centers, scales


def get_states_centers_and_scales_gauss(states: Contiguous):
    shape = states.array.shape
    states = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), states)   
    centers = jax.tree.map(lambda x: jnp.mean(x, axis=0, keepdims=True), states)

    centers_q = centers.array[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]]
    centers_q = centers_q.at[..., ::3].set(0.0)
    centers_q = centers_q.at[..., 1::3].set(0.9)
    centers_q = centers_q.at[..., 2::3].set(-1.8)
    centers.array = centers.array.at[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['v11'][0]:states.index_map['v23'][1]].set(0.0)
    centers.array = centers.array.at[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]].set(0.1)
    centers.array = centers.array.at[..., states.index_map['z'][0]:states.index_map['z'][1]].set(0.1)

    #print(f"States centers after: {centers.array}")

    scales = jax.tree.map(lambda x: jnp.std(x, axis=0, keepdims=True), states)
    scales_q = jnp.max(scales.array[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]])
    scales_v = jnp.max(scales.array[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]])
    scales_v_base = jnp.max(scales.array[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]])
    scales_ang = jnp.max(scales.array[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]])
    scales_6D = jnp.max(scales.array[..., states.index_map['v11'][0]:states.index_map['v23'][1]])
    scales_sdf = jnp.max(scales.array[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]])
    scales_z = jnp.max(scales.array[..., states.index_map['z'][0]:states.index_map['z'][1]])
    scales.array = scales.array.at[..., states.index_map['q_fl_hip'][0]:states.index_map['q_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., states.index_map['v_fl_hip'][0]:states.index_map['v_rr_ankle'][1]].set(scales_v)
    scales.array = scales.array.at[..., states.index_map['v_x'][0]:states.index_map['v_z'][1]].set(scales_v_base)
    scales.array = scales.array.at[..., states.index_map['v_roll'][0]:states.index_map['v_yaw'][1]].set(scales_ang)
    scales.array = scales.array.at[..., states.index_map['v11'][0]:states.index_map['v23'][1]].set(scales_6D)
    scales.array = scales.array.at[..., states.index_map['sdf_base_main'][0]:states.index_map['sdf_rr_hip'][1]].set(scales_sdf)
    scales.array = scales.array.at[..., states.index_map['z'][0]:states.index_map['z'][1]].set(scales_z)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
    states = jax.tree.map(lambda x: x.reshape(shape), states)
    return centers, scales


@register_pytree_node_class
class States(NormalizedContiguous):
    """This class stores the raw observation data for the go2 robot."""

    def __init__(self, states: Contiguous, centers: Contiguous, scales: Contiguous):
        super().__init__(states.array, states.index_map, self.__class__.__name__)
        self.centers = centers
        self.scales = scales

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map}, centers={self.centers}, scales={self.scales})"

    def flatten_static_aux(self):
        return (self.centers, self.scales)

    def unflatten_static_aux(self, aux):
        self.centers, self.scales = aux

    def normalized(self):
        return jax.tree.map(lambda x: (x - self.centers.array) / self.scales.array, self)

    def denormalized(self):
        return jax.tree.map(lambda x: x*self.scales.array + self.centers.array, self)

    @classmethod
    def from_data(cls, states: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        shape = states.array.shape

        states = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), states)   
        centers = jax.tree.map(lambda x: jnp.mean(x, axis=0, keepdims=True), states)
        scales = jax.tree.map(lambda x: jnp.std(x, axis=0, keepdims=True), states)
        scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
        states = jax.tree.map(lambda x: x.reshape(shape), states)
        return cls(states, centers, scales)

    @classmethod
    def from_data_robust(cls, states: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        shape = states.array.shape
        # print(f"States index map: {states.index_map}")
        # states = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), states)   
        # centers = jax.tree.map(lambda x: jnp.median(x, axis=0, keepdims=True), states)
        # scales = jax.tree.map(lambda x: median_absolute_deviation(x, axis=0, keepdims=True), states)
        # scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
        # states = jax.tree.map(lambda x: x.reshape(shape), states)
        centers, scales = get_states_centers_and_scales(states)
        return cls(states, centers, scales)
    
    def save(self, path):
        """Save the object to a file."""
        with open(path, "wb") as f:
            params = {"array": self.array, "index_map": self.index_map, "centers_array": self.centers.array, "scales_array": self.scales.array}
            dill.dump(params, f)

    @classmethod
    def load(cls, path):
        """Load the object from a file."""
        with open(path, "rb") as f:
            params = dill.load(f)
            array = params["array"]
            index_map = params["index_map"]
            centers_array = params["centers_array"]
            scales_array = params["scales_array"]
            obs = Contiguous(array, index_map)
            centers = Contiguous(centers_array, index_map)
            scales = Contiguous(scales_array, index_map)
            return cls(obs, centers, scales)




@register_pytree_node_class
class Actions(NormalizedContiguous):
    """This class stores the raw observation data for the go2 robot."""

    def __init__(self, actions: Contiguous, centers: Contiguous, scales: Contiguous):
        super().__init__(actions.array, actions.index_map, self.__class__.__name__)
        self.centers = centers
        self.scales = scales

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map}, centers={self.centers}, scales={self.scales})"

    def flatten_static_aux(self):
        return (self.centers, self.scales)

    def unflatten_static_aux(self, aux):
        self.centers, self.scales = aux

    def normalized(self):
        return jax.tree.map(lambda x: (x - self.centers.array) / self.scales.array, self)

    def denormalized(self):
        return jax.tree.map(lambda x: x*self.scales.array + self.centers.array, self)

    @classmethod
    def from_data(cls, actions: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        shape = actions.array.shape
        actions = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), actions)   
        centers = jax.tree.map(lambda x: jnp.mean(x, axis=0, keepdims=True), actions)
        # centered at 0, 0.9, -1.8
        centers = jax.tree.map(lambda x: jnp.zeros_like(x), centers)
        centers.array = centers.array.at[..., ::3].set(0.0)
        centers.array = centers.array.at[..., 1::3].set(0.9)
        centers.array = centers.array.at[..., 2::3].set(-1.8)

        scales = jax.tree.map(lambda x: jnp.std(x, axis=0, keepdims=True), actions)
        # min 
        max_scale = jnp.max(scales.array)
        scales = jax.tree.map(lambda x: jnp.ones_like(x)*max_scale, scales)
        scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
        actions = jax.tree.map(lambda x: x.reshape(shape), actions)
        return cls(actions, centers, scales)
    
    @classmethod
    def from_data_robust(cls, actions: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        shape = actions.array.shape
        actions = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), actions)   
        centers = jax.tree.map(lambda x: jnp.median(x, axis=0, keepdims=True), actions)
        # centered at 0, 0.9, -1.8
        centers = jax.tree.map(lambda x: jnp.zeros_like(x), centers)
        centers.array = centers.array.at[..., ::3].set(0.0)
        centers.array = centers.array.at[..., 1::3].set(0.9)
        centers.array = centers.array.at[..., 2::3].set(-1.8)

        scales = jax.tree.map(lambda x: median_absolute_deviation(x, axis=0, keepdims=True), actions)
        # min 
        max_scale = jnp.max(scales.array)
        scales = jax.tree.map(lambda x: jnp.ones_like(x)*max_scale, scales)
        scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
        actions = jax.tree.map(lambda x: x.reshape(shape), actions)
        return cls(actions, centers, scales)
    
    def save(self, path):
        """Save the object to a file."""
        with open(path, "wb") as f:
            params = {"array": self.array, "index_map": self.index_map, "centers_array": self.centers.array, "scales_array": self.scales.array}
            dill.dump(params, f)

    @classmethod
    def load(cls, path):
        """Load the object from a file."""
        with open(path, "rb") as f:
            params = dill.load(f)
            array = params["array"]
            index_map = params["index_map"]
            centers_array = params["centers_array"]
            scales_array = params["scales_array"]
            obs = Contiguous(array, index_map)
            centers = Contiguous(centers_array, index_map)
            scales = Contiguous(scales_array, index_map)
            return cls(obs, centers, scales)


def get_obs_centers_and_scales(obs: Contiguous):
    shape = obs.array.shape
    obs = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), obs)   
    centers = jax.tree.map(lambda x: jnp.median(x, axis=0, keepdims=True), obs)
    centers.array = centers.array.at[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]].set(0.0)
    centers_q = centers.array[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]]
    centers_q = centers_q.at[..., ::3].set(0.0)
    centers_q = centers_q.at[..., 1::3].set(0.9)
    centers_q = centers_q.at[..., 2::3].set(-1.8)
    centers.array = centers.array.at[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., obs.index_map['qd_fl_hip'][0]:obs.index_map['qd_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]].set(0.0)
    centers.array = centers.array.at[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]].set(0.0)
    centers.array = centers.array.at[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]].set(0.0)

    #print(f"Obs centers: {centers.array}")
    scales = jax.tree.map(lambda x: median_absolute_deviation(x, axis=0, keepdims=True), obs)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)

    scales_q = jnp.max(scales.array[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]])
    scales_v = jnp.max(scales.array[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]])
    scales_a = jnp.max(scales.array[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]])
    scales_6D = jnp.max(scales.array[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]])
    scales_ang = jnp.max(scales.array[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]])
    scales.array = scales.array.at[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., obs.index_map['qd_fl_hip'][0]:obs.index_map['qd_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]].set(scales_v)
    scales.array = scales.array.at[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]].set(scales_a)
    scales.array = scales.array.at[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]].set(scales_6D)
    scales.array = scales.array.at[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]].set(scales_ang)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
    #print(f"Obs scales: {scales.array}")
    obs = jax.tree.map(lambda x: x.reshape(shape), obs)
    return centers, scales


def get_obs_centers_and_scales_gauss(obs: Contiguous):
    shape = obs.array.shape
    obs = jax.tree.map(lambda x: x.reshape(-1, x.shape[-1]), obs)
    centers = jax.tree.map(lambda x: jnp.mean(x, axis=0, keepdims=True), obs)
    centers.array = centers.array.at[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]].set(0.0)
    centers_q = centers.array[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]]
    centers_q = centers_q.at[..., ::3].set(0.0)
    centers_q = centers_q.at[..., 1::3].set(0.9)
    centers_q = centers_q.at[..., 2::3].set(-1.8)
    centers.array = centers.array.at[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., obs.index_map['qd_fl_hip'][0]:obs.index_map['qd_rr_ankle'][1]].set(centers_q)
    centers.array = centers.array.at[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]].set(0.0)
    centers.array = centers.array.at[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]].set(0.0)
    centers.array = centers.array.at[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]].set(0.0)

    #print(f"Obs centers: {centers.array}")
    scales = jax.tree.map(lambda x: jnp.std(x, axis=0, keepdims=True), obs)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)

    scales_q = jnp.max(scales.array[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]])
    scales_v = jnp.max(scales.array[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]])
    scales_a = jnp.max(scales.array[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]])
    scales_6D = jnp.max(scales.array[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]])
    scales_ang = jnp.max(scales.array[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]])
    scales.array = scales.array.at[..., obs.index_map['q_fl_hip'][0]:obs.index_map['q_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., obs.index_map['qd_fl_hip'][0]:obs.index_map['qd_rr_ankle'][1]].set(scales_q)
    scales.array = scales.array.at[..., obs.index_map['v_fl_hip'][0]:obs.index_map['v_rr_ankle'][1]].set(scales_v)
    scales.array = scales.array.at[..., obs.index_map['a_x'][0]:obs.index_map['a_z'][1]].set(scales_a)
    scales.array = scales.array.at[..., obs.index_map['v11'][0]:obs.index_map['v23'][1]].set(scales_6D)
    scales.array = scales.array.at[..., obs.index_map['v_roll'][0]:obs.index_map['v_yaw'][1]].set(scales_ang)
    scales = jax.tree.map(lambda x: jnp.clip(x, 1e-5, None), scales)
    #print(f"Obs scales: {scales.array}")
    obs = jax.tree.map(lambda x: x.reshape(shape), obs)
    return centers, scales



@register_pytree_node_class
class Observations(NormalizedContiguous):
    """This class stores the raw observation data for the go2 robot."""

    def __init__(self, observations: Contiguous, centers: Contiguous, scales: Contiguous):
        super().__init__(observations.array, observations.index_map, self.__class__.__name__)
        self.centers = centers
        self.scales = scales

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map}, centers={self.centers}, scales={self.scales})"

    def flatten_static_aux(self):
        return (self.centers, self.scales)

    def unflatten_static_aux(self, aux):
        self.centers, self.scales = aux

    def normalized(self):
        return jax.tree.map(lambda x: (x - self.centers.array) / self.scales.array, self)

    def denormalized(self):
        return jax.tree.map(lambda x: x*self.scales.array + self.centers.array, self)

    @classmethod
    def from_data(cls, obs: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        centers, scales = get_obs_centers_and_scales_gauss(obs)
        return cls(obs, centers, scales)

    @classmethod
    def from_data_robust(cls, obs: Contiguous):
        """Create a Stattes object from a Contiguous object."""
        centers, scales = get_obs_centers_and_scales(obs)
        return cls(obs, centers, scales)
    
    def save(self, path):
        """Save the object to a file."""
        with open(path, "wb") as f:
            params = {"array": self.array, "index_map": self.index_map, "centers_array": self.centers.array, "scales_array": self.scales.array}
            dill.dump(params, f)

    @classmethod
    def load(cls, path):
        """Load the object from a file."""
        with open(path, "rb") as f:
            params = dill.load(f)
            array = params["array"]
            index_map = params["index_map"]
            centers_array = params["centers_array"]
            scales_array = params["scales_array"]
            obs = Contiguous(array, index_map)
            centers = Contiguous(centers_array, index_map)
            scales = Contiguous(scales_array, index_map)
            return cls(obs, centers, scales)


@register_pytree_node_class
class LiftedStates:
    """This class stores the lifted state representation for the robot.
    It combines the states and actions into a single array for use in neural networks.

    You cannot pass a LiftedStates instance as an argument to a jitted function because the tree defs are stored in the object.
    However, you can use it within a jitted function."""

    def __init__(self, states: States, actions: Actions, T: int, H: int):

        array = jnp.concatenate([states.array, actions.array], axis=-1)
        self.n_states = states.array.shape[-1]
        actions_index_map = jax.tree.map(lambda x: x + self.n_states, actions.index_map)
        index_map = {**states.index_map, **actions_index_map}

        self.index_map = index_map
        self.array = array
        self.name = self.__class__.__name__
        self.T = T
        self.H = H

        self.centers_array = jnp.concatenate([states.centers.array, actions.centers.array], axis=-1)
        self.scales_array = jnp.concatenate([states.scales.array, actions.scales.array], axis=-1)

        _ ,self.states_tree_def = jax.tree.flatten(states)
        _ ,self.actions_tree_def = jax.tree.flatten(actions)

    def tree_flatten(self):
        """Flatten the object for JAX."""
        dynamic_children = (self.array,)
        static_aux = (self.index_map, self.centers_array, self.scales_array, self.n_states, self.states_tree_def, self.actions_tree_def, self.T, self.H)
        return dynamic_children, static_aux
    
    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        """Unflatten the object for JAX."""
        index_map, centers_array, scales_array, n_states, states_tree_def, actions_tree_def, T, H = static_aux_data
        array, = dynamic_children

        obj = cls.__new__(cls)

        object.__setattr__(obj, "array", array)
        object.__setattr__(obj, "index_map", index_map)
        object.__setattr__(obj, "name", cls.__name__)
        object.__setattr__(obj, "centers_array", centers_array)
        object.__setattr__(obj, "scales_array", scales_array)
        object.__setattr__(obj, "n_states", n_states)
        object.__setattr__(obj, "states_tree_def", states_tree_def)
        object.__setattr__(obj, "actions_tree_def", actions_tree_def)
        object.__setattr__(obj, "T", T)
        object.__setattr__(obj, "H", H)

        return obj

    def normalized(self):
        return jax.tree.map(lambda x: (x - self.centers_array) / self.scales_array, self)

    def denormalized(self):
        return jax.tree.map(lambda x: x*self.scales_array + self.centers_array, self)
    
    def update_array_from_structs(self, states, actions):
        """Update the array with the dataclass instance."""
        states_dict_repr = flatten_struct(states)
        actions_dict_repr = flatten_struct(actions)
        all_vals = list(states_dict_repr.values()) + list(actions_dict_repr.values())
        new_array = jnp.concatenate(all_vals, axis=-1)
        self.array = new_array

    @property
    def states(self):
        return jax.tree.unflatten(self.states_tree_def, (self.array[..., :self.n_states], ))

    @property
    def actions(self):
        return jax.tree.unflatten(self.actions_tree_def, (self.array[..., self.n_states:], ))
    
    @property
    def pred_states(self):
        return jax.tree.unflatten(self.states_tree_def, (self.array[..., -self.T:, :self.n_states], ))

    def __repr__(self):
        return f"{self.name}(array={self.array}, index_map={self.index_map}, centers={self.centers_array}, scales={self.scales_array})"


@register_pytree_node_class
class NNData:
    """
    50 Hz version.
    All lengths (horizon, H, T, H_Obs) are specified in 50 Hz ticks.
    """
    horizon: int
    T: int
    H: int
    H_Obs: int
    state_trajectory: Optional[States] = None
    action_trajectory: Optional[Actions] = None
    obs_trajectory: Optional[Observations] = None

    def __init__(
        self,
        horizon: int,
        T: int,
        H: int,
        H_Obs: int,
        state_trajectory: Optional[States] = None,
        action_trajectory: Optional[Actions] = None,
        obs_trajectory: Optional[Observations] = None,
        discount: float = 0.8,
        seed: int = 0,
    ):
        self.horizon = horizon
        self.T = T
        self.H = H
        self.H_Obs = H_Obs
        self.state_trajectory = state_trajectory
        self.action_trajectory = action_trajectory
        self.obs_trajectory = obs_trajectory
        self.key = jax.random.PRNGKey(seed)

        assert self.H_Obs >= self.H, f"H_Obs {self.H_Obs} must be >= H {self.H}"

        slice_map = self.init_slice_map()
        object.__setattr__(self, "slice_map", jax.tree.map(lambda x: x, slice_map))
        self.total_steps = (self.horizon - 1) // 1  # 50 Hz steps
        self.discount = discount

    def init_slice_map(self):
        """
        Build index map for lifted states/actions/obs blocks at 50 Hz.
        """
        slice_map = {}
        past_state_starts = self._past_state_starts
        past_action_starts = self._past_action_starts
        future_state_starts = self._future_state_starts
        future_action_starts = self._future_action_starts
        past_obs_starts = self._past_obs_starts

        for block in range(self.n_blocks):
            past_states = f"past_states_block{block}"
            past_actions = f"past_actions_block{block}"
            past_obs = f"past_obs_block{block}"
            future_states = f"future_states_block{block}"
            future_actions = f"future_actions_block{block}"
            slice_map[past_obs] = (past_obs_starts[block], self.H_Obs)
            slice_map[past_states] = (past_state_starts[block], self.H)
            slice_map[past_actions] = (past_action_starts[block], self.H)
            slice_map[future_states] = (future_state_starts[block], self.T)
            slice_map[future_actions] = (future_action_starts[block], self.T)

        return slice_map

    # ------------------------- index schedules (50 Hz) -------------------------


    @property
    def _past_obs_starts(self):
        # start-of-block positions, step by T
        return jnp.arange(0, self.horizon + self.H, step=self.T)[:-1] + self.H + 1

    @property
    def _future_state_starts(self):
        # future window starts right after past window; obs occupies H_Obs before past
        return jnp.arange(self.H, self.horizon + self.H, step=self.T)[:-1] + self.H_Obs + 1

    @property
    def _future_action_starts(self):
        return jnp.arange(self.H, self.horizon + self.H, step=self.T)[:-1] + self.H_Obs 

    @property
    def _past_state_starts(self):
        return jnp.arange(0, self.horizon + self.H, step=self.T)[:-1] + self.H_Obs + 1

    @property
    def _past_action_starts(self):
        return jnp.arange(0, self.horizon + self.H, step=self.T)[:-1] + self.H_Obs 

    @property
    def n_blocks(self):
        # one block per stride of T across horizon (all in 50 Hz units now)
        return (self.horizon // self.T)


    # ------------------------- slicing helpers (50 Hz) -------------------------

    def _segment(self, idx, array):
        # window covers [past-obs | past | future] with no extra padding now
        start = idx - self.H - self.H_Obs
        length = self.horizon + self.H + self.H_Obs
        return jax.lax.dynamic_slice_in_dim(array, start, length, axis=1)


    # ------------------------- public accessors (50 Hz) ------------------------

    def past_states(self, idx):
        segment = jax.tree.map(lambda x: self._segment(idx, x), self.state_trajectory)
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, self.slice_map["past_states_block0"][0], self.slice_map["past_states_block0"][1], axis=1
            ),
            segment,
        )

    def past_states_50hz(self, idx):
        # Kept for symmetry with your original naming.
        return self.past_states(idx)


    def past_obs(self, idx):
        segment = jax.tree.map(lambda x: self._segment(idx, x), self.obs_trajectory)
        obs = []
        base = self.slice_map["past_obs_block0"][0]
        # at 50 Hz, each step is 1 index; include H past observations
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, base, self.slice_map["past_obs_block0"][1], axis=1
            ),
            segment,
        )


    def past_actions(self, idx):
        segment = jax.tree.map(lambda x: self._segment(idx, x), self.action_trajectory)
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, self.slice_map["past_actions_block0"][0], self.slice_map["past_actions_block0"][1], axis=1
            ),
            segment,
        )

    def future_states(self, idx):
        segment = jax.tree.map(lambda x: self._segment(idx, x), self.state_trajectory)
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, self.slice_map["future_states_block0"][0], self.slice_map["future_states_block0"][1], axis=1
            ),
            segment,
        )

    def future_actions(self, idx):
        segment = jax.tree.map(lambda x: self._segment(idx, x), self.action_trajectory)
        return jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(
                x, self.slice_map["future_actions_block0"][0], self.slice_map["future_actions_block0"][1], axis=1
            ),
            segment,
        )

    # ------------------------- dynamic attribute access ------------------------

    def __getattr__(self, name):
        slice_map = object.__getattribute__(self, "slice_map")
        if name in slice_map:
            if "past_actions_block" in name:
                return jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, slice_map[name][0], slice_map[name][1], axis=1),
                    self.action_trajectory,
                )
            if "past_states_block" in name:
                return jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, slice_map[name][0], slice_map[name][1], axis=1),
                    self.state_trajectory,
                )
            if "future_actions_block" in name:
                return jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, slice_map[name][0], slice_map[name][1], axis=1),
                    self.action_trajectory,
                )
            if "future_states_block" in name:
                return jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, slice_map[name][0], slice_map[name][1], axis=1),
                    self.state_trajectory,
                )
            if "past_obs_block" in name:
                obs = []
                base = slice_map[name][0]
                return jax.tree.map(
                    lambda x: jax.lax.dynamic_slice_in_dim(x, base, slice_map[name][1], axis=1),
                    self.obs_trajectory,
                )

        raise AttributeError(f"{name} not found")

    def __setattr__(self, name, value):
        if name in ("horizon", "T", "H", "state_trajectory", "action_trajectory", "obs_trajectory", "H_Obs", "key"):
            object.__setattr__(self, name, value)
        elif "slice_map" in self.__dict__ and name in self.slice_map:
            if "actions" in name:
                self.action_trajectory.array = jax.lax.dynamic_update_slice_in_dim(
                    self.action_trajectory.array, value, self.slice_map[name][0], axis=1
                )
            if "states" in name:
                self.state_trajectory.array = jax.lax.dynamic_update_slice_in_dim(
                    self.state_trajectory.array, value, self.slice_map[name][0], axis=1
                )
            if "obs" in name:
                self.obs_trajectory.array = jax.lax.dynamic_update_slice_in_dim(
                    self.obs_trajectory.array, value, self.slice_map[name][0], axis=1
                )
        else:
            object.__setattr__(self, name, value)

    # ------------------------- JAX PyTree plumbing -----------------------------

    def tree_flatten(self):
        dynamic_children = (self.state_trajectory, self.action_trajectory, self.obs_trajectory, self.key)
        static_aux = (self.slice_map, self.horizon, self.T, self.H, self.discount, self.total_steps, self.H_Obs)
        return dynamic_children, static_aux

    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        slice_map, horizon, T, H, discount, total_steps, H_Obs = static_aux_data
        state_trajectory, action_trajectory, obs_trajectory, key = dynamic_children

        obj = cls.__new__(cls)
        object.__setattr__(obj, "state_trajectory", state_trajectory)
        object.__setattr__(obj, "action_trajectory", action_trajectory)
        object.__setattr__(obj, "obs_trajectory", obs_trajectory)
        object.__setattr__(obj, "slice_map", slice_map)
        object.__setattr__(obj, "horizon", horizon)
        object.__setattr__(obj, "T", T)
        object.__setattr__(obj, "H", H)
        object.__setattr__(obj, "discount", discount)
        object.__setattr__(obj, "total_steps", total_steps)
        object.__setattr__(obj, "H_Obs", H_Obs)
        object.__setattr__(obj, "key", key)
        return obj

