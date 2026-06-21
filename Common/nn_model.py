import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jax import random
import dill
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional

def init_params_xavier(kernel_size, input_dim, output_dim, key, dtype=jnp.float32):
    if kernel_size > 0:
        limit = jnp.sqrt(6.0 / (kernel_size * (input_dim + output_dim)))
        w = jax.random.uniform(key, (kernel_size, input_dim, output_dim), minval=-limit, maxval=limit, dtype=dtype)
    else:
        limit = jnp.sqrt(6.0 / (input_dim + output_dim))
        w = jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit, dtype=dtype)
    return w

def init_params_he(kernel_size, input_dim, output_dim, key, dtype=jnp.float32):
    if kernel_size > 0:
        std = jnp.sqrt(2.0 / (kernel_size * input_dim))
        w = jax.random.normal(key, (kernel_size, input_dim, output_dim), dtype=dtype) * std
    else:
        std = jnp.sqrt(2.0 / input_dim)
        w = jax.random.normal(key, (input_dim, output_dim), dtype=dtype) * std
    return w

def init_params_WC(input_dim, output_dim, key, dtype=jnp.float32):
    fan_in = input_dim
    limit = 1 / jnp.sqrt(fan_in)
    w = jax.random.uniform(key, (input_dim, output_dim), minval=-limit, maxval=limit, dtype=dtype)
    return w

def init_bias(output_size, dtype=jnp.float32):
    """Initialize a single bias vector of length output_size."""
    return jnp.zeros(output_size, dtype=dtype)


def init_FC_model(hidden_dims, in_dim, out_dim, key_seq, dtype=jnp.float32):
    weights, biases = [], []
    prev_dim = in_dim
    for h_dim, k in zip(hidden_dims, key_seq[::2]):
        w = init_params_he(0, prev_dim, h_dim, k, dtype=dtype).T
        b = init_bias(h_dim, dtype=dtype)
        weights.append(w) 
        biases.append(b)
        prev_dim = h_dim
    w_final = init_params_xavier(0, prev_dim, out_dim, key_seq[-2], dtype=dtype).T #init_params_he(0, prev_dim, out_dim, key_seq[-2], dtype=dtype).T #
    b_final = init_bias(out_dim, dtype=dtype)
    weights.append(w_final)
    biases.append(b_final)
    return weights, biases

# @register_pytree_node_class
# class MultiHeadAttention:
#     def __init__(self, embed_dim, n_heads, smooth=True):
#         assert embed_dim % n_heads == 0

#         self.




        
    



@register_pytree_node_class
class MLPBase(ABC):
    def __init__(self, key=None, input_dim=None, output_dim=None, params=None,
                 hidden_scale=[1.5, 1.5, 1.5, 1.5], Lipschitz=False, Lipschitz_ub=1e12, J_Lipschitz_ub=1e12, activation='mish', 
                 inference_mode=False, dtype=jnp.float32, SNS=False, hard=True, order=1):

        self.Lipschitz = Lipschitz
        self.Lipschitz_ub = Lipschitz_ub

        self.J_Lipschitz_ub = J_Lipschitz_ub if J_Lipschitz_ub < 50 else None
        self.inference_mode = inference_mode
        n_layers = len(hidden_scale) + 1
        self.n_layers = n_layers
        self.hard = hard
        self.SNS = SNS
        self.order = order



        if self.Lipschitz_ub is not None:
            self.J_Lipschitz_ub = J_Lipschitz_ub
            # if J_Lipschitz_ub < 50:
            #     # min_c = jnp.power(
            #     #     jnp.asarray(Lipschitz_ub, dtype=jnp.float32),
            #     #     1.0 / jnp.asarray(n_layers, dtype=jnp.float32),
            #     # )
            #     # min_sum = jnp.asarray(n_layers, dtype=jnp.float32) * min_c
            #     # J_Lipschitz_ub = J_Lipschitz_ub * min_sum #* self.Lipschitz_ub
            #     self.J_Lipschitz_ub = 20.0 #J_Lipschitz_ub
        else:
            self.J_Lipschitz_ub = None

        if activation is None:
            self.activation = "mish"
        if activation not in ["softplus", "relu", "mish", "tanh", "elu", "sine"]:
            raise ValueError(f"Unsupported activation function: {activation}. Supported functions are 'softplus', 'relu', 'mish', 'tanh', and 'elu'.")
        
        self.activation = activation
        if params is not None:
            self.params = params
            W, _ = self.params
            self.input_dim = W[0].shape[1]
            self.output_dim = W[-1].shape[0]
            self.hidden_dims = [w.shape[1] for w in W[1:]]
            self.n_layers = len(self.hidden_dims) + 1
        else:
            assert key is not None and input_dim is not None and output_dim is not None
            self.input_dim = input_dim
            self.output_dim = output_dim

            dim = jnp.maximum(input_dim, output_dim)
            self.hidden_dims = [int(8 * jnp.ceil(hidden_scale[i] * dim / 8)) for i in range(n_layers - 1)]


            keys = jax.random.split(key, 2 * n_layers + 2)
            weights, biases = init_FC_model(self.hidden_dims, input_dim, output_dim, keys, dtype=dtype)

            #if self.Lipschitz:
                # if SNS:
                #     weights = [w / jnp.maximum(1e-8, self.infty_norm(w)) for w in weights]
                #     target = self.Lipschitz_ub ** (1.0 / len(weights))
                #     weights = [w * target for w in weights]

            self.params = [weights, biases]
            self.Lipschitz_constants = self.calc_Lipschitz_constants()
            if SNS:
                self.Lipschitz_constants = jax.tree.map(lambda x: jnp.log(x), self.Lipschitz_constants)


            self.Lipschitz_scale = jnp.log(jnp.ones(self.n_layers, dtype=jnp.float32))
            print(f"Lipschitz scale initialized to: {self.Lipschitz_scale}")

            # self.params = [weights, biases]
            # self.Lipschitz_constants = self.calc_Lipschitz_constants()

            # ub, residual = self.calc_residual()
            
            # make it so the residual is zero at initialization

    def calc_Lipschitz_constants(self):
        W, _ = self.params
        return [self.infty_norm(w) for w in W]

    @staticmethod
    def infty_norm(W):
        return jnp.max(jnp.sum(jnp.abs(W), axis=1))

    @staticmethod
    def Lipschitz_normalization(W, c):
        row_sums = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, c / row_sums)
        return W * scale[:, None]

    def fwd(self, x, W, b, c):
        if self.inference_mode:
            return jnp.dot(W, x) + b
        if self.Lipschitz:
            if self.SNS:
                c = jnp.exp(c)
            else:
                c = jax.nn.softplus(c)
            W = self.Lipschitz_normalization(W, c)
        return jnp.dot(W, x) + b


    def calc_residual(self):
        # Flatten and prepare Lipschitz constants
        if not self.Lipschitz:
            c = self.calc_Lipschitz_constants()
        else:
            c = self.Lipschitz_constants
        c, _ = jax.tree_util.tree_flatten(c)
        c = jax.tree_map(lambda x: x.reshape(1, -1), c)
        c = jnp.concatenate(c, axis=-1).flatten()

        # Enforce positivity
        if self.SNS:
            c = jnp.exp(c)
        else:
            c = jax.nn.softplus(c)



        g = jnp.prod(c)
        
        prefix = jnp.cumprod(c)
        rhs = jnp.sum(prefix)
        j = rhs * g

        
        if self.order == 1:
            c_target = self.Lipschitz_ub

            residual_f = (g - c_target) / c_target
            residual_j = 0

        elif self.order == 12:
            c_target = self.Lipschitz_ub

            residual_f = (g - c_target) / c_target

            c_ = self.Lipschitz_ub ** (1.0 / self.n_layers)
            c_cum_prod = jnp.cumprod(jnp.full((self.n_layers,), c_))
            d_target = jnp.sum(c_cum_prod) * self.Lipschitz_ub

            residual_j = (j - d_target) / d_target
            residual_j *= 2

        
        else:
            c_ = self.Lipschitz_ub ** (1.0 / self.n_layers)
            c_cum_prod = jnp.cumprod(jnp.full((self.n_layers,), c_))
            d_target = jnp.sum(c_cum_prod) * self.Lipschitz_ub

            residual_f = 0
            residual_j = (j - d_target) / d_target
            residual_j *= 2


        lipschitz_residual = jnp.maximum(0.0, residual_f)
        jacobian_residual =  jnp.maximum(0.0, residual_j)

        return g, j, lipschitz_residual, jacobian_residual
    
    def activate(self, x):
        if self.activation == "softplus":
            return jax.nn.softplus(x)
        elif self.activation == "relu":
            return jax.nn.relu(x)
        elif self.activation == "mish":
            return jax.nn.mish(x)
        elif self.activation == "tanh":
            return jnp.tanh(x)
        elif self.activation == "elu":
            return jax.nn.elu(x)
        elif self.activation == "sine":
            return jnp.sin(x)
        
    def forward(self, x):
        W, b = self.params
        c = self.Lipschitz_constants
        for i in range(self.n_layers):
            x = self.fwd(x, W[i], b[i], c[i])
            if i < self.n_layers - 1:
                x = self.activate(x)
        return x

    def set_inference_mode(self, inference_mode: bool, params: Tuple[Any, ...]):
        self.inference_mode = inference_mode

        if inference_mode and self.Lipschitz:
            W, b = params
            c = self.Lipschitz_constants
            if self.SNS:
                c = jax.tree.map(lambda x: jnp.exp(x), c)
            else:
                c = jax.tree.map(lambda x: jax.nn.softplus(x), c)
            Wn = [self.Lipschitz_normalization(w, ci) for w, ci in zip(W, c)]
            self.params = [Wn, b]
        else:
            self.params = params

    def clip_params(self):
        if self.Lipschitz:
            W, b = self.params
            c = self.Lipschitz_constants
            if self.SNS:
                c = jax.tree.map(lambda x: jnp.exp(x), c)
            else:
                c = jax.tree.map(lambda x: jax.nn.softplus(x), c)
            Wn = [self.Lipschitz_normalization(w, ci) for w, ci in zip(W, c)]
            self.params = [Wn, b]

    def flatten_dynamic_aux(self) -> Tuple[Any, ...]:
        return ()

    def flatten_static_aux(self) -> Tuple[Any, ...]:
        return ()

    def unflatten_dynamic_aux(self, aux: Tuple[Any, ...]):
        pass

    def unflatten_static_aux(self, aux: Tuple[Any, ...]):
        pass

    def tree_flatten(self):
        dynamic_children = (self.params, self.Lipschitz_constants, self.Lipschitz_scale) + tuple(self.flatten_dynamic_aux())
        static_aux = (self.input_dim, self.output_dim, self.Lipschitz, self.inference_mode, self.n_layers, self.hard, self.SNS,
                      tuple(self.hidden_dims), self.Lipschitz_ub, self.J_Lipschitz_ub, self.activation, self.order) + tuple(self.flatten_static_aux())
        return dynamic_children, static_aux

    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        input_dim, output_dim, Lipschitz, inference_mode, n_layers, hard, SNS, hidden_dims, Lipschitz_ub, J_Lipschitz_ub, activation, order, *static_extra = static_aux_data
        params, Lipschitz_constants, Lipschitz_scale, *dynamic_extra = dynamic_children

        obj = cls.__new__(cls)

        object.__setattr__(obj, "params", params)
        object.__setattr__(obj, "Lipschitz_constants", Lipschitz_constants)
        object.__setattr__(obj, "input_dim", input_dim)
        object.__setattr__(obj, "output_dim", output_dim)
        object.__setattr__(obj, "Lipschitz", Lipschitz)
        object.__setattr__(obj, "Lipschitz_ub", Lipschitz_ub)
        object.__setattr__(obj, "J_Lipschitz_ub", J_Lipschitz_ub)   
        object.__setattr__(obj, "hard", hard)
        object.__setattr__(obj, "SNS", SNS)
        object.__setattr__(obj, "Lipschitz_scale", Lipschitz_scale)
        object.__setattr__(obj, "inference_mode", inference_mode)
        object.__setattr__(obj, "n_layers", n_layers)
        object.__setattr__(obj, "hidden_dims", list(hidden_dims))
        object.__setattr__(obj, "activation", activation)
        object.__setattr__(obj, "order", order)

        obj.unflatten_dynamic_aux(tuple(dynamic_extra))
        obj.unflatten_static_aux(tuple(static_extra))
        return obj

    def save(self, path):
        path = path + "_params.npz"
        with open(path, "wb") as f:
            params = {
                "params": self.params,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "Lipschitz_constants": self.Lipschitz_constants,
                "Lipschitz_ub": self.Lipschitz_ub if hasattr(self, 'Lipschitz_ub') else None,
                "Lipschitz_scale": self.Lipschitz_scale if hasattr(self, 'Lipschitz_scale') else 1.0,
                "J_Lipschitz_ub": self.J_Lipschitz_ub if hasattr(self, 'J_Lipschitz_ub') else None,
                "hard": self.hard if hasattr(self, 'hard') else False,
                "SNS": self.SNS if hasattr(self, 'SNS') else False,
                "Lipschitz": self.Lipschitz, 
                "hidden_dims": self.hidden_dims,
                "n_layers": self.n_layers,
                "inference_mode": self.inference_mode,
                "activation": self.activation,
                "order": self.order if hasattr(self, 'order') else 1
            }

            dill.dump(params, f)
            
    @classmethod
    def load(cls, path):
        object = cls.__new__(cls)
        path = path + "_params.npz"
        with open(path, "rb") as f:
            params = dill.load(f)

            for key, value in params.items():
                setattr(object, key, value)

            if not hasattr(object, 'Lipschitz_ub'):
                object.Lipschitz_ub = 1e12
            if not hasattr(object, 'activation'):
                object.activation = "mish"
            if not hasattr(object, 'hard'):
                object.hard = False
            if not hasattr(object, 'Lipschitz_scale'):
                object.Lipschitz_scale = jnp.array(10.0, dtype=jnp.float32)
            if not hasattr(object, 'SNS'):
                object.SNS = False
            if not hasattr(object, 'J_Lipschitz_ub'):
                object.J_Lipschitz_ub = 1e12
            if not hasattr(object, 'order'):
                object.order = 1
        return object




def get_joint_data(lifted_states, lifted_actions):

    q_pos = lifted_states.array[..., lifted_states.index_map['q_fl_hip'][0]:lifted_states.index_map['q_rr_ankle'][1]]
    q_pos_d = jnp.concatenate([lifted_states.array[1:, lifted_states.n_states:], lifted_actions.array], axis=0)

    pos_error = q_pos[2:] - q_pos_d[2:]
    last_pos_error = q_pos[1:-1] - q_pos_d[1:-1]
    last_last_pos_error = q_pos[0:-2] - q_pos_d[0:-2]

    velocity = lifted_states.array[..., lifted_states.index_map['v_fl_hip'][0]:lifted_states.index_map['v_rr_ankle'][1]]

    joint_data = jnp.stack([
        pos_error, last_pos_error, last_last_pos_error,
        velocity[2:], velocity[1:-1], velocity[0:-2]
    ], axis=-1)
    return joint_data


def get_pd_torque(lifted_states, lifted_actions):
    kp = 20
    kd = 0.5

    pos = lifted_states.array[..., lifted_states.index_map['q_fl_hip'][0]:lifted_states.index_map['q_rr_ankle'][1]]
    pos_target = lifted_actions.array
    velocity = lifted_states.array[..., lifted_states.index_map['v_fl_hip'][0]:lifted_states.index_map['v_rr_ankle'][1]]
    torque = kp * (pos_target - pos[-1]) - kd * velocity[-1]
    actions = jax.tree.map(lambda x: torque, lifted_actions)

    return actions

def get_actuator_net_torque(actuator_net, lifted_states, lifted_actions):
    
    joint_data = get_joint_data(lifted_states, lifted_actions)

    torques = jax.vmap(jax.vmap(actuator_net.forward))(joint_data)
    actions = jax.tree.map(lambda x: torques.reshape(-1, 12), lifted_actions)
    return actions

def get_actuator_net_torque_rollout(actuator_net, lifted_states, lifted_actions):
    
    joint_data = get_joint_data(lifted_states, lifted_actions)

    torques = jax.vmap(actuator_net.forward)(joint_data[-1])
    actions = jax.tree.map(lambda x: torques, lifted_actions)

    return actions

torque_limits = {"fl_hip": (-24.0, 24.0),
                    "fl_knee": (-24.0, 24.0),
                    "fl_ankle": (-45.0, 45.0),
                    "fr_hip": (-24.0, 24.0),
                    "fr_knee": (-24.0, 24.0),
                    "fr_ankle": (-45.0, 45.0),
                    "rl_hip": (-24.0, 24.0),
                    "rl_knee": (-24.0, 24.0),
                    "rl_ankle": (-45.0, 45.0),
                    "rr_hip": (-24.0, 24.0),
                    "rr_knee": (-24.0, 24.0),
                    "rr_ankle": (-45.0, 45.0)}
tau_min = jnp.array([low for low, high in torque_limits.values()]).reshape(1, 12)
tau_max = jnp.array([high for low, high in torque_limits.values()]).reshape(1, 12)

@jax.custom_jvp
def clip_torques(torques):
    return jnp.clip(torques, tau_min, tau_max)

@clip_torques.defjvp
def _clip_torques_jvp(primals, tangents): # STE: gradient = identity
    torques, = primals
    dtorques, = tangents
    y = jnp.clip(torques, tau_min, tau_max)
    return y, dtorques

def normalize_torques(torques):
    return 2.0 * (torques - tau_min) / (tau_max - tau_min) - 1.0

@register_pytree_node_class
class MLPDynamics(MLPBase):
    def __init__(self, Lipschitz=False, Lipschitz_ub=1e12, J_Lipschitz_ub=1e12, activation="mish", T=10, H=10, horizon=18, n_states=8, n_actions=4, path=None, actuator_net=None,
                    hidden_scale=[1.5, 1.5, 1.5], seed=0, inference_mode=False, w_torque=False, SNS=False, dtype=jnp.float32, hard=False, order=1):
        
        self.T = T
        self.H = H
        self.n_states = n_states
        self.n_actions = n_actions
        self.horizon = horizon
        self.w_torque = w_torque

        self.actuator_net = actuator_net

        history_in = H * (n_states + n_actions)
        action_in = n_actions
        if self.w_torque:
            input_dim = (H - 2) * n_actions + H * n_states
        else:
            input_dim = history_in + action_in
        output_dim = T * n_states

        self.action_scales = jnp.zeros(1)
        self.Lipschitz = Lipschitz
        self.inference_mode = inference_mode

        key = jax.random.PRNGKey(seed)

        if path is None:
            super().__init__(key=key, input_dim=input_dim, output_dim=output_dim, 
                             hidden_scale=hidden_scale, Lipschitz=Lipschitz, inference_mode=inference_mode, Lipschitz_ub=Lipschitz_ub, J_Lipschitz_ub=J_Lipschitz_ub, activation=activation, SNS=SNS, dtype=dtype, hard=hard, order=order)
        else:
            with open(path, "rb") as f:
                data = dill.load(f)

                for key, value in data.items():
                    setattr(self, key, value)
                if not hasattr(self, 'Lipschitz_ub'):
                    self.Lipschitz_ub = 1e12
                if not hasattr(self, 'activation'):
                    self.activation = "mish"
                if not hasattr(self, 'hard'):
                    self.hard = False
                if not hasattr(self, 'SNS'):
                    self.SNS = False
                if not hasattr(self, 'J_Lipschitz_ub'):
                    self.J_Lipschitz_ub = 1e12
                if not hasattr(self, 'order'):
                    self.order = 1
            super().__init__(key=None, params=self.params, Lipschitz=self.Lipschitz, inference_mode=self.inference_mode, Lipschitz_ub=self.Lipschitz_ub, J_Lipschitz_ub=self.J_Lipschitz_ub, activation=self.activation, SNS=SNS, dtype=dtype, hard=self.hard, order=self.order)
            if not hasattr(self, 'Lipschitz_constants'):
                self.Lipschitz_constants = self.calc_Lipschitz_constants()
            if not hasattr(self, 'Lipschitz_scale'):
                self.Lipschitz_scale = jnp.array(10.0, dtype=jnp.float32)

    def process(self, lifted_states, lifted_actions):
        lifted_states_normalized = lifted_states.normalized()
        if self.w_torque:
            lifted_states_normalized_ = lifted_states_normalized.array[..., :lifted_states.n_states].flatten()
            torques = get_actuator_net_torque(self.actuator_net, lifted_states, lifted_actions)
            torques.array = clip_torques(torques.array)
            torques_normalized = normalize_torques(torques.array)
            torques_normalized_ = torques_normalized.flatten()
            lifted_actions_normalized_ = torques_normalized_
        else:
            lifted_states_normalized_ = lifted_states.normalized().array.flatten()
            lifted_actions_normalized_ = lifted_actions.normalized().array.flatten()
        return jnp.concatenate([lifted_states_normalized_, lifted_actions_normalized_], axis=-1)

    def _F_x(self, lifted_states, lifted_actions):
        lifted_input_normalized = self.process(lifted_states, lifted_actions)
        states = self.forward(lifted_input_normalized)
        states = states.reshape((self.T, self.n_states))
        return states
    
    def F_x(self, lifted_states, lifted_actions):
        """lifted_states and lifted_actions mapped to states"""
        states_normalized = lifted_states.states.normalized()
        current_state = states_normalized.array[-1]
        dstates = self._F_x(lifted_states, lifted_actions)
        states_normalized = jax.tree.map(lambda x: current_state + 0.1 * dstates, lifted_states.states)
        return states_normalized.denormalized()

    def dxdt(self, lifted_states, lifted_actions):
        d_states = self._F_x(lifted_states, lifted_actions)
        d_states = d_states*lifted_states.states.scales.array
        return jax.tree.map(lambda x: d_states, lifted_states.states)

    def F_X(self, lifted_states, lifted_actions):
        """lifted_states and lifted_actions mapped to lifted_states"""
        states = self.F_x(lifted_states, lifted_actions)
        full_states = jnp.concatenate([lifted_states.array[..., :lifted_states.n_states], states.array], axis=0)
        full_actions = jnp.concatenate([lifted_states.array[..., lifted_states.n_states:], lifted_actions.array], axis=0)
        past_states = full_states[-self.H:]
        past_actions = full_actions[-self.H:]
        lifted_states_out = jax.tree.map(lambda x: jnp.concatenate([past_states, past_actions], axis=-1), lifted_states)
        return states, lifted_states_out
    
    def rollout(self, lifted_states, lifted_actions):
        return self._rollout(lifted_states, lifted_actions)
    
    def _rollout(self, lifted_states, lifted_actions):
        def scan_fn(lifted_states, lifted_actions):
            states, lifted_states = self.F_X(lifted_states, lifted_actions)
            if self.actuator_net is not None:
                actions = get_actuator_net_torque_rollout(self.actuator_net, lifted_states, lifted_actions)
            else:
                actions = get_pd_torque(lifted_states, lifted_actions)
            return lifted_states, [states, actions]
        _, all_preds = jax.lax.scan(scan_fn, lifted_states, lifted_actions)
        return jax.tree.map(lambda x: x.reshape(-1, self.n_states), all_preds[0]), jax.tree.map(lambda x: x.reshape(-1, self.n_actions), all_preds[1])
    
    def save(self, path):
        path = path + "_params.npz"
        with open(path, "wb") as f:
            data = {
                "params": self.params,
                "T": self.T,
                "H": self.H,
                "n_states": self.n_states,
                "n_actions": self.n_actions,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "w_torque": self.w_torque,
                "Lipschitz_constants": self.Lipschitz_constants,
                "Lipschitz": self.Lipschitz, 
                "Lipschitz_ub": self.Lipschitz_ub if hasattr(self, 'Lipschitz_ub') else None,
                "J_Lipschitz_ub": self.J_Lipschitz_ub if hasattr(self, 'J_Lipschitz_ub') else None,
                "activation": self.activation if hasattr(self, 'activation') else "mish",
                "hard": self.hard if hasattr(self, 'hard') else False,
                "Lipschitz_scale": self.Lipschitz_scale if hasattr(self, 'Lipschitz_scale') else 1.0,
                "action_scales": self.action_scales if hasattr(self, 'action_scales') else jnp.array([1.0] * (self.n_actions * (self.H + 1)), dtype=jnp.float32),
                "order": self.order if hasattr(self, 'order') else 1,
                "hidden_dims": self.hidden_dims,
                "n_layers": self.n_layers,
                "horizon": self.horizon,
                "inference_mode": self.inference_mode
            }
            dill.dump(data, f)

    def flatten_static_aux(self):
        aux_data = (self.T, self.H, self.n_states, self.n_actions, self.horizon, self.actuator_net, self.w_torque)
        return aux_data
    
    def unflatten_static_aux(self, aux_data):
        self.T, self.H, self.n_states, self.n_actions, self.horizon, self.actuator_net, self.w_torque = aux_data

    def flatten_dynamic_aux(self):
        return (self.action_scales,)

    def unflatten_dynamic_aux(self, aux_data):
        self.action_scales, = aux_data

def preds_obs_states_map_naive(past_states_copy, past_observations_t, states, H, past_states_scales):

    # Measured

    past_states_copy.v_roll = past_observations_t.v_roll[-H:]
    past_states_copy.v_pitch = past_observations_t.v_pitch[-H:]
    past_states_copy.v_yaw = past_observations_t.v_yaw[-H:]

    past_states_copy.q_fl_hip = past_observations_t.q_fl_hip[-H:]
    past_states_copy.q_fr_hip = past_observations_t.q_fr_hip[-H:]
    past_states_copy.q_rl_hip = past_observations_t.q_rl_hip[-H:]
    past_states_copy.q_rr_hip = past_observations_t.q_rr_hip[-H:]
    past_states_copy.q_fl_knee = past_observations_t.q_fl_knee[-H:]
    past_states_copy.q_fr_knee = past_observations_t.q_fr_knee[-H:]
    past_states_copy.q_rl_knee = past_observations_t.q_rl_knee[-H:]
    past_states_copy.q_rr_knee = past_observations_t.q_rr_knee[-H:]
    past_states_copy.q_fl_ankle = past_observations_t.q_fl_ankle[-H:]
    past_states_copy.q_fr_ankle = past_observations_t.q_fr_ankle[-H:]
    past_states_copy.q_rl_ankle = past_observations_t.q_rl_ankle[-H:]
    past_states_copy.q_rr_ankle = past_observations_t.q_rr_ankle[-H:]

    past_states_copy.v_fl_hip = past_observations_t.v_fl_hip[-H:]
    past_states_copy.v_fr_hip = past_observations_t.v_fr_hip[-H:]
    past_states_copy.v_rl_hip = past_observations_t.v_rl_hip[-H:]
    past_states_copy.v_rr_hip = past_observations_t.v_rr_hip[-H:]
    past_states_copy.v_fl_knee = past_observations_t.v_fl_knee[-H:]
    past_states_copy.v_fr_knee = past_observations_t.v_fr_knee[-H:]
    past_states_copy.v_rl_knee = past_observations_t.v_rl_knee[-H:]
    past_states_copy.v_rr_knee = past_observations_t.v_rr_knee[-H:]
    past_states_copy.v_fl_ankle = past_observations_t.v_fl_ankle[-H:]
    past_states_copy.v_fr_ankle = past_observations_t.v_fr_ankle[-H:]
    past_states_copy.v_rl_ankle = past_observations_t.v_rl_ankle[-H:]
    past_states_copy.v_rr_ankle = past_observations_t.v_rr_ankle[-H:]

    past_states_copy.v11 = past_observations_t.v11[-H:]
    past_states_copy.v12 = past_observations_t.v12[-H:]
    past_states_copy.v13 = past_observations_t.v13[-H:]
    past_states_copy.v21 = past_observations_t.v21[-H:]
    past_states_copy.v22 = past_observations_t.v22[-H:]
    past_states_copy.v23 = past_observations_t.v23[-H:]

    # Predicted
    past_states_copy.z = states[:, 0][:, None] * past_states_scales.z
    past_states_copy.v_x = states[:, 1][:, None] * past_states_scales.v_x
    past_states_copy.v_y = states[:, 2][:, None] * past_states_scales.v_y
    past_states_copy.v_z = states[:, 3][:, None] * past_states_scales.v_z

    past_states_copy.sdf_base_main = past_states_copy.sdf_base_main + states[:, 4][:, None] * past_states_scales.sdf_base_main
    past_states_copy.sdf_base_head_top = past_states_copy.sdf_base_head_top + states[:, 5][:, None] * past_states_scales.sdf_base_head_top
    past_states_copy.sdf_base_head_bottom = past_states_copy.sdf_base_head_bottom +  states[:, 6][:, None] * past_states_scales.sdf_base_head_bottom

    past_states_copy.sdf_fl_foot = states[:, 7][:, None] * past_states_scales.sdf_fl_foot
    past_states_copy.sdf_fr_foot = states[:, 8][:, None] * past_states_scales.sdf_fr_foot
    past_states_copy.sdf_rl_foot = states[:, 9][:, None] * past_states_scales.sdf_rl_foot
    past_states_copy.sdf_rr_foot = states[:, 10][:, None] * past_states_scales.sdf_rr_foot

    past_states_copy.sdf_fl_shank_top = states[:, 11][:, None] * past_states_scales.sdf_fl_shank_top
    past_states_copy.sdf_fr_shank_top = states[:, 12][:, None] * past_states_scales.sdf_fr_shank_top
    past_states_copy.sdf_rl_shank_top = states[:, 13][:, None] * past_states_scales.sdf_rl_shank_top
    past_states_copy.sdf_rr_shank_top = states[:, 14][:, None] * past_states_scales.sdf_rr_shank_top

    past_states_copy.sdf_fl_shank_bottom = states[:, 15][:, None] * past_states_scales.sdf_fl_shank_bottom
    past_states_copy.sdf_fr_shank_bottom = states[:, 16][:, None] * past_states_scales.sdf_fr_shank_bottom
    past_states_copy.sdf_rl_shank_bottom = states[:, 17][:, None] * past_states_scales.sdf_rl_shank_bottom
    past_states_copy.sdf_rr_shank_bottom = states[:, 18][:, None] * past_states_scales.sdf_rr_shank_bottom

    past_states_copy.sdf_fl_thigh = states[:, 19][:, None] * past_states_scales.sdf_fl_thigh
    past_states_copy.sdf_fr_thigh = states[:, 20][:, None] * past_states_scales.sdf_fr_thigh
    past_states_copy.sdf_rl_thigh = states[:, 21][:, None] * past_states_scales.sdf_rl_thigh
    past_states_copy.sdf_rr_thigh = states[:, 22][:, None] * past_states_scales.sdf_rr_thigh

    past_states_copy.sdf_fl_hip = states[:, 23][:, None] * past_states_scales.sdf_fl_hip
    past_states_copy.sdf_fr_hip = states[:, 24][:, None] * past_states_scales.sdf_fr_hip
    past_states_copy.sdf_rl_hip = states[:, 25][:, None] * past_states_scales.sdf_rl_hip
    past_states_copy.sdf_rr_hip = states[:, 26][:, None] * past_states_scales.sdf_rr_hip

    return past_states_copy 
    
def preds_obs_states_map(past_states_copy, past_observations_t, d_states, H, past_states_scales):

    # Measured

    past_states_copy.v_roll = past_observations_t.v_roll[-H:]
    past_states_copy.v_pitch = past_observations_t.v_pitch[-H:]
    past_states_copy.v_yaw = past_observations_t.v_yaw[-H:]

    past_states_copy.q_fl_hip = past_observations_t.q_fl_hip[-H:]
    past_states_copy.q_fr_hip = past_observations_t.q_fr_hip[-H:]
    past_states_copy.q_rl_hip = past_observations_t.q_rl_hip[-H:]
    past_states_copy.q_rr_hip = past_observations_t.q_rr_hip[-H:]
    past_states_copy.q_fl_knee = past_observations_t.q_fl_knee[-H:]
    past_states_copy.q_fr_knee = past_observations_t.q_fr_knee[-H:]
    past_states_copy.q_rl_knee = past_observations_t.q_rl_knee[-H:]
    past_states_copy.q_rr_knee = past_observations_t.q_rr_knee[-H:]
    past_states_copy.q_fl_ankle = past_observations_t.q_fl_ankle[-H:]
    past_states_copy.q_fr_ankle = past_observations_t.q_fr_ankle[-H:]
    past_states_copy.q_rl_ankle = past_observations_t.q_rl_ankle[-H:]
    past_states_copy.q_rr_ankle = past_observations_t.q_rr_ankle[-H:]

    past_states_copy.v_fl_hip = past_observations_t.v_fl_hip[-H:]
    past_states_copy.v_fr_hip = past_observations_t.v_fr_hip[-H:]
    past_states_copy.v_rl_hip = past_observations_t.v_rl_hip[-H:]
    past_states_copy.v_rr_hip = past_observations_t.v_rr_hip[-H:]
    past_states_copy.v_fl_knee = past_observations_t.v_fl_knee[-H:]
    past_states_copy.v_fr_knee = past_observations_t.v_fr_knee[-H:]
    past_states_copy.v_rl_knee = past_observations_t.v_rl_knee[-H:]
    past_states_copy.v_rr_knee = past_observations_t.v_rr_knee[-H:]
    past_states_copy.v_fl_ankle = past_observations_t.v_fl_ankle[-H:]
    past_states_copy.v_fr_ankle = past_observations_t.v_fr_ankle[-H:]
    past_states_copy.v_rl_ankle = past_observations_t.v_rl_ankle[-H:]
    past_states_copy.v_rr_ankle = past_observations_t.v_rr_ankle[-H:]

    past_states_copy.v11 = past_observations_t.v11[-H:]
    past_states_copy.v12 = past_observations_t.v12[-H:]
    past_states_copy.v13 = past_observations_t.v13[-H:]
    past_states_copy.v21 = past_observations_t.v21[-H:]
    past_states_copy.v22 = past_observations_t.v22[-H:]
    past_states_copy.v23 = past_observations_t.v23[-H:]

    # Predicted
    past_states_copy.z = past_states_copy.z + d_states[:, 0][:, None] * past_states_scales.z
    past_states_copy.v_x = past_states_copy.v_x + d_states[:, 1][:, None] * past_states_scales.v_x
    past_states_copy.v_y = past_states_copy.v_y + d_states[:, 2][:, None] * past_states_scales.v_y
    past_states_copy.v_z = past_states_copy.v_z + d_states[:, 3][:, None] * past_states_scales.v_z

    past_states_copy.sdf_base_main = past_states_copy.sdf_base_main + d_states[:, 4][:, None] * past_states_scales.sdf_base_main
    past_states_copy.sdf_base_head_top = past_states_copy.sdf_base_head_top + d_states[:, 5][:, None] * past_states_scales.sdf_base_head_top
    past_states_copy.sdf_base_head_bottom = past_states_copy.sdf_base_head_bottom +  d_states[:, 6][:, None] * past_states_scales.sdf_base_head_bottom

    past_states_copy.sdf_fl_foot = past_states_copy.sdf_fl_foot + d_states[:, 7][:, None] * past_states_scales.sdf_fl_foot
    past_states_copy.sdf_fr_foot = past_states_copy.sdf_fr_foot + d_states[:, 8][:, None] * past_states_scales.sdf_fr_foot
    past_states_copy.sdf_rl_foot = past_states_copy.sdf_rl_foot + d_states[:, 9][:, None] * past_states_scales.sdf_rl_foot
    past_states_copy.sdf_rr_foot = past_states_copy.sdf_rr_foot + d_states[:, 10][:, None] * past_states_scales.sdf_rr_foot

    past_states_copy.sdf_fl_shank_top = past_states_copy.sdf_fl_shank_top + d_states[:, 11][:, None] * past_states_scales.sdf_fl_shank_top
    past_states_copy.sdf_fr_shank_top = past_states_copy.sdf_fr_shank_top + d_states[:, 12][:, None] * past_states_scales.sdf_fr_shank_top
    past_states_copy.sdf_rl_shank_top = past_states_copy.sdf_rl_shank_top + d_states[:, 13][:, None] * past_states_scales.sdf_rl_shank_top
    past_states_copy.sdf_rr_shank_top = past_states_copy.sdf_rr_shank_top + d_states[:, 14][:, None] * past_states_scales.sdf_rr_shank_top

    past_states_copy.sdf_fl_shank_bottom = past_states_copy.sdf_fl_shank_bottom + d_states[:, 15][:, None] * past_states_scales.sdf_fl_shank_bottom
    past_states_copy.sdf_fr_shank_bottom = past_states_copy.sdf_fr_shank_bottom + d_states[:, 16][:, None] * past_states_scales.sdf_fr_shank_bottom
    past_states_copy.sdf_rl_shank_bottom = past_states_copy.sdf_rl_shank_bottom + d_states[:, 17][:, None] * past_states_scales.sdf_rl_shank_bottom
    past_states_copy.sdf_rr_shank_bottom = past_states_copy.sdf_rr_shank_bottom + d_states[:, 18][:, None] * past_states_scales.sdf_rr_shank_bottom

    past_states_copy.sdf_fl_thigh = past_states_copy.sdf_fl_thigh + d_states[:, 19][:, None] * past_states_scales.sdf_fl_thigh
    past_states_copy.sdf_fr_thigh = past_states_copy.sdf_fr_thigh + d_states[:, 20][:, None] * past_states_scales.sdf_fr_thigh
    past_states_copy.sdf_rl_thigh = past_states_copy.sdf_rl_thigh + d_states[:, 21][:, None] * past_states_scales.sdf_rl_thigh
    past_states_copy.sdf_rr_thigh = past_states_copy.sdf_rr_thigh + d_states[:, 22][:, None] * past_states_scales.sdf_rr_thigh

    past_states_copy.sdf_fl_hip = past_states_copy.sdf_fl_hip + d_states[:, 23][:, None] * past_states_scales.sdf_fl_hip
    past_states_copy.sdf_fr_hip = past_states_copy.sdf_fr_hip + d_states[:, 24][:, None] * past_states_scales.sdf_fr_hip
    past_states_copy.sdf_rl_hip = past_states_copy.sdf_rl_hip + d_states[:, 25][:, None] * past_states_scales.sdf_rl_hip
    past_states_copy.sdf_rr_hip = past_states_copy.sdf_rr_hip + d_states[:, 26][:, None] * past_states_scales.sdf_rr_hip

    return past_states_copy 

def overwrite_past_states_tm1(past_observations_t, past_states_tm1, H):

    past_states_copy = jax.tree.map(lambda x: x,  past_states_tm1)

    past_states_copy.v_roll = past_observations_t.v_roll[-H-1:-1]
    past_states_copy.v_pitch = past_observations_t.v_pitch[-H-1:-1]
    past_states_copy.v_yaw = past_observations_t.v_yaw[-H-1:-1]

    past_states_copy.q_fl_hip = past_observations_t.q_fl_hip[-H-1:-1]
    past_states_copy.q_fr_hip = past_observations_t.q_fr_hip[-H-1:-1]
    past_states_copy.q_rl_hip = past_observations_t.q_rl_hip[-H-1:-1]
    past_states_copy.q_rr_hip = past_observations_t.q_rr_hip[-H-1:-1]
    past_states_copy.q_fl_knee = past_observations_t.q_fl_knee[-H-1:-1]
    past_states_copy.q_fr_knee = past_observations_t.q_fr_knee[-H-1:-1]
    past_states_copy.q_rl_knee = past_observations_t.q_rl_knee[-H-1:-1]
    past_states_copy.q_rr_knee = past_observations_t.q_rr_knee[-H-1:-1]
    past_states_copy.q_fl_ankle = past_observations_t.q_fl_ankle[-H-1:-1]
    past_states_copy.q_fr_ankle = past_observations_t.q_fr_ankle[-H-1:-1]
    past_states_copy.q_rl_ankle = past_observations_t.q_rl_ankle[-H-1:-1]
    past_states_copy.q_rr_ankle = past_observations_t.q_rr_ankle[-H-1:-1]

    past_states_copy.v_fl_hip = past_observations_t.v_fl_hip[-H-1:-1]
    past_states_copy.v_fr_hip = past_observations_t.v_fr_hip[-H-1:-1]
    past_states_copy.v_rl_hip = past_observations_t.v_rl_hip[-H-1:-1]
    past_states_copy.v_rr_hip = past_observations_t.v_rr_hip[-H-1:-1]
    past_states_copy.v_fl_knee = past_observations_t.v_fl_knee[-H-1:-1]
    past_states_copy.v_fr_knee = past_observations_t.v_fr_knee[-H-1:-1]
    past_states_copy.v_rl_knee = past_observations_t.v_rl_knee[-H-1:-1]
    past_states_copy.v_rr_knee = past_observations_t.v_rr_knee[-H-1:-1]
    past_states_copy.v_fl_ankle = past_observations_t.v_fl_ankle[-H-1:-1]
    past_states_copy.v_fr_ankle = past_observations_t.v_fr_ankle[-H-1:-1]
    past_states_copy.v_rl_ankle = past_observations_t.v_rl_ankle[-H-1:-1]
    past_states_copy.v_rr_ankle = past_observations_t.v_rr_ankle[-H-1:-1]

    past_states_copy.v11 = past_observations_t.v11[-H-1:-1]
    past_states_copy.v12 = past_observations_t.v12[-H-1:-1]
    past_states_copy.v13 = past_observations_t.v13[-H-1:-1]
    past_states_copy.v21 = past_observations_t.v21[-H-1:-1]
    past_states_copy.v22 = past_observations_t.v22[-H-1:-1]
    past_states_copy.v23 = past_observations_t.v23[-H-1:-1]

    return past_states_copy



def get_past_actions_current_obs_old(past_observations_t):

    obs_normalized = past_observations_t.normalized()
    current_obs = obs_normalized.array[-1].flatten()

    actions = jnp.concatenate([past_observations_t.qd_fl_hip[:-1].flatten(), past_observations_t.qd_fr_hip[:-1].flatten(), past_observations_t.qd_rl_hip[:-1].flatten(),
                            past_observations_t.qd_rr_hip[:-1].flatten(), past_observations_t.qd_fl_knee[:-1].flatten(), past_observations_t.qd_fr_knee[:-1].flatten(),
                            past_observations_t.qd_rl_knee[:-1].flatten(), past_observations_t.qd_rr_knee[:-1].flatten(), past_observations_t.qd_fl_ankle[:-1].flatten(),
                            past_observations_t.qd_fr_ankle[:-1].flatten(), past_observations_t.qd_rl_ankle[:-1].flatten(), past_observations_t.qd_rr_ankle[:-1].flatten()], axis=-1)

    return jnp.concatenate([actions, current_obs], axis=-1)

def get_past_actions_current_obs(past_observations_t):

    obs_normalized = past_observations_t.normalized()
    current_obs = obs_normalized.array[-1].flatten()
    a_x = obs_normalized.a_x[:-1].flatten()
    a_y = obs_normalized.a_y[:-1].flatten()
    a_z = obs_normalized.a_z[:-1].flatten()

    actions = jnp.concatenate([obs_normalized.qd_fl_hip[:-1].flatten(), obs_normalized.qd_fr_hip[:-1].flatten(), obs_normalized.qd_rl_hip[:-1].flatten(),
                            obs_normalized.qd_rr_hip[:-1].flatten(), obs_normalized.qd_fl_knee[:-1].flatten(), obs_normalized.qd_fr_knee[:-1].flatten(),
                            obs_normalized.qd_rl_knee[:-1].flatten(), obs_normalized.qd_rr_knee[:-1].flatten(), obs_normalized.qd_fl_ankle[:-1].flatten(),
                            obs_normalized.qd_fr_ankle[:-1].flatten(), obs_normalized.qd_rl_ankle[:-1].flatten(), obs_normalized.qd_rr_ankle[:-1].flatten()], axis=-1)

    return jnp.concatenate([actions, current_obs, a_x, a_y, a_z], axis=-1)

def get_past_actions_obs_actuator_net(actuator_net, past_observations_t):
    obs_normalized = past_observations_t.normalized()
    current_obs = obs_normalized.array[-1].flatten()
    a_x = obs_normalized.a_x[:-1].flatten()
    a_y = obs_normalized.a_y[:-1].flatten()
    a_z = obs_normalized.a_z[:-1].flatten()

    q_pos = jnp.concatenate([past_observations_t.q_fl_hip[:-1], past_observations_t.q_fr_hip[:-1], past_observations_t.q_rl_hip[:-1], past_observations_t.q_rr_hip[:-1],
                            past_observations_t.q_fl_knee[:-1], past_observations_t.q_fr_knee[:-1], past_observations_t.q_rl_knee[:-1], past_observations_t.q_rr_knee[:-1],
                            past_observations_t.q_fl_ankle[:-1], past_observations_t.q_fr_ankle[:-1], past_observations_t.q_rl_ankle[:-1], past_observations_t.q_rr_ankle[:-1]], axis=-1)

    q_pos_d = jnp.concatenate([past_observations_t.qd_fl_hip[:-1], past_observations_t.qd_fr_hip[:-1], past_observations_t.qd_rl_hip[:-1], past_observations_t.qd_rr_hip[:-1],
                            past_observations_t.qd_fl_knee[:-1], past_observations_t.qd_fr_knee[:-1], past_observations_t.qd_rl_knee[:-1], past_observations_t.qd_rr_knee[:-1],
                            past_observations_t.qd_fl_ankle[:-1], past_observations_t.qd_fr_ankle[:-1], past_observations_t.qd_rl_ankle[:-1], past_observations_t.qd_rr_ankle[:-1]], axis=-1)

    velocity = jnp.concatenate([past_observations_t.v_fl_hip[:-1], past_observations_t.v_fr_hip[:-1], past_observations_t.v_rl_hip[:-1], past_observations_t.v_rr_hip[:-1],
                                past_observations_t.v_fl_knee[:-1], past_observations_t.v_fr_knee[:-1], past_observations_t.v_rl_knee[:-1], past_observations_t.v_rr_knee[:-1],
                                past_observations_t.v_fl_ankle[:-1], past_observations_t.v_fr_ankle[:-1], past_observations_t.v_rl_ankle[:-1], past_observations_t.v_rr_ankle[:-1]], axis=-1)

    pos_error = q_pos[2:] - q_pos_d[2:]
    last_pos_error = q_pos[1:-1] - q_pos_d[1:-1]
    last_last_pos_error = q_pos[:-2] - q_pos_d[:-2]
    joint_data = jnp.stack([pos_error, last_pos_error, last_last_pos_error, velocity[2:], velocity[1:-1], velocity[:-2]], axis=-1)

    torques = jax.vmap(jax.vmap(actuator_net.forward))(joint_data)
    torques = clip_torques(torques.reshape(-1, 1, 12))
    torques = normalize_torques(torques)
    actions = torques.flatten()
    return jnp.concatenate([actions, current_obs, a_x, a_y, a_z], axis=-1)


@register_pytree_node_class
class MLPObserver(MLPBase):
    def __init__(self, Lipschitz=False, Lipschitz_ub=1e12, J_Lipschitz_ub=1e12, activation="mish", H_Obs=10, H=5, n_obs=8, n_actions=12, n_states=60, n_priv_states=27,  path=None, naive=False, reduced=False,
                    hidden_scale=[1.5, 1.5, 1.5], seed=0, inference_mode=False, actuator_net=None, SNS=False, dtype=jnp.float32, hard=False, order=1):

        self.H_Obs = H_Obs
        self.H = H
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_priv_states = n_priv_states
        self.naive = naive
        self.reduced = reduced
        self.actuator_net = actuator_net
        # measurements + previous state history estimate + current state estimate + grad sum innovation^2 wrt previous state history estimate + innovation
        if naive:
            input_dim = H_Obs * n_obs
            print(f"Using naive observer - input dim: {input_dim}")

        elif reduced:
            #action_obs_dim = (H_Obs - 3) * n_actions + n_obs + 3 * (H_Obs - 1)
            action_obs_dim = (H_Obs - 1) * n_actions + n_obs + 3 * (H_Obs - 1) 
            prev_est_dim = H * n_states
            dinnovation_dim = H * (n_states + n_actions) + n_actions
            innovation_dim = n_obs - n_actions
            state_pred_dim = n_states
            input_dim = action_obs_dim + dinnovation_dim + state_pred_dim + innovation_dim + prev_est_dim
            print(f"Using reduced observer - input dim: {input_dim}")
        else:
            obs_dim = H_Obs * n_obs
            prev_est_dim = H * n_states
            dinnovation_dim = H * (n_states) - n_actions
            innovation_dim = n_obs - n_actions
            state_pred_dim = n_states
            input_dim = obs_dim + dinnovation_dim + state_pred_dim + innovation_dim + prev_est_dim
            print(f"Using full observer - input dim: {input_dim}")

        output_dim = (n_priv_states * H)
        key = jax.random.PRNGKey(seed)

        if path is None:
            super().__init__(key=key, input_dim=input_dim, output_dim=output_dim,
                             hidden_scale=hidden_scale, Lipschitz=Lipschitz, inference_mode=inference_mode, Lipschitz_ub=Lipschitz_ub, J_Lipschitz_ub=J_Lipschitz_ub, activation=activation, SNS=SNS, dtype=dtype, hard=hard, order=order)
        else:
            with open(path, "rb") as f:
                data = dill.load(f)

                for key, value in data.items():
                    setattr(self, key, value)
                if not hasattr(self, 'Lipschitz_ub'):
                    self.Lipschitz_ub = 1e12
                if not hasattr(self, 'activation'):
                    self.activation = "mish"
                if not hasattr(self, 'hard'):
                    self.hard = False
                if not hasattr(self, 'SNS'):
                    self.SNS = False
                if not hasattr(self, 'J_Lipschitz_ub'):
                    self.J_Lipschitz_ub = 1e12
                if not hasattr(self, 'order'):
                    self.order = 1
            super().__init__(key=None, params=self.params, Lipschitz=self.Lipschitz, inference_mode=self.inference_mode, Lipschitz_ub=self.Lipschitz_ub, J_Lipschitz_ub=self.J_Lipschitz_ub, activation=self.activation, SNS=SNS, dtype=dtype, hard=hard, order=self.order)
            if not hasattr(self, 'Lipschitz_constants'):
                self.Lipschitz_constants = self.calc_Lipschitz_constants()
            if not hasattr(self, 'Lipschitz_scale'):
                self.Lipschitz_scale = jnp.array(10.0, dtype=jnp.float32)

    # def process(self, past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation):
    #     observations_normalized = past_observations_t.normalized()
    #     observations_normalized_ = observations_normalized.array.flatten()
    #     past_states_t_normalized = past_states_t.normalized()
    #     past_states_t_normalized_ = past_states_t_normalized.array[-1].flatten()
    #     past_states_tm1_normalized = past_states_tm1.normalized()
    #     past_states_tm1_normalized_ = past_states_tm1_normalized.array.flatten()
    #     innovation_normalized_ = innovation.flatten()
    #     dinnovation_normalized_ = dinnovation.array.flatten()
    #     return jnp.concatenate([observations_normalized_, past_states_t_normalized_, past_states_tm1_normalized_, innovation_normalized_, dinnovation_normalized_], axis=-1)

    def process_naive(self, past_observations_t):
        observations_normalized = past_observations_t.normalized()
        observations_normalized_ = observations_normalized.array.flatten()
        return observations_normalized_
    
    def process_reduced(self, past_observations_t, past_states_t, past_states_tm1, innovation, dinnovation):
        if self.actuator_net is None:
            past_obs_t_normalized_ = get_past_actions_current_obs(past_observations_t)
            #past_obs_t_normalized_ = get_past_actions_current_obs_old(past_observations_t)
        else:
            past_obs_t_normalized_ = get_past_actions_obs_actuator_net(self.actuator_net, past_observations_t)
        past_states_t_normalized = past_states_t.normalized()
        pred_states_t_normalized_ = past_states_t_normalized.array[-1].flatten()
        past_states_tm1_normalized = past_states_tm1.normalized()
        past_states_tm1_normalized_ = past_states_tm1_normalized.array.flatten()
        innovation_normalized_ = innovation.flatten()
        dinnovation_normalized_0 = dinnovation[0].array.flatten()
        if self.actuator_net is None:
            dinnovation_normalized_1 = dinnovation[1].array.flatten()
            dinnovation_normalized_2 = dinnovation[2].array.flatten()
            return jnp.concatenate([past_obs_t_normalized_, pred_states_t_normalized_, past_states_tm1_normalized_, innovation_normalized_, dinnovation_normalized_0, dinnovation_normalized_1, dinnovation_normalized_2], axis=-1)
        return jnp.concatenate([past_obs_t_normalized_, pred_states_t_normalized_, past_states_tm1_normalized_, innovation_normalized_, dinnovation_normalized_0], axis=-1)

    def process_full(self, past_observations_t, past_states_t, past_states_tm1, innovation, dinnovation):
        past_observations_t_normalized = past_observations_t.normalized()
        past_observations_t_normalized_ = past_observations_t_normalized.array.flatten()
        past_states_t_normalized = past_states_t.normalized()
        pred_states_t_normalized_ = past_states_t_normalized.array[-1].flatten()
        past_states_tm1_normalized = past_states_tm1.normalized()
        past_states_tm1_normalized_ = past_states_tm1_normalized.array.flatten()
        innovation_normalized_ = innovation.flatten()
        dinnovation_normalized_0 = dinnovation[0].array.flatten()
        #dinnovation_normalized_1 = dinnovation[1].array.flatten()
        #dinnovation_normalized_2 = dinnovation[2].array.flatten()
        return jnp.concatenate([past_observations_t_normalized_, pred_states_t_normalized_, past_states_tm1_normalized_, innovation_normalized_, dinnovation_normalized_0], axis=-1)

    def _F_x(self, past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation):
        if self.naive:
            input_normalized = self.process_naive(past_observations_t)
        elif self.reduced:
            past_states_tm1 = overwrite_past_states_tm1(past_observations_t, past_states_tm1, self.H)
            input_normalized = self.process_reduced(past_observations_t, past_states_t, past_states_tm1, innovation, dinnovation)
        else:
            past_states_tm1 = overwrite_past_states_tm1(past_observations_t, past_states_tm1, self.H)
            input_normalized = self.process_full(past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation)
        states = self.forward(input_normalized)
        states = states.reshape((self.H, self.n_priv_states))
        return states
    
    def F_x(self, past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation):
        """lifted_observations mapped to states"""
        if self.naive:
            states_update = self._F_x(past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation)
            past_states_copy = jax.tree.map(lambda x: x, past_states_t)
            return preds_obs_states_map_naive(past_states_copy, past_observations_t, states_update, self.H, past_states_t.scales)
        else:
            d_past_states_unscaled = self._F_x(past_states_t, past_observations_t, past_states_tm1, innovation, dinnovation)
            d_past_states_scaled = 0.1*d_past_states_unscaled
            past_states_copy = jax.tree.map(lambda x: x, past_states_t)
            return preds_obs_states_map(past_states_copy, past_observations_t, d_past_states_scaled, self.H, past_states_t.scales)


    def save(self, path):
        path = path + "_params.npz"
        with open(path, "wb") as f:
            params = {
                "params": self.params,
                "H": self.H,
                "H_Obs": self.H_Obs,
                "n_obs": self.n_obs,
                "n_states": self.n_states,
                "n_actions": self.n_actions,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "n_priv_states": self.n_priv_states,
                "naive": self.naive,
                "reduced": self.reduced,
                "Lipschitz_constants": self.Lipschitz_constants,
                "Lipschitz": self.Lipschitz, 
                "Lipschitz_ub": self.Lipschitz_ub,
                "J_Lipschitz_ub": self.J_Lipschitz_ub,
                "Lipschitz_scale": self.Lipschitz_scale,
                "order": self.order,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "n_layers": self.n_layers, 
                "inference_mode": self.inference_mode, 
                "actuator_net": self.actuator_net
            }

            dill.dump(params, f)
        
    def flatten_static_aux(self):
        aux_data = (self.H, self.H_Obs, self.n_obs, self.n_states, self.n_actions, self.n_priv_states, self.naive, self.reduced, self.actuator_net)
        return aux_data
    
    def unflatten_static_aux(self, aux_data):
        self.H, self.H_Obs, self.n_obs, self.n_states, self.n_actions, self.n_priv_states, self.naive, self.reduced, self.actuator_net = aux_data

@register_pytree_node_class
class NNModel:
    def __init__(self, dynamics: MLPDynamics, observer: MLPObserver):
        self.dynamics = dynamics
        self.observer = observer

    def save(self, path):
        self.dynamics.save(path + "_dynamics")
        self.observer.save(path + "_observer")

    def tree_flatten(self):
        dynamic_children = (self.dynamics, self.observer)
        static_aux = ()
        return dynamic_children, static_aux
    
    @classmethod
    def tree_unflatten(cls, static_aux_data, dynamic_children):
        dynamics, observer = dynamic_children
        obj = cls.__new__(cls)
        object.__setattr__(obj, "dynamics", dynamics)
        object.__setattr__(obj, "observer", observer)
        return obj
    
