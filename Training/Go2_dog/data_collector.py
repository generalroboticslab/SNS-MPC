import numpy as np
import mujoco
from Common.Go2_dog import *
from Common.runtime_paths import GO2_TORQUE_XML, SCENE_TORQUE_XML
from Training import *
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

class ThreadedGo2DataCollector(ThreadedOnlineDataCollectionModule):
    def __init__(self, *args, **kwargs):
        self.go2_xml_path = GO2_TORQUE_XML
        self.scene_xml_path = SCENE_TORQUE_XML

        if 'n_envs' not in kwargs:
            raise ValueError("num_envs must be provided in kwargs")
    
        self.num_envs = kwargs['n_envs']

        num_envs = self.num_envs
        num_envs_random = num_envs // 3
        num_envs_locomotion = num_envs // 3
        num_envs_random_pose = num_envs - num_envs_random - num_envs_locomotion
        self.num_envs_random = num_envs_random
        self.num_envs_locomotion = num_envs_locomotion
        self.num_envs_random_pose = num_envs_random_pose

        super().__init__(*args, **kwargs)


        self.get_new_commands = jax.jit(lambda key: get_new_commands_(self.horizon, 
                                                                    num_envs_random, 
                                                                    num_envs_locomotion, 
                                                                    num_envs_random_pose, 
                                                                    key))
        self.commands = self.reset_commands()


        # mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
        mujoco.mj_fwdPosition(self.mj_models[0], self.mj_data)

        self.n_eval_tasks = 5
    
    def new_mj_model(self):
        new_xml, spawn_position = get_random_xml_spawn_position(self)
        #print(f"new xml: {new_xml}")
        #new_xml = modify_go2_xml(self.scene_xml_path, self.go2_xml_path, scale=0.1)
        model =  mujoco.MjModel.from_xml_string(new_xml)
        model.opt.enableflags = 1
        # damp_ratio = np.random.uniform(0.9, 1.1)
        damp_ratio = 1.0
        model.opt.o_solref = np.array([0.01, damp_ratio])
        model.opt.integrator = 3
        model.opt.timestep = 0.005
        return model, spawn_position

    
    def reset_states(self):
        return reset_states_(self, self.spawn_pos)
    

    def set_weights(self):
        rand_env_weights = setup_weights()

        rand_env_cost_weights, rand_env_constraint_weights, rand_env_constraint_relaxation, global_cost_weights = repeat_weights(*rand_env_weights, self.horizon)
        rand_env_cost_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], self.num_envs, axis=0), rand_env_cost_weights)
        rand_env_constraint_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], self.num_envs, axis=0), rand_env_constraint_weights)
        rand_env_constraint_relaxation = jax.tree.map(lambda x: jnp.repeat(x[None, :], self.num_envs, axis=0), rand_env_constraint_relaxation)
        global_cost_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], self.num_envs, axis=0), global_cost_weights)

        cost_weights = rand_env_cost_weights
        constraint_weights = rand_env_constraint_weights
        constraint_relaxation = rand_env_constraint_relaxation

        self.cost_weights = CostWeights(cost_weights=cost_weights,
                            constraint_weights=constraint_weights,
                            constraint_relaxation=constraint_relaxation, 
                            global_cost_weights=global_cost_weights)
    
    def reset_commands(self):
        self.rng_key, key = jax.random.split(self.rng_key)
        self.commands = self.get_new_commands(key)
        self.commands = jax.tree.map(lambda x: x.reshape(self.num_envs, self.horizon, 1), self.commands)

    def set_evaluation_commands(self):
        self.evaluation_commands = evaluation_commands(self.horizon)

        assert len(self.evaluation_commands) == self.n_eval_tasks, "Number of evaluation tasks should be equal to the number of commands"

    def set_evaluation_weights(self):
        weights = evaluation_weights()
        cost_weight, constraint_weights, constraint_relaxation, global_cost_weights = repeat_weights(*weights, self.horizon)
        cost_weight = jax.tree.map(lambda x: x[None, :], cost_weight)
        constraint_weights = jax.tree.map(lambda x: x[None, :], constraint_weights)
        constraint_relaxation = jax.tree.map(lambda x: x[None, :], constraint_relaxation)
        global_cost_weights = jax.tree.map(lambda x: x[None, :], global_cost_weights)

        weights = CostWeights(cost_weights=cost_weight,
                    constraint_weights=constraint_weights, 
                    constraint_relaxation=constraint_relaxation, 
                    global_cost_weights=global_cost_weights)

        self.evaluation_cost_weights = []
        for i in range(self.n_eval_tasks):
            self.evaluation_cost_weights.append(weights)

    def set_evaluation_state(self):
        mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
        #self.mj_data.qpos[2] += 0.15

    def set_evaluation_mj_model(self):

        base_xml, spawn_position = get_base_xml_spawn_position(self)
        self.mj_models[0] = mujoco.MjModel.from_xml_string(base_xml)

        self.mj_models[0].opt.enableflags = 1
        self.mj_models[0].opt.o_solref = np.array([0.01, 1.0])
        self.mj_models[0].opt.integrator = 3
        self.mj_models[0].opt.timestep = 0.005


    def set_evaluation_task_names(self):
        self.evaluation_task_names = ["forward", "stand_still", "backward",  "left", "right"]
        assert len(self.evaluation_task_names) == self.n_eval_tasks, "Number of evaluation tasks should be equal to the number of commands"
