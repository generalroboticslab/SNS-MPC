import jax 
import jax.numpy as jnp
import numpy as np
import mujoco
from Common import MJData, MJDataset
from jax.tree_util import register_pytree_node_class
from collections import namedtuple

joint_angles = namedtuple("joint_angles", ["fl_hip", "fl_knee", "fl_ankle", "fr_hip", "fr_knee", "fr_ankle", "rl_hip", "rl_knee", "rl_ankle", "rr_hip", "rr_knee", "rr_ankle"])


@register_pytree_node_class
class Go2MJSensordata(MJData):
    def __init__(self, data, index_map, act_limits=None, torque_limits=None, joint_limits=None, nom_joint_pos=None):
        super().__init__(data, index_map, self.__class__.__name__)
        self.act_limits = act_limits
        self.torque_limits = torque_limits
        self.joint_limits = joint_limits
        self.nom_joint_pos = nom_joint_pos
        self.hip_rel_base = jnp.array([[0.1934, 0.142, 0.0], [0.1934, -0.142, 0.0], [-0.1934, 0.142, 0.0], [-0.1934, -0.142, 0.0]])
        
        self.sdf_to_normal = {"FL_distances": "FL_normals",
                            "FR_distances": "FR_normals",
                            "RL_distances": "RL_normals",
                            "RR_distances": "RR_normals"}

        self.sdf_names = {"sdf_base_main": "base_box_distances",
                            "sdf_base_head_top": "base_head_top_distances",
                            "sdf_base_head_bottom": "base_head_bottom_distances",
                            "sdf_z": "z_dummy_distances",
                            "sdf_fl_foot": "FL_distances",
                            "sdf_fr_foot": "FR_distances",
                            "sdf_rl_foot": "RL_distances",
                            "sdf_rr_foot": "RR_distances",
                            "sdf_fl_hip": "FL_hip_geom_distances",
                            "sdf_fr_hip": "FR_hip_geom_distances",
                            "sdf_rl_hip": "RL_hip_geom_distances",
                            "sdf_rr_hip": "RR_hip_geom_distances",
                            "sdf_fl_thigh": "FL_thigh_geom_distances",
                            "sdf_fr_thigh": "FR_thigh_geom_distances",
                            "sdf_rl_thigh": "RL_thigh_geom_distances",
                            "sdf_rr_thigh": "RR_thigh_geom_distances",
                            "sdf_fl_shank_top": "FL_shank_top_distances",
                            "sdf_fr_shank_top": "FR_shank_top_distances",
                            "sdf_rl_shank_top": "RL_shank_top_distances",
                            "sdf_rr_shank_top": "RR_shank_top_distances",
                            "sdf_fl_shank_bottom": "FL_shank_bottom_distances",
                            "sdf_fr_shank_bottom": "FR_shank_bottom_distances",
                            "sdf_rl_shank_bottom": "RL_shank_bottom_distances",
                            "sdf_rr_shank_bottom": "RR_shank_bottom_distances"}
        
    

    def flatten_static_aux(self):
        return (self.act_limits, self.torque_limits, self.joint_limits, self.nom_joint_pos, self.hip_rel_base, self.sdf_names)
    
    def unflatten_static_aux(self, aux):
        self.act_limits, self.torque_limits, self.joint_limits, self.nom_joint_pos, self.hip_rel_base, self.sdf_names = aux

    @classmethod
    def from_mjmodel(cls, mj_model, shape, dtype=jnp.float32):
        """Create a Go2sensordata instance from a Mujoco model.
            Shape of sensor is automatically passed to the constructor in the data collection module.
        """
        data = jnp.zeros(shape, dtype=dtype)

        foot_geoms=["FL", "FR", "RL", "RR"]
        hip_geoms=["FL_hip_geom", "FR_hip_geom", "RL_hip_geom", "RR_hip_geom"]
        thigh_geoms=["FL_thigh_geom", "FR_thigh_geom", "RL_thigh_geom", "RR_thigh_geom"]
        shank_geoms=["FL_shank_top", "FR_shank_top", "RL_shank_top", "RR_shank_top", 
                     "FL_shank_bottom", "FR_shank_bottom", "RL_shank_bottom", "RR_shank_bottom"]
        base_geoms=["base_box", "base_head_top", "base_head_bottom", "z_dummy"]
        geoms = foot_geoms + hip_geoms + thigh_geoms + shank_geoms + base_geoms

        index_map = {}
        joint_pos_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_hip_pos")]
        joint_vel_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_hip_vel")]
        imu_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_quat")]
        torque_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_hip_torque")]
        base_vel_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "base_vel")]
        base_world_pos_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "base_pos_w")]
        base_world_vel_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "base_vel_w")]

        # z_distance_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "z_to_floor_dist")]

        # foot_world_pos_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_foot_pos_w")]
        # foot_vel_world_adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "FL_foot_vel_w")]

        distance_adrs = []
        for i, geom in enumerate(geoms):
            adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, f"{geom}_to_floor_dist")]
            distance_adrs.append(adr)

        # foot_normal_adrs = []
        # for i, geom in enumerate(foot_geoms):
        #     adr = mj_model.sensor_adr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, f"{geom}_to_floor_normal")]
        #     foot_normal_adrs.append(adr)
        
        index_map = cls._setup_distance_adr_index_map(geoms, distance_adrs, index_map)
        #index_map = cls._setup_foot_normal_adr_index_map(foot_geoms, foot_normal_adrs, index_map)

        index_map = cls._setup_joint_pos_index_map(joint_pos_adr, index_map)
        index_map = cls._setup_joint_vel_index_map(joint_vel_adr, index_map)
        index_map = cls._setup_imu_index_map(imu_adr, index_map)
        index_map = cls._setup_torque_index_map(torque_adr, index_map)
        index_map = cls._setup_base_vel_index_map(base_vel_adr, index_map)
        index_map = cls._setup_base_world_pos_index_map(base_world_pos_adr, index_map)
        index_map = cls._setup_base_world_vel_index_map(base_world_vel_adr, index_map)

        
        index_map = jax.tree.map(lambda x: jnp.asarray(x), index_map)


        joint_limits = jnp.asarray(mj_model.jnt_range[1:, :], dtype=dtype)
        joint_limits = joint_angles(*joint_limits)
        act_limits = jnp.asarray(mj_model.actuator_ctrlrange, dtype=dtype)
        act_limits = joint_angles(*act_limits)
        torque_limits = jnp.asarray(mj_model.actuator_forcerange, dtype=dtype)
        torque_limits = joint_angles(*torque_limits)

        nom_joint_pos = mj_model.key_qpos[0, -12:]
        nom_joint_pos = jnp.asarray(nom_joint_pos, dtype=dtype)
        nom_joint_pos = joint_angles(*nom_joint_pos)

        return cls(data, index_map, act_limits, torque_limits, joint_limits, nom_joint_pos)
    
    def __getattr__(self, name):
        if name in self.sdf_names.keys():
            distance_name = self.sdf_names[name]
            i, j = self.index_map[distance_name]
            distances = self.array[..., i:j]
            min_distance = jnp.min(distances, axis=-1, keepdims=True)
            return min_distance
        elif name in self.index_map:
            i, j = self.index_map[name]
            return self.array[..., i:j]
        raise AttributeError(f"{name} not found")

    @staticmethod
    def _setup_foot_normal_adr_index_map(geoms, normal_adrs, index_map):
        """Setup the index map for normal sensors."""
        delta = normal_adrs[1] - normal_adrs[0]
        for i, geom in enumerate(geoms):
            name = f"{geom}_normals"
            start_adr = normal_adrs[i]
            end_adr = start_adr + delta
            index_map[name] = (start_adr, end_adr)

        return index_map
    
    @staticmethod
    def _setup_distance_adr_index_map(geoms, distance_adrs, index_map):
        """Setup the index map for distance sensors."""
        delta = distance_adrs[1] - distance_adrs[0]
        for i, geom in enumerate(geoms):
            name = f"{geom}_distances"
            start_adr = distance_adrs[i]
            end_adr = start_adr + delta
            index_map[name] = (start_adr, end_adr)

        return index_map
    
    @staticmethod
    def _setup_joint_pos_index_map(joint_pos_adr, index_map):
        """Setup the index map for joint position."""
        index_map["q_legs"] = (joint_pos_adr, joint_pos_adr + 12)

        index_map["q_fl_hip"] = (joint_pos_adr, joint_pos_adr + 1)
        index_map["q_fl_knee"] = (joint_pos_adr + 1, joint_pos_adr + 2)
        index_map["q_fl_ankle"] = (joint_pos_adr + 2, joint_pos_adr + 3)

        index_map["q_fr_hip"] = (joint_pos_adr + 3, joint_pos_adr + 4)
        index_map["q_fr_knee"] = (joint_pos_adr + 4, joint_pos_adr + 5)
        index_map["q_fr_ankle"] = (joint_pos_adr + 5, joint_pos_adr + 6)

        index_map["q_rl_hip"] = (joint_pos_adr + 6, joint_pos_adr + 7)
        index_map["q_rl_knee"] = (joint_pos_adr + 7, joint_pos_adr + 8)
        index_map["q_rl_ankle"] = (joint_pos_adr + 8, joint_pos_adr + 9)

        index_map["q_rr_hip"] = (joint_pos_adr + 9, joint_pos_adr + 10)
        index_map["q_rr_knee"] = (joint_pos_adr + 10, joint_pos_adr + 11)
        index_map["q_rr_ankle"] = (joint_pos_adr + 11, joint_pos_adr + 12)
        return index_map
    
    @staticmethod
    def _setup_joint_vel_index_map(joint_vel_adr, index_map):
        """Setup the index map for joint velocity."""
        index_map["v_legs"] = (joint_vel_adr, joint_vel_adr + 12)

        index_map["v_fl_hip"] = (joint_vel_adr, joint_vel_adr + 1)
        index_map["v_fl_knee"] = (joint_vel_adr + 1, joint_vel_adr + 2)
        index_map["v_fl_ankle"] = (joint_vel_adr + 2, joint_vel_adr + 3)

        index_map["v_fr_hip"] = (joint_vel_adr + 3, joint_vel_adr + 4)
        index_map["v_fr_knee"] = (joint_vel_adr + 4, joint_vel_adr + 5)
        index_map["v_fr_ankle"] = (joint_vel_adr + 5, joint_vel_adr + 6)

        index_map["v_rl_hip"] = (joint_vel_adr + 6, joint_vel_adr + 7)
        index_map["v_rl_knee"] = (joint_vel_adr + 7, joint_vel_adr + 8)
        index_map["v_rl_ankle"] = (joint_vel_adr + 8, joint_vel_adr + 9)

        index_map["v_rr_hip"] = (joint_vel_adr + 9, joint_vel_adr + 10)
        index_map["v_rr_knee"] = (joint_vel_adr + 10, joint_vel_adr + 11)
        index_map["v_rr_ankle"] = (joint_vel_adr + 11, joint_vel_adr + 12)

        return index_map
    
    @staticmethod
    def _setup_imu_index_map(imu_adr, index_map):
        """Setup the index map for IMU."""
        index_map["quat"] = (imu_adr, imu_adr + 4)
        index_map["quat_w"] = (imu_adr, imu_adr + 1)
        index_map["quat_x"] = (imu_adr + 1, imu_adr + 2)
        index_map["quat_y"] = (imu_adr + 2, imu_adr + 3)
        index_map["quat_z"] = (imu_adr + 3, imu_adr + 4)

        index_map["gyro"] = (imu_adr + 4, imu_adr + 7)
        index_map["v_roll"] = (imu_adr + 4, imu_adr + 5)
        index_map["v_pitch"] = (imu_adr + 5, imu_adr + 6)
        index_map["v_yaw"] = (imu_adr + 6, imu_adr + 7)

        index_map["imu_acc"] = (imu_adr + 7, imu_adr + 10)
        index_map["a_x"] = (imu_adr + 7, imu_adr + 8)
        index_map["a_y"] = (imu_adr + 8, imu_adr + 9)
        index_map["a_z"] = (imu_adr + 9, imu_adr + 10)

        return index_map
    
    @staticmethod
    def _setup_torque_index_map(torque_adr, index_map):
        """Setup the index map for torque."""
        index_map["torque_legs"] = (torque_adr, torque_adr + 12)

        index_map["torque_fl_hip"] = (torque_adr, torque_adr + 1)
        index_map["torque_fl_knee"] = (torque_adr + 1, torque_adr + 2)
        index_map["torque_fl_ankle"] = (torque_adr + 2, torque_adr + 3)

        index_map["torque_fr_hip"] = (torque_adr + 3, torque_adr + 4)
        index_map["torque_fr_knee"] = (torque_adr + 4, torque_adr + 5)
        index_map["torque_fr_ankle"] = (torque_adr + 5, torque_adr + 6)

        index_map["torque_rl_hip"] = (torque_adr + 6, torque_adr + 7)
        index_map["torque_rl_knee"] = (torque_adr + 7, torque_adr + 8)
        index_map["torque_rl_ankle"] = (torque_adr + 8, torque_adr + 9)

        index_map["torque_rr_hip"] = (torque_adr + 9, torque_adr + 10)
        index_map["torque_rr_knee"] = (torque_adr + 10, torque_adr + 11)
        index_map["torque_rr_ankle"] = (torque_adr + 11, torque_adr + 12)
        return index_map
    
    @staticmethod
    def _setup_base_vel_index_map(base_vel_adr, index_map):
        """Setup the index map for base velocity."""
        index_map["base_v"] = (base_vel_adr, base_vel_adr + 3)
        index_map["v_x"] = (base_vel_adr, base_vel_adr + 1)
        index_map["v_y"] = (base_vel_adr + 1, base_vel_adr + 2)
        index_map["v_z"] = (base_vel_adr + 2, base_vel_adr + 3)

        return index_map
    
    @staticmethod
    def _setup_base_world_vel_index_map(base_vel_adr, index_map):
        """Setup the index map for base velocity."""
        index_map["base_v_w"] = (base_vel_adr, base_vel_adr + 3)
        index_map["v_x_w"] = (base_vel_adr, base_vel_adr + 1)
        index_map["v_y_w"] = (base_vel_adr + 1, base_vel_adr + 2)
        index_map["v_z_w"] = (base_vel_adr + 2, base_vel_adr + 3)

        return index_map
    
    @staticmethod
    def _setup_base_world_pos_index_map(base_world_pos_adr, index_map):
        """Setup the index map for base world position."""
        index_map["base_xyz"] = (base_world_pos_adr, base_world_pos_adr + 3)
        index_map["x"] = (base_world_pos_adr, base_world_pos_adr + 1)
        index_map["y"] = (base_world_pos_adr + 1, base_world_pos_adr + 2)
        index_map["z"] = (base_world_pos_adr + 2, base_world_pos_adr + 3)

        return index_map
    
    


@register_pytree_node_class
class Go2MJActiondata(MJData):
    def __init__(self, data, index_map):
        super().__init__(data, index_map, self.__class__.__name__)

    @classmethod
    def from_mjmodel(cls, mj_model, shape, dtype=jnp.float32):
        """Create a Go2Actiondata instance from a Mujoco model."""
        data = jnp.zeros(shape, dtype=dtype)
        index_map = {}
        index_map = cls._setup_actuator_index_map(index_map)
        index_map = jax.tree.map(lambda x: jnp.asarray(x), index_map)
        return cls(data, index_map)
    
    @staticmethod
    def _setup_actuator_index_map(index_map):
        """Setup the index map for actuator."""
        index_map["fl_hip"] = (0,  1)
        index_map["fl_knee"] = (1,  2)
        index_map["fl_ankle"] = (2,  3)

        index_map["fr_hip"] = (3,  4)
        index_map["fr_knee"] = (4,  5)
        index_map["fr_ankle"] = (5,  6)

        index_map["rl_hip"] = (6,  7)
        index_map["rl_knee"] = (7,  8)
        index_map["rl_ankle"] = (8,  9)

        index_map["rr_hip"] = (9,  10)
        index_map["rr_knee"] = (10, 11)
        index_map["rr_ankle"] = (11, 12)

        return index_map
    


mj_dataset_cls = MJDataset(action_trajectory=Go2MJActiondata, sensor_trajectory=Go2MJSensordata)

#mj_dataset_cls = MJDataset(sensor_history=Go2MJSensordata, action_history=Go2MJActiondata, sensor_rollouts=Go2MJSensordata, action_rollouts=Go2MJActiondata)

if __name__ == "__main__":
    from Common.runtime_paths import SCENE_TORQUE_XML

    path = SCENE_TORQUE_XML
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)
    sensordata = mj_data.sensordata
    go2_sensordata = Go2MJSensordata.from_mjmodel(mj_model, (64, 10, mj_model.nsensordata))

    print(go2_sensordata.q_fl_hip)

    # print(mj_model.nsensor)
    # print(go2_sensordata.act_limits)
    # print(go2_sensordata.torque_limits)
    # print(go2_sensordata.joint_limits)
    # print(go2_sensordata.nom_joint_pos)
    # print(go2_sensordata.sdf_foot)

    flat, _ = jax.tree.flatten(go2_sensordata)

    #go2_dataset = MJDataset(sensor_history=go2_sensordata, action_history=go2_actiondata, sensor_rollouts=go2_sensordata, action_rollouts=go2_actiondata)

    # flat, _ = jax.tree.flatten(go2_dataset)

    # print(go2_actiondata.index_map)




    


    


    
