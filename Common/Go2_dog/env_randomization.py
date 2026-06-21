import numpy as np
import mujoco
import jax
import jax.numpy as jnp
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
from .commands import get_random_commands, get_random_locomotion_commands, get_random_pose_commands


def repeat_weights(cost_weights, terminal_cost_weights, constraint_weights, terminal_constraint_weights, constraint_relaxation, terminal_constraint_relaxtion, global_cost_weights, horizon):
    cost_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], horizon-1, axis=0), cost_weights)
    cost_weights = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, :]], axis=0), cost_weights, terminal_cost_weights)

    constraint_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], horizon-1, axis=0), constraint_weights)
    constraint_weights = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, :]], axis=0), constraint_weights, terminal_constraint_weights)

    constraint_relaxation = jax.tree.map(lambda x: jnp.repeat(x[None, :], horizon-1, axis=0), constraint_relaxation)
    constraint_relaxation = jax.tree.map(lambda x, y: jnp.concatenate([x, y[None, :]], axis=0), constraint_relaxation, terminal_constraint_relaxtion)

    global_cost_weights = jax.tree.map(lambda x: jnp.repeat(x[None, :], horizon, axis=0), global_cost_weights)

    return cost_weights, constraint_weights, constraint_relaxation, global_cost_weights

def get_new_commands_(horizon, num_envs_random, num_envs_locomotion, num_envs_random_pose, rng_key):
    key, rng_key = jax.random.split(rng_key)
    rng_key0, rng_key1, rng_key2 = jax.random.split(key, 3)
    # 1/3 fully random, 1/3 random locomotion, 1/3 random pose 

    random_commands_struct = get_random_commands(horizon, num_envs_random, rng_key0)
    locomotion_commands_struct = get_random_locomotion_commands(horizon, num_envs_locomotion, rng_key1)
    random_pose_commands_struct = get_random_pose_commands(horizon, num_envs_random_pose, rng_key2)

    commands_struct = jax.tree.map(
        lambda *args: jnp.concatenate(args, axis=0),
        random_commands_struct,
        locomotion_commands_struct,
        random_pose_commands_struct
    )
    return commands_struct




def reset_states_(self, spawn_points):
    mujoco.mj_resetDataKeyframe(self.mj_models[0], self.mj_data, 0)
    qpos = np.copy(self.mj_data.qpos)
    qpos = np.repeat(qpos[None, :], self.num_envs, axis=0)
    qvel = np.copy(self.mj_data.qvel)
    qvel = np.repeat(qvel[None, :], self.num_envs, axis=0)

    quat_noise_random_envs = np.random.normal(0, 0.4, (self.num_envs_random, 4))
    quat_noise_random_envs /= np.linalg.norm(quat_noise_random_envs, axis=1, keepdims=True)

    # height_noise_random_envs = np.random.uniform(0.2, 0.4, (self.num_envs_random, ))
    # height_noise_locomotion_envs = np.random.uniform(0.15, 0.2, (self.num_envs_locomotion, ))
    # height_noise_random_pose_envs = np.random.uniform(0.15, 0.2, (self.num_envs_random_pose, ))
    
    # random yaw
    yaw_noise_random_envs = np.random.uniform(-np.pi, np.pi, (self.num_envs_locomotion + self.num_envs_random_pose,))
    yaw_quats = R.from_euler('z', yaw_noise_random_envs).as_quat()  # shape: (N, 4), [x, y, z, w]

    # [w, x, y, z] order for mujoco
    yaw_quats = np.roll(yaw_quats, 1) 
    # assign to correct slice
    start_idx = self.num_envs_random
    end_idx = self.num_envs_random + self.num_envs_locomotion + self.num_envs_random_pose
    qpos[start_idx:end_idx, 3:7] = yaw_quats

    qpos[:self.num_envs_random, 3:7] = quat_noise_random_envs
    qpos[:, :3] = spawn_points

    qvel[:self.num_envs_random, 6:] = np.random.normal(0, 0.8, (self.num_envs_random, self.mj_models[0].nv-6))
    qvel[:self.num_envs_random, :3] += np.random.normal(0, 1.0, (self.num_envs_random, 3))
    qvel[:self.num_envs_random, 3:6] += np.random.normal(0, 2.4, (self.num_envs_random, 3))
    #qpos[:self.num_envs_random, 3:6] += np.random.normal(0, 0.6, (self.num_envs_random, 3))
    qpos[:self.num_envs_random, 7:] += np.random.normal(0, 0.3, (self.num_envs_random, self.mj_models[0].nq-7))

    return qpos, qvel

def get_base_xml(self):
    scene_tree = ET.parse(self.scene_xml_path)
    scene_root = scene_tree.getroot()

    robot_tree = ET.parse(self.go2_xml_path)
    robot_root = robot_tree.getroot()

    # Replace <include> with inlined robot XML
    for include in scene_root.findall("include"):
        if "file" in include.attrib:
            scene_root.remove(include)
            for child in list(robot_root):
                scene_root.append(child)
            break

    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        scene_root.insert(0, compiler)
    compiler.attrib["meshdir"] = "./Mj_models/Go2_dog/assets"

    string_xml = ET.tostring(scene_root, encoding="unicode")
    return string_xml

def get_random_xml_spawn_position(self):
    scene_generator = Go2SceneGenerator(self.scene_xml_path, self.go2_xml_path)
    scene_generator.randomize_floor()
    string_xml = scene_generator.get_merged_xml()
    return string_xml, scene_generator.spawn_position

def get_base_xml_spawn_position(self):
    scene_generator = Go2SceneGenerator(self.scene_xml_path, self.go2_xml_path)
    scene_generator.default_floor()
    string_xml = scene_generator.get_merged_xml()
    return string_xml, scene_generator.spawn_position

def get_random_terrain_xml_spawn_position(self):
    scene_generator = Go2SceneGenerator(self.scene_xml_path, self.go2_xml_path)
    rand = np.random.rand()
    if rand < 0.5:
        scene_generator.randomize_scene(terrain_type="rough")
    else:
        scene_generator.randomize_scene(terrain_type="stairs")
    string_xml = scene_generator.get_merged_xml()
    return string_xml, scene_generator.spawn_position

def get_base_xml_terrain_spawn_position(self, terrain_type="flat", seed=None):
    scene_generator = Go2SceneGenerator(self.scene_xml_path, self.go2_xml_path)
    scene_generator.default_scene(terrain_type=terrain_type, seed=seed)
    string_xml = scene_generator.get_merged_xml()
    return string_xml, scene_generator.spawn_position

def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)

    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )

def euler_to_rot(roll, pitch, yaw):
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ],
        dtype=np.float64,
    )

    rot_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


# 2d rotate
def rot2d(x, y, yaw):
    nx = x * np.cos(yaw) - y * np.sin(yaw)
    ny = x * np.sin(yaw) + y * np.cos(yaw)
    return nx, ny


# 3d rotate
def rot3d(pos, euler):
    R = euler_to_rot(euler[0], euler[1], euler[2])
    return R @ pos


def list_to_str(vec):
    return " ".join(str(s) for s in vec)


# change to Go2 scene generator
class Go2SceneGenerator:
    def __init__(self, scene_path, robot_path) -> None:
        self.scene = ET.parse(scene_path)
        self.robot = ET.parse(robot_path) 
        self.scene_root = self.scene.getroot()
        self.robot_root = self.robot.getroot()
        self.worldbody = self.scene_root.find("worldbody")
        self.scene_asset = self.scene_root.find("asset")
        self.sensor = self.scene_root.find("sensor")
        if self.sensor is None:
            self.sensor = ET.SubElement(self.scene_root, "sensor")

        # Store terrain box IDs
        self.rough_geom_names = []
        self.rough_geom_count = 0

    def randomize_floor(self):
        """
        Randomizes the scene by modifying the robot and terrain properties.
        """
        self.randomize_inertial()
        self.randomize_joint_properties()
        #self.randomize_pd_properties()
        self.randomize_foot_friction()
        self.randomize_foot_geometry()

        # Create terrain
        self.create_a_floor(random=True)

        # Add sensors to terrain
        self.AddSensorsToTerrain(cutoff=15.0)

    def default_floor(self):
        """
        Randomizes the scene by modifying the robot and terrain properties.
        """

        # Create terrain
        self.create_a_floor(random=False)

        # Add sensors to terrain
        self.AddSensorsToTerrain(cutoff=15.0)

    def randomize_scene(self, terrain_type="stairs"):
        """
        Randomizes the scene by modifying the robot and terrain properties.
        """
        self.randomize_inertial()
        self.randomize_joint_properties()
        #self.randomize_pd_properties()
        self.randomize_foot_friction()
        self.randomize_foot_geometry()

        # Create terrain
        self.create_terrain(terrain_type=terrain_type)  # or "flat"

        # Add sensors to terrain
        self.AddSensorsToTerrain(cutoff=15.0)

    def default_scene(self, terrain_type="stairs", seed=None):
        """
        Randomizes the scene by modifying the robot and terrain properties.
        """

        if seed is not None:
            np.random.seed(seed)
        # Create terrain
        self.create_terrain(terrain_type=terrain_type, random=False)  # or "flat"

        # Add sensors to terrain
        self.AddSensorsToTerrain(cutoff=15.0)

    def get_merged_xml(self):
        """
        Returns the merged XML string of the scene and robot.
        """
        # Replace <include> with inlined robot XML
        for include in self.scene_root.findall("include"):
            if "file" in include.attrib:
                self.scene_root.remove(include)
                for child in list(self.robot_root):
                    self.scene_root.append(child)
                break
        compiler = self.scene_root.find("compiler")
        if compiler is None:
            compiler = ET.Element("compiler")
            self.scene_root.insert(0, compiler)
        compiler.attrib["meshdir"] = "./Mj_models/Go2_dog/assets"
        return ET.tostring(self.scene_root, encoding="unicode")

    def randomize_inertial(self):
        """
        Randomizes inertial properties of the robot.
        """
        for body in self.robot_root.iter("body"):
            if body.attrib.get("name") == "base_link":
                for inertial in body.iter("inertial"):
                    if "mass" in inertial.attrib:
                        orig_mass = float(inertial.attrib["mass"])
                        minus = 0.025 * orig_mass
                        plus = 0.025 * orig_mass
                        new_mass = np.random.uniform(orig_mass - minus, orig_mass + plus)
                        new_mass = np.clip(new_mass, 0.001, np.inf)
                        inertial.attrib["mass"] = f"{new_mass:.6f}"
                    if "pos" in inertial.attrib:
                        orig_pos = np.fromstring(inertial.attrib["pos"], sep=' ')
                        max_deviation = np.array([0.06, 0.03, 0.015]) * 0.05
                        noise = np.random.randn(3)
                        noise /= np.linalg.norm(noise)
                        distance = np.random.uniform(0, 1)
                        noise *= distance * max_deviation
                        new_pos = orig_pos + noise
                        inertial.attrib["pos"] = " ".join(f"{x:.6f}" for x in new_pos)
            else:
                for inertial in body.iter("inertial"):
                    if "mass" in inertial.attrib:
                        orig_mass = float(inertial.attrib["mass"])
                        mass_scale = 0.025 * orig_mass
                        minus = mass_scale
                        plus = mass_scale
                        new_mass = np.random.uniform(orig_mass-minus, orig_mass+plus)
                        new_mass = np.clip(new_mass, 0.001, np.inf)
                        inertial.attrib["mass"] = f"{new_mass:.6f}"
                    if "pos" in inertial.attrib:
                        max_distance = 0.02 * 0.05
                        orig_pos = np.fromstring(inertial.attrib["pos"], sep=' ')
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        distance = np.random.uniform(0, max_distance)
                        noise = direction * distance
                        new_pos = orig_pos + noise
                        inertial.attrib["pos"] = " ".join(f"{x:.6f}" for x in new_pos)

    def randomize_joint_properties(self):
        """
        Randomizes joint properties of the robot.
        """
        
        for joint in self.robot_root.findall(".//worldbody/joint"):
            # dmp, fr = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.25)
            # joint.attrib["damping"] = f"{dmp:.6f}"
            # joint.attrib["frictionloss"] = f"{fr:.6f}"

            class_name = joint.attrib.get("class", "")
            if "hip" in class_name:
                dmp, fr = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.25)
                armature = np.random.uniform(0.0, 5e-5)
                joint.attrib["damping"] = f"{dmp:.6f}"
                joint.attrib["frictionloss"] = f"{fr:.6f}"
                joint.attrib["armature"] = f"{armature:.6f}"

            if "abduction" in class_name:
                dmp, fr = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.25)
                armature = np.random.uniform(0.0, 5e-5)
                joint.attrib["damping"] = f"{dmp:.6f}"
                joint.attrib["frictionloss"] = f"{fr:.6f}"
                joint.attrib["armature"] = f"{armature:.6f}"

            if "knee" in class_name:
                dmp, fr = np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.25)
                armature = np.random.uniform(0.0, 5e-5)
                joint.attrib["damping"] = f"{dmp:.6f}"
                joint.attrib["frictionloss"] = f"{fr:.6f}"
                joint.attrib["armature"] = f"{armature:.6f}"

    def randomize_pd_properties(self):
        """
        Randomizes position controller properties of the robot.
        """
        for pos in self.robot_root.findall(".//actuator/position"):
            kp = np.uniform(23.0, 27.0)
            kv = np.uniform(2.5, 3.5)
            tc = np.uniform(0.030, 0.050)

            pos.attrib["kp"] = f"{kp:.4f}"
            pos.attrib["kv"] = f"{kv:.4f}"
            pos.attrib["timeconst"] = f"{tc:.4f}"
            
    
    def randomize_foot_friction(self):
        """
        Randomizes foot friction properties of the robot.
        """
        sliding = np.random.uniform(0.2, 1.0)
        scaling = np.random.normal(1.0, 0.2, 2)
        rolling = np.clip(sliding * scaling[0], 0.1, 1.0)
        torsional = np.clip(sliding * scaling[1], 0.1, 1.0)
        for default in self.robot_root.iter("default"):
            if default.attrib.get("class") == "foot":
                for geom in default.iter("geom"):
                    if "friction" in geom.attrib:
                        noise = np.random.normal(0.0, 0.05, 3)
                        sliding_fric = max(0.1, sliding + noise[0])
                        rolling_fric = max(0.005, rolling + noise[1])
                        torsional_fric = max(0.005, torsional + noise[2])
                        geom.attrib["friction"] = " ".join(f"{v:.4f}" for v in [sliding_fric, rolling_fric, torsional_fric])

        # <option

        # for option in self.robot_root.iter("option"):
        #     imp = np.random.uniform(1.0, 20.0)
        #     option.attrib["impratio"] = f"{imp:.4f}"

    def randomize_foot_geometry(self):
        """
        Randomizes foot friction properties of the robot.
        """
        for default in self.robot_root.iter("default"):
            if default.attrib.get("class") == "foot":
                for geom in default.iter("geom"):
                    if "size" in geom.attrib:
                        size = np.fromstring(geom.attrib["size"], sep=' ')
                        size *= np.random.uniform(0.95, 1.05, size.shape)
                        geom.attrib["size"] = " ".join(f"{x:.6f}" for x in size)
    
    def create_a_floor(self, random=True):
            
            self.rotate_floor(random=random)
            spawn = np.random.choice(len(self.floor_spawn_positions))
            self.spawn_position = np.array(self.floor_spawn_positions[spawn])


    def create_terrain(self, terrain_type="flat", random=True):
    
        if terrain_type == "stairs":
            if random:
                depth = np.random.uniform(0.25, 0.4)
                height = np.random.uniform(0.1, 0.15)
            else:
                depth = 0.25
                height = 0.1
            x_bounds, y_bounds = self.AddStairPyramid(
                base_pos=[0., 0., 0.], #[0, 0, -0.5], #
                step_depth=depth,
                step_height=height,
                num_steps=2, #3 
                min_size=8
            )
            self.rotate_floor(stair_x_bounds=x_bounds, stair_y_bounds=y_bounds, random=random)

            # if random:
            rand = np.random.rand()
            if rand < 0.5:
                spawn = np.random.choice(len(self.stairs_spawn_positions))
                self.spawn_position = np.array(self.stairs_spawn_positions[spawn])
            else:
                spawn = np.random.choice(len(self.floor_spawn_positions))
                self.spawn_position = np.array(self.floor_spawn_positions[spawn])
            # else:
                # x = (x_bounds[1] - x_bounds[0]) / 2
                # y = (y_bounds[1] - y_bounds[0]) / 2
                # # find closest spawn position to the center of the pyramid
                # spawn = np.argmin([np.linalg.norm(np.array(pos[:2]) - np.array([x, y])) for pos in self.stairs_spawn_positions])
                # z = self.stairs_spawn_positions[spawn][2]
                # self.spawn_position = np.array([x, y, z])
        elif terrain_type == "rough":
            self.rotate_floor(random=random)
            rpy = self.ground_euler
            self.AddRoughGround(init_pos=[0, 0, 0.05],
                euler=[rpy[0], rpy[1], rpy[2]],
                box_size_rand=[0.3, 0.1, 0.1],
                separation=[0.5, 0.5],
                nums=[3, 5], 
                box_euler_rand=[0.1, 0.1, 0.1],
                separation_rand=[0.1, 0.1])
            rand = np.random.rand()
            spawn = np.random.choice(len(self.rough_ground_spawn_positions))
            self.spawn_position = np.array(self.rough_ground_spawn_positions[spawn])
            if rand > 0.5:
                self.spawn_position[0] += np.random.uniform(-1.0, 1.0)
                self.spawn_position[1] += np.random.uniform(-1.0, 1.0)

        elif terrain_type == "flat":
            # Add a flat floor
            depth = 0.3
            height = 0.15
            x_bounds, y_bounds = self.AddStairPyramid(
                base_pos=[0., 0., -8.0],
                step_depth=depth,
                step_height=height,
                num_steps=15, 
                min_size=8
            )
            self.rotate_floor(random=random)
            spawn = np.random.choice(len(self.floor_spawn_positions))
            self.spawn_position = np.array(self.floor_spawn_positions[spawn])

        elif terrain_type == "perlin":
            # Add a perlin noise terrain
            self.rotate_floor(random=random)
            self.AddPerlinHeighField(
                position=[1.0, 0.0, 0.0],  # position
                euler=[0.0, 0.0, 0.0],  # attitude
                size=[15.0, 15.0],  # width and length
                height_scale=0.6,  # max height
                negative_height=0.2,  # height in the negative direction of z axis
                image_width=640,  # height field image size
                img_height=640,
                smooth=100.0,  # smooth scale
                perlin_octaves=6,  # perlin noise parameter
                perlin_persistence=0.5,
                perlin_lacunarity=2.0,
                output_hfield_image="height_field.png")

            self.spawn_position = np.array([1.0, 0.0, 0.6])  # fixed spawn position for perlin terrain

    def rotate_floor(self, stair_x_bounds=None, stair_y_bounds=None, random=True):
        """
        Randomly rotates the floor in the scene.
        """
        if not hasattr(self, "spawn_positions"):
            self.floor_spawn_positions = []

        for geom in self.scene_root.iter("geom"):
            if geom.attrib.get("name") == "floor":
                axis = np.random.randn(3)
                axis /= np.linalg.norm(axis)

                if random:
                    angle = np.random.uniform(0.0, 0.5)
                else:
                    angle =  0.0 #
                rotation = R.from_rotvec(angle * axis)
                quat = rotation.as_quat()  # (x, y, z, w)
                quat_mujoco = np.roll(quat, 1)  # Convert to Mujoco format (w, x, y, z)
                geom.attrib["quat"] = " ".join(f"{x:.6f}" for x in quat_mujoco)

                normal_world = rotation.apply([0, 0, 1])  # rotated floor normal
                self.ground_euler = rotation.as_euler("xyz", degrees=False)

                # Choose x, y spawn position outside of the stair pyramid bounds
                if random:
                    x, y = np.random.uniform(-8.0, 8.0), np.random.uniform(-8.0, 8.0)
                    if stair_x_bounds is not None and stair_y_bounds is not None:
                        while (stair_x_bounds[0] < x < stair_x_bounds[1] and
                            stair_y_bounds[0] < y < stair_y_bounds[1]):
                            x, y = np.random.uniform(-8.0, 8.0), np.random.uniform(-8.0, 8.0)
                else:
                    # Use fixed position for flat terrain
                    x, y = -6.0, -6.0

                a, b, c = normal_world

                # Solve for z at that (x, y)
                z = -(a * x + b * y) / c
                floor_point = np.array([x, y, z])

                # Move 0.5m up from the floor
                spawn_point = floor_point + 0.5 * np.array([0, 0, 1])
                self.floor_spawn_positions.append(spawn_point)


    def AddRoughGround(self,
                init_pos=[1.0, 0.0, 0.0],
                euler=[0.0, -0.0, 0.0],
                nums=[10, 10],
                box_size=[0.5, 0.5, 0.5],
                box_euler=[0.0, 0.0, 0.0],
                separation=[0.2, 0.2],
                box_size_rand=[0.05, 0.05, 0.05],
                box_euler_rand=[0.2, 0.2, 0.2],
                separation_rand=[0.05, 0.05]):
        local_pos = [0.0, 0.0, -0.5 * box_size[2]]
        for i in range(nums[0]):
            local_pos[0] += separation[0] + separation_rand[0] * np.random.uniform(-1.0, 1.0)
            local_pos[1] = 0.0
            for j in range(nums[1]):
                new_box_size = np.array(box_size) + np.array(
                    box_size_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_box_euler = np.array(box_euler) + np.array(
                    box_euler_rand) * np.random.uniform(-1.0, 1.0, 3)
                new_sep = np.array(separation) + np.array(
                    separation_rand) * np.random.uniform(-1.0, 1.0, 2)

                local_pos[1] += new_sep[1]
                pos = rot3d(local_pos, euler) + np.array(init_pos)

                # Generate unique name for each terrain box
                geom_name = f"box_{self.rough_geom_count}"
                self.rough_geom_names.append(geom_name)
                self.rough_geom_count += 1
                z_pos = pos[2] + new_box_size[2]/2 + 0.5
                if not hasattr(self, "rough_ground_spawn_positions"):
                    self.rough_ground_spawn_positions = [] 
                self.rough_ground_spawn_positions.append((pos[0], pos[1], z_pos))

                self.AddBox(pos, new_box_euler, new_box_size, name=geom_name)
    
    def AddStairPyramid(self,
                        base_pos=[1.0, 0.0, 0.0],
                        step_depth=0.2,
                        step_height=0.1,
                        num_steps=5,
                        min_size=2,
                        yaw=0.0):
        """
        Creates a stair pyramid with one large box per level (instead of one box per tile).
        """
        if not hasattr(self, "stairs_spawn_positions"):
            self.stairs_spawn_positions = []

        max_side = min_size + num_steps - 1

        pyramid_bounds_x = [np.inf, -np.inf]
        pyramid_bounds_y = [np.inf, -np.inf]
        max_z_top = -np.inf

        for level in range(num_steps):
            side_len = max_side - level
            width = length = side_len * step_depth
            height = step_height

            z = base_pos[2] + level * step_height + height / 2
            world_pos = [base_pos[0], base_pos[1], z]
            name = f"box_{self.rough_geom_count}"

            self.AddBox(world_pos, [0.0, 0.0, yaw], [width, length, height], name=name)
            self.rough_geom_names.append(name)
            self.rough_geom_count += 1

            max_z_top = max(max_z_top, z + height / 2)

            min_x = world_pos[0] - width / 2
            max_x = world_pos[0] + width / 2
            min_y = world_pos[1] - length / 2
            max_y = world_pos[1] + length / 2
            spawn_z = z + height / 2 + 0.5

            pyramid_bounds_x = [min(pyramid_bounds_x[0], min_x), max(pyramid_bounds_x[1], max_x)]
            pyramid_bounds_y = [min(pyramid_bounds_y[0], min_y), max(pyramid_bounds_y[1], max_y)]

            # Sample spawn points along edges
            for _ in range(4):
                self.stairs_spawn_positions.append((np.random.uniform(min_x, max_x), min_y, spawn_z))  # bottom
                self.stairs_spawn_positions.append((np.random.uniform(min_x, max_x), max_y, spawn_z))  # top
                self.stairs_spawn_positions.append((min_x, np.random.uniform(min_y, max_y), spawn_z))  # left
                self.stairs_spawn_positions.append((max_x, np.random.uniform(min_y, max_y), spawn_z))  # right

        return pyramid_bounds_x, pyramid_bounds_y


    def AddBox(self,
            position=[1.0, 0.0, 0.0],
            euler=[0.0, 0.0, 0.0],
            size=[0.1, 0.1, 0.1],
            name=None,
            rgba=[0.4, 0.45, 0.5, 1.0]): 
        geo = ET.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))  # Mujoco uses half sizes
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        geo.attrib["material"] = "groundplane" #list_to_str(rgba)  # Add color
        if name:
            geo.attrib["name"] = name
        return geo

    def AddSensorsToTerrain(self, cutoff=15.0):
        foot_geoms=["FL", "FR", "RL", "RR"]
        hip_geoms=["FL_hip_geom", "FR_hip_geom", "RL_hip_geom", "RR_hip_geom"]
        thigh_geoms=["FL_thigh_geom", "FR_thigh_geom", "RL_thigh_geom", "RR_thigh_geom"]
        shank_geoms=["FL_shank_top", "FR_shank_top", "RL_shank_top", "RR_shank_top", 
                     "FL_shank_bottom", "FR_shank_bottom", "RL_shank_bottom", "RR_shank_bottom"]
        base_geoms=["base_box", "base_head_top", "base_head_bottom", "z_dummy"]
        geoms = foot_geoms + hip_geoms + thigh_geoms + shank_geoms + base_geoms
        for name in geoms:
            # sensor between foot and floor
            sensor = ET.SubElement(self.sensor, "distance")
            sensor.attrib["name"] = f"{name}_to_floor_dist"
            sensor.attrib["geom1"] = name
            sensor.attrib["geom2"] = "floor"
            sensor.attrib["cutoff"] = str(cutoff)
            for box in self.rough_geom_names:
                sensor = ET.SubElement(self.sensor, "distance")
                sensor.attrib["name"] = f"{name}_to_{box}_dist"
                sensor.attrib["geom1"] = name
                sensor.attrib["geom2"] = box
                sensor.attrib["cutoff"] = str(cutoff)
        
        # for name in foot_geoms:
        #     # sensor between foot and other objects
        #     sensor = ET.SubElement(self.sensor, "normal")
        #     sensor.attrib["name"] = f"{name}_to_floor_normal"
        #     sensor.attrib["geom1"] = name
        #     sensor.attrib["geom2"] = "floor"
        #     sensor.attrib["cutoff"] = str(cutoff)
        #     for box in self.rough_geom_names:
        #         sensor = ET.SubElement(self.sensor, "normal")
        #         sensor.attrib["name"] = f"{name}_to_{box}_normal"
        #         sensor.attrib["geom1"] = name
        #         sensor.attrib["geom2"] = box
        #         sensor.attrib["cutoff"] = str(cutoff)


    def AddPerlinHeighField(
            self,
            position=[1.0, 0.0, 0.0],  # position
            euler=[0.0, -0.0, 0.0],  # attitude
            size=[1.0, 1.0],  # width and length
            height_scale=0.2,  # max height
            negative_height=0.2,  # height in the negative direction of z axis
            image_width=128,  # height field image size
            img_height=128,
            smooth=100.0,  # smooth scale
            perlin_octaves=6,  # perlin noise parameter
            perlin_persistence=0.5,
            perlin_lacunarity=2.0,
            output_hfield_image="height_field.png"):

        # Generating height field based on perlin noise
        terrain_image = np.zeros((img_height, image_width), dtype=np.uint8)
        for y in range(image_width):
            for x in range(image_width):
                # Perlin noise
                import noise
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)
        import cv2
        cv2.imwrite("./Mj_models/Go2_dog/assets/" + output_hfield_image, terrain_image)

        hfield  = ET.SubElement(self.scene_asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] =  output_hfield_image

        geo = ET.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
                