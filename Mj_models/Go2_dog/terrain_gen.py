import xml.etree.ElementTree as xml_et
import numpy as np

ROBOT = "go2"
INPUT_SCENE_PATH = "./scene.xml"
OUTPUT_SCENE_PATH = "./scene_new.xml"
import cv2
import noise


# zyx euler angle to quaternion
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


# zyx euler angle to rotation matrix
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


class TerrainGenerator:

    def __init__(self) -> None:
        self.scene = xml_et.parse(INPUT_SCENE_PATH)
        self.root = self.scene.getroot()
        self.worldbody = self.root.find("worldbody")
        self.asset = self.root.find("asset")
        self.sensor = self.root.find("sensor")
        if self.sensor is None:
            self.sensor = xml_et.SubElement(self.root, "sensor")

        # Store terrain box IDs
        self.rough_geom_names = []
        self.rough_geom_count = 0

    # Add Box to scene
    def AddBox(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0],
               size=[0.1, 0.1, 0.1],
               name=None):
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = "box"
        geo.attrib["size"] = list_to_str(0.5 * np.array(size))  # Mujoco uses half sizes
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)
        if name:
            geo.attrib["name"] = name
            # # Track it for sensor generation
            # self.rough_geom_names.append(name)
            # self.rough_geom_count += 1
        return geo

    
    def AddGeometry(self,
               position=[1.0, 0.0, 0.0],
               euler=[0.0, 0.0, 0.0], 
               size=[0.1, 0.1],geo_type="box"):
        
        # geo_type supports "plane", "sphere", "capsule", "ellipsoid", "cylinder", "box"
        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["pos"] = list_to_str(position)
        geo.attrib["type"] = geo_type
        geo.attrib["size"] = list_to_str(
            0.5 * np.array(size))  # half size of box for mujoco
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)

    def AddStairs(self,
                  init_pos=[1.0, 0.0, 0.0],
                  yaw=0.0,
                  width=0.2,
                  height=0.15,
                  length=1.5,
                  stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            stair_name = f"stair_box_{self.rough_geom_count}"
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, height],
                        name=stair_name)


    def AddSuspendStairs(self,
                         init_pos=[1.0, 0.0, 0.0],
                         yaw=1.0,
                         width=0.2,
                         height=0.15,
                         length=1.5,
                         gap=0.1,
                         stair_nums=10):

        local_pos = [0.0, 0.0, -0.5 * height]
        for i in range(stair_nums):
            local_pos[0] += width
            local_pos[2] += height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self.AddBox([x + init_pos[0], y + init_pos[1], local_pos[2]],
                        [0.0, 0.0, yaw],
                        [width, length, abs(height - gap)])

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
                geom_name = f"rough_box_{self.rough_geom_count}"
                self.rough_geom_names.append(geom_name)
                self.rough_geom_count += 1

                self.AddBox(pos, new_box_euler, new_box_size, name=geom_name)

    def AddFootSensorsToTerrain(self, foot_geoms=["FR", "FL", "RR", "RL"], cutoff=1.0):
        for foot in foot_geoms:
            # sensor between foot and floor
            sensor = xml_et.SubElement(self.sensor, "distance")
            sensor.attrib["name"] = f"{foot}_to_floor_dist"
            sensor.attrib["geom1"] = foot
            sensor.attrib["geom2"] = "floor"
            sensor.attrib["cutoff"] = str(cutoff)
            for box in self.rough_geom_names:
                sensor = xml_et.SubElement(self.sensor, "distance")
                sensor.attrib["name"] = f"{foot}_to_{box}_dist"
                sensor.attrib["geom1"] = foot
                sensor.attrib["geom2"] = box
                sensor.attrib["cutoff"] = str(cutoff)

        for foot in foot_geoms:
            # sensor between foot and floor
            sensor = xml_et.SubElement(self.sensor, "normal")
            sensor.attrib["name"] = f"{foot}_to_floor_normal"
            sensor.attrib["geom1"] = foot
            sensor.attrib["geom2"] = "floor"
            sensor.attrib["cutoff"] = str(cutoff)
            for box in self.rough_geom_names:
                sensor = xml_et.SubElement(self.sensor, "normal")
                sensor.attrib["name"] = f"{foot}_to_{box}_normal"
                sensor.attrib["geom1"] = foot
                sensor.attrib["geom2"] = box
                sensor.attrib["cutoff"] = str(cutoff)

        for foot in foot_geoms:
            # sensor between foot and floor
            sensor = xml_et.SubElement(self.sensor, "fromto")
            sensor.attrib["name"] = f"{foot}_to_floor_fromto"
            sensor.attrib["geom1"] = foot
            sensor.attrib["geom2"] = "floor"
            sensor.attrib["cutoff"] = str(cutoff)
            for box in self.rough_geom_names:
                sensor = xml_et.SubElement(self.sensor, "fromto")
                sensor.attrib["name"] = f"{foot}_to_{box}_fromto"
                sensor.attrib["geom1"] = foot
                sensor.attrib["geom2"] = box
                sensor.attrib["cutoff"] = str(cutoff)

    def AddBaseSensorsToTerrain(self, cutoff=1.0):
        names = ["FR_hip_dummy", "FL_hip_dummy", "RR_hip_dummy", "RL_hip_dummy"]

        for name in names:
            sensor = xml_et.SubElement(self.sensor, "distance")
            sensor.attrib["name"] = f"{name}_to_floor_dist"
            sensor.attrib["geom1"] = name
            sensor.attrib["geom2"] = "floor"
            sensor.attrib["cutoff"] = str(cutoff)
            for box in self.rough_geom_names:
                sensor = xml_et.SubElement(self.sensor, "distance")
                sensor.attrib["name"] = f"{name}_to_{box}_dist"
                sensor.attrib["geom1"] = name
                sensor.attrib["geom2"] = box
                sensor.attrib["cutoff"] = str(cutoff)

    def Save(self):
        self.scene.write(OUTPUT_SCENE_PATH)

    
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
                noise_value = noise.pnoise2(x / smooth,
                                            y / smooth,
                                            octaves=perlin_octaves,
                                            persistence=perlin_persistence,
                                            lacunarity=perlin_lacunarity)
                terrain_image[y, x] = int((noise_value + 1) / 2 * 255)

        cv2.imwrite("./MJ_models/Go2_dog/" + output_hfield_image, terrain_image)

        hfield = xml_et.SubElement(self.asset, "hfield")
        hfield.attrib["name"] = "perlin_hfield"
        hfield.attrib["size"] = list_to_str(
            [size[0] / 2.0, size[1] / 2.0, height_scale, negative_height])
        hfield.attrib["file"] = "../" + output_hfield_image

        geo = xml_et.SubElement(self.worldbody, "geom")
        geo.attrib["type"] = "hfield"
        geo.attrib["hfield"] = "perlin_hfield"
        geo.attrib["pos"] = list_to_str(position)
        quat = euler_to_quat(euler[0], euler[1], euler[2])
        geo.attrib["quat"] = list_to_str(quat)


if __name__ == "__main__":
    tg = TerrainGenerator()

    # tg.AddRoughGround(init_pos=[-1.5, -1.5, 0.08],
    #               euler=[0, 0, 0.0],
    #               box_size_rand=[0.15, 0.15, 0.15],
    #               separation=[0.25, 0.25],
    #               nums=[10, 10], 
    #               separation_rand=[0.25, 0.25])
    
    # tg.AddRoughGround(init_pos=[-1.5, -1.5, 0.03],
    #               euler=[0, 0, 0.0],
    #               box_size_rand=[0.1, 0.1, 0.1],
    #               separation=[0.25, 0.25],
    #               nums=[10, 10], 
    #               separation_rand=[0.25, 0.25])
    

    # tg.AddRoughGround(init_pos=[-1.5, -1.5, 0.05],
    #             euler=[0, 0, 0.0],
    #             box_size_rand=[0.1, 0.1, 0.1],
    #             separation=[0.4, 0.4],
    #             nums=[10, 10], 
    #             box_euler_rand=[0, 0, 0],
    #             separation_rand=[0.0, 0.0])


    tg.AddBaseSensorsToTerrain(cutoff=15.)
    tg.AddFootSensorsToTerrain(foot_geoms=["FR", "FL", "RR", "RL"], cutoff=15.)
    tg.Save()