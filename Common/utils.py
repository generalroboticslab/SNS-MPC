import jax.numpy as jnp
import jax
import mujoco
import numpy as np


def log_so3(R):
    """Compute the logarithm map from SO(3) to so(3) in vector form (axis-angle)."""
    trace = jnp.trace(R)
    theta = jnp.arccos(jnp.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    
    # Small-angle approximation
    def small_angle():
        return 0.5 * jnp.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])

    # General case
    def general_case():
        skew = (R - R.T) / (2 * jnp.sin(theta))
        return theta * jnp.array([skew[2, 1], skew[0, 2], skew[1, 0]])

    return jax.lax.cond(jnp.isclose(theta, 0.0), small_angle, general_case)

def rotation_log_l2_cost(R, R_cmd):
    """L2 cost between R and R_cmd using log of SO(3)."""
    R_rel = R_cmd.T @ R
    log_R_rel = log_so3(R_rel)
    return jnp.sum(log_R_rel ** 2)


def rotation_matrix(axis, theta):
    axis = axis / jnp.linalg.norm(axis)
    x, y, z = axis
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1 - c
    return jnp.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
    ]).squeeze()


def world_to_local(quat, x):
    R = rotation_matrix_from_quat(quat)
    return jnp.dot(R.T, x)

def local_to_world(quat, x):
    R = rotation_matrix_from_quat(quat)
    return jnp.dot(R, x)

def normalize(v):
    return v / jnp.linalg.norm(v)


def rotation_matrix_from_quat(quat):
    quat = normalize(quat)
    w, x, y, z = quat
    return jnp.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                      [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                      [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])


def SO3_6d_vector_from_quat(quat):
    R = rotation_matrix_from_quat(quat)
    return jnp.array([
        R[0, 0], R[0, 1], R[0, 2],
        R[1, 0], R[1, 1], R[1, 2]
    ])

# def R_from_SO3_6d_vector(v):
#     v1, v2 = v[:3], v[3:]
#     r1 = normalize(v1)
#     r2 = normalize(v2 - jnp.dot(v2, r1) * r1)
#     r3 = jnp.cross(r1, r2)
#     R = jnp.stack((r1, r2, r3), axis=-1)

#     det = jnp.linalg.det(R)
#     R = jax.lax.cond(det < 0.0, lambda R: R.at[:, 2].set(-R[:, 2]), lambda R: R, R)
#     return R

def R_from_SO3_6d_vector(v):
    v1, v2 = v[:3], v[3:]
    r1 = normalize(v1)

    # Gram-Schmidt with sign correction
    proj = jnp.dot(v2, r1) * r1
    raw_r2 = v2 - proj
    r2 = normalize(raw_r2)

    # Canonicalize sign of r2 to match v2 direction
    sign = jnp.sign(jnp.dot(r2, v2))
    r2 = r2 * sign

    r3 = jnp.cross(r1, r2)
    return jnp.stack((r1, r2, r3), axis=-1)


class Camera(mujoco.MjvCamera):
    def __init__(self, name, model, data, renderer, options=None, lookat=None, distance=None, azimuth=None, elevation=None):
        super().__init__()
        self.model = model
        self.data = data
        self.renderer = renderer
        self.name = name
        self.options = options
        self.I = None
        self.E = None
        self.update_camera(lookat, distance, azimuth, elevation)

    def update_camera(self, lookat=None, distance=None, azimuth=None, elevation=None):
        """Update the camera's position and orientation."""
        if lookat is not None:
            self.lookat = lookat
        if distance is not None:
            self.distance = distance
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            self.elevation = elevation
        self.update_camera_matrices()

    def update_camera_matrices(self):
        """Returns the Intrinsics and Extrinsic matrices of the camera."""
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. It is a stereo camera, so we average over
        # the left and right channels. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.
        if self.options is None:
            self.renderer.update_scene(self.data, self)
        else:
            self.renderer.update_scene(self.data, self, scene_option=self.options)
        pos = np.mean([camera.pos for camera in self.renderer.scene.camera], axis=0)
        z = -np.mean([camera.forward for camera in self.renderer.scene.camera], axis=0)
        y = np.mean([camera.up for camera in self.renderer.scene.camera], axis=0)
        rot = np.vstack((np.cross(y, z), y, z))
        fov = self.model.vis.global_.fovy

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * self.renderer.height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (self.renderer.width - 1) / 2.0
        image[1, 2] =  (self.renderer.height - 1) / 2.0 

        intrinsic = image @ focal

        extrinsic = rotation @ translation 

        self.I = intrinsic
        self.E = extrinsic
    
    def get_depth_meters(self):
        """Returns the depth image in meters."""
        if self.options is not None:
            self.renderer.update_scene(self.data, self, scene_option=self.options)
        else:
            self.renderer.update_scene(self.data, self)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        return depth
    
    def get_depth_image(self):
        """Returns the depth image scaled so that is ready to be visualized as an image."""
        depth = self.get_depth_meters()
        # Shift nearest values to the origin.
        depth -= depth.min()
        # Scale by 2 mean distances of near rays.
        depth /= 2*depth[depth <= 1].mean()
        # Scale to [0, 255]
        pixels = 255*np.clip(depth, 0, 1)
        return pixels.astype(np.uint8)
    
    def get_rgb_image(self):
        """Returns the RGB image."""
        if self.options is not None:
            self.renderer.update_scene(self.data, self, scene_option=self.options)
        else:
            self.renderer.update_scene(self.data, self)
        rgb = self.renderer.render()
        return rgb

    


    