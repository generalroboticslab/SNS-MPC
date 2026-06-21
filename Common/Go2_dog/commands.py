import jax
import jax.numpy as jnp
from Common.Go2_dog import Go2CommandsStruct
#Costs: [11.065470695495605, 11.66569995880127, 12.889638900756836, 11.997454643249512], Mean: 11.904566049575806, Std: 0.6595708520563747
#Costs: [10.15470027923584, 10.15470027923584, 10.15470027923584, 10.15470027923584], Mean: 10.15470027923584, Std: 0.0 
min_commands = Go2CommandsStruct(
    v_x=jnp.array([-2.0]),
    v_y=jnp.array([-2.0]),
    v_yaw=jnp.array([-jnp.pi]),
    z=jnp.array([0.06]),
    _roll=jnp.array([-jnp.pi]),
    _pitch=jnp.array([-jnp.pi]), 
    _yaw=jnp.array([-jnp.pi]))

max_commands = Go2CommandsStruct(
    v_x=jnp.array([2.0]),
    v_y=jnp.array([2.0]),
    v_yaw=jnp.array([jnp.pi]),
    z=jnp.array([0.8]),
    _roll=jnp.array([jnp.pi]),
    _pitch=jnp.array([jnp.pi]), 
    _yaw=jnp.array([jnp.pi]))

def get_random_commands(horizon, num_envs, key):

    keys = jax.random.split(key, 7)
    

    random_commands = Go2CommandsStruct(v_x=jax.random.uniform(keys[0], (num_envs, ), minval=min_commands.v_x, maxval=max_commands.v_x),
                                    v_y=jax.random.uniform(keys[1], (num_envs, ), minval=min_commands.v_y, maxval=max_commands.v_y),
                                    v_yaw=jax.random.uniform(keys[2], (num_envs, ), minval=min_commands.v_yaw, maxval=max_commands.v_yaw),
                                    z=jax.random.uniform(keys[3], (num_envs, ), minval=min_commands.z, maxval=max_commands.z),
                                    _roll=jax.random.uniform(keys[4], (num_envs, ), minval=min_commands._roll, maxval=max_commands._roll),
                                    _pitch=jax.random.uniform(keys[5], (num_envs, ), minval=min_commands._pitch, maxval=max_commands._pitch),
                                    _yaw=jax.random.uniform(keys[6], (num_envs, ), minval=min_commands._yaw, maxval=max_commands._yaw), 
                                    global_time=jnp.zeros((num_envs, )),
                                    duty_ratio=jnp.ones((num_envs, )),
                                    cadence=jnp.ones((num_envs, )),
                                    swing_height=jnp.ones((num_envs, )) * 0.05,
                                    fl_phase=jnp.zeros((num_envs, )),
                                    fr_phase=jnp.ones((num_envs, )) * 0.5,
                                    rl_phase=jnp.zeros((num_envs, )),
                                    rr_phase=jnp.ones((num_envs, )) * 0.5, 
                                    fl_height_target=jnp.zeros((num_envs, )),
                                    fr_height_target=jnp.zeros((num_envs, )),
                                    rl_height_target=jnp.zeros((num_envs, )),
                                    rr_height_target=jnp.zeros((num_envs, )), 
                                    tripod=jnp.zeros((num_envs, )), 
                                    max_joint_v=jnp.ones((num_envs, )) * 1.0, 
                                    max_foot_height=jnp.ones((num_envs, )) * 0.08, 
                                    pitch_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    pitch_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    roll_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    roll_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    z_err_lb=-jnp.ones((num_envs, )) * 0.02,
                                    z_err_ub=jnp.ones((num_envs, )) * 0.08,
                                    v_x_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_x_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_z_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_z_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_yaw_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_yaw_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    yaw_invariant=jnp.zeros((num_envs, ))) 


    random_commands = jax.tree.map(lambda x: x[..., None].repeat(horizon, axis=1), random_commands)
    return random_commands

def get_random_locomotion_commands(horizon, num_envs, key):

    keys = jax.random.split(key, 9)
    
    # NONE of these are used because the cost weights are set to zero for these terms
    rand = jax.random.uniform(keys[4], (num_envs, ), minval=0.0, maxval=1.0)
    # trot or gallop
    fl_phase = jnp.where(rand < 0.5, 0.0, 0.0)
    fr_phase = jnp.where(rand < 0.5, 0.5, 0.05)
    rl_phase = jnp.where(rand < 0.5, 0.5, 0.4)
    rr_phase = jnp.where(rand < 0.5, 0.0, 0.45)

    swing_height = jax.random.uniform(keys[5], (num_envs, ), minval=0.05, maxval=0.1)
    cadence = jax.random.uniform(keys[6], (num_envs, ), minval=1.0, maxval=3.5)
    duty_ratio = jax.random.uniform(keys[7], (num_envs, ), minval=0.2, maxval=0.8)
    

    random_commands = Go2CommandsStruct(v_x=jax.random.uniform(keys[0], (num_envs, ), minval=min_commands.v_x, maxval=max_commands.v_x),
                                    v_y=jax.random.uniform(keys[1], (num_envs, ), minval=min_commands.v_y, maxval=max_commands.v_y),
                                    v_yaw=jax.random.uniform(keys[2], (num_envs, ), minval=min_commands.v_yaw, maxval=max_commands.v_yaw),
                                    z=jax.random.uniform(keys[3], (num_envs, ), minval=0.2, maxval=0.35),
                                    _roll=jnp.zeros((num_envs, )),
                                    _pitch=jnp.zeros((num_envs, )),
                                    _yaw=jax.random.uniform(keys[8], (num_envs, ), minval=-jnp.pi, maxval=jnp.pi),
                                    global_time=jnp.zeros((num_envs, )),
                                    duty_ratio=duty_ratio,
                                    cadence=cadence,
                                    swing_height=swing_height,
                                    fl_phase=fl_phase,
                                    fr_phase=fr_phase,
                                    rl_phase=rl_phase,
                                    rr_phase=rr_phase,
                                    fl_height_target=jnp.zeros((num_envs, )),
                                    fr_height_target=jnp.zeros((num_envs, )),
                                    rl_height_target=jnp.zeros((num_envs, )),
                                    rr_height_target=jnp.zeros((num_envs, )), 
                                    tripod=jnp.zeros((num_envs, )), 
                                    max_joint_v=jnp.ones((num_envs, )) * 1.0, 
                                    max_foot_height=jnp.ones((num_envs, )) * 0.08, 
                                    pitch_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    pitch_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    roll_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    roll_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    z_err_lb=-jnp.ones((num_envs, )) * 0.02,
                                    z_err_ub=jnp.ones((num_envs, )) * 0.08,
                                    v_x_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_x_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_z_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_z_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_yaw_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_yaw_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    yaw_invariant=jnp.zeros((num_envs, )))  


    random_commands = jax.tree.map(lambda x: x[..., None].repeat(horizon, axis=1), random_commands)
    return random_commands

def get_random_pose_commands(horizon, num_envs, key):

    keys = jax.random.split(key, 7)
    

    random_commands = Go2CommandsStruct(v_x=jnp.zeros((num_envs, )),
                                    v_y=jnp.zeros((num_envs, )),
                                    v_yaw=jnp.zeros((num_envs, )),
                                    z=jax.random.uniform(keys[3], (num_envs, ), minval=0.35, maxval=0.55),
                                    _roll=jax.random.uniform(keys[4], (num_envs, ), minval=-jnp.pi/ 4, maxval=jnp.pi / 4),
                                    _pitch=jax.random.uniform(keys[5], (num_envs, ), minval=-jnp.pi / 2, maxval=jnp.pi / 2),
                                    _yaw=jax.random.uniform(keys[6], (num_envs, ), minval=-jnp.pi, maxval=jnp.pi),
                                    global_time=jnp.zeros((num_envs, )),
                                    duty_ratio=jnp.ones((num_envs, )),
                                    cadence=jnp.ones((num_envs, )),
                                    swing_height=jnp.ones((num_envs, )) * 0.05,
                                    fl_phase=jnp.zeros((num_envs, )),
                                    fr_phase=jnp.ones((num_envs, )) * 0.5,
                                    rl_phase=jnp.zeros((num_envs, )),
                                    rr_phase=jnp.ones((num_envs, )) * 0.5, 
                                    fl_height_target=jnp.zeros((num_envs, )),
                                    fr_height_target=jnp.zeros((num_envs, )),
                                    rl_height_target=jnp.zeros((num_envs, )),
                                    rr_height_target=jnp.zeros((num_envs, )), 
                                    tripod=jnp.zeros((num_envs, )), 
                                    max_joint_v=jnp.ones((num_envs, )) * 1.0, 
                                    max_foot_height=jnp.ones((num_envs, )) * 0.08, 
                                    pitch_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    pitch_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    roll_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    roll_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_lb=-jnp.ones((num_envs, )) * 0.3,
                                    yaw_err_ub=jnp.ones((num_envs, )) * 0.3,
                                    z_err_lb=-jnp.ones((num_envs, )) * 0.02,
                                    z_err_ub=jnp.ones((num_envs, )) * 0.08,
                                    v_x_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_x_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_y_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    v_z_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_z_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_roll_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_lb=-jnp.ones((num_envs, )) * 0.8,
                                    v_pitch_err_ub=jnp.ones((num_envs, )) * 0.8,
                                    v_yaw_err_lb=-jnp.ones((num_envs, )) * 0.6,
                                    v_yaw_err_ub=jnp.ones((num_envs, )) * 0.6,
                                    yaw_invariant=jnp.zeros((num_envs, ))) 


    random_commands = jax.tree.map(lambda x: x[..., None].repeat(horizon, axis=1), random_commands)
    return random_commands


def setup_commands(horizon):
    commands = Go2CommandsStruct(v_x=1.0, 
                                v_y=0.0,
                                v_yaw=0.0,
                                z=0.27,
                                _roll=0.0,
                                _pitch=0.0, 
                                _yaw=0.0)
    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[..., None].repeat(horizon, axis=1), commands)
    return commands


def evaluation_commands(horizon):
    commands = []              

    commands.append(Go2CommandsStruct(v_x=1.0,
                            v_y=0.0,
                            v_yaw=0.0,
                            z=0.27,
                            _roll=0.0,
                            _pitch=0.0, 
                            _yaw=0.0))                          
    
    
    commands.append(Go2CommandsStruct(v_x=0.0,
                                v_y=0.0,
                                v_yaw=0.0,
                                z=0.27,
                                _roll=0.0,
                                _pitch=0.0, 
                                _yaw=0.0))
    
    
    commands.append(Go2CommandsStruct(v_x=-1.0,
                                v_y=0.0,
                                v_yaw=0.0,
                                z=0.27,
                                _roll=0.0,
                                _pitch=0.0, 
                                _yaw=0.0))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
                                v_y=1.0,
                                v_yaw=0.0,
                                z=0.27,
                                _roll=0.0,
                                _pitch=0.0, 
                                _yaw=0.0))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
                            v_y=-1.0,
                            v_yaw=0.0,
                            z=0.27,
                            _roll=0.0,
                            _pitch=0.0, 
                            _yaw=0.0))
    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands




def galloping_commands(horizon):
    commands = []

    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                     v_y=0.0,
    #                     v_yaw=0.0,
    #                     z=0.6,
    #                     _roll=0.0,
    #                     _pitch=3.14/2, 
    #                     _yaw=0.0))
    # 

    # commands.append(Go2CommandsStruct(v_x=0.25,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0))
    # gallop
    # duty_ratio = (0.4 - 0.36) / (1.75 - 2.0) * v_cmd + 0.4 = -0.16 * v_cmd + 0.4
    # commands.append(Go2CommandsStruct(v_x=2.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.1,
    #     duty_ratio=0.36,
    #     cadence=3.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.1, 
    #     rl_phase=0.5, 
    #     rr_phase=0.6))

    commands.append(Go2CommandsStruct(v_x=1.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.05,
        duty_ratio=0.45,
        cadence=2.0, 
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0))

    commands.append(Go2CommandsStruct(v_x=-2.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=3.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
        v_y=-2.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=4.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=4.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    
        # fl_phase: jnp.ndarray = jnp.array(0.0)
    # fr_phase: jnp.ndarray = jnp.array(0.05)
    # rl_phase: jnp.ndarray = jnp.array(0.4)
    # rr_phase: jnp.ndarray = jnp.array(0.45)
    
    # commands.append(Go2CommandsStruct(v_x=-1.0,
    #             v_y=0.0,
    #             v_yaw=0.0,
    #             z=0.27,
    #             _roll=0.0,
    #             _pitch=0.0, 
    #             _yaw=0.0, 
    #             swing_height=0.05))
    

    # commands.append(Go2CommandsStruct(v_x=0.0,
    #         v_y=-1.0,
    #         v_yaw=0.0,
    #         z=0.27,
    #         _roll=0.0,
    #         _pitch=0.0, 
    #         _yaw=0.0, 
    #         swing_height=0.05))
    

    
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                     v_y=0.0,
    #                     v_yaw=0.0,
    #                     z=0.27,
    #                     _roll=0.0,
    #                     _pitch=0.0, 
    #                     _yaw=0.0, 
    #                     swing_height=0.05))
    


    # rearing commands
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                         v_y=0.0,
    #                         v_yaw=0.0,
    #                         z=0.37,
    #                         _roll=0.0,
    #                         _pitch=-1.3, 
    #                         _yaw=0.0))       
    

    





    
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                         v_y=-1.0,
    #                         v_yaw=0.0,
    #                         z=0.27,
    #                         _roll=0.0,
    #                         _pitch=0.0, 
    #                         _yaw=0.0))
    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands


def rearing_commands(horizon):
    commands = []
    names = []
    names.append("rearing")
    commands.append(Go2CommandsStruct(v_x=0.0,
                        v_y=0.0,
                        v_yaw=0.0,
                        z=0.37,
                        _roll=0.0,
                        _pitch=-1.3, 
                        _yaw=0.0, 
                        max_log_R=0.2,
                        max_z=0.02,
                        max_v_xyz=0.8,
                        max_v_rpy=0.8, 
                        yaw_invariant=0.0, 
                        max_joint_v=10.0))
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names


def deploy_commands_trot(commands, v_x_cmd=0.0, v_y_cmd=0.0, v_yaw_cmd=0.0, 
                         max_foot_height=0.15, swing_height=0.1):

    horizon = 37

    def duty_ratio_trot(v_x_cmd):
        return -0.2 * (v_x_cmd - 1.0) + 0.45

    def cadence_trot(v_x_cmd):
        return (v_x_cmd - 1.0) + 2.0
    
    abs_v_x_cmd = jnp.abs(v_x_cmd)
    duty_ratio = duty_ratio_trot(abs_v_x_cmd)
    cadence = cadence_trot(abs_v_x_cmd)

    commands = jax.tree.map(lambda x: x, commands) 
    commands.v_x = jnp.array(v_x_cmd)
    commands.v_y = jnp.array(v_y_cmd)
    commands.v_yaw = jnp.array(v_yaw_cmd)
    commands.max_foot_height = jnp.array(max_foot_height)
    commands.swing_height = jnp.array(swing_height)
    commands.duty_ratio = jnp.array(duty_ratio)
    commands.cadence = jnp.array(cadence)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands


# from functools import partial
# gpus = jax.devices("gpu")
# @partial(jax.jit, device=gpus[0])
# def update_commands(weights, commands, v_x_cmd=0.0, v_y_cmd=0.0, v_yaw_cmd=0.0, 
#                          max_foot_height=0.02, swing_height=-0.02):

#     speed = jnp.sqrt(v_x_cmd ** 2 + v_y_cmd ** 2 + v_yaw_cmd ** 2)
#     trot_in_place = jax.lax.cond(speed < 0.1,
#                                   lambda: 1.0,
#                                   lambda: 0.0)

#     weights.cost_weights.z = jnp.ones_like(weights.cost_weights.z) * 5 
#     weights.cost_weights.log_R = jnp.ones_like(weights.cost_weights.log_R) * 1 
#     weights.cost_weights.v_lin = jnp.ones_like(weights.cost_weights.v_lin) * 3e-2 * trot_in_place  + (1 - trot_in_place) * 5e-2
#     weights.cost_weights.v_ang = jnp.ones_like(weights.cost_weights.v_ang) * 1e-3 
#     weights.cost_weights.nom_joint_q = jnp.ones_like(weights.cost_weights.nom_joint_q) * 1e-2 * trot_in_place
#     weights.cost_weights.nom_joint_v = jnp.ones_like(weights.cost_weights.nom_joint_v) * 1e-8
#     weights.cost_weights.nom_torque = jnp.ones_like(weights.cost_weights.nom_torque) * 1e-5 * trot_in_place + (1 - trot_in_place) * 1e-5
#     weights.cost_weights.gait = jnp.ones_like(weights.cost_weights.gait) * 2 * trot_in_place


#     weights.constraint_weights = jax.tree.map(lambda x: jnp.zeros_like(x), weights.constraint_weights)

#     weights.constraint_weights.stance = weights.constraint_weights.stance + 1e-4 * (1 - trot_in_place)
#     weights.constraint_weights.swing = weights.constraint_weights.swing + 1e-4 * (1 - trot_in_place)
#     weights.constraint_weights.max_foot_height = weights.constraint_weights.max_foot_height + 5e-5 * (1 - trot_in_place)

#     weights.constraint_weights.hip_q_lb = weights.constraint_weights.hip_q_lb + 1e-3 * (1 - trot_in_place)
#     weights.constraint_weights.hip_q_ub = weights.constraint_weights.hip_q_ub + 1e-3 * (1 - trot_in_place)
#     weights.constraint_weights.knee_q_lb = weights.constraint_weights.knee_q_lb + 2e-4 * (1 - trot_in_place)
#     weights.constraint_weights.knee_q_ub = weights.constraint_weights.knee_q_ub + 2e-4 * (1 - trot_in_place)
#     weights.constraint_weights.ankle_q_lb = weights.constraint_weights.ankle_q_lb + 1e-3 * (1 - trot_in_place)
#     weights.constraint_weights.ankle_q_ub = weights.constraint_weights.ankle_q_ub + 2e-3 * (1 - trot_in_place)
#     weights.constraint_weights.hip_v = weights.constraint_weights.hip_v + 1e-5 * (1 - trot_in_place)
#     weights.constraint_weights.knee_v = weights.constraint_weights.knee_v + 1e-5 * (1 - trot_in_place)
#     weights.constraint_weights.ankle_v = weights.constraint_weights.ankle_v + 1e-5 * (1 - trot_in_place)

#     weights.constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), weights.constraint_relaxation)

#     stance_wt, swing_wt, max_foot_height_wt = jax.lax.cond(trot_in_place == 1.0,
#                                                     lambda: (1.0, 1.0, 1.0),
#                                                     lambda: (0.01, 0.01, 0.05))

#     weights.constraint_relaxation.stance = weights.constraint_relaxation.stance * stance_wt
#     weights.constraint_relaxation.swing = weights.constraint_relaxation.swing * swing_wt
#     weights.constraint_relaxation.max_foot_height = weights.constraint_relaxation.max_foot_height * max_foot_height_wt

#     q_wt = jax.lax.cond(trot_in_place == 1.0, 
#                         lambda: 1.0,
#                         lambda: 0.08)
    
#     weights.constraint_relaxation.hip_q_lb = weights.constraint_relaxation.hip_q_lb * q_wt
#     weights.constraint_relaxation.hip_q_ub = weights.constraint_relaxation.hip_q_ub * q_wt
#     weights.constraint_relaxation.knee_q_lb = weights.constraint_relaxation.knee_q_lb * q_wt
#     weights.constraint_relaxation.knee_q_ub = weights.constraint_relaxation.knee_q_ub * q_wt
#     weights.constraint_relaxation.ankle_q_lb = weights.constraint_relaxation.ankle_q_lb * q_wt
#     weights.constraint_relaxation.ankle_q_ub = weights.constraint_relaxation.ankle_q_ub * q_wt
#     weights.constraint_relaxation.hip_v = weights.constraint_relaxation.hip_v * 1.0
#     weights.constraint_relaxation.knee_v = weights.constraint_relaxation.knee_v * 1.0
#     weights.constraint_relaxation.ankle_v = weights.constraint_relaxation.ankle_v * 2.0

#     commands = jax.tree.map(lambda x: x, commands) 
#     commands.v_x = jnp.ones_like(commands.v_x) * v_x_cmd
#     commands.v_y = jnp.ones_like(commands.v_y) * v_y_cmd
#     commands.v_yaw = jnp.ones_like(commands.v_yaw) * v_yaw_cmd
#     commands.max_foot_height = jnp.ones_like(commands.max_foot_height) * max_foot_height
#     commands.swing_height = jnp.ones_like(commands.swing_height) * swing_height
#     commands.duty_ratio = jnp.ones_like(commands.duty_ratio) * 0.45 #duty_ratio
#     commands.cadence = jnp.ones_like(commands.cadence) * 2.0 #cadence
#     return weights, commands

from functools import partial
gpus = jax.devices("gpu")
@partial(jax.jit, device=gpus[0])
def update_commands(weights, commands, v_x_cmd=0.0, v_y_cmd=0.0, v_yaw_cmd=0.0, 
                         max_foot_height=0.02, swing_height=-0.02):

    # speed = jnp.sqrt(v_x_cmd ** 2 + v_y_cmd ** 2 + v_yaw_cmd ** 2)
    # trot_in_place = jax.lax.cond(speed < 0.1,
    #                               lambda: 1.0,
    #                               lambda: 0.0)

    # weights.cost_weights.z = jnp.ones_like(weights.cost_weights.z) * 5 
    # weights.cost_weights.log_R = jnp.ones_like(weights.cost_weights.log_R) * 1 
    # weights.cost_weights.v_lin = jnp.ones_like(weights.cost_weights.v_lin) * 3e-2 #* trot_in_place  + (1 - trot_in_place) * 5e-2
    # weights.cost_weights.v_ang = jnp.ones_like(weights.cost_weights.v_ang) * 1e-3 
    # weights.cost_weights.nom_joint_q = jnp.ones_like(weights.cost_weights.nom_joint_q) * 1e-2 * trot_in_place
    # weights.cost_weights.nom_joint_v = jnp.ones_like(weights.cost_weights.nom_joint_v) * 1e-8
    # weights.cost_weights.nom_torque = jnp.ones_like(weights.cost_weights.nom_torque) * 1e-5 * trot_in_place + (1 - trot_in_place) * 1e-5
    # weights.cost_weights.gait = jnp.ones_like(weights.cost_weights.gait) * 2 * trot_in_place


    # weights.constraint_weights = jax.tree.map(lambda x: jnp.zeros_like(x), weights.constraint_weights)

    # weights.constraint_weights.stance = weights.constraint_weights.stance + 1e-4 * (1 - trot_in_place)
    # weights.constraint_weights.swing = weights.constraint_weights.swing + 1e-4 * (1 - trot_in_place)
    # weights.constraint_weights.max_foot_height = weights.constraint_weights.max_foot_height + 5e-5 * (1 - trot_in_place)

    # weights.constraint_weights.hip_q_lb = weights.constraint_weights.hip_q_lb + 1e-3 * (1 - trot_in_place)
    # weights.constraint_weights.hip_q_ub = weights.constraint_weights.hip_q_ub + 1e-3 * (1 - trot_in_place)
    # weights.constraint_weights.knee_q_lb = weights.constraint_weights.knee_q_lb + 2e-4 * (1 - trot_in_place)
    # weights.constraint_weights.knee_q_ub = weights.constraint_weights.knee_q_ub + 2e-4 * (1 - trot_in_place)
    # weights.constraint_weights.ankle_q_lb = weights.constraint_weights.ankle_q_lb + 1e-3 * (1 - trot_in_place)
    # weights.constraint_weights.ankle_q_ub = weights.constraint_weights.ankle_q_ub + 2e-3 * (1 - trot_in_place)
    # weights.constraint_weights.hip_v = weights.constraint_weights.hip_v + 1e-5 * (1 - trot_in_place)
    # weights.constraint_weights.knee_v = weights.constraint_weights.knee_v + 1e-5 * (1 - trot_in_place)
    # weights.constraint_weights.ankle_v = weights.constraint_weights.ankle_v + 1e-5 * (1 - trot_in_place)

    # weights.constraint_relaxation = jax.tree.map(lambda x: jnp.ones_like(x), weights.constraint_relaxation)

    # stance_wt, swing_wt, max_foot_height_wt = jax.lax.cond(trot_in_place == 1.0,
    #                                                 lambda: (1.0, 1.0, 1.0),
    #                                                 lambda: (0.01, 0.01, 0.05))

    # weights.constraint_relaxation.stance = weights.constraint_relaxation.stance * stance_wt
    # weights.constraint_relaxation.swing = weights.constraint_relaxation.swing * swing_wt
    # weights.constraint_relaxation.max_foot_height = weights.constraint_relaxation.max_foot_height * max_foot_height_wt

    # q_wt = jax.lax.cond(trot_in_place == 1.0, 
    #                     lambda: 1.0,
    #                     lambda: 0.08)
    
    # weights.constraint_relaxation.hip_q_lb = weights.constraint_relaxation.hip_q_lb * q_wt
    # weights.constraint_relaxation.hip_q_ub = weights.constraint_relaxation.hip_q_ub * q_wt
    # weights.constraint_relaxation.knee_q_lb = weights.constraint_relaxation.knee_q_lb * q_wt
    # weights.constraint_relaxation.knee_q_ub = weights.constraint_relaxation.knee_q_ub * q_wt
    # weights.constraint_relaxation.ankle_q_lb = weights.constraint_relaxation.ankle_q_lb * q_wt
    # weights.constraint_relaxation.ankle_q_ub = weights.constraint_relaxation.ankle_q_ub * q_wt
    # weights.constraint_relaxation.hip_v = weights.constraint_relaxation.hip_v * 1.0
    # weights.constraint_relaxation.knee_v = weights.constraint_relaxation.knee_v * 1.0
    # weights.constraint_relaxation.ankle_v = weights.constraint_relaxation.ankle_v * 2.0

    commands = jax.tree.map(lambda x: x, commands) 
    commands.v_x = jnp.ones_like(commands.v_x) * v_x_cmd
    commands.v_y = jnp.ones_like(commands.v_y) * v_y_cmd
    commands.v_yaw = jnp.ones_like(commands.v_yaw) * v_yaw_cmd
    commands.max_foot_height = jnp.ones_like(commands.max_foot_height) * max_foot_height
    commands.swing_height = jnp.ones_like(commands.swing_height) * swing_height
    commands.duty_ratio = jnp.ones_like(commands.duty_ratio) * 0.45 #duty_ratio
    commands.cadence = jnp.ones_like(commands.cadence) * 2.0 #cadence
    return weights, commands


def init_deploy_commands_trot(horizon):

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    duty_ratio = 0.45
    cadence = 2.0


    commands = Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6)


    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands



def galloping_evaluation(horizon):

    commands = []
    names = []


    def duty_ratio_rotary(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_rotary(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0

    v_x_cmd = 1.0
    v_y_cmd = 0.0
    duty_ratio = 0.3
    cadence = 3.0

    names.append("gallop_fwd")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.7, #0.5, #0.7 #fun hop
        fr_phase=0.8, #0.6, #0.8
        rl_phase=0.1,#0.1,
        rr_phase=0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.3,
        pitch_err_ub=0.3,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-0.5,
        yaw_err_ub=0.5,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    
    # v_x_cmd = 0.0
    # v_y_cmd = 0.0
    # duty_ratio = 0.45
    # cadence = 2.0

    # names.append("gallop_in_place")
    # commands.append(Go2CommandsStruct(v_x=v_x_cmd,
    #     v_y=v_y_cmd,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0,
    #     swing_height=0.08, #-0.02
    #     duty_ratio=duty_ratio, #0.45
    #     cadence=cadence, #2.0
    #     fl_phase=0.5, 
    #     fr_phase=0.6,
    #     rl_phase=0.1,
    #     rr_phase=0.0, 
    #     max_foot_height=0.15, #0.02
    #     max_joint_v=8.0,
    #     yaw_invariant=0.0,
    #     pitch_err_lb=-0.6,
    #     pitch_err_ub=0.6,
    #     roll_err_lb=-0.3,
    #     roll_err_ub=0.3,
    #     yaw_err_lb=-1.0,
    #     yaw_err_ub=1.0,
    #     z_err_lb=-0.02,
    #     z_err_ub=0.08,
    #     v_x_err_lb=-0.6,
    #     v_x_err_ub=0.6,
    #     v_y_err_lb=-0.6,
    #     v_y_err_ub=0.6,
    #     v_z_err_lb=-0.8,
    #     v_z_err_ub=0.8,
    #     v_roll_err_lb=-0.8,
    #     v_roll_err_ub=0.8,
    #     v_pitch_err_lb=-0.8,
    #     v_pitch_err_ub=0.8,
    #     v_yaw_err_lb=-0.6,
    #     v_yaw_err_ub=0.6))
    


    # v_x_cmd = 1.0
    # v_y_cmd = 0.0
    # v_yaw = 1.0
    # duty_ratio = 0.45
    # cadence = 2.0

    # names.append("gallop_turn_left")
    # commands.append(Go2CommandsStruct(v_x=v_x_cmd,
    #     v_y=v_y_cmd,
    #     v_yaw=v_yaw,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0,
    #     swing_height=0.08, #-0.02
    #     duty_ratio=duty_ratio, #0.45
    #     cadence=cadence, #2.0
    #     fl_phase=0.5, 
    #     fr_phase=0.6,
    #     rl_phase=0.1,
    #     rr_phase=0.0, 
    #     max_foot_height=0.15, #0.02
    #     max_joint_v=8.0,
    #     yaw_invariant=1.0,
    #     pitch_err_lb=-0.6,
    #     pitch_err_ub=0.6,
    #     roll_err_lb=-0.3,
    #     roll_err_ub=0.3,
    #     yaw_err_lb=-1.0,
    #     yaw_err_ub=1.0,
    #     z_err_lb=-0.02,
    #     z_err_ub=0.08,
    #     v_x_err_lb=-0.6,
    #     v_x_err_ub=0.6,
    #     v_y_err_lb=-0.6,
    #     v_y_err_ub=0.6,
    #     v_z_err_lb=-0.8,
    #     v_z_err_ub=0.8,
    #     v_roll_err_lb=-0.8,
    #     v_roll_err_ub=0.8,
    #     v_pitch_err_lb=-0.8,
    #     v_pitch_err_ub=0.8,
    #     v_yaw_err_lb=-0.6,
    #     v_yaw_err_ub=0.6))


    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands, names




def bounding_evaluation(horizon):

    commands = []
    names = []

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = 0.0
    duty_ratio = 0.45 # Good bounding controller !
    cadence = 2.0

    names.append("bounding_turn")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.32,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=0.0))
    
    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = 0.5
    duty_ratio = 0.4 # Good bounding controller !
    cadence = 3.0

    names.append("bounding_turn")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=0.0))
    

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = 0.0
    duty_ratio = 0.4 # Good bounding controller !
    cadence = 3.0

    names.append("bounding_in_place")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=0.0))
    


    



    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands, names




def standing_in_place(horizon):

    commands = []
    names = []

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = 0.0
    duty_ratio = 0.45 # Good bounding controller !
    cadence = 2.0

    names.append("standing_in_place")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.0, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=0.0))

    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands, names




def tripod_evaluation(horizon):

    commands = []
    names = []

    v_x_cmd = 0.5
    v_y_cmd = 0.0
    v_yaw_cmd = 0.0
    duty_ratio = 0.6
    cadence = 2.25

    names.append("tripod_in_place")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=1.0))
    


    # v_x_cmd = 1.0
    # v_y_cmd = 0.0
    # v_yaw_cmd = 0.0
    # duty_ratio = 0.4 # Good bounding controller !
    # cadence = 3.0

    # names.append("tripod_in_place")
    # commands.append(Go2CommandsStruct(v_x=v_x_cmd,
    #     v_y=v_y_cmd,
    #     v_yaw=v_yaw_cmd,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0,
    #     swing_height=0.1, #-0.02
    #     duty_ratio=duty_ratio, #0.45
    #     cadence=cadence, #2.0
    #     fl_phase=0.0, 
    #     fr_phase=0.0,
    #     rl_phase=0.5,
    #     rr_phase=0.5, 
    #     max_foot_height=0.15, #0.02
    #     max_joint_v=8.0,
    #     yaw_invariant=0.0,
    #     pitch_err_lb=-0.6,
    #     pitch_err_ub=0.6,
    #     roll_err_lb=-0.3,
    #     roll_err_ub=0.3,
    #     yaw_err_lb=-1.0,
    #     yaw_err_ub=1.0,
    #     z_err_lb=-0.02,
    #     z_err_ub=0.08,
    #     v_x_err_lb=-0.6,
    #     v_x_err_ub=0.6,
    #     v_y_err_lb=-0.6,
    #     v_y_err_ub=0.6,
    #     v_z_err_lb=-0.8,
    #     v_z_err_ub=0.8,
    #     v_roll_err_lb=-0.8,
    #     v_roll_err_ub=0.8,
    #     v_pitch_err_lb=-0.8,
    #     v_pitch_err_ub=0.8,
    #     v_yaw_err_lb=-0.6,
    #     v_yaw_err_ub=0.6, 
    #     tripod=0.0))
    

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = -0.5
    duty_ratio = 0.6
    cadence = 2.0

    names.append("tripod_turn_right")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=1.0))
    

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    v_yaw_cmd = 0.5
    duty_ratio = 0.6
    cadence = 2.0

    names.append("tripod_turn_left")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=v_yaw_cmd,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=1.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=1.0))
    

    v_x_cmd = 0.5
    v_y_cmd = 0.0
    duty_ratio = 0.6
    cadence = 2.0

    names.append("tripod_slow")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.1, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.0,
        rl_phase=0.5,
        rr_phase=0.5, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=1.0))







    



    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands, names



def trotting_evaluation(horizon):

    commands = []
    names = []
    

    v_x_cmd = 0.0
    v_y_cmd = 0.0
    duty_ratio = 0.45
    cadence = 2.0

    names.append("trot_in_place")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6, 
        tripod=0.0))
    
    v_x_cmd = 1.0
    v_y_cmd = 0.0
    duty_ratio = 0.45
    cadence = 2.0

    names.append("trot_fwd")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))

    v_x_cmd = 0.0
    v_y_cmd = 1.0
    duty_ratio = 0.45
    cadence = 2.0

    names.append("trot_left")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))

    v_x_cmd = 0.0
    v_y_cmd = -1.0
    duty_ratio = 0.45
    cadence = 2.0

    names.append("trot_right")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))

    
    v_x_cmd = -1.0
    v_y_cmd = 0.0
    duty_ratio = 0.45
    cadence = 2.0

    names.append("trot_bwd")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0,
        swing_height=0.08, #-0.02
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, #0.5, 
        fr_phase=0.5, #0.6,
        rl_phase=0.5, #0.1,
        rr_phase=0.0, #0.0, 
        max_foot_height=0.15, #0.02
        max_joint_v=8.0,
        yaw_invariant=0.0,
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))

    
    


    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    #commands = jax.tree.map(lambda x: x.reshape(1, horizon, 1), commands)
    return commands, names




def rough_terrain_commands(horizon):
    commands = []
    names = []

    def duty_ratio_trot(v_x_cmd):
        return -0.2 * (v_x_cmd - 1.0) + 0.45

    def cadence_trot(v_x_cmd):
        return 0.25 * (v_x_cmd - 1.0) + 2.0

    v_x_cmd = 1.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_trot(speed)
    cadence = cadence_trot(speed)

    names.append("rough_terrain_trot")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.35,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.12,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.4,
        pitch_err_ub=0.4,
        roll_err_lb=-0.2,
        roll_err_ub=0.2,
        yaw_err_lb=-0.8,
        yaw_err_ub=0.8,
        z_err_lb=-0.01,
        z_err_ub=0.07,
        v_x_err_lb=-0.4,
        v_x_err_ub=0.4,
        v_y_err_lb=-0.4,
        v_y_err_ub=0.4,
        v_z_err_lb=-0.6,
        v_z_err_ub=0.6,
        v_roll_err_lb=-0.6,
        v_roll_err_ub=0.6,
        v_pitch_err_lb=-0.6,
        v_pitch_err_ub=0.6,
        v_yaw_err_lb=-0.4,
        v_yaw_err_ub=0.4))


    def duty_ratio_rotary(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_rotary(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    duty_ratio = duty_ratio_rotary(v_x_cmd)
    cadence = cadence_rotary(v_x_cmd)
    
    # fl_phase=0.5, 
    # fr_phase=0.6, 
    # rl_phase=0.1, 
    # rr_phase=0.0, 
    
    names.append("rough_terrain_rotary_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.6, 
        fr_phase=0.5, 
        rl_phase=0.0, 
        rr_phase=0.1, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))


    def duty_ratio_pace(v_x_cmd):
        return -0.15 * (v_x_cmd - 1.0) + 0.4 

    def cadence_pace(v_x_cmd):
        return 0.25 * (v_x_cmd - 1.0) + 2.5
        
    v_x_cmd = 1.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_pace(speed)
    cadence = cadence_pace(speed)
    
    
    names.append("rough_terrain_pace")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.4, 
        fr_phase=0.8, 
        rl_phase=0.4, 
        rr_phase=0.8, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    

    def duty_ratio_canter(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.4 

    def cadence_canter(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_canter(speed)
    cadence = cadence_canter(speed)
    
    
    names.append("rough_terrain_canter")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.33, 
        rl_phase=0.33, 
        rr_phase=0.66, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))


    def duty_ratio_gallop(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_gallop(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 0.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_gallop(speed)
    cadence = cadence_gallop(speed)
    
    # fl_phase=0.6, 
    # fr_phase=0.5, 
    # rl_phase=0.1, 
    # rr_phase=0.0, 
    
    names.append("rough_terrain_transverse_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.5, 
        fr_phase=0.6, 
        rl_phase=0.0, 
        rr_phase=0.1, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))



    
   
    
    
    v_x_cmd = 0.0
    duty_ratio = duty_ratio_trot(v_x_cmd)
    cadence = cadence_trot(v_x_cmd)

    names.append("stand_still")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=-0.02,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.02,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    


    def duty_ratio_trot(v_x_cmd):
        return -0.2 * (v_x_cmd - 1.0) + 0.45

    def cadence_trot(v_x_cmd):
        return (v_x_cmd - 1.0) + 2.0
    
    v_x_cmd = 0.5
    duty_ratio = duty_ratio_trot(v_x_cmd)
    cadence = cadence_trot(v_x_cmd)

    names.append("stairs_trot")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.12,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    

    

    def duty_ratio_rotary(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_rotary(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    duty_ratio = duty_ratio_rotary(v_x_cmd)
    cadence = cadence_rotary(v_x_cmd)

    names.append("rough_terrain_rotary_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.5, 
        fr_phase=0.6, 
        rl_phase=0.1, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    
    


    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names
    
def rough_terrain_commands(horizon):
    commands = []
    names = []

    def duty_ratio_trot(v_x_cmd):
        return -0.2 * (v_x_cmd - 1.0) + 0.45

    def cadence_trot(v_x_cmd):
        return 0.25 * (v_x_cmd - 1.0) + 2.0

    v_x_cmd = 1.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_trot(speed)
    cadence = cadence_trot(speed)

    names.append("rough_terrain_trot")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.35,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.12,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.4,
        pitch_err_ub=0.4,
        roll_err_lb=-0.2,
        roll_err_ub=0.2,
        yaw_err_lb=-0.8,
        yaw_err_ub=0.8,
        z_err_lb=-0.01,
        z_err_ub=0.07,
        v_x_err_lb=-0.4,
        v_x_err_ub=0.4,
        v_y_err_lb=-0.4,
        v_y_err_ub=0.4,
        v_z_err_lb=-0.6,
        v_z_err_ub=0.6,
        v_roll_err_lb=-0.6,
        v_roll_err_ub=0.6,
        v_pitch_err_lb=-0.6,
        v_pitch_err_ub=0.6,
        v_yaw_err_lb=-0.4,
        v_yaw_err_ub=0.4))


    def duty_ratio_rotary(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_rotary(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    duty_ratio = duty_ratio_rotary(v_x_cmd)
    cadence = cadence_rotary(v_x_cmd)
    
    # fl_phase=0.5, 
    # fr_phase=0.6, 
    # rl_phase=0.1, 
    # rr_phase=0.0, 
    
    names.append("rough_terrain_rotary_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.6, 
        fr_phase=0.5, 
        rl_phase=0.0, 
        rr_phase=0.1, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))


    def duty_ratio_pace(v_x_cmd):
        return -0.15 * (v_x_cmd - 1.0) + 0.4 

    def cadence_pace(v_x_cmd):
        return 0.25 * (v_x_cmd - 1.0) + 2.5
        
    v_x_cmd = 1.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_pace(speed)
    cadence = cadence_pace(speed)
    
    
    names.append("rough_terrain_pace")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.4, 
        fr_phase=0.8, 
        rl_phase=0.4, 
        rr_phase=0.8, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    

    def duty_ratio_canter(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.4 

    def cadence_canter(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_canter(speed)
    cadence = cadence_canter(speed)
    
    
    names.append("rough_terrain_canter")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.33, 
        rl_phase=0.33, 
        rr_phase=0.66, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))


    def duty_ratio_gallop(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_gallop(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 0.0
    v_y_cmd = 0.0
    speed = jnp.sqrt(v_x_cmd**2 + v_y_cmd**2)
    duty_ratio = duty_ratio_gallop(speed)
    cadence = cadence_gallop(speed)
    
    # fl_phase=0.6, 
    # fr_phase=0.5, 
    # rl_phase=0.1, 
    # rr_phase=0.0, 
    
    names.append("rough_terrain_transverse_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=v_y_cmd,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.5, 
        fr_phase=0.6, 
        rl_phase=0.0, 
        rr_phase=0.1, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))



    
   
    
    
    v_x_cmd = 0.0
    duty_ratio = duty_ratio_trot(v_x_cmd)
    cadence = cadence_trot(v_x_cmd)

    names.append("stand_still")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=-0.02,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.02,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    


    def duty_ratio_trot(v_x_cmd):
        return -0.2 * (v_x_cmd - 1.0) + 0.45

    def cadence_trot(v_x_cmd):
        return (v_x_cmd - 1.0) + 2.0
    
    v_x_cmd = 0.5
    duty_ratio = duty_ratio_trot(v_x_cmd)
    cadence = cadence_trot(v_x_cmd)

    names.append("stairs_trot")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.12,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    

    

    def duty_ratio_rotary(v_x_cmd):
        return -0.15 * (v_x_cmd - 2.0) + 0.3 

    def cadence_rotary(v_x_cmd):
        return 0.25 * (v_x_cmd - 2.0) + 3.0
        
    v_x_cmd = 1.5
    duty_ratio = duty_ratio_rotary(v_x_cmd)
    cadence = cadence_rotary(v_x_cmd)

    names.append("rough_terrain_rotary_gallop")
    commands.append(Go2CommandsStruct(v_x=v_x_cmd,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14, 
        swing_height=0.1,
        duty_ratio=duty_ratio, #0.45
        cadence=cadence, #2.0
        fl_phase=0.5, 
        fr_phase=0.6, 
        rl_phase=0.1, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=8.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.6,
        pitch_err_ub=0.6,
        roll_err_lb=-0.3,
        roll_err_ub=0.3,
        yaw_err_lb=-1.0,
        yaw_err_ub=1.0,
        z_err_lb=-0.02,
        z_err_ub=0.08,
        v_x_err_lb=-0.6,
        v_x_err_ub=0.6,
        v_y_err_lb=-0.6,
        v_y_err_ub=0.6,
        v_z_err_lb=-0.8,
        v_z_err_ub=0.8,
        v_roll_err_lb=-0.8,
        v_roll_err_ub=0.8,
        v_pitch_err_lb=-0.8,
        v_pitch_err_ub=0.8,
        v_yaw_err_lb=-0.6,
        v_yaw_err_ub=0.6))
    
    


    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names

def walking_commands(horizon):
    commands = []
    names = []

    names.append("walk_forward")
    commands.append(Go2CommandsStruct(v_x=0.5,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.03,
        duty_ratio=0.75,
        cadence=1.0, 
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.75, 
        rr_phase=0.25, 
        max_foot_height=0.07, 
        max_log_R=0.25))
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names

    



def dial_vs_spline_shooter_commands(horizon):
    commands = []
    names = []

    names.append("trot_forward")
    commands.append(Go2CommandsStruct(v_x=1.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.05,
        duty_ratio=0.45,
        cadence=2.0, 
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0))

    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names



def trotting_commands(horizon):
    commands = []
    names = []

    names.append("trot_forward_high_step")
    commands.append(Go2CommandsStruct(v_x=1.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=3.14/2, 
        swing_height=0.1,
        duty_ratio=0.45,
        cadence=2.0, 
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0, 
        max_foot_height=0.15,
        max_joint_v=12.0,
        yaw_invariant=0.0, 
        pitch_err_lb=-0.4,
        pitch_err_ub=0.4,
        roll_err_lb=-0.4,
        roll_err_ub=0.4,
        yaw_err_lb=-0.1,
        yaw_err_ub=0.1,
        z_err_lb=-0.05,
        z_err_ub=0.05,
        v_x_err_lb=-0.4,
        v_x_err_ub=0.4,
        v_y_err_lb=-0.4,
        v_y_err_ub=0.4,
        v_z_err_lb=-0.3,
        v_z_err_ub=0.3,
        v_roll_err_lb=-0.4,
        v_roll_err_ub=0.4,
        v_pitch_err_lb=-0.4,
        v_pitch_err_ub=0.4,
        v_yaw_err_lb=-0.3,
        v_yaw_err_ub=0.3))

    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=3.14/2, 
    #     swing_height=0.1,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.15,
    #     max_joint_v=8.0,
    #     yaw_invariant=0.0, 
    #     pitch_err_lb=-0.05,
    #     pitch_err_ub=0.05,
    #     roll_err_lb=-0.05,
    #     roll_err_ub=0.05,
    #     yaw_err_lb=-0.5,
    #     yaw_err_ub=0.5,
    #     z_err_lb=-0.02,
    #     z_err_ub=0.05,
    #     v_x_err_lb=-0.4,
    #     v_x_err_ub=0.4,
    #     v_y_err_lb=-0.4,
    #     v_y_err_ub=0.4,
    #     v_z_err_lb=-0.3,
    #     v_z_err_ub=0.3,
    #     v_roll_err_lb=-0.3,
    #     v_roll_err_ub=0.3,
    #     v_pitch_err_lb=-0.3,
    #     v_pitch_err_ub=0.3,
    #     v_yaw_err_lb=-0.3,
    #     v_yaw_err_ub=0.3))
    
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=3.14/2, 
    #     swing_height=0.1,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.15,
    #     max_joint_v=8.0,
    #     yaw_invariant=0.0, 
    #     pitch_err_lb=-0.05,
    #     pitch_err_ub=0.05,
    #     roll_err_lb=-0.05,
    #     roll_err_ub=0.05,
    #     yaw_err_lb=-0.05,
    #     yaw_err_ub=0.05,
    #     z_err_lb=-0.02,
    #     z_err_ub=0.02,
    #     v_x_err_lb=-0.3,
    #     v_x_err_ub=0.3,
    #     v_y_err_lb=-0.3,
    #     v_y_err_ub=0.3,
    #     v_z_err_lb=-0.3,
    #     v_z_err_ub=0.3,
    #     v_roll_err_lb=-0.3,
    #     v_roll_err_ub=0.3,
    #     v_pitch_err_lb=-0.3,
    #     v_pitch_err_ub=0.3,
    #     v_yaw_err_lb=-0.3,
    #     v_yaw_err_ub=0.3))

    # names.append("trot_circle_high_step")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=1.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.08,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.2, 
    #     max_log_R=0.2, 
    #     max_z=0.05,
    #     max_v_xyz=0.3,
    #     max_v_rpy=0.3))  # max_v_xyz and max_v_r

    # names.append("trot_forward_high_step")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.08,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.2, 
    #     max_log_R=0.2, 
    #     max_z=0.05,
    #     max_v_xyz=0.3,
    #     max_v_rpy=0.3))  # max_v_xyz and max_v_r








    # names.append("trot_forward")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.07, 
    #     max_log_R=0.2, 
    #     max_z=0.05, 
    #     max_v_xyz=0.3,
    #     max_v_rpy=0.3))  # max_v_xyz and max_v_r
    

    # names.append("trot_forward_high_step")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.08,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.2, 
    #     max_log_R=0.2))




    


    # names.append("trot_forward")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.07))
    


    # names.append("trot_forward")
    # commands.append(Go2CommandsStruct(v_x=1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.05))
    

    # names.append("trot_backward")
    # commands.append(Go2CommandsStruct(v_x=-1.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.05))
    
    # names.append("trot_in_place")
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.05))
    
    # names.append("trot_right")
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #     v_y=-1.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.03,
    #     duty_ratio=0.45,
    #     cadence=2.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.5, 
    #     rl_phase=0.5, 
    #     rr_phase=0.0, 
    #     max_foot_height=0.05))
    
    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands, names

def deployment_commands(horizon):
    commands = []

    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                     v_y=0.0,
    #                     v_yaw=0.0,
    #                     z=0.6,
    #                     _roll=0.0,
    #                     _pitch=3.14/2, 
    #                     _yaw=0.0))
    # 

    # commands.append(Go2CommandsStruct(v_x=0.25,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0))
    # gallop
    # duty_ratio = (0.4 - 0.36) / (1.75 - 2.0) * v_cmd + 0.4 = -0.16 * v_cmd + 0.4
    # commands.append(Go2CommandsStruct(v_x=2.0,
    #     v_y=0.0,
    #     v_yaw=0.0,
    #     z=0.27,
    #     _roll=0.0,
    #     _pitch=0.0, 
    #     _yaw=0.0, 
    #     swing_height=0.1,
    #     duty_ratio=0.36,
    #     cadence=3.0, 
    #     fl_phase=0.0, 
    #     fr_phase=0.1, 
    #     rl_phase=0.5, 
    #     rr_phase=0.6))

    commands.append(Go2CommandsStruct(v_x=1.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.05,
        duty_ratio=0.45,
        cadence=2.0, 
        fl_phase=0.0, 
        fr_phase=0.5, 
        rl_phase=0.5, 
        rr_phase=0.0))

    commands.append(Go2CommandsStruct(v_x=-2.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=3.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
        v_y=-2.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=4.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    commands.append(Go2CommandsStruct(v_x=0.0,
        v_y=0.0,
        v_yaw=0.0,
        z=0.27,
        _roll=0.0,
        _pitch=0.0, 
        _yaw=0.0, 
        swing_height=0.1,
        duty_ratio=0.4,
        cadence=4.0, 
        fl_phase=0.0, 
        fr_phase=0.05, 
        rl_phase=0.4, 
        rr_phase=0.45))
    
    
        # fl_phase: jnp.ndarray = jnp.array(0.0)
    # fr_phase: jnp.ndarray = jnp.array(0.05)
    # rl_phase: jnp.ndarray = jnp.array(0.4)
    # rr_phase: jnp.ndarray = jnp.array(0.45)
    
    # commands.append(Go2CommandsStruct(v_x=-1.0,
    #             v_y=0.0,
    #             v_yaw=0.0,
    #             z=0.27,
    #             _roll=0.0,
    #             _pitch=0.0, 
    #             _yaw=0.0, 
    #             swing_height=0.05))
    

    # commands.append(Go2CommandsStruct(v_x=0.0,
    #         v_y=-1.0,
    #         v_yaw=0.0,
    #         z=0.27,
    #         _roll=0.0,
    #         _pitch=0.0, 
    #         _yaw=0.0, 
    #         swing_height=0.05))
    

    
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                     v_y=0.0,
    #                     v_yaw=0.0,
    #                     z=0.27,
    #                     _roll=0.0,
    #                     _pitch=0.0, 
    #                     _yaw=0.0, 
    #                     swing_height=0.05))
    


    # rearing commands
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                         v_y=0.0,
    #                         v_yaw=0.0,
    #                         z=0.37,
    #                         _roll=0.0,
    #                         _pitch=-1.3, 
    #                         _yaw=0.0))       
    

    





    
    # commands.append(Go2CommandsStruct(v_x=0.0,
    #                         v_y=-1.0,
    #                         v_yaw=0.0,
    #                         z=0.27,
    #                         _roll=0.0,
    #                         _pitch=0.0, 
    #                         _yaw=0.0))
    
    
    commands = jax.tree.map(jnp.array, commands)
    commands = jax.tree.map(lambda x: x.reshape(1), commands)
    commands = jax.tree.map(lambda x: x[None, ...].repeat(horizon, axis=0), commands)
    return commands



   
# def update_gait_commands(commands, dt=0.02):
#     """Update the airtime commands based on the current states and actions."""
#     horizon = commands.global_time.shape[0]
#     global_time = commands.global_time[0]
#     global_time += dt
#     global_times = jnp.linspace(global_time, global_time + dt * (horizon - 1), horizon)
#     phase_times = global_times * commands.cadence[0] * 2 * jnp.pi

#     fl_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.fl_phase, commands.duty_ratio)
#     fr_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.fr_phase, commands.duty_ratio)
#     rl_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.rl_phase, commands.duty_ratio)
#     rr_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.rr_phase, commands.duty_ratio)

#     commands.fl_height_target = fl_height_target
#     commands.fr_height_target = fr_height_target
#     commands.rl_height_target = rl_height_target
#     commands.rr_height_target = rr_height_target

#     commands.global_time = global_times

#     return commands

def step_height(time, footphase, duty_ratio):
    pi = jnp.pi
    angle = (time + pi - footphase) % (2 * pi) - pi
    angle *= 0.5 / (1.0 - duty_ratio)
    angle = jnp.clip(angle, -pi / 2, pi / 2)
    value = jnp.cos(angle)
    outs = jnp.where(jnp.abs(value) < 1e-6, 0.0, value)
    return jnp.where(duty_ratio < 1, outs, 0.0)

def update_gait_commands(commands, dt=0.02):
    """Update the airtime commands based on the current states and actions."""
    horizon = commands.global_time.shape[0]
    global_time = commands.global_time[0]
    global_time += dt
    times = jnp.linspace(0, dt * (horizon - 1), horizon) 
    global_times = global_time + times
    phase_times = global_times * commands.cadence[0] * 2 * jnp.pi

    fl_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.fl_phase, commands.duty_ratio)
    fr_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.fr_phase, commands.duty_ratio)
    rl_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.rl_phase, commands.duty_ratio)
    rr_height_target = commands.swing_height * step_height(phase_times, 2*jnp.pi*commands.rr_phase, commands.duty_ratio)

    commands.fl_height_target = fl_height_target
    commands.fr_height_target = fr_height_target
    commands.rl_height_target = rl_height_target
    commands.rr_height_target = rr_height_target


    commands.global_time = global_times

    v_yaw_cmd = commands.v_yaw[0]
    commands._yaw = commands._yaw[1] + v_yaw_cmd * times
    return commands
