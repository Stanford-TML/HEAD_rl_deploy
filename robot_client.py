import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator

import torch
import numpy as np
import argparse
import time
import pybullet as pb
from pynput import keyboard
from scipy.spatial.transform import Rotation, Slerp
from threading import Lock
from mj_kinematics import MjRobot, MjRobot29, G1_LINKS_NO_FEET
from unitree_model import UnitreeRobot, UnitreeRobot29
from utils import heading_zup, quatmultiply, quatconj, axang2quat, rotatepoint
import traceback
import math
from numba import njit
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_tracking.py",
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument("--motion", type=str, default="data/ref_min.pkl",
    help="Motion file for tracking humanoid.")
parser.add_argument("--use_omega", action="store_true", default=False)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument("--obs_horizon", type=int, default=5)
parser.add_argument("--obs_frame_skip", type=int, default=4)
parser.add_argument("--net", type=str, required=True,
    help="Network interface for robot.")

settings = parser.parse_args()

# Setup logger

robot = UnitreeRobot29(dt=1/settings.fps, net=settings.net, low_pd=True)

v = torch.zeros((1,3,13), dtype=torch.float32, device = "cpu")
use_feet = torch.zeros(1,1, dtype=torch.float32, device = "cpu")
vlock = Lock()

R = torch.tensor([[0,-1],
                  [1,0]], dtype=torch.float32, device = "cpu")

wz = Rotation.from_euler("xyz", [0.0, 0.0, 0.02])
wx = Rotation.from_euler("xyz", [0.02, 0.0, 0.0])
wy = Rotation.from_euler("xyz", [0.0, 0.02, 0.0])
v = torch.zeros((1,3,13), dtype=torch.float32, device = "cpu")
target = torch.zeros((1,3,13), dtype=torch.float32, device = "cpu")
use_feet = torch.zeros(1,1, dtype=torch.float32, device = "cpu")
vlock = Lock()
### Helper functions
def on_press(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.esc:
            vlock.release()
            robot.damping_state()
            return False
    except:
        pass
    vlock.release()

def on_release(key):
    pass

def get_q(env):
    q = env.joint_tensor.detach().squeeze().cpu()[env.actuated_dofs.cpu(),0].numpy()
    #print("DEBUG q shape:", q.shape)
    return q

def get_root_pose(env):
    root_pose = env.root_tensor[0,0,:7].detach().cpu().numpy()
    return root_pose[:3], root_pose[3:7]

def process_action(action, env):
    action = action.detach().cpu().squeeze().numpy()
    a = action * env.action_scale.cpu().numpy() + env.action_offset.cpu().numpy()
    return a

def get_gravity_orientation(quaternion, convention='xyzw'):
    if convention == 'wxyz':
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]
    elif convention == 'xyzw':
        qx = quaternion[0]
        qy = quaternion[1]
        qz = quaternion[2]
        qw = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

@njit(cache=True)
def dehead_quaternion_from_gravity(gravity):
    """
    Compute a 'deheaded' (tilt-only) quaternion from a gravity vector.
    
    The computed quaternion rotates the reference down vector [0, 0, -1] to 
    align with the measured gravity. This effectively removes the yaw (heading)
    component, leaving only the tilt (roll and pitch).
    
    Parameters:
        gravity (array-like): 3-element gravity vector (need not be normalized).
        
    Returns:
        np.ndarray: A quaternion in [x, y, z, w] format representing the tilt.
                  When g == [0,0,-1], the identity quaternion is returned.
    """
    # Extract components and compute norm (avoiding np.linalg.norm)
    gx = gravity[0]
    gy = gravity[1]
    gz = gravity[2]
    g_norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    if g_norm < 1e-8:
        # Cannot determine orientation from a zero or near-zero vector.
        raise ValueError("Gravity vector is zero or near-zero.")
    gx /= g_norm
    gy /= g_norm
    gz /= g_norm

    # The reference down vector is [0, 0, -1].
    # Compute dot product: dot = down • g = 0*gx + 0*gy + (-1)*gz = -gz.
    dot = -gz
    # Clamp dot to the interval [-1, 1] to avoid numerical issues.
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    # Angle between g and down.
    angle = math.acos(dot)

    # Compute the rotation axis using the cross product.
    # For down = [0, 0, -1], cross(g, down) = [-gy, gx, 0]
    ax = -gy
    ay = gx
    az = 0.0

    # Compute the norm of the axis.
    axis_norm = math.sqrt(ax * ax + ay * ay + az * az)
    if axis_norm < 1e-6:
        # The vectors are parallel or anti-parallel.
        # When g is aligned with down (i.e. [0,0,-1]), no rotation is needed.
        # When g is opposite to down (i.e. [0,0,1]), use a 180° rotation about [1, 0, 0].
        if dot > 0:  # g is the same as down.
            # Identity quaternion (no tilt)
            out = np.empty(4, dtype=np.float64)
            out[0] = 0.0
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 1.0
            return out
        else:
            out = np.empty(4, dtype=np.float64)
            out[0] = 1.0
            out[1] = 0.0
            out[2] = 0.0
            out[3] = 0.0
            return out

    # Normalize the rotation axis.
    ax /= axis_norm
    ay /= axis_norm
    # az remains zero.

    # Compute half-angle values.
    half_angle = angle * 0.5
    s = math.sin(half_angle)
    w = math.cos(half_angle)
    
    # Build quaternion: [x, y, z, w].
    q = np.empty(4, dtype=np.float64)
    q[0] = ax * s
    q[1] = ay * s
    q[2] = 0.0  # since az is zero
    q[3] = w
    return q

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
TRAINING_PARAMS = dict(
    horizon = 8,
    num_envs = 1024,
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000,
    save_interval = 1000,
    log_interval = 50,
    terminate_reward = -1,
    control_mode="position"
)

def test(env, model):
    model.eval()
    env.eval()
    env.reset()
    ts = time.time()
    global target
    # env.force_init_state(np.array([50]), np.array([125.0])) # 79
    # env.reset_done()
    
    mj_kin = MjRobot29("resources/robots/g1/g1_29dof_rev_1_0.xml", dt=env.step_time, link_names=G1_LINKS_NO_FEET, low_pd=True)
    s = 0
    transit = 100
    q, root_pose = mj_kin.get_robot_state()
    
    q[18] = np.pi/6
    q[25] = np.pi/6
    q[17] = np.pi/6
    q[24] = -np.pi/6
    mj_kin.set_robot_state(q, root_pose)
    obs_queue = env.obs_queue
    while True:
        now = time.time()
        if now - ts > 1./env.fps: # Guarantee 30 Hz
            ts = now
            if s == 0:    
                target[0] = torch.from_numpy(mj_kin.get_tracking_site_with_orn()).float().cpu()[0]
            robot.redis_client.set("current_target", pickle.dumps(target[0].cpu().numpy()))
            mj_kin.set_keypoints(target[0].cpu().numpy())
            
            q, root_pose, dq, omega = mj_kin.get_robot_state(include_vel=True, include_omega=True)
            dq = torch.from_numpy(dq).float().cpu().unsqueeze(0)
            omega = torch.from_numpy(omega).float().cpu().unsqueeze(0)
            
            
            obs_queue.push(target)
            obs_target = []
            obs_root_pose = root_pose.copy()
            
            head_pose = mj_kin.get_head_pose()
            origin = torch.from_numpy(head_pose[:3]).float().cpu().view(1,1,-1)
            orient = torch.from_numpy(head_pose[3:]).float().cpu().view(1,1,-1)
            #origin[...,2] = 0.0
            heading = heading_zup(orient)
            #breakpoint()
            orient_inv = axang2quat(env.up_dir_tensor[:1].cpu(), -heading)
            for i in range(env.obs_horizon):
                current_obs_target = torch.zeros_like(obs_queue[i]).to("cpu")
                current_obs_target[...,:3] = rotatepoint(orient_inv, obs_queue[i][...,:3] - origin)
                
                current_obs_target[...,3:7] = quatmultiply(orient_inv, target[:,:,3:7])
                obs_target.append(current_obs_target)
            g = get_gravity_orientation(root_pose[3:], convention='xyzw')
            obs_root_pose[3:] = dehead_quaternion_from_gravity(g) # there are non-neglectable numerical error......
            obs_root_pose[:3] = np.zeros(3)
            robot.set_pb_robot_state(robot.robot_ref, q, root_pose)

            state_hist = mj_kin.get_state_hist(q, obs_root_pose)
            
            obs, seq_len = env.process_real_observation(obs_target, state_hist, local_obs = True)
            if env.use_feet:
                obs = torch.hstack([obs, use_feet])
                print("use feet:", use_feet)
            
            if settings.use_omega:
                obs = torch.hstack([obs, 0.05*dq, 0.25*omega])
            else:
                obs = torch.hstack([obs, 0.05*dq])
            actions = model.act(obs, seq_len-1)
            
            action = process_action(actions, env)
            #print(action)
            mj_kin.step_physics(action)
            #print("Mujoco Action:", action)
            s += 1
            if s > transit:
                break
            #breakpoint()
    
    mj_kin.use_sim = True
    robot.zero_torque_state()
    robot.set_robot_state(q)
    robot.maintain_state(q, root_pose, link="head") # Wait for activation
    ts = time.time()
    # imu_gs = []
    # mocap_gs = []
    # dqs = []
    # sts = time.time()
    step = 0
    init_step = 0
    robot.redis_client.set("server_ready", "false")
    while not env.request_quit:
        now = time.time()
        if now - ts > 1./env.fps: # Guarantee fps Hz
            ts = now
            if robot.redis_client.get("server_ready").decode() == "true":
                    raw_target = robot.redis_client.get("target")
                    if raw_target is not None:
                        target_ = pickle.loads(robot.redis_client.get("target"))
                        target[0] = torch.from_numpy(target_).float().cpu()
            robot.redis_client.set("current_target", pickle.dumps(target[0].cpu().numpy()))
            q, root_pose, dq = robot.get_robot_state(link="head", include_vel=True)
            dq = torch.from_numpy(dq).float().cpu().unsqueeze(0)
            omega = torch.from_numpy(robot.get_omega()).float().cpu().unsqueeze(0)
            #ts_msg = time.time()
            obs_queue.push(target)
            obs_target = []
            obs_root_pose = root_pose.copy()
            
            head_pos, head_orn = robot.get_head_site()
            origin = torch.from_numpy(head_pos).float().cpu().view(1,1,-1)
            orient = torch.from_numpy(head_orn).float().cpu().view(1,1,-1)
            
            #origin[...,2] = 0.0
            heading = heading_zup(orient)
            #breakpoint()
            orient_inv = axang2quat(env.up_dir_tensor[:1].cpu(), -heading)
            for i in range(env.obs_horizon):
                current_obs_target = torch.zeros_like(obs_queue[i]).to("cpu")
                current_obs_target[...,:3] = rotatepoint(orient_inv, obs_queue[i][...,:3] - origin)
                
                current_obs_target[...,3:7] = quatmultiply(orient_inv, obs_queue[i][...,3:7])
                obs_target.append(current_obs_target)
            g = get_gravity_orientation(robot.get_imu_quat(), convention='wxyz')
            obs_root_pose[3:] = dehead_quaternion_from_gravity(g) # there are non-neglectable numerical error......
            obs_root_pose[:3] = np.zeros(3)
            #ts_obs = time.time()
            state_hist = mj_kin.get_state_hist(q, obs_root_pose)
            obs, seq_len = env.process_real_observation(obs_target, state_hist, local_obs = True)
            #ts_nn = time.time()
            
            if settings.use_omega:
                obs = torch.hstack([obs, 0.05*dq, 0.25*omega])
            else:
                obs = torch.hstack([obs, 0.05*dq])
            
            actions = model.act(obs, seq_len - 1)
            action = process_action(actions, env)
            robot.step_robot(action)
            mj_kin.set_keypoints(target[0].cpu().numpy())
            mj_kin.set_robot_state(q,root_pose)
            step += 1
            print("Time taken:", 1/(time.time()-now))

if __name__ == "__main__":
    if os.path.splitext(settings.config)[-1] in [".pkl", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(
            motion_file = settings.config
        )
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print(TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())
    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        raise NotImplementedError
    print(env_cls, config.env_params)

    num_envs = 1
    if config.env_params["use_feet"]:
        config.env_params["character_model"] = "resources/robots/g1/g1_27dof_rev_1_0_track_feet.xml"
    if settings.motion is not None:
        config.env_params["motion_file"] = [settings.motion]

    config.env_params["obs_tracking_quat"] = True
    config.env_params["track_hands"] = True
    config.env_params["fps"] = settings.fps
    config.env_params["obs_horizon"] = settings.obs_horizon
    config.env_params["obs_frame_skip"] = settings.obs_frame_skip
    env = env_cls(num_envs,
        discriminators=discriminators,
        compute_device=0, 
        render_to=None,
        test1=False, test2=False,
        **config.env_params
    )
    #breakpoint()
    env.episode_length = 500000
    value_dim = len(env.discriminators)+env.rew_dim
    #breakpoint()
    
    goal_dim = env.goal_dim + 29 + (3 if settings.use_omega else 0)
    model = ACModel(env.state_dim, env.act_dim, goal_dim, value_dim,
                    use_rnn=config.training_params["use_rnn"],
                    ob_horizon = config.env_params["ob_horizon"], smaller_nn=False,concate_s_g=False)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device("cpu")
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    settings.ckpt = f"exp_out/{settings.ckpt}"
    if settings.ckpt is not None and os.path.exists(settings.ckpt):
        if os.path.isdir(settings.ckpt):
            ckpt = os.path.join(settings.ckpt, "ckpt")
        else:
            ckpt = settings.ckpt
            settings.ckpt = os.path.dirname(ckpt)
        if os.path.exists(ckpt):
            print("Load model from {}".format(ckpt))
            state_dict = torch.load(ckpt, map_location=torch.device("cpu"))
            #model.load_state_dict(state_dict["model"])
            model.load_actor_dict(state_dict["model"])
    #env.render()
    test(env, model)