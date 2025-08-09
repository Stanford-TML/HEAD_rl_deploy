
import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import argparse
import time
import pickle
import pybullet as pb
from mj_kinematics import MjRobot, MjRobot29, G1_LINKS_NO_FEET, add_random_rotation_noise
from utils import heading_zup, quatmultiply, quatconj, axang2quat, rotatepoint, get_gravity_orientation, dehead_quaternion_from_gravity
import traceback
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_tracking.py",
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument("--motion", type=str, default="data/ref_min.pkl")
parser.add_argument("--use_omega", action="store_true", default=False)
parser.add_argument("--small", action="store_true", default=False)
parser.add_argument("--obs_horizon", type=int, default=5)
parser.add_argument("--obs_frame_skip", type=int, default=4)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--obs_action", action="store_true", default=False)
settings = parser.parse_args()
# Setup logger

R = torch.tensor([[0,-1],
                  [1,0]], dtype=torch.float32, device = "cpu")

wz = Rotation.from_euler("xyz", [0.0, 0.0, 0.01*np.pi])
wx = Rotation.from_euler("xyz", [0.01*np.pi, 0.0, 0.0])
wy = Rotation.from_euler("xyz", [0.0, 0.01*np.pi, 0.0])
v = torch.zeros((1,3,13), dtype=torch.float32, device = "cpu")
target = torch.zeros((1,3,13), dtype=torch.float32, device = "cpu")
use_feet = torch.zeros(1,1, dtype=torch.float32, device = "cpu")

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
    #env.reset()
    ts = time.time()
    step = 0
    global target
    # env.force_init_state(np.array([79]), np.array([125.0])) # 79
    # env.reset_done()
    mj_kin = MjRobot29("resources/robots/g1/g1_29dof_rev_1_0.xml", dt = 1/settings.fps, link_names=G1_LINKS_NO_FEET, low_pd=True)
    q, root_pose = mj_kin.get_robot_state()
    
    q[18] = np.pi/6
    q[25] = np.pi/6
    q[17] = np.pi/6
    q[24] = -np.pi/6
    
    mj_kin.set_robot_state(q, root_pose)
    init = True
    import redis
    redis_client = redis.Redis("localhost", port=6379, db=0)
    redis_client.set("server_ready", "false")
    obs_queue = env.obs_queue
    # print(mj_kin.get_tracking_site_with_orn())
    # breakpoint()
    mj_kin.attach_redis_client(redis_client)
    last_action = np.zeros(29, dtype=np.float32)
    while True:
        now = time.time()
        if now - ts > 1./settings.fps: # Guarantee fps Hz
            ts = now
            if step == 0:
                target[0] = torch.from_numpy(mj_kin.get_tracking_site_with_orn()).float().cpu()[0]
            else:
                if redis_client.get("server_ready").decode() == "true":
                    raw_target = redis_client.get("target")
                    if raw_target is not None:
                        target_ = pickle.loads(raw_target)
                        target[0] = torch.from_numpy(target_).float().cpu()

                
            redis_client.set("current_target", pickle.dumps(target[0].cpu().numpy()))
            redis_client.set("current_eef", pickle.dumps(mj_kin.get_tracking_site_with_orn()[0]))
            mj_kin.set_keypoints(target[0].cpu().numpy())
            obs_root_pose = root_pose.copy()
            
            
            q, root_pose, dq, omega = mj_kin.get_robot_state(include_vel=True, include_omega=True)
            
            dq = torch.from_numpy(dq).float().cpu().unsqueeze(0)
            omega = torch.from_numpy(omega).float().cpu().unsqueeze(0)
            
            head_pose = mj_kin.get_head_pose()
            origin = torch.from_numpy(head_pose[:3]).float().cpu().view(1,1,-1)
            orient = torch.from_numpy(head_pose[3:]).float().cpu().view(1,1,-1)
            #origin[...,2] = 0.0
            heading = heading_zup(orient)
            #breakpoint()
            orient_inv = axang2quat(env.up_dir_tensor[:1].cpu(), -heading)
            obs_queue.push(target)
            obs_target = [] #target.clone()

            # Key file to change if want to remove mocap. Currently 3 frames from current frame
            for i in range(env.obs_horizon):
                current_obs_target = torch.zeros_like(obs_queue[i]).to("cpu")
                current_obs_target[...,:3] = rotatepoint(orient_inv, obs_queue[i][...,:3] - origin) + torch.randn_like(obs_queue[i][...,:3]) * 0.0    
                current_obs_target[...,3:7] = quatmultiply(orient_inv, obs_queue[i][...,3:7])
                obs_target.append(current_obs_target)
            g = get_gravity_orientation(root_pose[3:])
            obs_root_pose[3:] = dehead_quaternion_from_gravity(g) # there are non-neglectable numerical error......
            obs_root_pose[:3] = np.zeros(3)

            #root_pose[3:] = add_random_rotation_noise(root_pose[3:], 0.01)
            state_hist = mj_kin.get_state_hist(q, obs_root_pose)

                

            # if init:
            #     prev_state = state_hist
            #     init = False
            
            obs, seq_len = env.process_real_observation(obs_target, state_hist, local_obs=True)
            #prev_state = state_hist
            if env.use_feet:
                obs = torch.hstack([obs, use_feet])
                print("use feet:", use_feet)
            
            if settings.use_omega:
                obs = torch.hstack([obs, dq*0.05, omega*0.25])
            
            if settings.obs_action:
                obs = torch.hstack([obs, torch.from_numpy(last_action*0.25).float().cpu().unsqueeze(0)])
            obs = torch.hstack([obs, dq*0.05])
            
            actions = model.act(obs, seq_len-1)
            last_action = actions.detach().cpu().squeeze().numpy()
            action = process_action(actions, env)
            #print(action)
            if init:
                prev_action = action
                init = False
            tsim = time.time()
            mj_kin.step_physics(action)
            prev_action = action
            #print("Mujoco Action:", action)
            step += 1
            tc = time.time()
            # print("Real fps:",1/(tc-ts), tc - tsim)
            #breakpoint()

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
    env.goal_dim += 29 + (3 if settings.use_omega else 0) + (29 if settings.obs_action else 0)
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim,
                    use_rnn=config.training_params["use_rnn"],
                    ob_horizon = config.env_params["ob_horizon"], smaller_nn=settings.small,concate_s_g=settings.small)
    device = torch.device("cpu")
    model.to(device)

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
            print(">>>>>>>>>>>"* 10)
            print("Load model successfully")
            epoch = state_dict["epoch"]
            print("Load epoch {}".format(epoch))
            print(">>>>>>>>>>>"* 10)

    #env.render()
    test(env, model)