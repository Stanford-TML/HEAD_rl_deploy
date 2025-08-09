import redis
import numpy as np
import pybullet as pb
from pynput import keyboard
import traceback
from threading import Lock
import pickle
import joblib
import time
from scipy.spatial.transform import Rotation

import os
file_dir = os.path.dirname(os.path.abspath(__file__))

vlock = Lock()
v = np.zeros(1, dtype=np.float32)
def on_press(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.ctrl:
            v[0] = 1
    except Exception as e:
        traceback.print_exc()
    vlock.release()

def on_release(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.ctrl:
            v[0] = 0
    except Exception as e:
        traceback.print_exc()
    vlock.release()

def rotatepoint(q, v):
    # q_v = [v[0], v[1], v[2], 0]
    # return quatmultiply(quatmultiply(q, q_v), quatconj(q))[:-1]
    #
    # https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    q_r = q[3:4]
    q_xyz = q[:3]
    t = 2*np.cross(q_xyz, v)
    return v + q_r * t + np.cross(q_xyz, t)

def heading_zup(quat):
    ref_dir = np.zeros_like(quat[:3])
    ref_dir[0] = 1
    ref_dir = rotatepoint(quat, ref_dir)
    return np.arctan2(ref_dir[1], ref_dir[0])

def set_robot_state(robot, q, root_pos, root_orn):
    idx=0
    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)
        if joint_info[2] == pb.JOINT_REVOLUTE:
            pb.resetJointState(robot, i, q[idx])
            idx += 1
        else:
            pass
    pb.resetBasePositionAndOrientation(robot, root_pos, root_orn)

def get_eef_pos(robot):
    eef_pos = np.zeros((3,13))
    eef_pos[0,:3] = pb.getLinkState(robot, 17)[0]
    eef_pos[0,3:7] = pb.getLinkState(robot, 17)[1]
    eef_pos[1,:3] = pb.getLinkState(robot, 28)[0]
    eef_pos[1,3:7] = pb.getLinkState(robot, 28)[1]
    eef_pos[2,:3] = pb.getLinkState(robot, 36)[0]
    eef_pos[2,3:7] = pb.getLinkState(robot, 36)[1]
    return eef_pos

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--motion_id", type=int, default=5)
args = parser.parse_args()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
amass_data = joblib.load(f"{file_dir}/../data/amass_curated_large_noback.pkl")
motion_id = args.motion_id #18 #5
traj = amass_data[f"motion_{motion_id}"]
c = pb.connect(pb.GUI)
robot = pb.loadURDF(f"{file_dir}/../resources/robots/g1/g1_29dof_rev_1_0.urdf")

init_root_pos = traj["root_trans_offset"][0]
init_root_pos[2] = 0
init_root_orn = traj["root_rot"][0]
init_heading = heading_zup(init_root_orn)
orient_inv = Rotation.from_euler("z", -init_heading)
target = None
while target is None:
    raw_target = pickle.loads(redis_client.get("current_target"))
    if (raw_target > 0).any():
        target = raw_target
    time.sleep(0.1)
targets = []
for i in range(len(traj["root_rot"])):
    root_pos = orient_inv.apply(traj["root_trans_offset"][i] - init_root_pos)
    root_orn = (orient_inv*Rotation.from_quat(traj["root_rot"][i])).as_quat()
    set_robot_state(robot, traj["dof"][i], root_pos, root_orn)
    targets.append(get_eef_pos(robot))
    time.sleep(0.03)

idx = 1
while True:
    vlock.acquire()
    flag = v[0]
    vlock.release()
    if flag:
        if idx == len(targets):
            target = targets[-1]
        else:
            target = targets[idx]
        idx += 1
    redis_client.set("target", pickle.dumps(target))
    redis_client.set("server_ready", "true")
    time.sleep(0.03)
