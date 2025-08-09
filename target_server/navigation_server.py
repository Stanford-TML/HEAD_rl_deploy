import redis
import numpy as np
import pybullet as pb
from pynput import keyboard
import traceback
from threading import Lock
import pickle
import joblib
import time
from scipy.spatial.transform import Rotation, Slerp
vlock = Lock()
v = np.zeros(1, dtype=np.float32)
def on_press(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.alt:
            v[0] = 1
    except Exception as e:
        traceback.print_exc()
    vlock.release()

def on_release(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.alt:
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

redis_client = redis.Redis(host='localhost', port=6379, db=0)
robot_client = redis.Redis(host="192.168.123.222", port=6379, db=0)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

c = pb.connect(pb.GUI)
heads = [pb.loadURDF("../resources/robots/g1/head.urdf", useFixedBase=True) for _ in range(10)]
redis_client.set("current_target", pickle.dumps(np.zeros(3)))
target = None
while target is None:
    raw_target = pickle.loads(redis_client.get("current_target"))
    if (raw_target > 0).any():
        target = raw_target
        init_target = target.copy()
    time.sleep(0.1)
interp = 10
while True:
    head_plan = pickle.loads(redis_client.get("traj_pose"))
    target = pickle.loads(redis_client.get("current_target"))
    head_plan[:,2] = 0 # remove z component
    # visualize plan
    for i in range(len(head_plan)):
        heading = heading_zup(head_plan[i][3:7])
        head_plan[i][3:7] = Rotation.from_euler('z', heading).as_quat()
        # add first frame target rotation
        pb.resetBasePositionAndOrientation(heads[i], head_plan[i][:3], head_plan[i][3:])
        head_plan[i][:3] = Rotation.from_quat(target[0,3:7]).apply(head_plan[i][:3])
    idx = 0
    plan_horizon = 3
    robot_client.set("disable_navi", "false")
    while True:
        while robot_client.get("disable_navi").decode() == "true":
            print("waiting for navi to be enabled",end="\r")
            time.sleep(0.1)
        vlock.acquire()
        flag = v[0]
        vlock.release()
        if flag:
            new_target = target.copy()
            cycle = idx // interp
            if cycle < min(len(head_plan) - 1, plan_horizon):
                alpha = (idx - cycle*interp) / interp
                # Should add initial rotation
                new_target[0,:3] = target[0,:3] + head_plan[cycle][:3] * (1-alpha) + head_plan[cycle+1][:3] * alpha
                # slerp interpolate quaternion
                times = [0, 1]
                key_rots = Rotation.from_quat([head_plan[cycle][3:], head_plan[cycle+1][3:]])
                slerp = Slerp(times, key_rots)
                # slerp expects an array-like; we take the first element of the result.
                interp_rot = slerp([alpha])[0]
                new_target[0,3:7] = (interp_rot*Rotation.from_quat(target[0,3:7])).as_quat()
                # update wrist pose
                new_target[1,:3] = interp_rot.apply(target[1,:3] - target[0,:3]) + new_target[0,:3]
                new_target[2,:3] = interp_rot.apply(target[2,:3] - target[0,:3]) + new_target[0,:3]
                redis_client.set("target", pickle.dumps(new_target))
                redis_client.set("server_ready", "true")
                idx += 1
            else:
                break
        time.sleep(0.04)
