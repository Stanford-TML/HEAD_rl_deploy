import redis
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from threading import Lock
import pickle
import time
vlock = Lock()
v = np.zeros((3,13), dtype=np.float32)

left_modes_joints = {
    0: np.zeros(9),
    1: np.array([-0.,          0.,          0.,         -0.,          0.,          0.,
                 -1.57079633, -0.78539816, -1.25663706]),
    2: np.array([-0.,          1.57079633,  1.04719755, -0.,          0.,          0.,
                 -1.25663706, -0.,          0.78539816]),
    3: np.array([-0.        ,  1.57079633,  1.04719755, -0.52359878,  1.57079633,  1.04719755,
                 -1.57079633, -0.,          0.78539816])
}

right_modes_joints = {
    0: np.zeros(9),
    1: np.array([0.,          0.,          0.,
                  0.,          0.,          0.,          1.57079633,  0.78539816, -1.25663706]),
    2: np.array([0.,          0.,          0.,
                  0.,          1.57079633,  1.04719755,  1.25663706,  0.,          0.78539816]),
    3: np.array([0.52359878,  1.57079633,  1.04719755,
                  0.        ,  1.57079633,  1.04719755,  1.57079633,  0.,          0.78539816])
}

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

redis_client = redis.Redis(host='localhost', port=6379, db=0)
hand_client = redis.Redis("192.168.123.222", port=6379, db=0)
current_pose = pickle.loads(redis_client.get("root_pose"))

current_pos = current_pose[:3]
heading = heading_zup(current_pose[3:7])
current_rot = Rotation.from_euler('xyz', [0, 0, heading])

init_target = np.array([[0.00, 0.00, 1.24, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.22, 0.15, 0.68, 0.00, 0.38, -0.00, 0.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                        [0.22, -0.15, 0.68, -0.00, 0.38, 0.00, 0.92, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]])
target = init_target.copy()
init_target[:, :2] = (current_rot.apply(target[:, :3]) + current_pos)[:,:2]
init_target[:, 3:7] = (current_rot * Rotation.from_quat(target[:, 3:7])).as_quat()
hand_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[0],right_modes_joints[0]])))

current_target = pickle.loads(redis_client.get("current_target"))
interp = 100
head_slerp = Slerp([0, 1], Rotation.from_quat([current_target[0,3:7],init_target[0,3:7]]))
lhand_slerp= Slerp([0, 1], Rotation.from_quat([current_target[1,3:7],init_target[1,3:7]]))
rhand_slerp= Slerp([0, 1], Rotation.from_quat([current_target[2,3:7],init_target[2,3:7]]))
current_hand_mode = np.array([0.0, 0.0])
redis_client.set("current_hand_mode", pickle.dumps(current_hand_mode))
for i in range(interp):
    target = np.zeros((3,13))
    target[:,:3] = (1-i/interp)*current_target[:,:3] + (i/interp)*init_target[:,:3]
    # use slerp to interpolate orientation
    target[0,3:7] = head_slerp(i/interp).as_quat()
    target[1,3:7] = lhand_slerp(i/interp).as_quat()
    target[2,3:7] = rhand_slerp(i/interp).as_quat()
    redis_client.set("target", pickle.dumps(target))
    redis_client.set("server_ready", "true")
    time.sleep(0.03)
