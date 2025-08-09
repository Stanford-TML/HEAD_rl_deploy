import mujoco
import mujoco.viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pickle
G1_LINKS = [
    'pelvis', 
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_track_site',    # idx: 5
    'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_track_site',   # idx: 12
    'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link',
    'torso_link',
    'head_track_site',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'left_hand_track_site',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 
    'right_hand_track_site'
]

G1_LINKS_NO_FEET = [
    'pelvis', 
    'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link', 'left_ankle_roll_link',
    'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link', 'right_ankle_roll_link',
    'waist_yaw_link', 'waist_roll_link',
    'torso_link',
    'head_track_site',
    'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link',
    'left_elbow_link', 
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'left_hand_track_site',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link', 
    'right_hand_track_site'
]

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    #print(" Wrist ref cur:",target_q[[19,20,21,26,27,28]],q[[19,20,21,26,27,28]])
    #print("Wrist vel:", dq[[19,20,21,26,27,28]])
    return (target_q - q) * kp + (target_dq - dq) * kd

KP = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               300, # waist yaw
               100, 100, 50, 50, 20, 20, 20, # left arm
               100, 100, 50, 50, 20, 20, 20]) # right arm

KD = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      4,
      2, 2, 2, 2, 1, 1, 1,
      2, 2, 2, 2, 1, 1, 1])


KP_29 = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               150, 300, 300, # waist yaw
               100, 100, 50, 50, 20, 20, 20, # left arm
               100, 100, 50, 50, 20, 20, 20]) # right arm

KD_29 = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      4, 4, 4,
      2, 2, 2, 2, 1, 1, 1,
      2, 2, 2, 2, 1, 1, 1])
# SVL PD
LOW_KP = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               150, # waist yaw
               40, 40, 40, 40, 20, 20, 20, # left arm
               40, 40, 40, 40, 20, 20, 20]) # right arm

LOW_KD = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      4,
      5, 5, 5, 5, 2, 2, 2,
      5, 5, 5, 5, 2, 2, 2])

LOW_KP_29 = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               120, 120, 120, # waist yaw
               40, 40, 40, 40, 20, 20, 20, # left arm
               40, 40, 40, 40, 20, 20, 20]) # right arm

# LOW_KD_29 = np.array([2, 2, 2, 4, 2, 2, 
#       2, 2, 2, 4, 2, 2,
#       4, 4, 4,
#       5, 5, 5, 5, 2, 2, 2,
#       5, 5, 5, 5, 2, 2, 2])

LOW_KD_29 = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      8, 8, 8,
      12, 12, 12, 12, 8, 8, 8,
      12, 12, 12, 12, 8, 8, 8])

# KP = np.array([100, 100, 100, 150, 40, 40, # Left leg
#                100, 100, 100, 150, 40, 40, # Right leg
#                150, #300, 300, # waist yaw
#                40, 40, 40, 40, 20, 20, 20, # left arm
#                40, 40, 40, 40, 20, 20, 20]) # right arm

# KD = np.array([2, 2, 2, 4, 2, 2, 
#       2, 2, 2, 4, 2, 2,
#       4, #4, 4,
#       5, 5, 5, 5, 1, 1, 1,
#       5, 5, 5, 5, 1, 1, 1])

# KP = np.array([499.99997, 499.99997, 499.99997, 300.0, 50.0, 50.0, 499.99997, 499.99997, 499.99997, 300.0, 50.0, 50.0, 499.99997, 499.99997, 499.99997, 300.0, 300.0, 300.0, 200.0, 200.0, 200.0, 200.0, 300.0, 300.0, 300.0, 200.0, 200.0, 200.0, 200.0]) 

# KD = np.array([50.0, 50.0, 50.0, 30.0, 5.0, 5.0, 50.0, 50.0, 50.0, 30.0, 5.0, 5.0, 50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 10.0, 10.0, 10.0, 30.0, 30.0, 30.0, 20.0, 10.0, 10.0, 10.0])

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    
    Parameters
    ----------
    q1, q2 : array-like, shape (4,)
        Quaternions in [w, x, y, z] format.
    
    Returns
    -------
    product : ndarray, shape (4,)
        The quaternion product q1 * q2 in [w, x, y, z] format.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def add_random_rotation_noise(quat, noise_std):
    """
    Add random rotation noise to a quaternion.
    
    The function interprets the input quaternion as [x, y, z, w] and adds a random
    rotation by sampling:
      - A random unit axis uniformly on the sphere.
      - A rotation angle from a normal distribution with standard deviation `noise_std` (radians).
      
    The noise rotation is applied by left-multiplying the original quaternion.
    
    Parameters
    ----------
    quat : array-like, shape (4,)
        The input quaternion in [x, y, z, w] format.
    noise_std : float
        Standard deviation of the rotation noise (in radians).
        
    Returns
    -------
    noisy_quat : ndarray, shape (4,)
        The quaternion with added rotation noise, in [x, y, z, w] format.
    """
    # Ensure input is a NumPy array and normalized.
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    
    # Convert from [x, y, z, w] to [w, x, y, z] for computation.
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)
    
    # Sample a random unit axis.
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    
    # Sample a random noise angle from a normal distribution.
    angle = np.random.normal(loc=0.0, scale=noise_std)
    
    # Build the noise quaternion in [w, x, y, z] format.
    half_angle = angle / 2.0
    noise_q = np.concatenate(([np.cos(half_angle)], np.sin(half_angle) * axis))
    
    # Apply the noise by quaternion multiplication: noise * quat.
    noisy_q_wxyz = quaternion_multiply(noise_q, quat_wxyz)
    noisy_q_wxyz /= np.linalg.norm(noisy_q_wxyz)  # Ensure normalization
    
    # Convert back to [x, y, z, w] format.
    noisy_quat = np.array([noisy_q_wxyz[1], noisy_q_wxyz[2], noisy_q_wxyz[3], noisy_q_wxyz[0]])
    return noisy_quat

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

class MjRobot:
    def __init__(self, xml_path, dt = 1/30., substeps=10, link_names = G1_LINKS, use_sim=True, low_pd=False):
        with open(xml_path, 'r') as f:
            xml = f.read()
        self.low_pd = low_pd
        self.link_names = link_names
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.kin_data = mujoco.MjData(self.model)
        self.full_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.kin_data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.full_data)
        self.use_sim = use_sim
        self.substeps = substeps
        self.model.opt.timestep = dt / self.substeps
        self.model.opt.impratio = 10
        self.kp1 = self.model.body("kp1").mocapid[0]
        self.kp2 = self.model.body("kp2").mocapid[0]
        self.kp3 = self.model.body("kp3").mocapid[0]

    def forward_kinematics(self, q, root_pose):
        #old_qpos = self.kin_data.qpos.copy()
        self.kin_data.qpos[:7] = root_pose[[0,1,2,6,3,4,5]] # mujoco: w,x,y,z; isaac gym: x,y,z,w
        self.kin_data.qpos[7:34] = q
        mujoco.mj_kinematics(self.model, self.kin_data)
        # Should obtain all relevant link position, orientationa and velocity
        link_pos = np.stack([self.kin_data.body(link).xpos for link in self.link_names])
        link_orn = np.stack([self.kin_data.body(link).xquat for link in self.link_names])[:,[1,2,3,0]]
        mujoco.mj_kinematics(self.model, self.kin_data)
        if not self.use_sim:
            self.full_data.qpos[:] = self.kin_data.qpos.copy()
            mujoco.mj_kinematics(self.model, self.full_data)
            self.viewer.sync()
        #self.kin_data.qpos = old_qpos
        
        return link_pos, link_orn, np.zeros((len(link_pos), 6))
    
    def get_state_hist_with_more_info(self, q, root_pose):
        link_pos, link_orn, link_vel = self.forward_kinematics(q, root_pose)
        state_hist = torch.from_numpy(np.hstack([link_pos, link_orn, link_vel])).reshape(1,-1)
        return state_hist, link_pos, link_orn, link_vel
    
    def get_state_hist(self, q, root_pose):
        link_pos, link_orn, link_vel = self.forward_kinematics(q, root_pose)
        state_hist = torch.from_numpy(np.hstack([link_pos, link_orn, link_vel])).reshape(1,-1)
        return state_hist

    def set_keypoints(self, keypoints):
        self.full_data.mocap_pos[self.kp1] = keypoints[0,:3]
        self.full_data.mocap_pos[self.kp2] = keypoints[1,:3]
        self.full_data.mocap_pos[self.kp3] = keypoints[2,:3]
        mujoco.mj_kinematics(self.model, self.full_data)
        self.viewer.sync()

    # Should take multiple substeps
    def step_physics(self, action):
        kp = LOW_KP if self.low_pd else KP
        kd = LOW_KD if self.low_pd else KD
        for _ in range(self.substeps):
            tau = pd_control(action, self.full_data.qpos[7:34], kp, np.zeros(len(action)), self.full_data.qvel[6:33], kd)
            #breakpoint()
            self.full_data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.full_data)
            self.viewer.sync()

    # Assume robot in static state, no velocity
    def set_robot_state(self, q, root_pose):
        self.full_data.qpos[:7] = root_pose[[0,1,2,6,3,4,5]].copy()
        self.full_data.qpos[7:34] = q.copy()
        mujoco.mj_kinematics(self.model, self.full_data)
        self.viewer.sync()

    def get_robot_state(self, include_vel=False, include_omega=False):
        q = self.full_data.qpos[7:34].copy()
        root_pose = self.full_data.qpos[:7][[0,1,2,4,5,6,3]].copy()
        #breakpoint()
        rets = [q, root_pose]
        # if has attribute of redis_client
        if hasattr(self, 'redis_client'):
            self.redis_client.set("joint_angle", pickle.dumps(q))
            self.redis_client.set("root_pose", pickle.dumps(root_pose))
        if include_vel:
            dq = self.full_data.qvel[6:33].copy()
            rets.append(dq)
        if include_omega:
            omega = self.full_data.qvel[3:6]
            rets.append(omega)
        return rets
    
    def get_head_pose(self, no_mocap=False):
        head_pos = self.full_data.body("head_track_site").xpos.copy()
        head_orn = self.full_data.body("head_track_site").xquat.copy()
        head_orn = head_orn[[1,2,3,0]]
        if no_mocap: # compute head in deheaded root frame
            root_pos = self.full_data.body("pelvis").xpos.copy()
            root_orn = self.full_data.body("pelvis").xquat.copy()[[1,2,3,0]]
            heading = heading_zup(root_orn)
            dehead_rot = Rotation.from_euler("z", -heading)
            head_pos = dehead_rot.apply(head_pos - root_pos) + np.array([0, 0, root_pos[2]]) # only add z can be replaced by collision checking later
            head_orn = (dehead_rot*Rotation.from_quat(head_orn)).as_quat()
        return np.hstack([head_pos, head_orn])

    def get_root_z(self): # TO BE IMPLEMENTED
        pass

    def get_tracking_site(self, local=False):
        head = self.full_data.body("head_track_site").xpos.copy()
        left_wrist = self.full_data.body("left_hand_track_site").xpos.copy()
        right_wrist = self.full_data.body("right_hand_track_site").xpos.copy()
        target = np.zeros((1,3,13))
        target[:,0,:3] = head
        target[:,1,:3] = left_wrist
        target[:,2,:3] = right_wrist
        if local:
            heading = heading_zup(self.full_data.qpos[3:7][[1,2,3,0]].copy())
            dehead_rot = Rotation.from_euler("z", -heading)
            origin = self.full_data.body("pelvis").xpos.copy()
            target[:,0,:3] = dehead_rot.apply(target[:,0,:3] - origin) + np.array([0, 0, origin[2]]) # only add z can be replaced by collision checking later
            target[:,1,:3] = dehead_rot.apply(target[:,1,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,2,:3] = dehead_rot.apply(target[:,2,:3] - origin) + np.array([0, 0, origin[2]])
        return target
    
    def get_tracking_site_with_orn(self, no_mocap=False):
        head_pos = self.full_data.body("head_track_site").xpos.copy()
        left_wrist_pos = self.full_data.body("left_hand_track_site").xpos.copy()
        right_wrist_pos = self.full_data.body("right_hand_track_site").xpos.copy()
        head_orn = self.full_data.body("head_track_site").xquat.copy()
        left_wrist_orn = self.full_data.body("left_hand_track_site").xquat.copy()
        right_wrist_orn = self.full_data.body("right_hand_track_site").xquat.copy()
        target = np.zeros((1,3,13))
        target[:,0,:3] = head_pos
        target[:,1,:3] = left_wrist_pos
        target[:,2,:3] = right_wrist_pos
        target[:,0,3:7] = head_orn[[1,2,3,0]]
        target[:,1,3:7] = left_wrist_orn[[1,2,3,0]]
        target[:,2,3:7] = right_wrist_orn[[1,2,3,0]]
        if no_mocap:
            heading = heading_zup(self.full_data.qpos[3:7][[1,2,3,0]].copy())
            dehead_rot = Rotation.from_euler("z", -heading)
            origin = self.full_data.body("pelvis").xpos.copy()
            target[:,0,:3] = dehead_rot.apply(target[:,0,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,1,:3] = dehead_rot.apply(target[:,1,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,2,:3] = dehead_rot.apply(target[:,2,:3] - origin) + np.array([0, 0, origin[2]])
            target[0,0,3:7] = (dehead_rot*Rotation.from_quat(target[0,0,3:7])).as_quat()
            target[0,1,3:7] = (dehead_rot*Rotation.from_quat(target[0,1,3:7])).as_quat()
            target[0,2,3:7] = (dehead_rot*Rotation.from_quat(target[0,2,3:7])).as_quat()
        return target
    
    def attach_redis_client(self, redis_client):
        self.redis_client = redis_client

class MjRobot29:
    def __init__(self, xml_path, dt = 1/30., substeps=10, link_names = G1_LINKS, use_sim=True, low_pd=False):
        with open(xml_path, 'r') as f:
            xml = f.read()
        self.low_pd = low_pd
        self.link_names = link_names
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.kin_data = mujoco.MjData(self.model)
        self.full_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.kin_data)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.full_data)
        self.use_sim = use_sim
        self.substeps = substeps
        self.model.opt.timestep = dt / self.substeps
        self.model.opt.impratio = 10
        self.kp1 = self.model.body("kp1").mocapid[0]
        self.kp2 = self.model.body("kp2").mocapid[0]
        self.kp3 = self.model.body("kp3").mocapid[0]

    def forward_kinematics(self, q, root_pose):
        #old_qpos = self.kin_data.qpos.copy()
        self.kin_data.qpos[:7] = root_pose[[0,1,2,6,3,4,5]] # mujoco: w,x,y,z; isaac gym: x,y,z,w
        self.kin_data.qpos[7:36] = q
        mujoco.mj_kinematics(self.model, self.kin_data)
        # Should obtain all relevant link position, orientationa and velocity
        link_pos = np.stack([self.kin_data.body(link).xpos for link in self.link_names])
        link_orn = np.stack([self.kin_data.body(link).xquat for link in self.link_names])[:,[1,2,3,0]]
        mujoco.mj_kinematics(self.model, self.kin_data)
        if not self.use_sim:
            self.full_data.qpos[:] = self.kin_data.qpos.copy()
            mujoco.mj_kinematics(self.model, self.full_data)
            self.viewer.sync()
        #self.kin_data.qpos = old_qpos
        
        return link_pos, link_orn, np.zeros((len(link_pos), 6))
    
    def get_state_hist_with_more_info(self, q, root_pose):
        link_pos, link_orn, link_vel = self.forward_kinematics(q, root_pose)
        state_hist = torch.from_numpy(np.hstack([link_pos, link_orn, link_vel])).reshape(1,-1)
        return state_hist, link_pos, link_orn, link_vel
    
    def get_state_hist(self, q, root_pose):
        link_pos, link_orn, link_vel = self.forward_kinematics(q, root_pose)
        state_hist = torch.from_numpy(np.hstack([link_pos, link_orn, link_vel])).reshape(1,-1)
        return state_hist

    def set_keypoints(self, keypoints):
        self.full_data.mocap_pos[self.kp1] = keypoints[0,:3]
        self.full_data.mocap_pos[self.kp2] = keypoints[1,:3]
        self.full_data.mocap_pos[self.kp3] = keypoints[2,:3]
        mujoco.mj_kinematics(self.model, self.full_data)
        self.viewer.sync()

    # Should take multiple substeps
    def step_physics(self, action):
        kp = LOW_KP_29 if self.low_pd else KP_29
        kd = LOW_KD_29 if self.low_pd else KD_29
        for _ in range(self.substeps):
            tau = pd_control(action, self.full_data.qpos[7:36], kp, np.zeros(len(action)), self.full_data.qvel[6:35], kd)
            #breakpoint()
            self.full_data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.full_data)
            self.viewer.sync()

    # Assume robot in static state, no velocity
    def set_robot_state(self, q, root_pose):
        self.full_data.qpos[:7] = root_pose[[0,1,2,6,3,4,5]].copy()
        self.full_data.qpos[7:36] = q.copy()
        mujoco.mj_kinematics(self.model, self.full_data)
        self.viewer.sync()

    def get_robot_state(self, include_vel=False, include_omega=False):
        q = self.full_data.qpos[7:36].copy()
        root_pose = self.full_data.qpos[:7][[0,1,2,4,5,6,3]].copy()
        #breakpoint()
        rets = [q, root_pose]
        # if has attribute of redis_client
        if hasattr(self, 'redis_client'):
            self.redis_client.set("joint_angle", pickle.dumps(q))
            self.redis_client.set("root_pose", pickle.dumps(root_pose))
        if include_vel:
            dq = self.full_data.qvel[6:35].copy()
            rets.append(dq)
        if include_omega:
            omega = self.full_data.qvel[3:6]
            rets.append(omega)
        return rets
    
    def get_head_pose(self, no_mocap=False):
        head_pos = self.full_data.body("head_track_site").xpos.copy()
        head_orn = self.full_data.body("head_track_site").xquat.copy()
        head_orn = head_orn[[1,2,3,0]]
        if no_mocap: # compute head in deheaded root frame
            root_pos = self.full_data.body("pelvis").xpos.copy()
            root_orn = self.full_data.body("pelvis").xquat.copy()[[1,2,3,0]]
            heading = heading_zup(root_orn)
            dehead_rot = Rotation.from_euler("z", -heading)
            head_pos = dehead_rot.apply(head_pos - root_pos) + np.array([0, 0, root_pos[2]]) # only add z can be replaced by collision checking later
            head_orn = (dehead_rot*Rotation.from_quat(head_orn)).as_quat()
        return np.hstack([head_pos, head_orn])

    def get_root_z(self): # TO BE IMPLEMENTED
        pass

    def get_tracking_site(self, local=False):
        head = self.full_data.body("head_track_site").xpos.copy()
        left_wrist = self.full_data.body("left_hand_track_site").xpos.copy()
        right_wrist = self.full_data.body("right_hand_track_site").xpos.copy()
        target = np.zeros((1,3,13))
        target[:,0,:3] = head
        target[:,1,:3] = left_wrist
        target[:,2,:3] = right_wrist
        if local:
            heading = heading_zup(self.full_data.qpos[3:7][[1,2,3,0]].copy())
            dehead_rot = Rotation.from_euler("z", -heading)
            origin = self.full_data.body("pelvis").xpos.copy()
            target[:,0,:3] = dehead_rot.apply(target[:,0,:3] - origin) + np.array([0, 0, origin[2]]) # only add z can be replaced by collision checking later
            target[:,1,:3] = dehead_rot.apply(target[:,1,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,2,:3] = dehead_rot.apply(target[:,2,:3] - origin) + np.array([0, 0, origin[2]])
        return target
    
    def get_tracking_site_with_orn(self, no_mocap=False):
        head_pos = self.full_data.body("head_track_site").xpos.copy()
        left_wrist_pos = self.full_data.body("left_hand_track_site").xpos.copy()
        right_wrist_pos = self.full_data.body("right_hand_track_site").xpos.copy()
        head_orn = self.full_data.body("head_track_site").xquat.copy()
        left_wrist_orn = self.full_data.body("left_hand_track_site").xquat.copy()
        right_wrist_orn = self.full_data.body("right_hand_track_site").xquat.copy()
        target = np.zeros((1,3,13))
        target[:,0,:3] = head_pos
        target[:,1,:3] = left_wrist_pos
        target[:,2,:3] = right_wrist_pos
        target[:,0,3:7] = head_orn[[1,2,3,0]]
        target[:,1,3:7] = left_wrist_orn[[1,2,3,0]]
        target[:,2,3:7] = right_wrist_orn[[1,2,3,0]]
        if no_mocap:
            heading = heading_zup(self.full_data.qpos[3:7][[1,2,3,0]].copy())
            dehead_rot = Rotation.from_euler("z", -heading)
            origin = self.full_data.body("pelvis").xpos.copy()
            target[:,0,:3] = dehead_rot.apply(target[:,0,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,1,:3] = dehead_rot.apply(target[:,1,:3] - origin) + np.array([0, 0, origin[2]])
            target[:,2,:3] = dehead_rot.apply(target[:,2,:3] - origin) + np.array([0, 0, origin[2]])
            target[0,0,3:7] = (dehead_rot*Rotation.from_quat(target[0,0,3:7])).as_quat()
            target[0,1,3:7] = (dehead_rot*Rotation.from_quat(target[0,1,3:7])).as_quat()
            target[0,2,3:7] = (dehead_rot*Rotation.from_quat(target[0,2,3:7])).as_quat()
        return target
    
    def attach_redis_client(self, redis_client):
        self.redis_client = redis_client

if __name__ == "__main__":
    model = MjRobot("resources/robots/g1/g1_27dof_rev_1_0.xml", use_sim=True)
    q = np.zeros(29)
    dq = np.zeros(29)
    root_pose = np.array([0., 0., 0., 1., 0., 0., 0.])
    root_vel = np.zeros(6)
    input()
    while True:
        link_pos, link_orn, link_vel = model.forward_kinematics(q, dq, root_pose, root_vel)
        print(link_pos.shape, link_orn.shape, link_vel.shape)

