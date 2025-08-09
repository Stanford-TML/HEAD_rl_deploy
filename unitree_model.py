import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
import redis
import pickle
from threading import Lock
import pybullet as pb
from scipy.spatial.transform import Rotation

import unitree_sdk2py
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.remote_controller import RemoteController, KeyMap

G1_LINKS = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 
            'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 
            'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 
            'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 
            'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 
            'torso_link', 'head_track_site', 'left_shoulder_pitch_link', 
            'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 
            'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 
            'left_hand_track_site', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 
            'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 
            'right_wrist_pitch_link', 'right_wrist_yaw_link', 'right_hand_track_site']

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    #print(" Wrist ref cur:",target_q[[19,20,21,26,27,28]],q[[19,20,21,26,27,28]])
    #print("Wrist vel:", dq[[19,20,21,26,27,28]])
    return (target_q - q) * kp + (target_dq - dq) * kd

KP = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               300, 300, 300, # waist yaw
               100, 100, 50, 50, 20, 20, 20, # left arm
               100, 100, 50, 50, 20, 20, 20]) # right arm

KD = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      4, 4, 4,
      2, 2, 2, 2, 1, 1, 1,
      2, 2, 2, 2, 1, 1, 1])

# SVL PD
LOW_KP = np.array([100, 100, 100, 150, 40, 40, # Left leg
               100, 100, 100, 150, 40, 40, # Right leg
               120, 120, 120, # waist yaw
               40, 40, 40, 40, 20, 20, 20, # left arm
               40, 40, 40, 40, 20, 20, 20]) # right arm

# LOW_KD = np.array([2, 2, 2, 4, 2, 2, 
#       2, 2, 2, 4, 2, 2,
#       4, 4, 4,
#       5, 5, 5, 5, 2, 2, 2,
#       5, 5, 5, 5, 2, 2, 2])

LOW_KD = np.array([2, 2, 2, 4, 2, 2, 
      2, 2, 2, 4, 2, 2,
      8, 8, 8,
      12, 12, 12, 12, 8, 8, 8,
      12, 12, 12, 12, 8, 8, 8])

LOCK_WAIST_IDX = [0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11,
                  12, 
                  15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28]


PB_JOINT_IDX = [17,18,19,20,21,22,
                23,24,25,26,27,28,
                0,
                10, 11, 12, 13, 14, 15, 16,
                3, 4, 5, 6, 7, 8, 9]

# PB_JOINT_IDX_29 = [17,18,19,20,21,22,
#                 23,24,25,26,27,28,
#                 0,1,2,
#                 10, 11, 12, 13, 14, 15, 16,
#                 3, 4, 5, 6, 7, 8, 9]
PB_JOINT_IDX_29 = list(range(29))
# KP = np.array([499.99997, 499.99997, 499.99997, 300.0, 50.0, 50.0, 499.99997, 499.99997, 499.99997, 300.0, 50.0, 50.0, 499.99997, 499.99997, 499.99997, 300.0, 300.0, 300.0, 200.0, 200.0, 200.0, 200.0, 300.0, 300.0, 300.0, 200.0, 200.0, 200.0, 200.0]) 

# KD = np.array([50.0, 50.0, 50.0, 30.0, 5.0, 5.0, 50.0, 50.0, 50.0, 30.0, 5.0, 5.0, 50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 10.0, 10.0, 10.0, 30.0, 30.0, 30.0, 20.0, 10.0, 10.0, 10.0])

def get_root_pose_from_link(robot_id, link_id, joint_angles, target_link_pos, target_link_orn):
    """
    Given current joint angles and a target link's desired pose,
    compute the root frame pose.
    Parameters:
    robot_id (int): PyBullet ID of the robot.
    link_id (int): Index of the target link.
    joint_indices (list[int]): Joint indices to set.
    joint_angles (list[float]): Corresponding joint angles.
    target_link_pose (tuple): Desired pose of the link (position, quaternion).
    Returns:
    root_pose (tuple): Computed pose of the robot's root frame (position, quaternion).
    """
    # Reset joints to specified angles
    #pb.resetBasePositionAndOrientation(robot_id, [0, 0, 0], [0, 0, 0, 1])
    current_root_pos, current_root_orn = pb.getBasePositionAndOrientation(robot_id)
    current_root_orn = Rotation.from_quat(current_root_orn)
    current_root_pos = np.array(current_root_pos)
    j = 0
    for i in range(pb.getNumJoints(robot_id)):
        if pb.getJointInfo(robot_id, i)[2] == pb.JOINT_REVOLUTE:
            pb.resetJointState(robot_id, i, joint_angles[j])
            j += 1        
    # Get the current pose of the link
    state = pb.getLinkState(robot_id, link_id, computeForwardKinematics=True)
    current_link_pos, current_link_orn = np.array(state[4]), np.array(state[5])
    current_link_pos = current_root_orn.inv().apply(current_link_pos - current_root_pos)
    current_link_orn = (current_root_orn.inv() * Rotation.from_quat(current_link_orn)).as_quat()
    # Convert poses to transformation matrices
    T_link_world = np.eye(4)
    T_link_world[:3, :3] = Rotation.from_quat(current_link_orn).as_matrix()
    T_link_world[:3, 3] = current_link_pos
    T_target_link = np.eye(4)
    T_target_link[:3, :3] = Rotation.from_quat(target_link_orn).as_matrix()
    T_target_link[:3, 3] = target_link_pos
    # Compute inverse transform to find root pose
    T_root_world = T_target_link @ np.linalg.inv(T_link_world)
    root_pos = T_root_world[:3, 3]
    root_orn = Rotation.from_matrix(T_root_world[:3, :3]).as_quat()
    pb.resetBasePositionAndOrientation(robot_id, root_pos, root_orn)
    return root_pos, root_orn

class UnitreeRobot:
    def __init__(self, dt = 1/50., net=None, one_robot=False, low_pd=False):
        self.qj = np.zeros(27, dtype=np.float32)
        self.dqj = np.zeros(27, dtype=np.float32)
        self.action = np.zeros(27, dtype=np.float32)
        self.target_dof_pos = np.zeros(29, dtype = np.float32)
        self.counter = 0
        self.control_dt = dt
        self.low_pd = low_pd
        assert net is not None
        ChannelFactoryInitialize(0, net)

        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.remote_controller = RemoteController()
        self.control_lock = Lock()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine = 0

        c = pb.connect(pb.GUI)
        if not one_robot:
            self.robot_ref = pb.loadURDF("resources/robots/g1/g1_29dof_rev_1_0.urdf")
        self.robot_cur = pb.loadURDF("resources/robots/g1/g1_29dof_rev_1_0.urdf")

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.low_state_subscriber_ = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_subscriber_.Init(self.LowStateHgHandler, 10)
        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        self.terminated = False

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")
    
    def get_robot_state(self, link="root", include_vel=False):
        for i, idx in enumerate(LOCK_WAIST_IDX):
            self.qj[i] = self.low_state.motor_state[idx].q
            self.dqj[i] = self.low_state.motor_state[idx].dq
        if link == "root":
            root_pos = pickle.loads(self.redis_client.get("root_pos"))
            root_quat = pickle.loads(self.redis_client.get("root_quat"))
        elif link == "head":
            self.root_pos = pickle.loads(self.redis_client.get("head_pos"))
            self.head_quat = pickle.loads(self.redis_client.get("head_quat"))
            r = Rotation.from_euler("xyz", [0, 0, -self.qj[12]])
            root_quat = (Rotation.from_quat(self.head_quat) * r).as_quat()
        root_pose = np.concatenate([self.root_pos, root_quat])
        self.redis_client.set("joint_angle", pickle.dumps(self.qj))
        self.redis_client.set("root_pose", pickle.dumps(root_pose))
        if include_vel:
            return self.qj, np.concatenate([self.root_pos, root_quat]), self.dqj
        # Should broad cast joint angle and root pose
        return self.qj, root_pose

    def get_head_site(self):
        head_pos = self.root_pos + Rotation.from_quat(self.head_quat).apply(np.array([0.,0.,0.444]))
        head_quat = self.head_quat
        return head_pos, head_quat

    def get_imu_quat(self):
        return self.low_state.imu_state.quaternion
    
    def get_omega(self):
        return np.array(self.low_state.imu_state.gyroscope)

    def pd_control(self, target_q, kp, kd):
        assert len(target_q) == 29
        for i in range(len(target_q)):
            self.low_cmd.motor_cmd[i].q = target_q[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = kp[i]
            self.low_cmd.motor_cmd[i].kd = kd[i]
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd(self.low_cmd)

    # Assume robot in static state, no velocity
    def set_robot_state(self, q): # q 27 dof
        total_time = 2
        num_step = int(total_time/self.control_dt)
        
        init_q = np.zeros(29, dtype=np.float32)
        dof_idx = np.arange(29)
        target_q = np.zeros(29, dtype=np.float32)
        target_q[LOCK_WAIST_IDX] = q
        for i in dof_idx:
            init_q[i] = self.low_state.motor_state[i].q

        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha, LOW_KP if self.low_pd else KP, LOW_KD if self.low_pd else KD)
            time.sleep(self.control_dt)

    def set_pb_robot_state(self, robot, q, root_pose):
        j = 0
        for i in range(pb.getNumJoints(robot)):
            if pb.getJointInfo(robot, i)[2] == pb.JOINT_REVOLUTE:
                if j not in [13,14]:
                    pb.resetJointState(robot, i, q[j])
                else:
                    pb.resetJointState(robot, i, 0)
        pb.resetBasePositionAndOrientation(robot, root_pose[:3]+Rotation.from_quat(root_pose[3:]).apply(np.array([0.,0.,-0.02])), root_pose[3:])

    def maintain_state(self, q, root_pose, link="root"):
        print("Maintaining state, wait for A signal...")
        target_q = np.zeros(29, dtype=np.float32)
        target_q[LOCK_WAIST_IDX] = q
        self.set_pb_robot_state(self.robot_ref, q, root_pose)
        while self.remote_controller.button[KeyMap.A] != 1:
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            self.pd_control(target_q, LOW_KP if self.low_pd else KP, LOW_KD if self.low_pd else KD)
            cur_q, cur_root_pose = self.get_robot_state(link=link)
            self.set_pb_robot_state(self.robot_cur, cur_q, cur_root_pose)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def damping_state(self):
        print("Entering damping state")
        self.control_lock.acquire()
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        self.terminated = True
        self.control_lock.release() # Never release lock as this is final command

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def single_zero_torque(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)

    def step_robot(self, action):
        self.control_lock.acquire()
        if self.remote_controller.button[KeyMap.select] == 1:
            self.damping_state()
            self.terminated = True
        if self.terminated:
            print("Exiting...")
            exit(-1)
        target_q = np.zeros(29, dtype=np.float32)
        target_q[LOCK_WAIST_IDX] = action
        self.pd_control(target_q, LOW_KP if self.low_pd else KP, LOW_KD if self.low_pd else KD)
        # time.sleep(self.control_dt)
        self.control_lock.release()

class UnitreeRobot29:
    def __init__(self, dt = 1/50., net=None, one_robot=False, low_pd=False):
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.action = np.zeros(29, dtype=np.float32)
        self.target_dof_pos = np.zeros(29, dtype = np.float32)
        self.counter = 0
        self.control_dt = dt
        self.low_pd = low_pd
        assert net is not None
        ChannelFactoryInitialize(0, net)

        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.remote_controller = RemoteController()
        self.control_lock = Lock()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine = 0

        c = pb.connect(pb.GUI)
        if not one_robot:
            self.robot_ref = pb.loadURDF("resources/robots/g1/g1_29dof_rev_1_0.urdf")
        self.robot_cur = pb.loadURDF("resources/robots/g1/g1_29dof_rev_1_0.urdf")

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()
        self.low_state_subscriber_ = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.low_state_subscriber_.Init(self.LowStateHgHandler, 10)
        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        self.terminated = False

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")
    
    def get_robot_state(self, link="root", include_vel=False):
        for i in range(29):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        if link == "root":
            root_pos = pickle.loads(self.redis_client.get("root_pos"))
            root_quat = pickle.loads(self.redis_client.get("root_quat"))
        elif link == "head":
            self.head_pos = pickle.loads(self.redis_client.get("head_pos"))
            self.head_quat = pickle.loads(self.redis_client.get("head_quat"))
            root_pos, root_quat = get_root_pose_from_link(self.robot_cur, 17, self.qj, self.head_pos, self.head_quat)
            # r = Rotation.from_euler("xyz", [0, 0, -self.qj[12]])
            # root_quat = (Rotation.from_quat(self.head_quat) * r).as_quat()
        root_pose = np.concatenate([root_pos, root_quat])
        self.redis_client.set("joint_angle", pickle.dumps(self.qj))
        self.redis_client.set("root_pose", pickle.dumps(root_pose))
        if include_vel:
            return self.qj, np.concatenate([root_pos, root_quat]), self.dqj
        # Should broad cast joint angle and root pose
        return self.qj, root_pose

    def get_head_site(self):
        head_pos = self.head_pos + Rotation.from_quat(self.head_quat).apply(np.array([0.,0.,0.444]))
        head_quat = self.head_quat
        return head_pos, head_quat

    def get_imu_quat(self):
        return self.low_state.imu_state.quaternion
    
    def get_omega(self):
        return np.array(self.low_state.imu_state.gyroscope)

    def pd_control(self, target_q, kp, kd):
        assert len(target_q) == 29
        for i in range(len(target_q)):
            self.low_cmd.motor_cmd[i].q = target_q[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = kp[i]
            self.low_cmd.motor_cmd[i].kd = kd[i]
            self.low_cmd.motor_cmd[i].tau = 0
        self.send_cmd(self.low_cmd)

    # Assume robot in static state, no velocity
    def set_robot_state(self, q): # q 27 dof
        total_time = 2
        num_step = int(total_time/self.control_dt)
        
        init_q = np.zeros(29, dtype=np.float32)
        dof_idx = np.arange(29)
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        for i in dof_idx:
            init_q[i] = self.low_state.motor_state[i].q

        for i in range(num_step):
            alpha = i / num_step
            self.pd_control(init_q*(1-alpha)+target_q*alpha, LOW_KP, LOW_KD)
            time.sleep(self.control_dt)

    def set_pb_robot_state(self, robot, q, root_pose):
        j = 0
        for i in range(pb.getNumJoints(robot)):
            if pb.getJointInfo(robot, i)[2] == pb.JOINT_REVOLUTE:
                pb.resetJointState(robot, i, q[j])
                j += 1
        pb.resetBasePositionAndOrientation(robot, root_pose[:3], root_pose[3:])

    def maintain_state(self, q, root_pose, link="root"):
        print("Maintaining state, wait for A signal...")
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = q
        self.set_pb_robot_state(self.robot_ref, q, root_pose)
        while self.remote_controller.button[KeyMap.A] != 1:
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            self.pd_control(target_q, LOW_KP, LOW_KD)
            cur_q, cur_root_pose = self.get_robot_state(link=link)
            self.set_pb_robot_state(self.robot_cur, cur_q, cur_root_pose)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def damping_state(self):
        print("Entering damping state")
        self.control_lock.acquire()
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        self.terminated = True
        self.control_lock.release() # Never release lock as this is final command

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            self.control_lock.acquire()
            if self.terminated:
                print("Exiting...")
                exit(-1)
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)
            self.control_lock.release()

    def single_zero_torque(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.control_dt)

    def step_robot(self, action):
        self.control_lock.acquire()
        if self.remote_controller.button[KeyMap.select] == 1:
            self.damping_state()
            self.terminated = True
        if self.terminated:
            print("Exiting...")
            exit(-1)
        target_q = np.zeros(29, dtype=np.float32)
        target_q[:] = action
        self.pd_control(target_q, LOW_KP, LOW_KD)
        # time.sleep(self.control_dt)
        self.control_lock.release()



