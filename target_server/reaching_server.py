import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
import mink

import numpy as np
import redis.client
from scipy.spatial.transform import Rotation
import redis
import pickle
from pynput import keyboard
from threading import Lock
import cv2
import pybullet as pb
import open3d as o3d
from mink_server import MinkSolver
from hand_utils import left_modes_joints, right_modes_joints
LOCK_WAIST_IDX = [0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11,
                  12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28]




class CameraToPoint:
    ISAAC_JOINT_MAP = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                       "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                       "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", 
                       "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint","left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                       "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint","right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
    camera_orn_offset = Rotation.from_euler('xyz', [0, np.pi/90, 0])
    camera_pos_offset = np.array([0, 0, 0])
    def __init__(self, camera_client):
        pb.connect(pb.DIRECT)
        self.redis_client = camera_client
        self.intrinsics = pickle.loads(self.redis_client.get("rs_intrinsics"))
        self.fx, self.fy, self.cx, self.cy = self.intrinsics
        self.robot_model = pb.loadURDF("../resources/robots/g1/g1_27dof_rev_1_0_d435.urdf", flags=pb.URDF_MERGE_FIXED_LINKS)
        self.joint_order = []
        special_joints = ["left_wrist_yaw_joint", "right_wrist_yaw_joint", "d435_joint"]
        self.special_joint_idx = {}
        for i in range(pb.getNumJoints(self.robot_model)):
            joint_info = pb.getJointInfo(self.robot_model, i)
            if joint_info[1].decode() in special_joints:
                self.special_joint_idx[joint_info[1].decode()] = i
            if joint_info[2] == pb.JOINT_REVOLUTE:
                self.joint_order.append(joint_info[1].decode())
        #breakpoint()
        self.reindex = [CameraToPoint.ISAAC_JOINT_MAP.index(joint_name) for joint_name in self.joint_order]
        self.pcd = o3d.geometry.PointCloud()
        #self.vis = o3d.visualization.Visualizer()
        #self.vis.create_window()
        #self.vis_init = True

    def get_image(self):
        jpeg_bytes = self.redis_client.get("manip_camera_stream")
        depth_image = pickle.loads(self.redis_client.get("manip_camera_depth"))
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) 
        return color_image, depth_image
    
    def select_point(self, color_image):
        self._get_camera_pose()
        point = [0, 0]
        def on_mouse(event, x, y, flags, userdata):
            if event != cv2.EVENT_LBUTTONDOWN:
                return    
            cv2.circle(color_image, (x,y), 5, (0, 255, 0), -1)
            point[:] = [x, y]
            print("Point:", point)
        cv2.namedWindow("Select")
        cv2.setMouseCallback("Select", on_mouse)
        while point == [0, 0]:
            cv2.imshow("Select", color_image)
            cv2.waitKey(1)
        return point

    def wait_for_point(self):
        self.redis_client.set("goal_opencv_xy_in_raw_manip_cam_frame", pickle.dumps(np.array([-1,-1])))
        while pickle.loads(self.redis_client.get("goal_opencv_xy_in_raw_manip_cam_frame"))[0] < 0:
            time.sleep(0.01)
        point = pickle.loads(self.redis_client.get("goal_opencv_xy_in_raw_manip_cam_frame"))
        self._get_camera_pose()
        return point.tolist()

    def _set_pb_robot_state(self, q, root_pose):
        q = q[self.reindex]
        jid = 0
        for i in range(len(q)):
            if pb.getJointInfo(self.robot_model, jid)[2] == pb.JOINT_REVOLUTE:
                pb.resetJointState(self.robot_model, jid, q[i])
            else:
                jid += 1
                pb.resetJointState(self.robot_model, jid, q[i])
            jid += 1
        pb.resetBasePositionAndOrientation(self.robot_model, root_pose[:3], root_pose[3:])

    def _get_camera_pose(self):
        root_pose = pickle.loads(self.redis_client.get("root_pose"))
        robot_q = pickle.loads(self.redis_client.get("joint_angle"))
        self._set_pb_robot_state(robot_q,root_pose)
        camera_info = pb.getLinkState(self.robot_model, self.special_joint_idx["d435_joint"])
        camera_pos = camera_info[0] + Rotation.from_quat(camera_info[1]).apply(CameraToPoint.camera_pos_offset)
        camera_orn = Rotation.from_quat(camera_info[1]) * CameraToPoint.camera_orn_offset
        self.camera_pos, self.camera_orn = camera_pos, camera_orn

    def get_3dpoint(self, point, depth, toworld=False):
        # Convert pixel coordinates to normalized device coordinates
        z = depth[point[1], point[0]]
        x = (point[0] - self.cx)*z / self.fx
        y = (point[1] - self.cy)*z / self.fy
        #point = np.array([x, y, z])
        point = np.array([z+0.1, -x, -y]) # add 0.05 offset to guarantee to touch the object
        if toworld:
            # transform point to world frame
            point = self.camera_orn.apply(point) + self.camera_pos

        point[-1] += 0.05
        return point
    
    def compute_pcd(self, color_image, depth_image, point,toworld=False):
        # Convert pixel coordinates to normalized device coordinates
        h, w = depth_image.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_image.flatten()
        x = ((x.flatten() - self.cx) * z) / self.fx
        y = ((y.flatten() - self.cy) * z) / self.fy
        points = np.vstack((z, -x, -y)).T
        if toworld:
            # transform point to world frame
            points = self.camera_orn.apply(points) + self.camera_pos
        # visualize pcd
        
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        #o3d.visualization.draw_geometries([self.pcd, frame])
        
    
dist_thresh = 0.5
if __name__ == "__main__":
    import time
    ik_solver = MinkSolver()
    ts = time.time()
    updated = False
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    pcd_client = redis.Redis(host="192.168.123.222", port=6379, db=0)
    camera = CameraToPoint(pcd_client)
    target = pickle.loads(redis_client.get("current_target"))
    q = pickle.loads(redis_client.get("joint_angle"))
    root_pose = pickle.loads(redis_client.get("root_pose"))
    pcd_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[0],right_modes_joints[0]])))
    ik_solver.set_robot_state(q, root_pose)
    ik_solver.solve(target)

    while True:
        point2d = camera.wait_for_point()
        target = pickle.loads(redis_client.get("current_target"))
        q = pickle.loads(redis_client.get("joint_angle"))
        root_pose = pickle.loads(redis_client.get("root_pose"))
        root_orn = Rotation.from_quat(root_pose[3:7])
        
        color_image, depth_image = camera.get_image()

        #point2d = camera.select_point(color_image)
        point = camera.get_3dpoint(point2d, depth_image, toworld=True)
        # root2point = point - root_pose[:3]
        # root2point_len = np.linalg.norm(root2point)
        # root_level_delta = root2point * (1/np.linalg.norm(root2point)) * (root2point_len - dist_thresh)


        ik_solver.set_robot_state(q, root_pose)
        ik_solver.solve(target)
        #camera.compute_pcd(color_image, depth_image, point, toworld=True)
        # linear interpolate left hand from current target to point
        dist_left = np.linalg.norm(target[1,:3] - point)
        dist_right = np.linalg.norm(target[2,:3] - point)

        if dist_left >= dist_right:
            hand_id = 2
            pcd_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[0],right_modes_joints[3]])))
        else:
            hand_id = 1
            pcd_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[3],right_modes_joints[0]])))
        interp = 25
        traj_hand = []
        zoffset = 0.3
        yoffset = -0.2 if hand_id == 1 else 0.2
        for i in range(interp):
            traj_hand.append(target[hand_id,:3]+root_orn.apply(np.array([0, i*yoffset/interp, i*zoffset/interp])))
        for i in range(interp):
            traj_hand.append((1-i/interp)*traj_hand[interp-1][:3] + (i/interp)*point)
        idx = 0
        print("Use hand id:", hand_id)
        while True:
            # color_image, depth_image = camera.get_image()
            # point = camera.get_3dpoint(point2d, depth_image, toworld=True)
            # camera.compute_pcd(color_image, depth_image, point, toworld=True)
            if idx < len(traj_hand)-1 and idx >= 0:
                idx += 1
            else:
                idx = -1
                if hand_id == 1:
                    pcd_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[4],right_modes_joints[0]])))
                else:
                    pcd_client.set("both_target_q", pickle.dumps(np.hstack([left_modes_joints[0],right_modes_joints[4]])))
            target[hand_id,:3] = traj_hand[idx]
            solved_target = ik_solver.solve(target)
            redis_client.set("target", pickle.dumps(solved_target))
            redis_client.set("server_ready", "true")
            time.sleep(0.1)
