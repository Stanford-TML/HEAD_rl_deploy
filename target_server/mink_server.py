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

import os
file_dir = os.path.dirname(os.path.abspath(__file__))

LOCK_WAIST_IDX = [0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, 11,
                  12, 13, 14,
                  15, 16, 17, 18, 19, 20, 21,
                  22, 23, 24, 25, 26, 27, 28]

vlock = Lock()
v = np.zeros((3,13), dtype=np.float32)
def on_press(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.up:
            v[:,0] = 0.02
        elif key == keyboard.Key.down:
            v[:,0] = -0.02
        elif key == keyboard.Key.left:
            v[:,1] = -0.02
        elif key == keyboard.Key.right:
            v[:,1] = 0.02
        elif key == keyboard.Key.shift:
            v[:,2] = 0.02
        elif key == keyboard.Key.ctrl:
            v[:,2] = -0.02
        elif key.char == "w":
            v[1:,0] = 0.01
        elif key.char == "s":
            v[1:,0] = -0.01
        elif key.char == "q":
            v[1,1] = -0.01
            v[2,1] = 0.01
        elif key.char == "e":
            v[1,1] = 0.01
            v[2,1] = -0.01
        elif key.char == "a":
            v[1:,2] = -0.01
        elif key.char == "d":
            v[1:,2] = 0.01
    except:
        pass
    vlock.release()

def on_release(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.up:
            v[:,0] = 0.0
        elif key == keyboard.Key.down:
            v[:,0] = 0.0
        elif key == keyboard.Key.left:
            v[:,1] = 0.0
        elif key == keyboard.Key.right:
            v[:,1] = 0.0
        elif key == keyboard.Key.shift:
            v[:,2] = 0.0
        elif key == keyboard.Key.ctrl:
            v[:,2] = 0.0
        elif key.char == "w":
            v[1:,0] = 0.0
        elif key.char == "s":
            v[1:,0] = 0.0
        elif key.char == "q":
            v[1,1] = 0.0
            v[2,1] = 0.0
        elif key.char == "e":
            v[1,1] = 0.0
            v[2,1] = 0.0
        elif key.char == "a":
            v[1:,2] = 0.0
        elif key.char == "d":
            v[1:,2] = 0.0
    except:
        pass
    vlock.release()


class MinkSolver:
    def __init__(self):
        model = mujoco.MjModel.from_xml_path(f"{file_dir}/../resources/robots/g1/mink_scene.xml")

        configuration = mink.Configuration(model)
        self.feet = ["right_foot", "left_foot"]
        self.hands = ["right_palm", "left_palm"]

        self.tasks = [
            pelvis_orientation_task := mink.FrameTask(
                frame_name="pelvis",
                frame_type="body",
                position_cost=0.2,
                orientation_cost=0.5,
                lm_damping=1.0,
            ),
            torso_orientation_task := mink.FrameTask(
                frame_name="torso_link",
                frame_type="body",
                position_cost=0.,
                orientation_cost=0.5,
                lm_damping=1.0,
            ),
            posture_task := mink.PostureTask(model, cost=0.8),
            #posture_task := mink.PostureTask(model, cost=damping_cost),
            com_task := mink.ComTask(cost=10.0),
        ]
        self.com_task = com_task
        self.pelvis_orientation_task = pelvis_orientation_task
        self.torso_orientation_task = torso_orientation_task
        self.posture_task = posture_task
        self.feet_tasks = []
        for foot in self.feet:
            task = mink.FrameTask(
                frame_name=foot,
                frame_type="site",
                position_cost=10.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )
            self.feet_tasks.append(task)
        self.tasks.extend(self.feet_tasks)
        self.hand_tasks = []
        for hand in self.hands:
            task = mink.FrameTask(
                frame_name=hand,
                frame_type="site",
                position_cost=5.0,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            self.hand_tasks.append(task)
        self.tasks.extend(self.hand_tasks)

        # Enable collision avoidance between the following geoms.
        # left hand - table, right hand - table
        # left hand - left thigh, right hand - right thigh
        collision_pairs = [
            # (["left_hand_collision", "right_hand_collision"], ["table"]),
            (["left_hand_collision"], ["left_thigh"]),
            (["right_hand_collision"], ["right_thigh"]),
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        self.limits = [
            mink.ConfigurationLimit(model),
            collision_avoidance_limit,
        ]

        self.com_mid = model.body("com_target").mocapid[0]
        self.feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in self.feet]
        self.hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in self.hands]

        self.model = configuration.model
        self.data = configuration.data
        self.solver = "osqp"
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        configuration.update_from_keyframe("stand")
        #configuration.update_from_keyframe("zeros")
        # configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(self.hands, self.feet):
            mink.move_mocap_to_frame(self.model, self.data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")
        self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
        self.configuration = configuration
        self.rate = RateLimiter(frequency=200.0, warn=False)

    def solve(self, target=None):
        # task = mink.FrameTask(
        #         frame_name="right_palm",
        #         frame_type="site",
        #         position_cost=1.0,
        #         orientation_cost=1.0,
        #         lm_damping=1.0,
        #     )
        # self.hand_tasks[0] = task
        # self.tasks[-2] = task
        self.com_task.set_target(self.data.mocap_pos[self.com_mid])
        if target is not None:
            self.data.mocap_pos[self.hands_mid[1]] = target[1,:3]
            self.data.mocap_pos[self.hands_mid[0]] = target[2,:3]
        for i, (hand_task, foot_task) in enumerate(zip(self.hand_tasks, self.feet_tasks)):
            foot_task.set_target(mink.SE3.from_mocap_id(self.data, self.feet_mid[i]))
            hand_task.set_target(mink.SE3.from_mocap_id(self.data, self.hands_mid[i]))

        vel = mink.solve_ik(
            self.configuration, self.tasks, self.rate.dt, self.solver, 1e-1, limits=self.limits
        )
        self.configuration.integrate_inplace(vel, self.rate.dt)
        mujoco.mj_camlight(self.model, self.data)

        # Note the below are optional: they are used to visualize the output of the
        # fromto sensor which is used by the collision avoidance constraint.
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_sensorPos(self.model, self.data)
        self.viewer.sync()
        self.rate.sleep()
        return self.get_target()

    def get_SE3(self, name):
        link_pos = self.data.body(name).xpos.copy()
        link_orn = self.data.body(name).xquat.copy()
        return mink.SE3.from_rotation_and_translation(translation=link_pos, rotation=mink.SO3(link_orn))

    # one step
    def set_robot_state(self, q, root_pos):
        full_q = np.zeros(29)
        full_q[LOCK_WAIST_IDX] = q
        self.data.qpos[7:36] = q
        self.data.qpos[:7] = root_pos[[0,1,2,6,3,4,5]]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_sensorPos(self.model, self.data)
        for hand, foot in zip(self.hands, self.feet):
            mink.move_mocap_to_frame(self.model, self.data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(self.model, self.data, f"{hand}_target", hand, "site")
        #self.pelvis_orientation_task.set_target(mink.SE3.from_rotation_and_translation)
        mink.move_mocap_to_frame(self.model, self.data, "com_target", "pelvis", "body")
        self.data.mocap_pos[self.com_mid] = self.data.subtree_com[1]
        self.pelvis_orientation_task.set_target(self.get_SE3("pelvis"))
        self.torso_orientation_task.set_target(self.get_SE3("torso_link"))
        self.viewer.sync()

    def get_target(self):
        target = np.zeros((3, 13))
        target[1, :3] = self.data.site("left_palm").xpos.copy()
        target[2, :3] = self.data.site("right_palm").xpos.copy()
        target[0, :3] = self.data.body("head_track_site").xpos.copy()
        target[0, 3:7] = Rotation.from_matrix(self.data.body("head_track_site").xmat.reshape(3,3)).as_quat()
        target[1, 3:7] = Rotation.from_matrix(self.data.site("left_palm").xmat.reshape(3,3)).as_quat()
        target[2, 3:7] = Rotation.from_matrix(self.data.site("right_palm").xmat.reshape(3,3)).as_quat()
        return target

if __name__ == "__main__":
    import time
    ik_solver = MinkSolver()
    ts = time.time()
    updated = False
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    target = pickle.loads(redis_client.get("current_target"))
    q = pickle.loads(redis_client.get("joint_angle"))
    root_pose = pickle.loads(redis_client.get("root_pose"))
    ik_solver.set_robot_state(q, root_pose)
    ik_solver.solve(target)
    while True:
        vlock.acquire()
        target += v
        vlock.release()
        solved_target = ik_solver.solve()
        #target[1:,3:7] = solved_target[1:,3:7]
        redis_client.set("target", pickle.dumps(solved_target))
        redis_client.set("server_ready", "true")
        
