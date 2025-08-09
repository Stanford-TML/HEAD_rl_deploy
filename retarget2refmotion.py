import isaacgym
import argparse, sys, os
import joblib
import numpy as np
from scipy.spatial.transform import Rotation
from ref_motion import load_mjcf, compute_motion
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, help="target file")
parser.add_argument("--data", nargs="?", type=str, default="amass_curated_large_noback.pkl")
parser.add_argument("--character", nargs="?", type=str, default="resources/robots/g1/g1_27dof_rev_1_0_track.xml")
parser.add_argument("--split", type=int, default=0, help="split the whole data into several parts")
parser.add_argument("--mirror", action="store_true", default=False, help="mirror the motion")
args = parser.parse_args()

skeleton = load_mjcf(args.character)
FPS = 30


data = joblib.load(args.data)
mjcf_joint_order = ['left_hip_pitch_link', 
                    'left_hip_roll_link', 
                    'left_hip_yaw_link', 
                    'left_knee_link', 
                    'left_ankle_pitch_link', 
                    'left_ankle_roll_link', 
                    'right_hip_pitch_link', 
                    'right_hip_roll_link', 
                    'right_hip_yaw_link', 
                    'right_knee_link', 
                    'right_ankle_pitch_link', 
                    'right_ankle_roll_link', 
                    'waist_yaw_link', 
                    'waist_roll_link', 
                    'torso_link', 
                    'left_shoulder_pitch_link', 
                    'left_shoulder_roll_link', 
                    'left_shoulder_yaw_link', 
                    'left_elbow_link', 
                    'left_wrist_roll_link', 
                    'left_wrist_pitch_link', 
                    'left_wrist_yaw_link', 
                    'right_shoulder_pitch_link', 
                    'right_shoulder_roll_link', 
                    'right_shoulder_yaw_link', 
                    'right_elbow_link', 
                    'right_wrist_roll_link', 
                    'right_wrist_pitch_link', 
                    'right_wrist_yaw_link'] # 29
mirror_joint_order = ['right_hip_pitch_link',
                      'right_hip_roll_link',
                      'right_hip_yaw_link',
                        'right_knee_link',
                        'right_ankle_pitch_link',
                        'right_ankle_roll_link',
                        'left_hip_pitch_link',
                        'left_hip_roll_link',
                        'left_hip_yaw_link',
                        'left_knee_link',
                        'left_ankle_pitch_link',
                        'left_ankle_roll_link',
                        'waist_yaw_link',
                        # 'waist_roll_link',
                        # 'torso_link',
                        'right_shoulder_pitch_link',
                        'right_shoulder_roll_link',
                        'right_shoulder_yaw_link',
                        'right_elbow_link',
                        'right_wrist_roll_link',
                        'right_wrist_pitch_link',
                        'right_wrist_yaw_link',
                        'left_shoulder_pitch_link',
                        'left_shoulder_roll_link',
                        'left_shoulder_yaw_link',
                        'left_elbow_link',
                        'left_wrist_roll_link',
                        'left_wrist_pitch_link']
G1_SIGN_MIRROR = np.array([1,-1, -1, 1, 1, -1,
                           1,-1, -1, 1, 1, -1,
                           -1, #-1, 1,
                           1, -1, -1, 1, -1, 1, -1,
                           1, -1, -1, 1, -1, 1, -1])
n_data = len(data)
max_name_len = max(len(n) for n in data.keys())
motions = []
split = max(1, args.split)

data_iter = iter(data.items())
for bid, batch in enumerate(np.array_split(np.arange(n_data), split)):
    motions = []
    for i in batch:
        #if i == 20: break
        n, traj = next(data_iter)
        sys.stdout.write("\r{}/{}: {}{}".format(i+1, n_data, n, " "*(max_name_len-len(n))))
        frames = []
        for fid in range(len(traj["dof"])):
            dof = traj["dof"][fid]
            traj["root_trans_offset"][fid][2] += 0.05
            frame = {"pelvis": [traj["root_trans_offset"][fid], traj["root_rot"][fid]]}
            # frame = ["\"pelvis\": [["+", ".join(list(map(str, traj["root_trans_offset"][fid])))+"], ["+", ".join(list(map(str, traj["root_rot"][fid])))+"]]"]
            for i, link_name in enumerate(mjcf_joint_order):
                if "roll" in link_name:
                    axis = [1, 0, 0]
                elif "pitch" in link_name:
                    axis = [0, 1, 0]
                elif "yaw" in link_name:
                    axis = [0, 0, 1]
                else: # torso, elbow, knee
                    axis = [0, 1, 0]
                angle = dof[i]
                q = Rotation.from_rotvec(angle*np.array(axis)).as_quat() # x, y, z, w
                frame[link_name] = q
            #     frame.append("\"{}\": [".format(link_name)+", ".join(list(map(str, q)))+"]")
            # frames.append(", ".join(frame))
            frames.append(frame)
            
        r, t = [], []
        for frame in frames:
            r.append([])
            t.append([])
            for joint in skeleton.nodes:
                if joint in frame:
                    q = frame[joint]
                    if len(q) == 2:
                        p, q = q[0], q[1]
                        assert (len(p) == 3 and len(q) == 4) or (len(p) == 4 and len(q) == 3)
                        if len(p) == 4 and len(q) == 3:
                            p, q = q, p
                    elif len(q) == 3:
                        # translation
                        p, q = q, [0.,0.,0.,1.]
                    elif len(q) == 4:
                        p = [0.,0.,0.]
                    else:
                        assert len(frame[joint]) in [2,3,4]
                else:
                    q = [0.,0.,0.,1.]
                    p = [0.,0.,0.]
                r[-1].append(q)
                t[-1].append(p)
        r = torch.from_numpy(np.array(r))
        t = torch.from_numpy(np.array(t))
        m = compute_motion(FPS, skeleton, r, t)
        motions.append(m)

        print("\nDumping...")
        joblib.dump(motions, args.target+f"_{bid}.pkl")
    if args.mirror:
        data_iter = iter(data.items())
        #motions = []
        for i in batch:
            #if i == 20: break
            n, traj = next(data_iter)
            sys.stdout.write("\r{}/{}: {}{}".format(i+1, n_data, n, " "*(max_name_len-len(n))))
            frames = []
            for fid in range(len(traj["dof"])):
                dof = traj["dof"][fid]
                #traj["root_trans_offset"][fid][2] #+= 0.05
                traj["root_trans_offset"][fid][1] *= -1
                #breakpoint()
                #traj["root_rot"][fid][1] *= -1
                traj["root_rot"][fid][0] *= -1
                traj["root_rot"][fid][2] *= -1
                #breakpoint()
                frame = {"pelvis": [traj["root_trans_offset"][fid], traj["root_rot"][fid]]}
                # frame = ["\"pelvis\": [["+", ".join(list(map(str, traj["root_trans_offset"][fid])))+"], ["+", ".join(list(map(str, traj["root_rot"][fid])))+"]]"]
                for i, link_name in enumerate(mirror_joint_order):
                    if "roll" in link_name:
                        axis = [1, 0, 0]
                    elif "pitch" in link_name:
                        axis = [0, 1, 0]
                    elif "yaw" in link_name:
                        axis = [0, 0, 1]
                    else: # torso, elbow, knee
                        axis = [0, 1, 0]
                    angle = G1_SIGN_MIRROR[i]*dof[i]
                    q = Rotation.from_rotvec(angle*np.array(axis)).as_quat() # x, y, z, w
                    frame[link_name] = q 
                #     frame.append("\"{}\": [".format(link_name)+", ".join(list(map(str, q)))+"]")
                # frames.append(", ".join(frame))
                frames.append(frame)
                
            r, t = [], []
            for frame in frames:
                r.append([])
                t.append([])
                for joint in skeleton.nodes:
                    if joint in frame:
                        q = frame[joint]
                        if len(q) == 2:
                            p, q = q[0], q[1]
                            assert (len(p) == 3 and len(q) == 4) or (len(p) == 4 and len(q) == 3)
                            if len(p) == 4 and len(q) == 3:
                                p, q = q, p
                        elif len(q) == 3:
                            # translation
                            p, q = q, [0.,0.,0.,1.]
                        elif len(q) == 4:
                            p = [0.,0.,0.]
                        else:
                            assert len(frame[joint]) in [2,3,4]
                    else:
                        q = [0.,0.,0.,1.]
                        p = [0.,0.,0.]
                    r[-1].append(q)
                    t[-1].append(p)
            r = torch.from_numpy(np.array(r))
            t = torch.from_numpy(np.array(t))
            m = compute_motion(FPS, skeleton, r, t)
            motions.append(m)

            print("\nDumping...")
            joblib.dump(motions, args.target+f"_{bid}.pkl")


#     with open("data/isaacgym.v2/{}.json".format(n), "w") as f:
#         f.write("""{
#     "fps": 30,
#     "frames": [
#         {""" + "},\n        {".join(frames) + """}
#     ]
# }""")

