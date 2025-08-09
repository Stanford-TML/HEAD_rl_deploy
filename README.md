# G1 Sim2Sim & Sim2Real deployment
> Official implementation of deployment part for Hand Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching [Paper](https://arxiv.org/abs/2508.03068) and [Website](https://stanford-tml.github.io/HEAD).


### Installation
1. Install Unitree SDK python repo [link](https://github.com/unitreerobotics/unitree_sdk2) (If there is any problem, try install from git repo instead of pypi)
2. Install `isaac_gym` and other python dependencies `pip install -r requirements.txt`
3. We use redis for interprocess communication between target server, mocap module and whole body controller, please install `redis-server`

### Stream head pose via mocap
Please read this to setup mocap and streaming head pose [link](https://github.com/Ericcsr/G1_localization.git)

### Run whole body control policy
#### Sim2Sim test policy trained in IsaacGym in Mujoco
Create a folder inside `exp_out`, with the name of `<checkpoint_dir>`, put the trained policy checkpoint inside the new folder
```
python mujoco_client.py  --ckpt <checkpoint_dir>
```
#### Sim2Real test policy in real robot
1. Connect to G1 via ethernet cable
2. Get network interface via `ifconfig`, it should look like `enp8s0` in following output
```
enp8s0: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
        ether fc:5c:ee:19:93:a6  txqueuelen 1000  (Ethernet)
        RX packets 1282866  bytes 1674739173 (1.6 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 661  bytes 343250 (343.2 KB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
1. Run code
```
python robot_client.py --ckpt <trained_ckeckpoint> --net <net_interface>
```
1. The code will first run simulation to initialize robot pose, after 100 steps, press `start` on remote controller to initialize real robot to default pose. Adjust robot pose to make sure real robot aligned with final state in simulation in pybullet visualizer. When robot is aligned, press A to activate robot control
2. **WARNING**: G1 Robot has a large range of motion and powerful motors that can be very dangerous. When doing experiment, it is strongly advised to be accompanied by another labmate. Press `esc` on keyboard or `select` on remote controller will immediately stop all joint movement, make sure there is always someone who can access keyboard.

### Run target server
Please make sure whole body controller is already running, running target server before starting whole body controller can produce undefined behavior
#### Keyboard control
```
python target_server/keyboard_server.py
```

#### Navigation and reaching
In first terminal run navigation server
```
python target_server/navigation_server.py
```
In another terminal run reaching server
```
python target_server/reaching_server.py
```

## Contribution
1. [Sirui Chen](https://ericcsr.github.io) and [Zi-Ang Cao](https://zi-ang-cao.github.io/) develop this sim2sim and sim2real deployment code repo
2. [Yanjie Ze](yanjieze.com) Provide useful suggestions for building infra and sim2sim testing

## Citation
```
@article{chen2025hand,
  title={Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching},
  author={Chen, Sirui and Ye, Yufei and Cao, Zi-Ang and Lew, Jennifer and Xu, Pei and Liu, C Karen},
  journal={arXiv preprint arXiv:2508.03068},
  year={2025}
}
```