from typing import Callable, Optional, Union, List, Dict, Any
import os
from isaacgym import gymapi, gymtorch
import torch

from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized, quat2axang, expmap2quat, orientation_error, mirror_quaternion
from isaacgym.torch_utils import torch_rand_float
def parse_kwarg(kwargs: dict, key: str, default_val: Any):
    return kwargs[key] if key in kwargs else default_val

class ObsQueue:
    def __init__(self, max_size, stride=4):
        self.max_size = max_size * stride
        self.stride = stride
        self._queue = []

    def push(self, obs):
        if len(self._queue) == 0:
            self._queue = self._queue + [obs] * self.max_size
        elif len(self._queue) == self.max_size:
            self._queue.pop(0)
            self._queue.append(obs.clone())
    def __getitem__(self, idx):
        return self._queue[idx*self.stride]    

    def __len__(self):
        return len(self._queue) // self.stride

class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: Optional[bool]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight

class Env(object):
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS= 0, -4.5, 2.0
    CAMERA_FOLLOWING = True

    def __init__(self,
        n_envs: int, fps: int=30, frameskip: int=2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,

        render_to: Optional[str] = None,

        **kwargs
    ):
        self.viewer = None
        self.render_to = render_to

        assert(control_mode in ["position", "torque", "free", "pd"])
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1./self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(compute_device)
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        if graphics_device is None:
            graphics_device = compute_device
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if type(self.character_model) == str:
            self.character_model = [self.character_model]

        self.use_domain_rand = parse_kwarg(kwargs, "use_domain_rand", False)
        self.friction_range = parse_kwarg(kwargs, "friction_range", (0.5, 1.25))
        self.mass_range = parse_kwarg(kwargs, "mass_range", (-0.5, 0.5))
        self.stiffness_range = parse_kwarg(kwargs, "stiffness_range", (0.8, 1.1))
        self.damping_range = parse_kwarg(kwargs, "damping_range", (0.7, 1.3))
        self.torque_limit = parse_kwarg(kwargs, "torque_limit", None)
        self.obs_vel = parse_kwarg(kwargs, "obs_vel", True)
        self.merge = parse_kwarg(kwargs, "merge", "add")
        sim_params = self.setup_sim_params()
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
        self.add_ground()
        self.envs, self.actors, self.actuated_dofs = self.create_envs(n_envs)
        n_actors_per_env = self.gym.get_actor_count(self.envs[0])
        self.actor_ids = torch.arange(n_actors_per_env * len(self.envs), dtype=torch.int32, device=self.device).view(len(self.envs), -1)
        controllable_actors = []
        for i in range(self.gym.get_actor_count(self.envs[0])):
            dof = self.gym.get_actor_dof_count(self.envs[0], i)
            if dof > 0: controllable_actors.append(i)
        self.actor_ids_having_dofs = \
            n_actors_per_env * torch.arange(len(self.envs), dtype=torch.int32, device=self.device).unsqueeze(-1) + \
            torch.tensor(controllable_actors, dtype=torch.int32, device=self.device).unsqueeze(-2)
        self.setup_action_normalizer()
        self.create_tensors()

        

        self.gym.prepare_sim(self.sim)

        self.root_tensor.fill_(0)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
        self.joint_tensor.fill_(0)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_tensor))
        self.root_updated_actors, self.dof_updated_actors = [], []
        self.refresh_tensors()
        self.train()
        self.viewer_pause = False
        self.viewer_advance = False
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        self.cam_target = gymapi.Vec3(*self.vector_up(1.0, [base_pos[0], base_pos[1], base_pos[2]]))
        self.last_action = None
        self.simulation_step = 0
        self.lifetime = torch.zeros(len(self.envs), dtype=torch.int64, device=self.device)
        self.done = torch.ones(len(self.envs), dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)

        self.act_dim = self.action_scale.size(-1)
        self.ob_dim = self.observe().size(-1) + (1 if self.use_feet else 0)
        self.rew_dim = self.reward().size(-1)

        for i in range(self.gym.get_actor_count(self.envs[0])):
            rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], i)
            print("Links", sorted(rigid_body.items(), key=lambda x:x[1]), len(rigid_body))
            dof = self.gym.get_actor_dof_dict(self.envs[0], i)
            print("Joints", sorted(dof.items(), key=lambda x:x[1]), len(dof))

    def __del__(self):
        if hasattr(self, "viewer") and self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
        if hasattr(self, "sim"):
            self.gym.destroy_sim(self.sim)

    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def vector_up(self, val: float, base_vector=None):
        if base_vector is None:
            base_vector = [0., 0., 0.]
        base_vector[self.UP_AXIS] = val
        return base_vector
    
    def setup_sim_params(self, physx_params=dict()):
        p = gymapi.SimParams()
        p.dt = self.step_time/self.frameskip
        p.substeps = self.substeps
        p.up_axis = gymapi.UP_AXIS_Z if self.UP_AXIS == 2 else gymapi.UP_AXIS_Y
        p.gravity = gymapi.Vec3(*self.vector_up(-9.81))
        p.num_client_threads = 0
        p.physx.num_threads = 10
        p.physx.solver_type = 1
        p.physx.num_subscenes = 4  # works only for CPU 
        p.physx.num_position_iterations = 4
        p.physx.num_velocity_iterations = 0
        p.physx.contact_offset = 0.01
        p.physx.rest_offset = 0.0
        p.physx.bounce_threshold_velocity = 0.2
        p.physx.max_depenetration_velocity = 1.0
        p.physx.default_buffer_size_multiplier = 5.0
        p.physx.max_gpu_contact_pairs = 8*1024*1024
        # FIXME IsaacGym Pr4 will provide unreliable results when collecting from all substeps
        p.physx.contact_collection = \
            gymapi.ContactCollection(gymapi.ContactCollection.CC_LAST_SUBSTEP) 
        #gymapi.ContactCollection(gymapi.ContactCollection.CC_ALL_SUBSTEPS)
        for k, v in physx_params.items():
            setattr(p.physx, k, v)
        p.use_gpu_pipeline = True # force to enable GPU
        p.physx.use_gpu = True
        return p

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def create_envs(self, n: int, start_height: float=0.89,  actuate_all_dofs: bool=True, asset_options: Dict[str, Any]=dict()):
        if self.control_mode == "position":
            control_mode = gymapi.DOF_MODE_POS
        elif self.control_mode == "torque" or self.control_mode == "pd":
            control_mode = gymapi.DOF_MODE_EFFORT
        else:
            control_mode = gymapi.DOF_MODE_NONE

        envs, actors = [], []
        env_spacing = 3

        actor_assets = []
        controllable_dofs = []
        for character_model in self.character_model:
            asset_opt = gymapi.AssetOptions()
            asset_opt.angular_damping = 0.01
            asset_opt.max_angular_velocity = 100.0
            asset_opt.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
            for k, v in asset_options.items():
                setattr(asset_opt, k, v)

            # when loading mesh in mjcf, the convex hull settings does not work
            # asset_opt.vhacd_enabled = True
            # asset_opt.vhacd_params.max_convex_hulls = 32
            # asset_opt.vhacd_params.max_num_vertices_per_ch = 64
            # asset_opt.vhacd_params.resolution = 300000
                
            asset = self.gym.load_asset(self.sim,
                os.path.abspath(os.path.dirname(character_model)),
                os.path.basename(character_model),
                asset_opt)
            actor_assets.append(asset)
            if actuate_all_dofs:
                controllable_dofs.append([i for i in range(self.gym.get_asset_dof_count(asset))])
            else:
                actuators = []
                for i in range(self.gym.get_asset_actuator_count(asset)):
                    name = self.gym.get_asset_actuator_joint_name(asset, i)
                    actuators.append(self.gym.find_asset_dof_index(asset, name))
                    if actuators[-1] == -1:
                        raise ValueError("Failed to find joint with name {}".format(name))
                controllable_dofs.append(sorted(actuators) if len(actuators) else [])

        spacing_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        spacing_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        n_envs_per_row = int(n**0.5)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.vector_up(start_height))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        total_rigids = sum([self.gym.get_asset_rigid_body_count(asset) for asset in actor_assets])
        total_shapes = sum([self.gym.get_asset_rigid_shape_count(asset) for asset in actor_assets])

        if self.use_domain_rand:
            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(asset)
            friction_buckets = torch_rand_float(self.friction_range[0], self.friction_range[1], (64,1), device="cpu")

        actuated_dofs = []
        self.joint_stiffness = []
        self.joint_damping = []
        
        for env_id in range(n):
            env = self.gym.create_env(self.sim, spacing_lower, spacing_upper, n_envs_per_row)
            self.gym.begin_aggregate(env, total_rigids, total_shapes, True)
            joint_stiffness = []
            joint_damping = []
            for aid, (asset, dofs) in enumerate(zip(actor_assets, controllable_dofs)):
                actor = self.gym.create_actor(env, asset, start_pose, "humanoid", env_id, -1, 0)
                dof_prop = self.gym.get_asset_dof_properties(asset)
                for k in range(len(dof_prop)):
                    if k in dofs:
                        dof_prop[k]["driveMode"] = control_mode
                        dof_prop[k]["effort"] = self.torque_limit[k] if self.torque_limit is not None else dof_prop[k]["effort"]
                        joint_stiffness.append(dof_prop[k]["stiffness"])
                        joint_damping.append(dof_prop[k]["damping"] * 3.0)
                        if self.use_domain_rand:
                            dof_prop[k]["stiffness"] *= np.random.uniform(self.stiffness_range[0], self.stiffness_range[1])
                            dof_prop[k]["damping"] *= np.random.uniform(self.damping_range[0], self.damping_range[1])
                        if control_mode == gymapi.DOF_MODE_EFFORT: # BUG: Seems that position control co-exist with effort control, should disable positino control first
                            dof_prop[k]["stiffness"] = 0
                            dof_prop[k]["damping"] = 0
                    else:
                        dof_prop[k]["driveMode"] = gymapi.DOF_MODE_NONE
                        dof_prop[k]["stiffness"] = 0
                        dof_prop[k]["damping"] = 0
                self.gym.set_actor_dof_properties(env, actor, dof_prop)
                if env_id == n-1:
                    actors.append(actor)
                    actuated_dofs.append(dofs)
            self.gym.end_aggregate(env)
            envs.append(env)
            self.joint_stiffness.append(joint_stiffness)
            self.joint_damping.append(joint_damping)
            # Domain randomization
            if self.use_domain_rand:
                frictions = friction_buckets[torch.randint(0, 64,(1,)).squeeze()]
                for s in range(len(rigid_shape_props_asset)):
                    rigid_shape_props_asset[s].friction = frictions
                rigid_body_props_asset = self.gym.get_actor_rigid_body_properties(env, actor)
                rigid_body_props_asset[0].mass += np.random.uniform(self.mass_range[0], self.mass_range[1])
                self.gym.set_actor_rigid_body_properties(env, actor, rigid_body_props_asset)
                self.gym.set_actor_rigid_shape_properties(env, actor, rigid_shape_props_asset)
            
        #breakpoint()
        self.joint_stiffness = torch.tensor(self.joint_stiffness, dtype=torch.float32, device=self.device)
        self.joint_damping = torch.tensor(self.joint_damping, dtype=torch.float32, device=self.device)
        self.torque_limit = torch.tensor(self.torque_limit, dtype=torch.float32, device=self.device)
        #breakpoint()
        return envs, actors, actuated_dofs

    def render(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(*self.vector_up(self.camera_pos[2], 
            [base_pos[0]+self.camera_pos[0], base_pos[1]+self.camera_pos[1], base_pos[2]+self.camera_pos[1]]))
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_F, "TOGGLE_CAMERA_FOLLOWING")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_P, "TOGGLE_PAUSE")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "SINGLE_STEP_ADVANCE")
    
    def update_viewer(self):
        self.gym.poll_viewer_events(self.viewer)
        for event in self.gym.query_viewer_action_events(self.viewer):
            if event.action == "QUIT" and event.value > 0:
                exit()
            if event.action == "TOGGLE_CAMERA_FOLLOWING" and event.value > 0:
                self.camera_following = not self.camera_following
            if event.action == "TOGGLE_PAUSE" and event.value > 0:
                self.viewer_pause = not self.viewer_pause
            if event.action == "SINGLE_STEP_ADVANCE" and event.value > 0:
                self.viewer_advance = not self.viewer_advance
        if self.camera_following: self.update_camera()
        self.gym.step_graphics(self.sim)
        self.gym.clear_lines(self.viewer)

    def update_camera(self):
        tar_env = len(self.envs)//4 + int(len(self.envs)**0.5)//2
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, self.envs[tar_env])
        dx, dy = cam_trans.p.x - self.cam_target.x, cam_trans.p.y - self.cam_target.y
        base_pos = self.root_tensor[tar_env, 0, :3].cpu().detach()
        cam_pos = gymapi.Vec3(base_pos[0]+dx, base_pos[1]+dy, cam_trans.p.z)
        self.cam_target = gymapi.Vec3(base_pos[0], base_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[tar_env], cam_pos, self.cam_target)

    def refresh_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def create_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_tensor = gymtorch.wrap_tensor(root_tensor)
        self.root_tensor = root_tensor.view(len(self.envs), -1, 13)

        num_links = self.gym.get_env_rigid_body_count(self.envs[0])
        link_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        link_tensor = gymtorch.wrap_tensor(link_tensor)
        self.link_tensor = link_tensor.view(len(self.envs), num_links, -1)

        num_dof = self.gym.get_env_dof_count(self.envs[0])
        joint_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        joint_tensor = gymtorch.wrap_tensor(joint_tensor)
        self.joint_tensor = joint_tensor.view(len(self.envs), num_dof, -1)  # n_envs x n_dof x 2

        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.contact_force_tensor = contact_force_tensor.view(len(self.envs), -1, 3)

        if self.actuated_dofs.size(-1) == self.joint_tensor.size(1):
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros_like(self.joint_tensor[..., 0])

    def setup_action_normalizer(self):
        actuated_dof = []
        dof_cnts = 0
        action_lower, action_upper = [], []
        action_scale = []
        for i, dofs in zip(range(self.gym.get_actor_count(self.envs[0])), self.actuated_dofs):
            actor = self.gym.get_actor_handle(self.envs[0], i)
            dof_prop = self.gym.get_actor_dof_properties(self.envs[0], actor)
            if len(dof_prop) < 1: continue
            if self.control_mode == "torque":
                action_lower.extend([-dof_prop["effort"][j] for j in dofs])
                action_upper.extend([dof_prop["effort"][j] for j in dofs])
                action_scale.extend([1]*len(dofs))
            else: # self.control_mode == "position":
                action_lower.extend([min(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_upper.extend([max(dof_prop["lower"][j], dof_prop["upper"][j]) for j in dofs])
                action_scale.extend([2]*len(dofs))
            for j in dofs:
                actuated_dof.append(dof_cnts+j)
            dof_cnts += len(dof_prop)
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale *= 0.8 * np.subtract(action_upper, action_lower)
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale, dtype=torch.float32, device=self.device)
        self.actuated_dofs = torch.tensor(actuated_dof, dtype=torch.int64, device=self.device)

    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        self.last_action = a
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dofs] = a
        return self.action_tensor

    def reset(self):
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        self.obs = None

        self.i = 0

    def reset_done(self):
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == len(self.envs) or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
    
    def reset_envs(self, env_ids):
        """
        reset root pos and orientation as well as joint angles
        """
        print("Environment resetted!")
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)
        self.root_tensor[env_ids] = ref_root_tensor
        self.link_tensor[env_ids] = ref_link_tensor
        if self.action_tensor is None:
            self.joint_tensor[env_ids] = ref_joint_tensor
        else:
            self.joint_tensor[env_ids.unsqueeze(-1), self.actuated_dofs] = ref_joint_tensor
        self.root_updated_actors.append(self.actor_ids[env_ids].flatten())
        self.dof_updated_actors.append(self.actor_ids_having_dofs[env_ids].flatten())
        self.lifetime[env_ids] = 0

    def do_simulation(self):
        # root tensor inside isaacgym would be overwritten
        # when set_actor_root_state_tensor is called multiple times before doing simulation
        if self.root_updated_actors:
            actor_ids = torch.unique(torch.cat(self.root_updated_actors))
            if actor_ids.numel() == self.actor_ids.numel():
                self.gym.set_actor_root_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor)
                )
            else:
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.root_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.root_updated_actors.clear()
        if self.dof_updated_actors:
            actor_ids = torch.unique(torch.cat(self.dof_updated_actors))
            if actor_ids.numel() == self.actor_ids_having_dofs.numel():
                self.gym.set_dof_state_tensor(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor)
                )
            else:
                self.gym.set_dof_state_tensor_indexed(self.sim,
                    gymtorch.unwrap_tensor(self.joint_tensor),
                    gymtorch.unwrap_tensor(actor_ids), actor_ids.numel()
                )
            self.dof_updated_actors.clear()
        for _ in range(self.frameskip):
            self.gym.simulate(self.sim)
        self.simulation_step += 1

    def apply_perturbation(self):
        """
        Apply perturbation to robot, by default, no perturbation
        """
        pass

    def step(self, actions):
        # For visualization
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            # print("BEFORE", self.root_tensor[0])
            self.do_simulation()
            self.refresh_tensors()
            self.apply_perturbation()
            # print("AFTER", self.root_tensor[0])
            self.lifetime += 1
            if self.viewer is not None:
                self.gym.fetch_results(self.sim, True)
                self.viewer_advance = False

            if self.render_to:
                self.gym.write_viewer_image_to_file(self.viewer, "{}/frame{:04d}.png".format(self.render_to, self.i))
                self.i += 1
                # if self.i == 1800:
                #     exit()

        if self.viewer is not None:
            self.update_viewer()
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)    # sync to simulation dt

        rewards = self.reward()
        terminate = self.termination_check()                    # N
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check() # NOTE: Will reset goal if overtime.
        # if self.render_to and overtime.item():
        #     exit()
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        self.info["terminate"] = terminate
        self.obs = self.observe()
        self.request_quit = False if self.viewer is None else self.gym.query_viewer_has_closed(self.viewer)
        return self.obs, rewards, self.done, self.info

    def apply_actions(self, actions):
        actions = self.process_actions(actions)
        if self.control_mode == "position":
            #breakpoint()
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_position_target_tensor(self.sim, actions)
        elif self.control_mode == "torque":
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        elif self.control_mode == "pd":
            #breakpoint()
            actions = self.joint_stiffness * (actions - self.joint_tensor[:,self.actuated_dofs, 0]) #- self.joint_damping * self.joint_tensor[:,self.actuated_dofs, 1]
            #actions.clamp_(max=self.torque_limit, min=-self.torque_limit)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, actions)
        else:
            actions = torch.stack((actions, torch.zeros_like(actions)), -1)
            actions = gymtorch.unwrap_tensor(actions)
            self.gym.set_dof_state_tensor(self.sim, actions)

    def init_state(self, env_ids):
        pass
    
    def observe(self, env_ids=None):
        pass
    
    def overtime_check(self, override=False):
        if self.episode_length is not None and not override:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return None

    def termination_check(self):
        return torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)

    def reward(self):
        """
        Dummy Reward as this is an base class
        """
        return torch.ones((len(self.envs), 0), dtype=torch.float32, device=self.device)


from ref_motion import ReferenceMotion
import numpy as np


class ICCGANHumanoid(Env):

    CHARACTER_MODEL = os.path.join("assets", "humanoid.xml")
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2

    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None

    OB_HORIZON = 4
    KEY_LINKS = None    # All links
    PARENT_LINK = None  # root link


    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
    **kwargs):
        contactable_links = parse_kwarg(kwargs, "contactable_links", self.CONTACTABLE_LINKS)
        goal_reward_weight = parse_kwarg(kwargs, "goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = parse_kwarg(kwargs, "enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = parse_kwarg(kwargs, "goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = parse_kwarg(kwargs, "ob_horizon", self.OB_HORIZON)
        self.key_links = parse_kwarg(kwargs, "key_links", self.KEY_LINKS)
        self.parent_link = parse_kwarg(kwargs, "parent_link", self.PARENT_LINK)
        self.continuous_track = parse_kwarg(kwargs, "continuous_track", False)
        super().__init__(*args, **kwargs)
        
        n_envs = len(self.envs)
        n_links = self.char_link_tensor.size(1)
        n_dofs = self.char_joint_tensor.size(1)

        if contactable_links is None:
            self.contactable_links = None
        else:
            # contact = np.zeros((n_envs, n_links), dtype=bool)
            contact = np.full((n_envs, n_links), 0.15)
            if type(contactable_links) != dict:
                contactable_links = {link: -10000 for link in contactable_links}
            for link, h in contactable_links.items():
                lids = []
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid >= 0:
                        contact[:, lid] = h
                        lids.append(lid)
                if not lids: print("[Warning] Unrecognized contactable link {}".format(link))
            self.contactable_links = torch.tensor(contact, dtype=torch.float32).to(self.contact_force_tensor.device)

        if goal_reward_weight is not None:
            reward_weights = torch.empty((len(self.envs), self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(goal_reward_weight, "__len__"):
                goal_reward_weight = [goal_reward_weight]
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight))
            for i, w in zip(range(self.rew_dim), goal_reward_weight):
                reward_weights[:, i] = w
        elif self.rew_dim:
            goal_reward_weight = []
            assert self.rew_dim == len(goal_reward_weight), "{} vs {}".format(self.rew_dim, len(goal_reward_weight)) 

        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert(total_weights <= 1), "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None

        self.discriminators = dict()
        max_ob_horizon = self.ob_horizon+1
        for i, (id, config) in enumerate(discriminators.items()):
            if config.key_links is None:
                key_links = None
            else:
                key_links = []
                for link in config.key_links:
                    for actor in self.actors:
                        lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                        if lid != -1:
                            key_links.append(lid)
                            break
                    assert lid != -1, "Unrecognized key link {}".format(link)
                key_links = sorted(key_links)
            if config.parent_link is None:
                parent_link = None
            else:
                for j in self.actors:
                    parent_link = self.gym.find_actor_rigid_body_handle(self.envs[0], j, config.parent_link)
                    if parent_link != -1: break
                assert parent_link != -1, "Unrecognized parent link {}".format(parent_link)
            assert key_links is None or all(lid >= 0 for lid in key_links)
            assert parent_link is None or parent_link >= 0
            config.parent_link = parent_link
            config.key_links = key_links
            
            if config.motion_file is None:
                config.motion_file = motion_file
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon+1
            config.id = i
            config.name = id
            self.discriminators[id] = config
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
            max_ob_horizon = max(max_ob_horizon, config.ob_horizon)

        if max_ob_horizon != self.state_hist.size(0):
            self.state_hist = torch.zeros((max_ob_horizon, *self.state_hist.shape[1:]),
                dtype=self.root_tensor.dtype, device=self.device)
        if self.reward_weights is None:
            self.reward_weights = torch.ones((n_envs, 1), dtype=torch.float32, device=self.device)
        elif self.rew_dim > 0:
            if self.rew_dim > 1:
                self.reward_weights *= (1-reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1-reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
            
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.goal_dim = self.GOAL_DIM
        self.state_dim = (self.ob_dim-self.goal_dim)//self.ob_horizon
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
            self.info["disc_obs_expert"] = self.info["disc_obs"]        # dummy result
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}

        self.ref_motion, self.root_links = self.build_motion_lib(motion_file)
        self.sampling_workers = []
        self.real_samples = []

    def build_motion_lib(self, motion_file):
        ref_motion = ReferenceMotion(motion_file=motion_file, character_model=self.character_model, device=self.device)
        root_links = [i for i, p in enumerate(ref_motion.skeleton.parents) if p == -1]
        return ref_motion, root_links
    
    def __del__(self):
        if hasattr(self, "sampling_workers"):
            for p in self.sampling_workers:
                p.terminate()
            for p in self.sampling_workers:
                p.join()
        super().__del__()

    def reset_done(self):
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def reset(self):
        if self.goal_tensor is not None:
            self.goal_tensor.zero_()
            if self.goal_timer is not None: self.goal_timer.zero_()
        super().reset()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)
        
    def reset_goal(self, env_ids):
        pass
    
    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, dones, info

    def overtime_check(self):
        if self.goal_timer is not None and not self.continuous_track:
            self.goal_timer -= 1
            env_ids = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(env_ids) > 0: self.reset_goal(env_ids)
        return super().overtime_check(override=self.continuous_track)

    def termination_check(self):
        if self.contactable_links is None:
            return torch.zeros_like(self.done)

        contacted = torch.any(self.char_contact_force_tensor.abs() > 1., dim=-1)      # N x n_links
        too_low = self.link_pos[..., self.UP_AXIS] < self.contactable_links    # N x n_links

        terminate = torch.any(torch.logical_and(contacted, too_low), -1)    # N x
        terminate *= (self.lifetime > 1)
        return terminate

    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor
    
    def create_tensors(self):
        super().create_tensors()
        n_dofs = sum([self.gym.get_actor_dof_count(self.envs[0], actor) for actor in self.actors])
        n_links = sum([self.gym.get_actor_rigid_body_count(self.envs[0], actor) for actor in self.actors])
        self.root_pos, self.root_orient = self.root_tensor[:, 0, :3], self.root_tensor[:, 0, 3:7]
        self.root_lin_vel, self.root_ang_vel = self.root_tensor[:, 0, 7:10], self.root_tensor[:, 0, 10:13]
        self.char_root_tensor = self.root_tensor[:, 0]
        if self.link_tensor.size(1) > n_links:
            self.link_pos, self.link_orient = self.link_tensor[:, :n_links, :3], self.link_tensor[:, :n_links, 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[:, :n_links, 7:10], self.link_tensor[:, :n_links, 10:13]
            self.char_link_tensor = self.link_tensor[:, :n_links]
        else:
            self.link_pos, self.link_orient = self.link_tensor[..., :3], self.link_tensor[..., 3:7]
            self.link_lin_vel, self.link_ang_vel = self.link_tensor[..., 7:10], self.link_tensor[..., 10:13]
            self.char_link_tensor = self.link_tensor
        if self.joint_tensor.size(1) > n_dofs:
            self.joint_pos, self.joint_vel = self.joint_tensor[:, :n_dofs, 0], self.joint_tensor[:, :n_dofs, 1]
            self.char_joint_tensor = self.joint_tensor[:, :n_dofs]
        else:
            self.joint_pos, self.joint_vel = self.joint_tensor[..., 0], self.joint_tensor[..., 1]
            self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor[:, :n_links]
    
        self.state_hist = torch.empty((self.ob_horizon+1, len(self.envs), n_links*13),
            dtype=self.root_tensor.dtype, device=self.device)

        if self.key_links is None:
            self.key_links = None
        else:
            key_links = []
            for link in self.key_links:
                for actor in self.actors:
                    lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, link)
                    if lid != -1:
                        key_links.append(lid)
                        break
                assert lid != -1, "Unrecognized key link {}".format(link)
            self.key_links = key_links
        if self.parent_link is None:
            self.parent_link = None
        else:
            for actor in self.actors:
                lid = self.gym.find_actor_rigid_body_handle(self.envs[0], actor, self.parent_link)
                if lid != -1:
                    parent_link = lid
                    break
            assert lid != -1, "Unrecognized parent link {}".format(self.parent_link)
            self.parent_link = parent_link
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((len(self.envs), dim), dtype=self.root_tensor.dtype, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((len(self.envs), self.goal_tensor_dim), dtype=self.root_tensor.dtype, device=self.device)
        else:
            self.goal_tensor = None
        self.goal_timer = torch.zeros((len(self.envs), ), dtype=torch.int32, device=self.device) if self.enable_goal_timer else None

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.char_link_tensor[env_ids].view(n_envs, -1)
        return self._observe(env_ids)
    
    def _observe(self, env_ids):
        if env_ids is None:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens
            ).flatten(start_dim=1)
        else:
            return observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids]
            ).flatten(start_dim=1)

    def observe_disc(self, state):
        seq_len = self.info["ob_seq_lens"]+1
        res = dict()
        if torch.is_tensor(state):
            # fake
            for id, disc in self.discriminators.items():
                res[id] = observe_iccgan(state[-disc.ob_horizon:], seq_len)
            return res
        else:
            # real
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_iccgan(s[-disc.ob_horizon:], seq_len, disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos)
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def fetch_real_samples(self):
        if not self.real_samples:
            if not self.sampling_workers:
                self.disc_ref_motion = {}
                import torch.multiprocessing as mp
                mp.set_start_method("spawn")
                manager = mp.Manager()
                seed = np.random.get_state()[1][0]
                for n, config in self.discriminators.items():
                    q = manager.Queue(maxsize=1)
                    self.disc_ref_motion[n] = q
                    key_links = None if config.key_links is None else config.key_links
                    if key_links is None:  # all links are key links and observable
                        parent_link_index = config.parent_link
                        key_links_index = None
                    elif config.parent_link is None: # parent link is the root, ensure it appears as the first in the key link list
                        parent_link_index = None
                        if 0 in key_links:
                            key_links = [0] + [_ for _ in key_links if _ != 0] # root link is the first key links
                            key_links_index = None # all links in the key link list are key links for observation
                        else:
                            key_links = [0] + key_links # the root link in the key link list but not for observation
                            key_links_index = list(range(1, len(key_links)+1))
                    else:
                        if config.parent_link in key_links:
                            key_links_index = None
                        else:
                            key_links_index = list(range(1, len(key_links)+1))
                            key_links = [config.parent_link] + key_links
                        parent_link_index = key_links.index(config.parent_link)
                    p = mp.Process(target=self.__class__.ref_motion_sample, args=(q,
                        seed+1+config.id, self.step_time, len(self.envs), config.ob_horizon, key_links_index, parent_link_index, config.local_pos, config.replay_speed,
                        dict(motion_file=config.motion_file, character_model=self.character_model,
                            key_links=key_links, device=self.device
                        )
                    ))
                    p.start()
                    self.sampling_workers.append(p)

            self.real_samples = [{n: None for n in self.disc_ref_motion.keys()} for _ in range(128)]
            for n, q in self.disc_ref_motion.items():
                for i, v in enumerate(q.get()):
                    self.real_samples[i][n] = v.to(self.device)
        return self.real_samples.pop()

    @staticmethod
    def ref_motion_sample(queue, seed, step_time, n_inst, ob_horizon, key_links, parent_link, local_pos, replay_speed, kwargs):
        np.random.seed(seed)
        torch.set_num_threads(1)
        lib = ReferenceMotion(**kwargs)
        if replay_speed is not None:
            replay_speed = eval(replay_speed)
        while True:
            obs = []
            for _ in range(128):
                if replay_speed is None:
                    dt = step_time
                else:
                    dt = step_time * replay_speed(n_inst)
                motion_ids, motion_times0 = lib.sample(n_inst, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate((motion_times0, *[motion_times0+dt*i for i in range(1, ob_horizon)]))
                link_tensor = lib.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = link_tensor.view(ob_horizon, n_inst, -1)
                ob = observe_iccgan(samples, None, key_links, parent_link, include_velocity=False, local_pos=local_pos)
                obs.append(ob.cpu())
            queue.put(obs)

@torch.jit.script
def observe_iccgan(state_hist: torch.Tensor, seq_len: torch.Tensor):
    # state_hist: L x N x (1+N_links) x 13

    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)
    
    link_tensor = state_hist.view(n_hist, n_inst, -1, 13) #  [pos, quat, lin_vel, ang_vel]
    
    link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]

    root_tensor = state_hist[..., :13]
    
    origin = root_tensor[:,:, :3]          # L x N x 3
    orient = root_tensor[:,:,3:7]          # L x N x 4
    

    heading = heading_zup(orient)               # (L x) N
    up_dir = torch.zeros_like(origin)
    up_dir[..., UP_AXIS] = 1                    # (L x) N x 3
    orient_inv = axang2quat(up_dir, -heading)   # (L x) N x 4
    orient_inv = orient_inv.view(-1, n_inst, 1, 4)   # L x N x 1 x 4 or 1 x N x 1 x 4

    origin = origin.clone()
    origin.unsqueeze_(-2)                       # (L x) N x 1 x 3

    ob_link_pos = link_pos - origin                                     # L x N x n_links x 3 
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)              # L x N x n_links x 4
    ob = torch.cat((ob_link_pos, ob_link_orient), -1)               # L x N x n_links x 7
    ob = ob.view(n_hist, n_inst, -1)                                    # L x N x (n_links x 7 or 13)

    ob1 = ob.permute(1, 0, 2)                                           # N x L x (n_links x 7 or 13)

    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2


class TrackingHumanoidPartial(ICCGANHumanoid):
    CAMERA_POS= 0, 4.5, 2.0
    ENABLE_GOAL_TIMER = True

    feet_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    def __init__(self, *args, **kwargs):
        self.test1 = kwargs["test1"] if "test1" in kwargs else False
        self.test2 = kwargs["test2"] if "test2" in kwargs else False

        # perturbation and domain randomization related
        self.max_push_vel = parse_kwarg(kwargs, "max_push_vel", 0)
        self.push_interval = parse_kwarg(kwargs, "push_interval", 200)
        self.obs_tracking_quat = parse_kwarg(kwargs, "obs_tracking_quat", False)

        self.random_init = parse_kwarg(kwargs, "random_init", False)
        self.loop_motion = parse_kwarg(kwargs, "loop_motion", False)
        self.rew_weights = parse_kwarg(kwargs, "reward_weights", None)
        self.verbose = parse_kwarg(kwargs, "verbose", True)
        self.pos_tracking_weights = kwargs["pos_tracking_weights"]
        self.tracking_hands = parse_kwarg(kwargs, "track_hands", True)
        self.reg_joints = kwargs["reg_joints"]
        self.reg_q  = torch.tensor(kwargs["reg_q"])
        assert all([_ >= 0 for _ in self.pos_tracking_weights.values()])
        self.feet_links = None
        self.real_target = None
        self.use_dq = parse_kwarg(kwargs, "use_dq", False)
        self.obs_horizon = parse_kwarg(kwargs, "obs_horizon", 1)
        self.obs_action = parse_kwarg(kwargs, "obs_action", False)
        self.obs_goal = parse_kwarg(kwargs, "obs_goal", False)
        self.obs_frame_skip = parse_kwarg(kwargs, "obs_frame_skip", 2)
        self.obs_queue = ObsQueue(max_size=self.obs_horizon, stride=self.obs_frame_skip)
        if not self.obs_tracking_quat:
            self.GOAL_DIM = 3*len(self.pos_tracking_weights) * self.obs_horizon
        else:
            self.GOAL_DIM = 7*len(self.pos_tracking_weights) * self.obs_horizon
        self.use_feet = parse_kwarg(kwargs, "use_feet", False)
        self.GOAL_DIM += 1 if self.use_feet else 0
        if not self.tracking_hands and self.obs_tracking_quat:
            self.GOAL_DIM -= 8 * self.obs_horizon
    
        self.force_motion = False
        super().__init__(*args, **kwargs)

    def create_tensors(self):
        super().create_tensors()
        self.motion_tracking_ids = np.zeros((len(self.envs)), dtype=int)
        self.motion_tracking_times = np.zeros((len(self.envs)), dtype=float)
        self.up_dir_tensor = torch.zeros(((len(self.envs), 1, 3)), dtype=torch.float, device=self.device)
        self.up_dir_tensor[..., 2] = 1
        self.reg_q = self.reg_q.to(self.device)
        links, n_links = dict(), 0
        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], 0)
        for n, id in rigid_body.items():
            handle = id + n_links
            if n in links:
                links[n].append(handle)
            else:
                links[n] = [handle]
        
        # ID of pos tracking links, sum is to flatten the list
        pos_tracking_links = sum([links[k] for k, v in self.pos_tracking_weights.items()], [])
        # Feet links
        self.feet_links = [links[k][0] for k in TrackingHumanoidPartial.feet_link_names]
        # Weights of pos tracking links, sum is to flatten the list
        pos_tracking_weights = sum([[v]*len(links[k]) for k, v in self.pos_tracking_weights.items()], [])    
        self.pos_tracking_links = sorted(pos_tracking_links)
        pos_tracking_weights = np.array([v for _, v in sorted(zip(pos_tracking_links, pos_tracking_weights))], dtype=float)
        pos_tracking_weights /= np.sum(pos_tracking_weights)
        self.pos_tracking_weights = torch.tensor(pos_tracking_weights, dtype=torch.float, device=self.device)
        
        self.tracking_root_pos_offset0 = torch.zeros((len(self.envs), 1, 2), dtype=torch.float, device=self.device) # root position in the source reference motion (horizontally)
        self.tracking_root_pos_offset1 = torch.zeros((len(self.envs), 1, 2), dtype=torch.float, device=self.device) # root position in the target space (horizontally)
        self.tracking_root_orient_offset = torch.zeros((len(self.envs), 1, 4), dtype=torch.float, device=self.device) # root rotation from the source reference motion to the target space
        # the target tracking pos = rotatepoint(self.tracking_root_orient_offset, tracking pos in the refernce motion - self.tracking_root_pos_offset0) + self.tracking_root_pos_offset1
        self.feet_air_time = torch.zeros(len(self.envs), len(self.feet_links)).to(self.device)
        self.last_contacts = torch.zeros(len(self.envs), len(self.feet_links), dtype=torch.bool).to(self.device)
        # dummy reference motion for observe function calling during __init__
        self.env_enum = torch.arange(len(self.envs)).to(self.device)
        self.joint_state_hist = torch.zeros((3, 1, 33 + (27 if self.obs_action else 0) + (13 if self.obs_goal else 0)), dtype=torch.float, device=self.device)

        class DummyRefMotion():
            def __init__(self, n_links, device):
                self.n_links = n_links
                self.device = device
            def state(self, _, __, with_joint_tensor=False):
                return torch.empty((_.shape[0], self.n_links, 13), device=self.device)
        self.ref_motion_tracking = DummyRefMotion(len(self.pos_tracking_links), self.device)

    def build_motion_lib(self, motion_file):
        ref_motion, root_links = super().build_motion_lib(motion_file)
        self.ref_motion_tracking = ReferenceMotion(motion_file=motion_file,
                                    character_model=self.character_model,
                                    key_links = self.pos_tracking_links,
                                    load_joint_data = False,
                                    device=self.device)
        self.ref_motion_tracking_root = ReferenceMotion(motion_file=motion_file,
                                    character_model=self.character_model,
                                    key_links = [root_links[0]],
                                    load_joint_data = False,
                                    device=self.device)
        return ref_motion, root_links
    
    def force_init_state(self, motion_ids, motion_times):
        self.forced_motion_id = motion_ids
        self.forced_motion_time = motion_times
        self.force_motion = True

    def init_state(self, env_ids):
        if self.force_motion == False:
            motion_ids, motion_times = self.ref_motion.sample(len(env_ids), truncate_time=1.)
        else:
            motion_ids, motion_times = self.forced_motion_id, self.forced_motion_time
        # if not self.training or not self.random_init:
        #     motion_times[:] = 0
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        self._reset_goal(env_ids, motion_ids, motion_times)
        # the reference motion space is the character's current space
        # no need to convert coordinate system
        self.tracking_root_pos_offset0[env_ids] = 0
        self.tracking_root_pos_offset1[env_ids] = 0
        self.tracking_root_orient_offset[env_ids, :, :3] = 0
        self.tracking_root_orient_offset[env_ids, :, 3] = 1
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor
    
    def reset_goal(self, env_ids):
        if self.has_init_goal:
            self.has_init_goal = False
        else:
            motion_ids, motion_times = self.ref_motion_tracking.sample(len(env_ids), truncate_time=1)
            self._reset_goal(env_ids, motion_ids, motion_times)
            ref_root_pose = self.ref_motion_tracking_root.state(motion_ids, motion_times, with_joint_tensor=False)
            root_pose = self.root_tensor[env_ids, self.root_links[0]]
            heading0 = heading_zup(ref_root_pose[..., 3:7])
            heading1 = heading_zup(root_pose[..., 3:7]).unsqueeze_(1)
            self.tracking_root_pos_offset0[env_ids] = -ref_root_pose[...,:2]
            self.tracking_root_orient_offset[env_ids] = axang2quat(self.up_dir_tensor[:ref_root_pose.size(0)], heading1-heading0)
            self.tracking_root_pos_offset1[env_ids] = root_pose[...,:2]

    def _reset_goal(self, env_ids, motion_ids, motion_times):
        env_ids_ = env_ids.cpu().numpy()
        self.motion_tracking_ids[env_ids_] = motion_ids
        self.motion_tracking_times[env_ids_] = motion_times
        nframes = torch.tensor((self.ref_motion_tracking.motion_length[motion_ids]-motion_times)//self.step_time,
                dtype=torch.int, device=self.device)
        if len(env_ids) == len(self.envs):
            self.goal_timer.copy_(nframes)
        else:
            self.goal_timer[env_ids] = nframes
        self.has_init_goal = True

    def set_real_target(self, real_target, state_hist = None):
        self.real_target = real_target
        self.received_state_hist = state_hist

    def get_api_state(self):
        q = self.joint_pos.detach().cpu().numpy()[0]
        #dq = self.joint_vel.detach().cpu().numpy()[0]
        root_pose = np.hstack([self.root_pos.detach().cpu().numpy(),self.root_orient.detach().cpu().numpy()])[0]
        #root_vel = np.hstack([self.root_lin_vel.detach().cpu().numpy(),self.root_ang_vel.detach().cpu().numpy()])[0]
        return q, root_pose

    def observe(self, env_ids=None):
        self.ob_seq_lens = self.lifetime+1 #(self.lifetime+1).clip(max=self.state_hist.size(0)-1)
        n_envs = len(self.envs)
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
        else:
            n_envs = len(env_ids)
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.char_link_tensor[env_ids].view(n_envs, -1)
        if self.real_target is None:
            return self._observe(env_ids)
        else:
            return self._observe_real(self.real_target, state_hist = self.received_state_hist)

    def _observe(self, env_ids):
        if env_ids is None:
            link_tensor = self.link_tensor
            n_envs = link_tensor.size(0)
            motion_ids = self.motion_tracking_ids
            motion_times = self.motion_tracking_times
            tracking_target = []
            if self.obs_tracking_quat:
                tracking_orientation = []
            for i in range(self.obs_horizon):
                tracking_target_ = self.ref_motion_tracking.state(motion_ids, motion_times+2*i*self.step_time, with_joint_tensor=False)
                if self.obs_tracking_quat:
                    tracking_orientation_ = tracking_target_[..., 3:7]
                    tracking_orientation_ = quatmultiply(self.tracking_root_orient_offset, tracking_orientation_)
                    tracking_orientation.append(tracking_orientation_)
                tracking_target_ = tracking_target_[..., :3]
                tracking_target_[..., :2] += self.tracking_root_pos_offset0
                tracking_target_ = rotatepoint(self.tracking_root_orient_offset, tracking_target_)
                tracking_target_[..., :2] += self.tracking_root_pos_offset1
                tracking_target.append(tracking_target_)
            #tracking_target = self.ref_motion_tracking.state(motion_ids, motion_times, with_joint_tensor=False)
            self.motion_tracking_times += self.step_time
            # tracking_target = tracking_target[..., :3]
            # tracking_target[..., :2] += self.tracking_root_pos_offset0
            # tracking_target = rotatepoint(self.tracking_root_orient_offset, tracking_target)
            # tracking_target[..., :2] += self.tracking_root_pos_offset1
            #breakpoint()
            self.tracking_target = tracking_target[0]#torch.cat(tracking_target, dim=1)
            if self.obs_tracking_quat:
                self.tracking_orientation = tracking_orientation[0]
            ob = observe_iccgan(self.state_hist[-self.ob_horizon:], self.ob_seq_lens)
        else:
            link_tensor = self.link_tensor[env_ids]
            n_envs = link_tensor.size(0)
            env_ids_ = env_ids.cpu().numpy()
            motion_ids = self.motion_tracking_ids[env_ids_]
            motion_times = self.motion_tracking_times[env_ids_]
            tracking_target = []
            if self.obs_tracking_quat:
                tracking_orientation = []
            for i in range(self.obs_horizon):
                tracking_target_ = self.ref_motion_tracking.state(motion_ids, motion_times+2*i*self.step_time, with_joint_tensor=False)
                if self.obs_tracking_quat:
                    tracking_orientation_ = tracking_target_[..., 3:7]
                    tracking_orientation_ = quatmultiply(self.tracking_root_orient_offset[env_ids], tracking_orientation_)
                    tracking_orientation.append(tracking_orientation_)
                tracking_target_ = tracking_target_[..., :3]
                tracking_target_[..., :2] += self.tracking_root_pos_offset0[env_ids]
                tracking_target_ = rotatepoint(self.tracking_root_orient_offset[env_ids], tracking_target_)
                tracking_target_[..., :2] += self.tracking_root_pos_offset1[env_ids]
                tracking_target.append(tracking_target_)
            #tracking_target = self.ref_motion_tracking.state(motion_ids, motion_times, with_joint_tensor=False)
            self.motion_tracking_times[env_ids_] += self.step_time
            # tracking_target = tracking_target[..., :3]
            # tracking_target[..., :2] += self.tracking_root_pos_offset0[env_ids]
            # tracking_target = rotatepoint(self.tracking_root_orient_offset[env_ids], tracking_target)
            # tracking_target[..., :2] += self.tracking_root_pos_offset1[env_ids]
            self.tracking_target[env_ids] = tracking_target[0]#torch.cat(tracking_target, dim=1)
            if self.obs_tracking_quat:
                self.tracking_orientation[env_ids] = tracking_orientation[0]
            ob = observe_iccgan(self.state_hist[-self.ob_horizon:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link, include_velocity=self.obs_vel)
        
        orient = link_tensor[..., :1, 3:7]
        origin = link_tensor[:, :1, :3].clone()
        origin[..., 2] = 0
        heading = heading_zup(orient)

        orient_inv = axang2quat(self.up_dir_tensor[:link_tensor.size(0)], -heading)
        p = rotatepoint(orient_inv, torch.cat(tracking_target,dim=1) - origin)
        if self.obs_tracking_quat:
            q_dists = []
            
            #link_local = quatmultiply(orient_inv, link_tensor[:, self.pos_tracking_links, 3:7])
            for i in range(self.obs_horizon):
                r = quatmultiply(orient_inv, tracking_orientation[i]) # target quaternion in root frame
                # dq = quatmultiply(
                #     link_local, quatconj(r) # diff from target quaternion to current quaternion in root frame
                # )
                # if not self.tracking_hands:
                #     dq = dq[:, :1]
                if not self.tracking_hands:
                    q_dists.append(r[:,:1])
                else:
                    q_dists.append(r)
            q_dist = torch.cat(q_dists, dim=1)
        dist = p.square().sum(-1, keepdim=True).sqrt_()
        p = torch.where(dist > 5, p/dist*5, p)
        if self.obs_tracking_quat:
            return torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1), q_dist.view(n_envs, -1)), -1)
        else:
            return torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1)), -1)

    def process_real_observation(self, target, state_hist, local_obs=False):
        self.state_hist[:-1] = self.state_hist[1:].clone()
        self.ob_seq_lens += 1
        if self.ob_seq_lens == 2:
            self.state_hist = self.state_hist.cpu()
        """
        real_target: [1, 3, 7]
        """
        if state_hist is not None: # override state history for real robot experiment
            self.state_hist[-1] = state_hist.clone()
            self.ob_seq_lens = self.ob_seq_lens.cpu()
        link_tensor = self.link_tensor
        n_envs = link_tensor.size(0)
        #self.obs_queue.push(target) # If self.obs_track_quat, real target has orientation
        tracking_target = []
        if self.obs_tracking_quat:
            tracking_orientation = []
        for i in range(self.obs_horizon):
            tracking_target_ = target[i][...,:3] # Must clone or BUG.
            tracking_target_ = tracking_target_[..., :3]
            if self.obs_tracking_quat:
                tracking_orientation_ = target[i][..., 3:7]
                tracking_orientation.append(tracking_orientation_)
            tracking_target.append(tracking_target_)
        self.motion_tracking_times += self.step_time
        self.tracking_target = tracking_target[0] #actual tracking target for compute reward
        if self.obs_tracking_quat:
            self.tracking_orientation = tracking_orientation[0]
        ob = observe_iccgan(self.state_hist[-self.ob_horizon:], self.ob_seq_lens)
        
        if self.obs_tracking_quat:
            q_dists = []
            for i in range(self.obs_horizon):
                r = tracking_orientation[i]
                if not self.tracking_hands:
                    q_dists.append(r[:,:1])
                else:
                    q_dists.append(r)
            q_dist = torch.cat(q_dists, dim=1)
        p = torch.cat(tracking_target,dim=1)
        dist = p.square().sum(-1, keepdim=True).sqrt_()
        p = torch.where(dist > 5, p/dist*5, p)
        if self.obs_tracking_quat:
            obs = torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1), q_dist.view(n_envs, -1)), -1)
        else:
            obs = torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1)), -1)
        return obs, self.ob_seq_lens

    def process_joint_real_observations(self, target, q, dq, g, omega, action=None):
        self.ob_seq_lens += 1
        if self.lifetime == 0:
            self.joint_state_hist[...,:3] = omega * 0.25
            self.joint_state_hist[...,3:6] = g.clone()
            self.joint_state_hist[...,6:33] = q.clone()
            #self.joint_state_hist[...,33:60] = dq * 0.05
        else:
            self.joint_state_hist[:-1] = self.joint_state_hist[1:].clone()
            self.joint_state_hist[-1,:3] = omega.clone() * 0.25
            self.joint_state_hist[-1,3:6] = g.clone()
            self.joint_state_hist[-1,6:33] = q.clone()
            #self.joint_state_hist[-1,33:60] = dq * 0.05
        # get target obs
        link_tensor = self.link_tensor
        n_envs = link_tensor.size(0)
        #self.obs_queue.push(target) # If self.obs_track_quat, real target has orientation
        tracking_target = []
        if self.obs_tracking_quat:
            tracking_orientation = []
        for i in range(self.obs_horizon):
            tracking_target_ = target[i][:,:,:3] # Must clone or BUG.
            tracking_target_ = tracking_target_[..., :3]
            if self.obs_tracking_quat:
                tracking_orientation_ = target[i][..., 3:7]
                tracking_orientation.append(tracking_orientation_)
            tracking_target.append(tracking_target_)
        self.motion_tracking_times += self.step_time
        self.tracking_target = tracking_target[0] #actual tracking target for compute reward
        if self.obs_tracking_quat:
            self.tracking_orientation = tracking_orientation[0]
        if self.obs_tracking_quat:
            q_dists = []
            for i in range(self.obs_horizon):
                r = tracking_orientation[i]
                if not self.tracking_hands:
                    q_dists.append(r[:,:1])
                else:
                    q_dists.append(r)
            q_dist = torch.cat(q_dists, dim=1)
        p = torch.cat(tracking_target,dim=1)
        dist = p.square().sum(-1, keepdim=True).sqrt_()
        p = torch.where(dist > 5, p/dist*5, p)
        ob = self.joint_state_hist[1:].clone().permute(1,0,2)
        if self.obs_tracking_quat:
            obs = torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1), q_dist.view(n_envs, -1), action.view(n_envs,-1)), -1)
        else:
            obs = torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1)), -1)
        return obs, self.ob_seq_lens

    def apply_perturbation(self):
        if self.max_push_vel > 0:
            self._push_robots()

    # Random perturbations
    def _push_robots(self):
        max_vel = self.max_push_vel
        mask = (self.lifetime % self.push_interval == 0) * (self.lifetime > 0)
        # convert mask to env_id
        env_id = self.env_enum[mask]
        self.root_tensor[env_id,0,7:9] = torch_rand_float(-max_vel, max_vel, (len(env_id),2), device = self.device.type)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))

    # Start of reward functions
    def _energy_reward(self):
        rew = 0
        if self.last_action is not None:
            last_action = self.last_action
            # Compute torque
            torque = (last_action - self.joint_tensor[..., 0]) * self.joint_stiffness
            last_qvel = self.joint_tensor[..., 1] # 0: position, 1: velocity
            energy_cost = 0.1*torch.sum(torch.abs(torque * last_qvel), dim=-1) + 0.005* torch.sum(torque**2, dim=-1)
            rew -= energy_cost
        return rew
    
    def _eef_tracking_reward(self):
        e_p = self.link_tensor[:, self.pos_tracking_links, :3] - self.tracking_target
        if self.obs_tracking_quat:
            e_q = quatmultiply(self.link_tensor[:, self.pos_tracking_links[1:], 3:7], quatconj(self.tracking_orientation[:,1:])) # only consider wrist
            e_r = (quat2axang(e_q)[1]).square_() # sum over all trackpoints
            e_p = e_p.square_().sum(-1)
            self.tracking_error = e_p # only consider position tracking error
            # if not self.training:
            #     print(self.simulation_step, "Tracking Error:", e_p.sqrt().mean().item(), e_r.sqrt().mean().item())
            rew = torch.exp(-5*(e_p*self.pos_tracking_weights).sum(-1)-e_r.sum(-1)) # assume weight of orientation is 0.2
        else:
            e_p = e_p.square_().sum(-1)
            self.tracking_error = e_p
            # if not self.training:
            #     print(self.simulation_step, "Tracking Error:", e_p.sqrt().mean().item())
            rew = torch.exp(-5*(e_p*self.pos_tracking_weights).sum(-1))
        
        return rew

    def _joint_deviation_reward(self):
        # Penalize joint deviation
        diff_q = (self.joint_tensor[:, self.reg_joints, 0] - self.reg_q).norm(dim=-1)
        return -diff_q

    def _feet_contact_force_reward(self, max_contact_force = 450.):
        # Reward gentle contact, penalize hard contact
        if self.feet_links is not None:
            contact_forces = self.char_contact_force_tensor[:, self.feet_links]
            contact_force_norm = contact_forces.norm(dim=-1)
            return -torch.sum((contact_force_norm - max_contact_force).clamp(min=0.), dim=1)
        else:
            return 0

    def _feet_in_air_reward(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        if self.feet_links is not None:
            contact = (self.char_contact_force_tensor[:, self.feet_links, 2] > 1.).squeeze(dim=-1)
            contact_filt = torch.logical_or(contact, self.last_contacts) 
            self.last_contacts = contact
            first_contact = (self.feet_air_time > 0.) * contact_filt
            self.feet_air_time += self.step_time
            rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
            #rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command # Should based on head pose?
            self.feet_air_time *= ~contact_filt
            return rew_airTime
        else:
            return 0

    def __feet_sliding_reward(self):
        # Assume last contact has been updated
        # self.last_contacts [N, 2]
        if self.feet_links is not None:
            # Get feet speed
            feet_speed = self.link_lin_vel[:, self.feet_links, :2].norm(dim=-1)
            # if in contact, penalize linear velocity
            rew = -feet_speed * self.last_contacts
            return rew.sum(dim=-1)
        else:
            return 0

    def reward(self):
        """
        reward of tracking target link position
        """
        rew = self._eef_tracking_reward() * self.rew_weights["eef_tracking"]
        rew += self._energy_reward() * self.rew_weights["energy"]
        rew += self._feet_in_air_reward() * self.rew_weights["feet_in_air"] # seems to be important
        rew += self.__feet_sliding_reward() * self.rew_weights["feet_sliding"]
        rew += self._joint_deviation_reward() * self.rew_weights["joint_deviation"]
        #rew += self._feet_contact_force_reward() * self.rew_weights["feet_contact_force"]
        return rew.unsqueeze_(-1)

    def termination_check(self):
        term = super().termination_check()
        too_far = torch.any(self.tracking_error > 5, -1)
        return term.logical_or_(too_far)
    
    def update_viewer(self):
        super().update_viewer()
        target = self.tracking_target.cpu().numpy()

        n_lines = 200
        phi = np.linspace(0, 2*np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        dx = 0.05 * (sin_phi[:, None] * cos_theta[None, :])
        dy = 0.05 * (sin_phi[:, None] * sin_theta[None, :])
        dz = 0.05 * cos_phi
        dx.shape = (-1, )
        dy.shape = (-1, )
        n_lines = len(dx)

        for e, pp in zip(self.envs, target):
            for p in pp:
                l = np.stack([
                    np.stack((p[0], p[1], p[2], p[0]+x, p[1]+y, p[2]+dz[i%len(dz)]))
                    for i, (x, y) in enumerate(zip(dx, dy))
                ])
                self.gym.add_lines(self.viewer, e, n_lines, np.float32(l), np.float32([[1.,0.,0.] for _ in range(n_lines)]))

    def update_viewer_(self):
        super().update_viewer()
        target_pos = self.tracking_target.cpu().numpy()
        target_orn = self.tracking_orientation.cpu()

        x_axis = torch.tensor([0.1, 0., 0.])
        y_axis = torch.tensor([0., 0.1, 0.])
        z_axis = torch.tensor([0., 0., 0.1])
        n_lines = 3

        repeat = 2
        delta = np.stack(
                    np.meshgrid(np.linspace(-0.005, 0.005, repeat), 
                                np.linspace(-0.005, 0.005, repeat), 
                                np.linspace(-0.005, 0.005, repeat), indexing='ij'),
                    axis=-1).reshape(-1,3)
        n_lines = len(delta) * 3
        for e, pp, oo in zip(self.envs, target_pos, target_orn):
            for p, o in zip(pp, oo):
                l = []
                c = []
                for d in delta:
                    pd = p + d
                    l.extend([np.stack((pd, pd+rotatepoint(o,x_axis).numpy())),
                               np.stack((pd, pd+rotatepoint(o,y_axis).numpy())),
                               np.stack((pd, pd+rotatepoint(o,z_axis).numpy()))])
                    c.extend([[1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
                l = np.stack(l).astype(np.float32)
                c = np.stack(c).astype(np.float32)
                self.gym.add_lines(self.viewer, e, n_lines, l,c)

class TrackingHumanoidPartialIK(ICCGANHumanoid):
    CAMERA_POS= 0, 4.5, 2.0
    ENABLE_GOAL_TIMER = True

    feet_link_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    left_arm_joints = [15,16,17,18,19,20,21]
    right_arm_joints = [22,23,24,25,26,27,28]
    def __init__(self, *args, **kwargs):
        self.test1 = kwargs["test1"] if "test1" in kwargs else False
        self.test2 = kwargs["test2"] if "test2" in kwargs else False

        self.random_init = parse_kwarg(kwargs, "random_init", False)
        self.loop_motion = parse_kwarg(kwargs, "loop_motion", False)
        self.rew_weights = parse_kwarg(kwargs, "reward_weights", None)
        self.dampings = parse_kwarg(kwargs, "ik_dampings", None)
        self.pos_tracking_weights = kwargs["pos_tracking_weights"]
        self.reg_joints = kwargs["reg_joints"]
        self.reg_q  = torch.tensor(kwargs["reg_q"])
        self.dampings = torch.tensor(self.dampings)
        assert all([_ >= 0 for _ in self.pos_tracking_weights.values()])
        self.GOAL_DIM = 3*len(self.pos_tracking_weights)
        self.feet_links = None
        super().__init__(*args, **kwargs)

    def create_tensors(self):
        super().create_tensors()
        self.motion_tracking_ids = np.zeros((len(self.envs)), dtype=int)
        self.motion_tracking_times = np.zeros((len(self.envs)), dtype=float)
        self.up_dir_tensor = torch.zeros(((len(self.envs), 1, 3)), dtype=torch.float, device=self.device)
        self.up_dir_tensor[..., 2] = 1
        self.reg_q = self.reg_q.to(self.device)
        self.dampings = self.dampings.to(self.device)
        links, n_links = dict(), 0
        rigid_body = self.gym.get_actor_rigid_body_dict(self.envs[0], 0)
        for n, id in rigid_body.items():
            handle = id + n_links
            if n in links:
                links[n].append(handle)
            else:
                links[n] = [handle]
        
        # ID of pos tracking links, sum is to flatten the list
        pos_tracking_links = sum([links[k] for k, v in self.pos_tracking_weights.items()], [])
        # Feet links
        self.feet_links = [links[k][0] for k in TrackingHumanoidPartial.feet_link_names]
        # Weights of pos tracking links, sum is to flatten the list
        pos_tracking_weights = sum([[v]*len(links[k]) for k, v in self.pos_tracking_weights.items()], [])    
        self.pos_tracking_links = sorted(pos_tracking_links)
        pos_tracking_weights = np.array([v for _, v in sorted(zip(pos_tracking_links, pos_tracking_weights))], dtype=float)
        pos_tracking_weights /= np.sum(pos_tracking_weights)
        self.pos_tracking_weights = torch.tensor(pos_tracking_weights, dtype=torch.float, device=self.device)
        
        # Create jacobian tensor
        num_dof = self.gym.get_env_dof_count(self.envs[0])
        num_body = self.gym.get_actor_rigid_body_count(self.envs[0], 0)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "humanoid")
        self.jacobian = gymtorch.wrap_tensor(_jacobian).view(len(self.envs), num_body, 6, num_dof+6)
        self.tracking_root_pos_offset0 = torch.zeros((len(self.envs), 1, 2), dtype=torch.float, device=self.device) # root position in the source reference motion (horizontally)
        self.tracking_root_pos_offset1 = torch.zeros((len(self.envs), 1, 2), dtype=torch.float, device=self.device) # root position in the target space (horizontally)
        self.tracking_root_orient_offset = torch.zeros((len(self.envs), 1, 4), dtype=torch.float, device=self.device) # root rotation from the source reference motion to the target space
        # the target tracking pos = rotatepoint(self.tracking_root_orient_offset, tracking pos in the refernce motion - self.tracking_root_pos_offset0) + self.tracking_root_pos_offset1
        self.feet_air_time = torch.zeros(len(self.envs), len(self.feet_links)).to(self.device)
        self.last_contacts = torch.zeros(len(self.envs), len(self.feet_links), dtype=torch.bool).to(self.device)
        # dummy reference motion for observe function calling during __init__
        class DummyRefMotion():
            def __init__(self, n_links, device):
                self.n_links = n_links
                self.device = device
            def state(self, _, __, with_joint_tensor=False):
                return torch.empty((_.shape[0], self.n_links, 13), device=self.device)
        self.ref_motion_tracking = DummyRefMotion(len(self.pos_tracking_links), self.device)

    def refresh_tensors(self):
        super().refresh_tensors()
        self.gym.refresh_jacobian_tensors(self.sim)

    def build_motion_lib(self, motion_file):
        ref_motion, root_links = super().build_motion_lib(motion_file)
        self.ref_motion_tracking = ReferenceMotion(motion_file=motion_file,
                                    character_model=self.character_model,
                                    key_links = self.pos_tracking_links,
                                    load_joint_data = False,
                                    device=self.device)
        self.ref_motion_tracking_root = ReferenceMotion(motion_file=motion_file,
                                    character_model=self.character_model,
                                    key_links = [root_links[0]],
                                    load_joint_data = False,
                                    device=self.device)
        return ref_motion, root_links
    
    def init_state(self, env_ids):
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids), truncate_time=1.)
        # if not self.training or not self.random_init:
        #     motion_times[:] = 0
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        self._reset_goal(env_ids, motion_ids, motion_times)
        # the reference motion space is the character's current space
        # no need to convert coordinate system
        self.tracking_root_pos_offset0[env_ids] = 0
        self.tracking_root_pos_offset1[env_ids] = 0
        self.tracking_root_orient_offset[env_ids, :, :3] = 0
        self.tracking_root_orient_offset[env_ids, :, 3] = 1
        self.feet_air_time[env_ids] = 0.
        self.last_contacts[env_ids] = False
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor
    
    def reset_goal(self, env_ids):
        if self.has_init_goal:
            self.has_init_goal = False
        else:
            motion_ids, motion_times = self.ref_motion_tracking.sample(len(env_ids), truncate_time=1)
            self._reset_goal(env_ids, motion_ids, motion_times)
            ref_root_pose = self.ref_motion_tracking_root.state(motion_ids, motion_times, with_joint_tensor=False)
            root_pose = self.root_tensor[env_ids, self.root_links[0]]
            heading0 = heading_zup(ref_root_pose[..., 3:7])
            heading1 = heading_zup(root_pose[..., 3:7]).unsqueeze_(1)
            self.tracking_root_pos_offset0[env_ids] = -ref_root_pose[...,:2]
            self.tracking_root_orient_offset[env_ids] = axang2quat(self.up_dir_tensor[:ref_root_pose.size(0)], heading1-heading0)
            self.tracking_root_pos_offset1[env_ids] = root_pose[...,:2]

    def _reset_goal(self, env_ids, motion_ids, motion_times):
        env_ids_ = env_ids.cpu().numpy()
        self.motion_tracking_ids[env_ids_] = motion_ids
        self.motion_tracking_times[env_ids_] = motion_times
        nframes = torch.tensor((self.ref_motion_tracking.motion_length[motion_ids]-motion_times)//self.step_time,
                dtype=torch.int, device=self.device)
        if len(env_ids) == len(self.envs):
            self.goal_timer.copy_(nframes)
        else:
            self.goal_timer[env_ids] = nframes
        self.has_init_goal = True

    def _observe(self, env_ids):
        if env_ids is None:
            link_tensor = self.link_tensor
            n_envs = link_tensor.size(0)
            motion_ids = self.motion_tracking_ids
            motion_times = self.motion_tracking_times
            tracking_target_ = self.ref_motion_tracking.state(motion_ids, motion_times, with_joint_tensor=False)
            self.motion_tracking_times += self.step_time
            tracking_target = tracking_target_[..., :3]
            tracking_target[..., :2] += self.tracking_root_pos_offset0
            tracking_target = rotatepoint(self.tracking_root_orient_offset, tracking_target)
            tracking_target[..., :2] += self.tracking_root_pos_offset1

            # Update tracking orientation
            tracking_orientation = tracking_target_[..., 3:7]
            tracking_orientation = quatmultiply(self.tracking_root_orient_offset, tracking_orientation)
            
            self.tracking_target = tracking_target
            self.tracking_orientation = tracking_orientation
            ob = observe_iccgan(self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link)
        else:
            link_tensor = self.link_tensor[env_ids]
            n_envs = link_tensor.size(0)
            env_ids_ = env_ids.cpu().numpy()
            motion_ids = self.motion_tracking_ids[env_ids_]
            motion_times = self.motion_tracking_times[env_ids_]
            tracking_target_ = self.ref_motion_tracking.state(motion_ids, motion_times, with_joint_tensor=False)
            self.motion_tracking_times[env_ids_] += self.step_time
            tracking_target = tracking_target_[..., :3]
            tracking_target[..., :2] += self.tracking_root_pos_offset0[env_ids]
            tracking_target = rotatepoint(self.tracking_root_orient_offset[env_ids], tracking_target)
            tracking_target[..., :2] += self.tracking_root_pos_offset1[env_ids]

            # Update tracking orientation
            tracking_orientation = tracking_target_[..., 3:7]
            tracking_orientation = quatmultiply(self.tracking_root_orient_offset[env_ids], tracking_orientation) # apply new orientation based on root orientation

            self.tracking_target[env_ids] = tracking_target
            self.tracking_orientation[env_ids] = tracking_orientation
            ob = observe_iccgan(self.state_hist[-self.ob_horizon:, env_ids], self.ob_seq_lens[env_ids], self.key_links, self.parent_link)
        
        orient = link_tensor[..., :1, 3:7]
        origin = link_tensor[:, :1, :3].clone()
        origin[..., 2] = 0
        heading = heading_zup(orient)

        orient_inv = axang2quat(self.up_dir_tensor[:link_tensor.size(0)], -heading)
        p = rotatepoint(orient_inv, tracking_target - origin)
        dist = p.square().sum(-1, keepdim=True).sqrt_()
        p = torch.where(dist > 5, p/dist*5, p)
        
        return torch.cat((ob.flatten(start_dim=1), p.view(n_envs, -1)), -1)

    def solve_ik(self, wrist_target_error, damping, jacobian,handedness = "left"):
        # Solve arm IK for wrist
        if handedness == "left":
            eef_id = self.pos_tracking_links[1] # 0 is head, 1 is left wrist, 2 is right wrist
            arm_joints = self.left_arm_joints
        else:
            eef_id = self.pos_tracking_links[2] # 0 is head, 1 is left wrist, 2 is right wrist
            arm_joints = self.right_arm_joints
        jac = jacobian[:, eef_id, :, arm_joints]
        jac_T = jac.transpose(1,2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (jac_T @ torch.inverse(jac @ jac_T + lmbda) @ wrist_target_error).view(len(self.envs), 7)
        return u
        
    def process_actions(self, actions):
        a = actions*self.action_scale + self.action_offset
        # Should add IK controller to override arm actions
        left_pos_error = self.tracking_target[:,1] - self.link_tensor[:, self.pos_tracking_links[1], :3]
        left_orn_error = orientation_error(self.tracking_orientation[:,1], self.link_tensor[:, self.pos_tracking_links[1], 3:7])
        left_wrist_error = torch.cat([left_pos_error, left_orn_error], dim=-1).unsqueeze(-1)
        l_delta_q = self.solve_ik(left_wrist_error, self.dampings, self.jacobian[:,:,:,6:]) # 6 is to skip root joint
        a[:,self.left_arm_joints] = self.joint_tensor[:,self.left_arm_joints,0] + l_delta_q
        right_pos_error = self.tracking_target[:,2] - self.link_tensor[:, self.pos_tracking_links[2], :3]
        right_orn_error = orientation_error(self.tracking_orientation[:,2], self.link_tensor[:, self.pos_tracking_links[2], 3:7])
        right_wrist_error = torch.cat([right_pos_error, right_orn_error], dim=-1).unsqueeze(-1)
        r_delta_q = self.solve_ik(right_wrist_error, self.dampings, self.jacobian[:,:,:,6:], handedness="right")
        a[:,self.right_arm_joints] = self.joint_tensor[:,self.right_arm_joints,0] + r_delta_q
        self.last_action = a
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dofs] = a
        return self.action_tensor

    # Start of reward functions
    def _energy_reward(self):
        rew = 0
        if self.last_action is not None:
            last_action = self.last_action
            # Compute torque
            torque = (last_action - self.joint_tensor[..., 0]) * self.joint_stiffness
            last_qvel = self.joint_tensor[..., 1] # 0: position, 1: velocity
            energy_cost = 0.1*torch.sum(torch.abs(torque * last_qvel), dim=-1) + 0.005* torch.sum(torque**2, dim=-1)
            rew -= energy_cost
        return rew
    
    def _eef_tracking_reward(self):
        e_p = self.link_tensor[:, self.pos_tracking_links, :3] - self.tracking_target
        e_p = e_p.square_().sum(-1)
        self.tracking_error = e_p
        rew = torch.exp(-5*(e_p*self.pos_tracking_weights).sum(-1))
        # if not self.training:
        #     print(self.simulation_step, "Tracking Error:", e_p.sqrt().mean().item())
        return rew

    def _joint_deviation_reward(self):
        # Penalize joint deviation
        diff_q = (self.joint_tensor[:, self.reg_joints, 0] - self.reg_q).norm(dim=-1)
        return -diff_q

    def _feet_contact_force_reward(self, max_contact_force = 450.):
        # Reward gentle contact, penalize hard contact
        if self.feet_links is not None:
            contact_forces = self.char_contact_force_tensor[:, self.feet_links]
            contact_force_norm = contact_forces.norm(dim=-1)
            return -torch.sum((contact_force_norm - max_contact_force).clamp(min=0.), dim=1)
        else:
            return 0


    def _feet_in_air_reward(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        if self.feet_links is not None:
            contact = (self.char_contact_force_tensor[:, self.feet_links, 2] > 1.).squeeze(dim=-1)
            contact_filt = torch.logical_or(contact, self.last_contacts) 
            self.last_contacts = contact
            first_contact = (self.feet_air_time > 0.) * contact_filt
            self.feet_air_time += self.step_time
            rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
            #rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command # Should based on head pose?
            self.feet_air_time *= ~contact_filt
            return rew_airTime
        else:
            return 0

    def __feet_sliding_reward(self):
        # Assume last contact has been updated
        # self.last_contacts [N, 2]
        if self.feet_links is not None:
            # Get feet speed
            feet_speed = self.link_lin_vel[:, self.feet_links, :2].norm(dim=-1)
            # if in contact, penalize linear velocity
            rew = -feet_speed * self.last_contacts
            return rew.sum(dim=-1)
        else:
            return 0

    def reward(self):
        """
        reward of tracking target link position
        """
        rew = self._eef_tracking_reward() * self.rew_weights["eef_tracking"]
        rew += self._energy_reward() * self.rew_weights["energy"]
        rew += self._feet_in_air_reward() * self.rew_weights["feet_in_air"] # seems to be important
        rew += self.__feet_sliding_reward() * self.rew_weights["feet_sliding"]
        rew += self._joint_deviation_reward() * self.rew_weights["joint_deviation"]
        rew += self._feet_contact_force_reward() * self.rew_weights["feet_contact_force"]
        return rew.unsqueeze_(-1)

    def termination_check(self):
        term = super().termination_check()
        too_far = torch.any(self.tracking_error > 5, -1)
        return term.logical_or_(too_far)
    
    def update_viewer(self):
        super().update_viewer()
        target = self.tracking_target.cpu().numpy()

        n_lines = 200
        phi = np.linspace(0, 2*np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        dx = 0.05 * (sin_phi[:, None] * cos_theta[None, :])
        dy = 0.05 * (sin_phi[:, None] * sin_theta[None, :])
        dz = 0.05 * cos_phi
        dx.shape = (-1, )
        dy.shape = (-1, )
        n_lines = len(dx)

        for e, pp in zip(self.envs, target):
            for p in pp:
                l = np.stack([
                    np.stack((p[0], p[1], p[2], p[0]+x, p[1]+y, p[2]+dz[i%len(dz)]))
                    for i, (x, y) in enumerate(zip(dx, dy))
                ])
                self.gym.add_lines(self.viewer, e, n_lines, np.float32(l), np.float32([[1.,0.,0.] for _ in range(n_lines)]))


    
class ICCGANHumanoidDemo(ICCGANHumanoid):
    OB_HORIZON = 1
    CAMERA_POS= 0, 4.5, 2.0

    def __init__(self, *args, **kwargs):
        self.controllable = True
        if not self.controllable:
            if len(args) > 4:
                args = [_ for _ in args]
                args[4] = "free"
            else:
                kwargs["control_mode"] = "free"
        self.cnt = 0
        super().__init__(*args, **kwargs)

    def termination_check(self):
        return torch.from_numpy(self.finished).type(torch.bool).to(self.device)
    
    def create_envs(self, n):
        return super().create_envs(n,
            asset_options=dict(
                fix_base_link = True,
                disable_gravity = True
        ))

    def reset(self):
        super().reset()
        self.motion_ids = torch.zeros(len(self.envs), dtype=torch.int32, device=self.device) + self.cnt
        self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device) - self.step_time

    def apply_actions(self, actions):
        env_ids = torch.arange(len(self.envs), dtype=torch.int64)
        ref_root_tensor, ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)

        # remove velocity
        ref_root_tensor[..., 7:12] = 0
        ref_joint_tensor[..., 1] = 0
        ref_link_tensor[..., 7:12] = 0

        self.root_tensor[env_ids] = ref_root_tensor
        self.joint_tensor[env_ids] = ref_joint_tensor
        self.link_tensor[env_ids] = ref_link_tensor

        self.gym.set_actor_root_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.root_tensor)
        )
        self.gym.set_dof_state_tensor(self.sim,
            gymtorch.unwrap_tensor(self.joint_tensor.view(self.joint_tensor.size(0), -1))
        )

    def reset_done(self):
        obs, info = super().reset_done()
        if self.done[0]:
            self.motion_ids = torch.zeros(len(self.envs), dtype=torch.int32, device=self.device) + self.cnt
            self.cnt += 1
            print("traj:",self.cnt)
            self.motion_times = torch.zeros(len(self.envs), dtype=torch.float32, device=self.device) - self.step_time
            self.finished = torch.zeros(len(self.envs), dtype=torch.bool, device=self.device)
        return obs, info

    def step(self, actions):
        obs, rews, dones, info = super().step(actions)
        self.done = torch.from_numpy(self.finished).to(self.device)
        return obs, rews, self.finished, info


    def init_state(self, env_ids):
        self.motion_times[env_ids] += self.step_time
        self.motion_times[env_ids].clip_(min=0)
        motion_ids = self.motion_ids[env_ids].cpu().numpy()
        motion_times = self.motion_times[env_ids].cpu().numpy()
        self.finished = motion_times > self.ref_motion.motion_length[motion_ids]
        motion_times[self.finished] = 0
        self.motion_times = torch.from_numpy(motion_times).to(self.device)
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        return ref_link_tensor[:, self.root_links], ref_link_tensor, ref_joint_tensor

