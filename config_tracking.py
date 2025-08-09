env_cls = "TrackingHumanoidPartial"
# env_cls = "ICCGANHumanoidDemo"

env_params = dict(
    fps = 30,
    random_init = True,
    verbose = False,
    continuous_track = True,
    episode_length = 600,
    goal_reward_weight = 0.6, # reward weight for tracking, comment to replay
    character_model = "resources/robots/g1/g1_29dof_rev_1_0_track.xml",
    contactable_links = ["right_ankle_roll_link", "left_ankle_roll_link"],
    reg_joints = [4, 10, 1, 7], # left_ankle_pitch_joint, right_ankle_pitch_joint
    reg_q = [0.0, 0.0, 0.1,-0.1],
    pos_tracking_weights = {
        "right_hand_track_site": 1,
        "left_hand_track_site": 1, 
        "head_track_site": 1
    },
    reward_weights = {
        "eef_tracking": 1.0,
        "energy": 0.0001,
        "feet_in_air": 0.5,
        "feet_sliding": 0.5,
        "joint_deviation": 0.1,
        "feet_contact_force": 0.001
    },
    ob_horizon = 6,
    max_push_vel = 0.0,
    push_interval = 60,
    obs_horizon=3,
    use_domain_rand = False,
    friction_range = [0.5, 1.25],
    mass_range = [-1.0, 1.0],
    stiffness_range = [0.8, 1.1],
    damping_range = [0.7, 1.3],
    torque_limit = [88,139,88,139,50,50,
                    88,139,88,139,50,50,
                    88, 50, 50, 
                    25, 25, 25, 25, 25, 5, 5,
                    25, 25, 25, 25, 25, 5, 5],
    control_mode = "position",
    obs_vel = False,
    obs_tracking_quat = False,
    use_feet = False,
    filter = True
)

training_params = dict(
    max_epochs =  1000000,
    save_interval = 50000,
    terminate = -25,
    use_rnn = True,
)

discriminators = {
    "_/full": dict(
        parent_link = None,
    )
}