import genesis as gs
import numpy as np

kp = 30.0
kd = 0.5

"Initializes Genesis with the CPU backend."
gs.init(backend=gs.cpu)

"Creates a new simulation scene"
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.02,
        substeps=4,
    ),
    show_viewer=False,
)

"Adds a flat ground plane to the scene"
scene.add_entity(
    gs.morphs.URDF(
        file="urdf/plane/plane.urdf",
        fixed=True,
    )
)

"Loads the Franka Emika Panda robot arm using its MJCF XML file"
# Go2 = scene.add_entity(gs.morphs.URDF(file='urdf/go2/urdf/go2.urdf'
# , pos=[0.0, 0.0, 0.42], quat=[1.0, 0.0, 0.0, 0.0]))

Go2_test = scene.add_entity(
    gs.morphs.MJCF(
        file="externals/mujoco_menagerie/unitree_go2/go2_test.xml", pos=[0.0, 1.0, 0.32], quat=[1.0, 0.0, 0.0, 0.0]
    )
)

"Finalizes the scene setup"
scene.build()

# Declaring the joint names in the desired order
joint_names = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Defining the Joint Name IDs
motors_dof_idx = [Go2_test.get_joint(name).dof_start for name in joint_names]
kp_array = np.full(len(motors_dof_idx), kp)
kd_array = np.full(len(motors_dof_idx), kd)

# Go2.set_dofs_kp(kp_array, motors_dof_idx)
# Go2.set_dofs_kv(kd_array, motors_dof_idx)
Go2_test.set_dofs_kp(kp_array, motors_dof_idx)
Go2_test.set_dofs_kv(kd_array, motors_dof_idx)

# Initial position
initial_position = np.array(
    [
        0.0,
        0.8,
        -1.5,  # FL_hip, FL_thigh, FL_calf
        0.0,
        0.8,
        -1.5,  # FR_hip, FR_thigh, FR_calf
        0.0,
        1.0,
        -1.5,  # RL_hip, RL_thigh, RL_calf
        0.0,
        1.0,
        -1.5,  # RR_hip, RR_thigh, RR_calf
    ],
    dtype=np.float32,
)

# Go2.set_pos([0.0, 0.0, 0.42], zero_velocity=True)
# Go2.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
# Go2.set_dofs_position(initial_position, motors_dof_idx, zero_velocity=True)
Go2_test.set_pos([0.0, 1.0, 0.42], zero_velocity=True)
Go2_test.set_quat([1.0, 0.0, 0.0, 0.0], zero_velocity=True)
Go2_test.set_dofs_position(initial_position, motors_dof_idx, zero_velocity=True)

# Hard reset the initial position
# import ipdb; ipdb.set_trace()
for i in range(500):
    torques = (
        kp * (initial_position - Go2_test.get_dofs_position(motors_dof_idx).detach().numpy())
        - kd * Go2_test.get_dofs_velocity(motors_dof_idx).detach().numpy()
    )
    Go2_test.control_dofs_force(torques, motors_dof_idx)
    # Go2.control_dofs_position(initial_position, motors_dof_idx)
    # Go2_test.control_dofs_position(initial_position, motors_dof_idx)
    scene.step()

# Analyze the robot
robot_mass = sum(link.inertial_mass for link in Go2_test.links)
print(f"Calculated total robot mass: {robot_mass:.2f} kg")

print("\n--- Link Inertial Properties ---")
for link in Go2_test.links:
    print(f"\nLink: {link.name}")
    print(f"  Mass: {link.inertial_mass}")
    print(f"  Inertia tensor:\n{link.inertial_i}")
    print(f"  COM position: {link.inertial_pos}")
    print(f"  COM orientation (quat): {link.inertial_quat}")
print("---------------------------------\n")

# Fetch all properties at once, assuming they return values for all DOFs

valid_joints = {}
for j in Go2_test.joints:
    if isinstance(j.dof_idx_local, int):
        valid_joints[j.name] = j.dof_idx_local

jnt_names = list(valid_joints.keys())
dofs_idx_list = list(valid_joints.values())

dofs_idx = np.array(dofs_idx_list, dtype=np.int32)

all_pos_lower, all_pos_upper = Go2_test.get_dofs_limit()
all_force_lower, all_force_upper = Go2_test.get_dofs_force_range()
all_armature = Go2_test.get_dofs_armature()
all_kp = Go2_test.get_dofs_kp()
all_kv = Go2_test.get_dofs_kv()
all_damping = Go2_test.get_dofs_damping()
all_stiffness = Go2_test.get_dofs_stiffness()
all_invweight = Go2_test.get_dofs_invweight()

print("\n--- Joint Properties ---")
for name in jnt_names:
    joint = Go2_test.get_joint(name)
    dof_idx = joint.dofs_idx_local
    print(f"\nJoint: {name}")
    print(f"  Position Limits: [{all_pos_lower[dof_idx].item():.4f}, {all_pos_upper[dof_idx].item():.4f}]")
    print(f"  Force Limits: [{all_force_lower[dof_idx].item():.2f}, {all_force_upper[dof_idx].item():.2f}]")
    print(f"  Armature: {all_armature[dof_idx].item():.4f}")
    print(f"  Kp: {all_kp[dof_idx].item():.2f}")
    print(f"  Kv: {all_kv[dof_idx].item():.2f}")
    print(f"  Damping: {all_damping[dof_idx].item():.4f}")
    print(f"  Stiffness: {all_stiffness[dof_idx].item():.4f}")
    print(f"  Inverse Weight: {all_invweight[dof_idx].item():.4f}")
print("---------------------------------\n")
