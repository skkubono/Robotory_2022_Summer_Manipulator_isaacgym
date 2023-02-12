"""
220701(금) ur10.urdf파일 받아서 world에 나타내기위한 작업시작
... forward_kinematics_demo 파일에서 joint angle을 통해 end-effector(ee_link)위치 계산 성공
220715(금) 속도로 set.joint position 나타내고, 어느정도 데이터 받아오는 것을 완료, attractor및 
           쓸모없는 코드 삭제
"""

# gymapi
# from turtle import forward
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
import numpy as np
import math
import random
import forward_kinematics_demo as f_k_d
import torch
 
# Initialize gym
gym = gymapi.acquire_gym()

# Parse arguments
args = gymutil.parse_arguments(description="ur10")

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.solver_type = 5
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 15
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

# sim_params.use_gpu_pipeline = True  ############ 이거 True로 하면 RL처럼 에러뜸
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()
up_axis = "z"
# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0) # if up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
gym.add_ground(sim, plane_params)

# Load ur10 asset
asset_root = "../../assets"
asset_file = "urdf/ur10.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True    # 이거있어야 각 링크가 연결된 형태로 보임
asset_options.armature = 0.01
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# Set up the env grid
num_envs = 1
envs_per_row  = 2   
spacing = 1
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# Some common handles for later use
envs = []
ur10_handles = []
ur10_hand = "ee_link"
base_link = []
shoulder_link = []
hand_idxs = []
ee_handles = []


# set torch device
device = args.sim_device if args.use_gpu_pipeline else 'cpu'
ee_pose = gymapi.Transform()
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107) # 이거안하니까 매니퓰레이터 렉걸린거마냥 진동하면서 움직임

print("Creating %d environments" % num_envs)
num_per_row = int(math.sqrt(num_envs))

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)
    # add actor
    ur10_handle = gym.create_actor(env, asset, pose, "ur10", i, 2)
    # body_dict = gym.get_actor_rigid_body_dict(env, ur10_handle)
    # props = gym.get_actor_rigid_body_states(env, ur10_handle, gymapi.STATE_POS)
    # hand_handle = body = gym.find_actor_rigid_body_handle(env, ur10_handle, ur10_hand)
    # obj_handle =  gym.find_actor_rigid_body_handle(env, ur10_handle, "ee_link")

    hand_idx = gym.find_actor_rigid_body_index(env, ur10_handle, "ee_link", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    ur10_handles.append(ur10_handle)

# get joint limits and ranges for ur10
ur10_dof_props = gym.get_actor_dof_properties(envs[0], ur10_handles[0])
ur10_lower_limits = ur10_dof_props['lower']
ur10_upper_limits = ur10_dof_props['upper']
ur10_ranges = ur10_upper_limits - ur10_lower_limits
ur10_mids = 0.5 * (ur10_upper_limits + ur10_lower_limits)
ur10_num_dofs = len(ur10_dof_props)

# override default stiffness and damping values
ur10_dof_props['stiffness'].fill(1000.0)
ur10_dof_props['damping'].fill(1000.0)
# Give a desired pose for first 2 robot joints to improve stability
ur10_dof_props["driveMode"][0:6] = gymapi.DOF_MODE_NONE # 드라이브모드 속도 위치
ur10_dof_props["driveMode"][7:] = gymapi.DOF_MODE_VEL
ur10_dof_props['stiffness'][7:] = 1e10
ur10_dof_props['damping'][7:] = 1.0
# print(ur10_dof_props)

for i in range(num_envs):
    gym.set_actor_dof_properties(envs[i], ur10_handles[i], ur10_dof_props)

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)
# print('============rb_states========',rb_states)

# Point camera at environments
cam_pos = gymapi.Vec3(-2.0, 2.0, -2)
cam_target = gymapi.Vec3(2.0, 1.3, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


# print('ur10 handle: ', ur10_handle)
joint_handle1 = gym.find_actor_joint_handle(env, ur10_handle,'shoulder_pan_joint')
joint_handle2 = gym.find_actor_joint_handle(env, ur10_handle,'shoulder_lift_joint')
joint_handle3 = gym.find_actor_joint_handle(env, ur10_handle,'elbow_joint')
joint_handle4 = gym.find_actor_joint_handle(env, ur10_handle,'wrist_1_joint')
joint_handle5 = gym.find_actor_joint_handle(env, ur10_handle,'wrist_2_joint')
joint_handle6 = gym.find_actor_joint_handle(env, ur10_handle,'wrist_3_joint')

def set_position(aa):
    gym.set_joint_target_position(env, joint_handle1,aa[0])
    gym.set_joint_target_position(env, joint_handle2,aa[1]) # +로하면 바닥에 박힘
    gym.set_joint_target_position(env, joint_handle3,aa[2]) # +는 자기 축기준 시계방향           
    gym.set_joint_target_position(env, joint_handle4,aa[3]) # +는 자기축기준 반시계방향
    gym.set_joint_target_position(env, joint_handle5,aa[4]) # + 는 자기 축기준 반시계
    gym.set_joint_target_position(env, joint_handle6,aa[5]) # 딱히 방향이 중요하진 않음 ㅇㅇ

def forwardkinematics(qc):
    revised_val = f_k_d.kinematic_func(qc)
    revised_val[0] = -revised_val[0]
    revised_tem = revised_val[1]
    revised_val[1] = revised_val[2]
    revised_val[2] = revised_tem
    return revised_val

############################# 실험용 ########################################

# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pos = gymapi.Transform(r=sphere_rot)
sphere_pos.p.x = 1.3
sphere_pos.p.y = 0
sphere_pos.p.z = -2.3
sphere_geom = gymutil.WireframeSphereGeometry(0.05, 20, 20, sphere_pos, color=(1, 0, 0))

gymutil.draw_lines(axes_geom, gym, viewer, env, sphere_pos)
gymutil.draw_lines(sphere_geom, gym, viewer, env, sphere_pos)
         

############################################################################


# Time to wait in seconds before moving robot
next_ur10_update_time = 0

# simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Every 0.01 seconds (dt) the pose of the attactor is updated
    t = gym.get_sim_time(sim)
    qc=[]
    if t >= next_ur10_update_time:
        # gym.set_joint_target_velocity(env, 6, 0.0001)
        ang_list1=[]
        for i in range(6):
            ang_list1.append(random.uniform(0.1,0.4))

        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(num_envs, -1, 13) # 13개 텐서로 나오는데, 1보다 작아서그런가 0.으로나
        
        hand_pos = rb_states[hand_idxs, :3][0]
        hand_rot = rb_states[hand_idxs, 3:7]
        hand_vel = rb_states[hand_idxs, 7:]
        # print('hand_pos : ',hand_pos)
        # print('hand_rot : ',hand_rot)
        # print('hand_vel : ',hand_vel)
        # print('hand_idxs whole : ',rb_states[hand_idxs])
        set_position([0, -1.23, 0.785, 0.785, 1.57, 0])
        if t>3:
            set_position([1.234, -1.57, 0.433, -0.785, 0, 1.57])
      
        dof_state_tensor = gym.acquire_dof_state_tensor(sim)####get
        dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        num_ur10_dofs = gym.get_asset_dof_count(asset)
        print('=======num ur10 dofs======', num_ur10_dofs)
        ur10_dof_state = dof_state.view(num_envs, -1, 2)[:, :num_ur10_dofs]        
        # ur10_dof_pos = ur10_dof_state[..., 0]
        # ur10_dof_vel = ur10_dof_state[..., 1]
        rigid_body_tensor = gym.acquire_rigid_body_state_tensor(sim)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

        actor_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
        root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(num_envs, -1, 13)
        # print(' ur10_rigid_body_count_value :        ', )
        print(' 내가원하는것1111 :            ',hand_pos, '원하는것 끝1111')
        # print(' 내가원하는것2222 :            ',ur10_dof_targets, '원하는것 끝2222')
        
        # qc에 값 1~6번 joint angle값 넣어줌
        for i in range(1,7):
            qc.append(gym.get_joint_position(env,i))
        # print('rigid_body_states:      ',hand_pos)    
        # print('ang_list1 :                  ',torch.Tensor(ang_list1))
        # print('ang_list1_to_position :      ',torch.Tensor(forwardkinematics(ang_list1)))
        print('hand_pos:        ',hand_pos)
        print('forward_kinematics_cal :',forwardkinematics(qc),'\n')
        # update_ur10(t)
        next_ur10_update_time += 3.0
        # gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)
        
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# 7. Clean Up
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

# reward input & output 설정