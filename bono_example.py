# Simulation Setup
"""
아이작 짐은 지능형 에이전트(agent)를 훈련시키고, 여러 움직임을 할 수 있도록 도와준다.
데이터 중심의 API구조를 가지고 있으며, C++의 도움을 받고 사용자가 Python으로 입력한 정보를 기반으로 구동된다.
(flat한 data를 기반으로 구동되므로 numpy와의 호환성이 좋다.)
위의 방식을 사용하여 PyTorch처럼 CPU나 GPU Tensor를 시뮬레이션 데이터와 서로 주고 받는다.
"""

#gymapi
"""
아이작의 주된 기능(supporting data & constants)은 gymapi 모듈에 들어가있다.
"""
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import random
import math
gym = gymapi.acquire_gym()                  # gym의 API를 불러온다.

#1. Creating a Simulation
"""
gym이라는 변수 자체는 별다르게 하는게 없다. 그냥 Gym의 API를 불러올 뿐이다.
Simulation을 만들기 위해 creat_sim이라는 command를 통해 불러온다.
$ creat_sim(CUDA GPU, 렌더링 GPU, 사용할 Simulation 종류, Simulation 변수들)
>>> r: Simulation Handle
    t: SIM
"""
compute_device_id = 0           #CUDA 계산에 들어가는 GPU를 설정한다.
graphics_device_id = 0          #Rendering에 들어가는 GPU를 설정한다.
#----------------------------------------------------------------------------------


# 2. Simulation Parameters
"""
Simulation Parameter들은 Simulation의 디테일한 물리현상들을 설정할 수 있게 한다.
에이전트의 안정성과 수행능력을 향상시키기 위해 이런 변수를 올바르게 사용하는 것이 좋다.
"""
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--num_envs", "type": int, "default": 36, "help": "Number of environments to create"},
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])


# 2-1. set common parameters (Backend와 관계없이 설정되는 Parameter)
sim_params          = gymapi.SimParams()          # get default set of parameters
sim_params.dt       = 1 / 60                      # dt설정
sim_params.substeps = 2                           # substep 설정
sim_params.up_axis  = gymapi.UP_AXIS_Z            # Simulation의 Up-Axis설정
sim_params.gravity  = gymapi.Vec3(0.0, 0.0, -9.8) # Simulation의 Gravity-direction 설정

# 2-2. set PhysX-specific parameters
sim_params.physx.use_gpu = True                # Simulation을 PhysX backend에서 gpu사용
sim_params.physx.solver_type = 1               # Simulation 내에서 문제를 해결할 때 사용할 Equation. 현재는 jacobian 사용
"""
property: solver_type
0 = XPBD(GPU)                   1 = Newton Jacobi(GPU)      2 = Newton LDLT(CPU)        
3 = Newton PCG(CPU)             4 = Newton PCG(GPU)         5 = Newton PCR(GPU)         
6 = Newton Gauss Seidel(CPU)    7 = Newton NNCG(GPU)
"""
sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()         
initial_state = np.copy(                                    # 나중에 reset 시킬 때의 초기상태를 변수로 저장한다.
    gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


sim_params.physx.num_position_iterations = 6    # Position 해석을 위해 몇번을 계산할지 Range [1,255]
sim_params.physx.num_velocity_iterations = 1    # Velocity 해석을 위해 몇번을 계산할지 Range [1,255]
sim_params.physx.contact_offset = 0.01          # contact_offset보다 거리가 짧다면 접점 생성
sim_params.physx.rest_offset = 0.0              # 두개의 도형이 rest_offset과 동일할 때 정지
# [dt, flex, gravity, num_client_threads, physX, stress_visualization,
# stress_visualization_max & min, substeps, up_axis, use_gpu_pipeline]

# # set Flex-specific parameters (Flex-version인데 Pass(주석처리))
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

# 2-3. create sim with these parameters
# sim = gym.create_sim(           # Simulation 만듬
#         compute_device_id,      # CUDA 연산
#         graphics_device_id,     # Rendering용 : headlss 모드에서는 사용X. -1로 설정하면 됨
#         gymapi.SIM_PHYSX,       # 어떤 background를 사용할지 정할 수 있다.
#                                 # SIM_Physx: Rigid, SIM_Flex: Soft 가 있다.
#         sim_params)             # 2-1, 2-2에서 설정한 모든 parameter들을 simulation에 넣어준다.
#----------------------------------------------------------------------------------


# 3. Creating a Ground Plane
"""
무중력을 가정하는 환경이 아니라면, 접지면이 필요하다.
"""

# 3-1. configure the ground plane
plane_params                  = gymapi.PlaneParams()    # 평면에 대한 Parameter 설정을한다.
plane_params.normal           = gymapi.Vec3(0, 0, 1)    # 법선 벡터를 표시한다.(0,0,1) = Z-axis와 수직!
# plane_params.distance         = 0                       # 평면의 원점 좌표계로부터의 거리
# plane_params.static_friction  = 1                       # 평면의 static 마찰계수 [정지마찰계수]
# plane_params.dynamic_friction = 1                       # 평면의 dynamic 마찰계수 [운동마찰계수]
# plane_params.restitution      = 0                       # 평면의 restitution 반력계수 [충돌, 탄성]

# 3-2. create the ground plane
gym.add_ground(sim, plane_params)                       # Simulation에 Plane을 Parameter와 함께 입력한다.
#----------------------------------------------------------------------------------

# 4. Loading Assets
"""
Assets를 불러우는데 파일 형식은 URDF,MJCF,USD의 형식이 존재한다.
이런 파일형식에는 본체, 충돌모양, 첨부파일, 관절 및 자유도(DOF)의 정의를 포함하게 된다.
AssetOptions()는 모델의 물리 및 시각적 특성에 영향을 미치므로 안정성과 성능에 영향을 미친다.
"""

# 4-1. Asset Load 
asset_root = "../../assets"                                         # asset directory 지정 (상대, 절대 경로 모두 가능)
asset_file = "urdf/franka_description/robots/franka_panda.urdf"     # 불러올 urdf파일을 지정
# "urdf/franka_description/robots/franka_panda.urdf"
# "urdf/ball.urdf"
# 4-2. Asset options
asset_options               = gymapi.AssetOptions()                 # asset에 대한 추가적인 parameter들을 설정한다.
asset_options.fix_base_link = True                                  # asset을 고정으로 배치한다. fix_base_link값을 조정하면 모델을 basis에 고정하거나 자유롭게 움직일 수 있다.
asset_options.flip_visual_attachments = True                        # franka_attractor 모델링이 연결돼서 나타남. 이것을 설정안하면 link마다 떨어져있다.
asset_options.armature      = 0.01                                  # 본체와 링크에 대한 inertia tensor의 대각선 요소에 추가되는 값이다. 시뮬레이션의 안정성 향상에 기여한다.
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)  # Simulation에 입력하고, 해당하는 경로를 넣어준다.
#----------------------------------------------------------------------------------


# 5. Environments and Actors
"""
IsaacGym은 의미있는 계산을 수행하는데 더 많은 시간이 소요된다.
Agent의 위치와 속성을 제어하는 방법을 소개한다.
각각의 Actor들은 필수적으로 (env)Environment에 배치돼야한다.
"""

# 5-1. set up the env grid
num_envs      = 4                                               # env 생성의 반복 횟수
envs_per_row  = 2                                                # 총 8줄의 env를 추가로 생성
env_spacing   = 1.0                                              # env사이의 공간
env_lower     = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper     = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
# scale         = 2

# 5-2. cache some common handles for later use  / 나중에 진행되는 events에 대해 저장한다.
envs          = []
actor_handles = []
attractor_properties = gymapi.AttractorProperties()
franka_hand = "panda_hand"
attractor_handles = []
attractor_properties = gymapi.AttractorProperties()
attractor_properties.stiffness = 5e5
attractor_properties.damping = 5e3


# Create helper geometry used for visualization
# Create an wireframe axis
axes_geom = gymutil.AxesGeometry(0.1)
# Create an wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
sphere_geom = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))

# 5-3. create and populate the environments  /  한번에 다양한 env와 actor를 생성한다.
for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)
    pose = gymapi.Transform()
    # add franka
    franka_handle = gym.create_actor(env, asset, pose, "franka", i, 2)
    body_dict = gym.get_actor_rigid_body_dict(env, franka_handle)
    props = gym.get_actor_rigid_body_states(env, franka_handle, gymapi.STATE_POS)
    hand_handle = body = gym.find_actor_rigid_body_handle(env, franka_handle, franka_hand)
    
    # Initialize the attractor
    attractor_properties.target = props['pose'][:][body_dict[franka_hand]]
    attractor_properties.target.p.y -= 0.1
    attractor_properties.target.p.z = 0.1
    attractor_properties.rigid_handle = hand_handle

    # Draw axes and sphere at attractor location
    gymutil.draw_lines(axes_geom, gym, viewer, env, attractor_properties.target)
    gymutil.draw_lines(sphere_geom, gym, viewer, env, attractor_properties.target)

    actor_handles.append(franka_handle)
    attractor_handle = gym.create_rigid_body_attractor(env, attractor_properties)
    attractor_handles.append(attractor_handle)
# for i in range(num_envs):                                                  # Simulation 공간에 포함되는 고유 좌표공간을 만든다. (청크같은 개념)
#     env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
#     envs.append(env)
#     height = 1                                     #  
#     pose   = gymapi.Transform()                                            # env에 들어갈 asset의 pose를 설정한다. 
#     pose.p = gymapi.Vec3(0.0, height, 0.0)                                 # Position을 설정한다.
#     """    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)     # Rotation - Quaternion을 사용한다.
#                                                                # sim_params의 up-axis를 기준으로 원하는 방향 회전이 가능하다.
#     """
#     actor_handle = gym.create_actor(env, asset, pose, "MyActor", i, 1)
#     gym.set_actor_scale(env, actor_handle, scale)                                     # scaling
#     actor_handles.append(actor_handle)
""" 
    $ create_actor(
    env라는 청크에,
    asset이 포함된 모델을 입력,
    어느곳에 어떻게 넣을지 지정,
    이름은 MyActor이며,
    collision_group: 충돌 그룹 설정,{동일한 No.끼리 충돌한다.}
    collision_filter: 충돌 필터 설정)
    -- 하위 두개 항목은 물리 시뮬레이션에서 중요한 역할을 한다.--
    collision_group : 동일한 env에서의 Actor간 충돌을 설정한다. *-1(특수충돌 : 공유개체)은 모든 객체와 충돌한다.
    collision_filter: body간의 충돌을 필터링한다. env안에 다물체의 자체충돌과 Scene의 특정 충돌을 설정할때 사용한다.
"""
#----------------------------------------------------------------------------------


# 6. Running the Simulation
"""
env와 다른 Parameter들을 설정하고 나서 시뮬레이션 시작 가능하다.
해당하는 Simulation의 fps는 위에서 설정한 dt의 영향을 받는다.
뷰어에 대한 전체화면 모드는 F11로 전환가능하다.
"""
# 6-1. create Viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()         
# initial_state = np.copy(                                    # 나중에 reset 시킬 때의 초기상태를 변수로 저장한다.
#     gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))

# Point camera at environments
cam_pos = gymapi.Vec3(4.0, -8.0, 8.0)
cam_target = gymapi.Vec3(1.0, 2.0, 1.0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# 6-2. with the simulation / 키보드와 마우스를 활용할 수 있도록
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "space_shoot")
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")
gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "mouse_shoot")

# 6-3. set & make simulation
while not gym.query_viewer_has_closed(viewer):              # viewer가 닫힐 때 simulation 종료
    gym.simulate(sim)                                       # step the physics
    gym.fetch_results(sim,True)

    for evt in gym.query_viewer_action_events(viewer):      # viewer내에서 키보드, 마우스의 기능을 추가 
                                                            # simulation내에서 추가적으로 필요한 기능이 있다면 해당 형식을 참고하여 삽입하면된다.
        if evt.action == "reset" and evt.value > 0 :        # R키를 누르면 reset된다.
            gym.set_sim_rigid_body_states_tensor(sim)

        elif (evt.action == "space_shoot" or evt.action == "mouse_shoot") and evt.value > 0 :
            if evt.action == "mouse_shoot":                 # 마우스 좌표를 알려주는 기능
                pos         = gym.get_viewer_mouse_position(viewer)
                window_size = gym.get_viewer_size(viewer)
                xcoord      = round(pos.x * window_size.x)
                ycoord      = round(pos.y * window_size.y)
                print(f"Fired projectile with mouse at coords: {xcoord} {ycoord}")

    gym.step_graphics(sim)                                  # update the viewer  / 시뮬레이션의 시각적 표현을 물리 상태와 동기화한다. (simulation을 보기위해 필수적)
    gym.draw_viewer(viewer, sim, True)                      # 최신 스냅샷에서 렌더링 수행
    """
    이렇게만 구성된 simulation은 dt증분이 실시간보다 빠르게 처리되기 때문에, 오차가 종종 발생한다.
    시각적인 업데이트 빈도를 실시간으로 동기화하기위해 추가 구문이 필요하다.
    이는 고성능 컴퓨터에서는 계산에 관한 속도저하를 일으키는 요소가 될 수도 있다. (Viewer와 동반시)
    """
    gym.sync_frame_time(sim)                                # 실시간 Viewer와 실제 simulation의 sync를 맞춘다.

    # 7. Clean Up  /  끝나고 나서 다음과 같이 해제시켜야 한다. 
                     # viewer가 끝난다고해서 simulation이 끝나는 것은 아니기때문에 코드로 정확하게 simulation을 종료시켜야한다.
print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)