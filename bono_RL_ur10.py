# ur10 기준 강화학습!
# 07/12 시작
# 07/15 임시 코드 작성 완료. 터미널창에서 무수한 에러가 반겨줌. 
# 07/25 시뮬레이션 일단 켜는데까지는 성공, 거의 2주걸림. CPU로 돌리고있는데, env많이하면 렉 걸림
# 07/28 도대체 왜 rigid_body_states에서 앞의 3개 -> xyz값이 왜 안받아와지는지 모르겠다.
# 08/01 잘못하고 있다는 것을 깨닫고, 논문 다시 읽고, 코드갈아엎기 시작.
# 08/02 리워드 수정. 일정 거리 이하일때 리워드가 아닌, straight forward하게 거리에 비례해 리워드 주는것으로 변경, 속도제어로 변경.
# 08/03 코드가 안됨. TypeError: cannot create weak reference to 'numpy.ufunc' object
# 08/10 forwardkinematics 에서 1m더해줘야함.(해결) ball표시의 축이 안맞음, 가속도가 너무 크게나옴 
#        - 속도를 봐야하지 않을까?, 리셋이 제대로 안됨, 리셋속도 =0(해결)

# gymapi
from unittest import result
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
import numpy as np
import random
import math
import forward_kinematics_demo as f_k_d
import torch
import os
from tasks.base.vec_task import VecTask


class Ur10(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        
        self.action_scale = self.cfg["env"]["actionScale"]
        self.reset_dist = self.cfg["env"]["resetDist"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        
        self.num_props = self.cfg["env"]["numProps"]
        self.lambda_err = self.cfg["env"]["lambda_err"]
        self.lambda_eff = self.cfg["env"]["lambda_eff"]
        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        # self.distX_offset = 0.04
        self.dt = 1/60.
        self.cfg["env"]["numObservations"] = 12  # 마지막에 obs_buf 크기설정해주는것
        self.cfg["env"]["numActions"] = 6
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # 각 항목에 대해 gym gpu state tensor 받음
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) 
                    # actor의 rootstate tensor을 받아옴. shpae : (num_environments, num_actors * 13),  pos(3), rot(4), lin_vel(3), ang_vel(3)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) 
                    # Buffer has shape (num_environments, num_dofs * 2). Each DOF state contains position and velocity.
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) 
                    # rigid body states buffer 받아옴. shape : (num_environments, num_rigid_bodies * 13), pos(3), rot(4), lin_vel(3), ang_vel(3)
        
        # 각 항목들에 대해서 update를 실시
        self.gym.refresh_actor_root_state_tensor(self.sim)  # Updates actor root state buffer
        self.gym.refresh_dof_state_tensor(self.sim)         # Updates DOF state buffer
        self.gym.refresh_rigid_body_state_tensor(self.sim)  # Updates rigid body states buffer
        
        # create some wrapper tensors for different slices 
        self.ur10_default_dof_pos = to_torch([0.0, -1.57, 0.785, 0.785, 1.57, 0], device=self.device) # tensor([0., 0., 0., 0., 0., 0.], device='cuda:0')
        # [0.0, -1.57, 0.785, 0.785, 1.57, 0]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # (1536,2) # rigid-body-states pos(3), rot(4), lin_vel(3), ang_vel(3)
        
        self.ur10_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur10_dofs] # torch.Size([256, 6, 2])
    
        self.ur10_dof_pos = self.ur10_dof_state[..., 0] # env개수당, 6개 현재 각도 표시  
        self.ur10_dof_vel = self.ur10_dof_state[..., 1] # 6개 속도
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor) # .view(self.num_envs, -1, 13)
        # gymtorch.wrap_tensor(rigid_body_tensor)는 128행 (16env x 8joint?), 13열 
        # self.num_bodies = self.rigid_body_states.shape[1]  #  shape[0] 16, shape[1] 8 나옴 중복이라 쓸데없을듯
        
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13) # 13개 텐서로 나오는데, 1보다 작아서그런가 0.으로나옴 / 아니면 0인가?

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur10_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.ur10_dof_input_velocity = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        # self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
    
        self.current_xyz_hand_pos = torch.zeros_like(self.ur10_dof_targets[:,:3]) # xyz 3개의 pos를 담기위한 그릇 만들어줌
        self.last_dof_vel = torch.zeros_like(self.ur10_dof_vel)
        self.goal_target_pos = torch.zeros_like(self.ur10_dof_targets[:,:3])

        self.joint_handle6 = self.gym.find_actor_joint_handle(self.env_ptr, self.ur10_actor,'wrist_3_joint')

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self): 
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]
        # self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))


    def _create_ground_plane(self): 
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row): 
        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        #gymapi.Vec3(-spacing, 0.0, -spacing)  # if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # ur10 asset path
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        ur10_asset_file = "urdf/ur10.urdf"
        # box_asset_file = "bono_box.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            ur10_asset_file = self.cfg["env"]["asset"].get("assetFileNameUr10", ur10_asset_file)
            # box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)

        asset_path = os.path.join(asset_root, ur10_asset_file)
        asset_root = os.path.dirname(asset_path)
        ur10_asset_file = os.path.basename(asset_path)
        
        # create ball asset
        self.ball_radius = 0.1
        ball_asset_options = gymapi.AssetOptions()
        ball_asset_options.disable_gravity = True
        
        ball_asset = self.gym.create_sphere(self.sim, self.ball_radius, ball_asset_options)
        

        # load ur10 asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True    # 이거있어야 각 링크가 연결된 형태로 보임
        asset_options.armature = 0.01
        asset_options.disable_gravity = True
        asset_options.collapse_fixed_joints = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL#DOF_MODE_VEL  # 속도제어 #PD control
        asset_options.use_mesh_materials = True         # 이거는 일단 그냥 넣어봄
        ur10_asset = self.gym.load_asset(self.sim, asset_root, ur10_asset_file, asset_options) # 앞에서 설정한 조건으로 ur10 asset load
        self.ur10_asset = self.gym.load_asset(self.sim, asset_root, ur10_asset_file, asset_options)
        # load box asset
        # asset_options = gymapi.AssetOptions()
        # asset_options.flip_visual_attachments = True
        # asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.armature = 0.005
        # box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file, asset_options)

        self.num_ur10_bodies = self.gym.get_asset_rigid_body_count(ur10_asset) # 9 나옴 # int형으로 rigidbody개수셈
        self.num_ur10_dofs = self.gym.get_asset_dof_count(ur10_asset)          # 6 나옴 # asset파일기준, int형으로 dof반환
        # self.num_box_bodies = self.gym.get_asset_rigid_body_count(box_asset)   # 1 나옴
        # self.num_box_dofs = self.gym.get_asset_dof_count(box_asset)            # 0 나옴

        ur10_dof_stiffness = to_torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device)
        ur10_dof_damping = to_torch([200.0, 200.0, 200.0, 200.0, 200.0, 200.0], dtype=torch.float, device=self.device)
        # ur10_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400], dtype=torch.float, device=self.device)
        # ur10_dof_damping = to_torch([80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)
        # set ur10 dof properties
        ur10_dof_props = self.gym.get_asset_dof_properties(ur10_asset)
        self.ur10_dof_lower_limits = []
        self.ur10_dof_upper_limits = [] 
        for i in range(self.num_ur10_dofs):
            ur10_dof_props['driveMode'][i] = gymapi.DOF_MODE_VEL
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur10_dof_props['stiffness'][i] = ur10_dof_stiffness[i]
                ur10_dof_props['damping'][i] = ur10_dof_damping[i]
            else:
                ur10_dof_props['stiffness'][i] = 0.0  
                ur10_dof_props['damping'][i] = 200.0 # 50.0

            self.ur10_dof_lower_limits.append(ur10_dof_props['lower'][i]) # 6개의 텐서값 모두 -6.2831855
            self.ur10_dof_upper_limits.append(ur10_dof_props['upper'][i]) # 6개의 텐서값 모두  6.2831855 

        # set box dof props
        # box_dof_props = self.gym.get_asset_dof_properties(box_asset)
        # for i in range(self.num_box_dofs):
        #     box_dof_props['damping'][i] = 0.0

        # set torch device
        # self.ur10_rot = gymapi.Quat.from_euler_zyx(0, 0, 0.5 * math.pi) ####여기
        self.ur10_pose = gymapi.Transform() # r=self.ur10_rot
        self.ur10_pose.p = gymapi.Vec3(0, 0.0, 1.0) # parallel transeference

        self.ball_pose = gymapi.Transform()
        self.ball_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        # self.ur10_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107) # 이거안하니까 매니퓰레이터 렉걸린거마냥 진동하면서 움직임
        # box_pose = gymapi.Transform()
        # box_pose.p = gymapi.Vec3(0.0, 0.5, 0.0)
        self.envs = []
        self.ur10s = []
        self.hand_idxs = []
        self.obj_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.env_ptr = env_ptr
            self.ur10_actor = self.gym.create_actor(env_ptr, ur10_asset, self.ur10_pose, "ur10", i, 2)
            # box_actor = self.gym.create_actor(env_ptr, box_asset, box_pose, "bono_box", i, 0)
            hand_idx = self.gym.find_actor_rigid_body_index(env_ptr, self.ur10_actor, "ee_link", gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env_ptr, self.ur10_actor, ur10_dof_props)
            # self.gym.set_actor_dof_properties(env_ptr, box_actor, box_dof_props)
            self.hand_idxs.append(hand_idx)
            self.envs.append(env_ptr)
            self.ur10s.append(self.ur10_actor)

            # ball_handle = self.gym.create_actor(env_ptr, ball_asset, self.ball_pose, "ball", i, 0, 0)
            # self.obj_handles.append(ball_handle)            
            # self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.99, 0.66, 0.25))
            
        self.ur10_dof_lower_limits = to_torch(self.ur10_dof_lower_limits, device=self.device)
        self.ur10_dof_upper_limits = to_torch(self.ur10_dof_upper_limits, device=self.device)
        self.ur10_dof_speed_scales = torch.ones_like(self.ur10_dof_lower_limits)

        
    def _set_goal_pos(self, env_ids):
        self.torus_xyz = []
    # torus area 실험용 
        r_maj = 0.45  # major radius
        r_min = [0.15, 0.20, 0.30, 0.30]  # minor radii
        rot_ang = [math.radians(90), math.radians(120), math.radians(120), math.radians(180)] # [1.5707963267948966, 2.0943951023931953, 2.0943951023931953, 3.141592653589793]
        rot_i = 0

        rand_t = (rot_ang[rot_i] * torch.rand(len(env_ids), device=self.device))
        rand_theta = (2 * math.pi * torch.rand(len(env_ids), device=self.device)) # 360도 회전
        for i in range(len(env_ids)):
            self.torus_xyz.append([(r_maj + r_min[rot_i] * math.cos(rand_t[i])) * math.cos(rand_theta[i]),
                            (r_maj + r_min[rot_i] * math.cos(rand_t[i])) * math.sin(rand_theta[i]),
                             1 + r_min[rot_i] * math.sin(rand_t[i])])
        
        # rand_t = (rot_ang[rot_i] * torch.rand(1, device=self.device))
        # rand_theta = (2 * math.pi * torch.rand(1, device=self.device))
        # self.torus_xyz = ([(r_maj + r_min[rot_i] * math.cos(rand_t)) * math.cos(rand_theta),
        #                      (r_maj + r_min[rot_i] * math.cos(rand_t)) * math.sin(rand_theta),
        #                       1 + r_min[rot_i] * math.sin(rand_t)])
        torus_xyz_tensor = to_torch(self.torus_xyz, dtype=torch.float, device=self.device)
        
        self.goal_target_pos[env_ids, :] = torus_xyz_tensor# to_torch([0.3, 0.6, 0.9], dtype=torch.float, device=self.device)
# torus_xyz_tensor # to_torch(self.torus_xyz, dtype=torch.float, device=self.device)

        return self.goal_target_pos[env_ids]
  

    def _forwardkinematics(self, qc):
        self.revised_val = f_k_d.kinematic_func(qc)
         ### z=1 고려해줌 ## origin change
        return self.revised_val


    def compute_reward(self): 
        # self.current_xyz_hand_pos = self.obs_buf[:, :3]
        ur10_dof_vel = self.obs_buf[:, 3:9]
        ee_error = self.obs_buf[:, 9:]
        
        vel_diff = torch.norm(self.ur10_dof_vel - self.last_dof_vel, p=2, dim=-1)
        joint_accel = vel_diff / self.dt
        # retrieve environment observations from buffer
        self.rew_buf[:], self.reset_buf[:] = compute_ur10_reward(
            self.reset_buf, self.progress_buf, self.actions, self.current_xyz_hand_pos, ee_error, joint_accel,
            self.num_envs, self.max_episode_length, self.lambda_err, self.lambda_eff
        )
        # vel_diff = torch.norm(self.current_joint_vel - self.current_joint_vel_dt_1, p=2, dim=-1)
        result_env42 = open('result_env42.txt', 'a')
        # print(' ur10_dof_input_velocity :', self.ur10_dof_input_velocity[41], file = result_env42)
        # print(' ur10_dof_vel :', ur10_dof_vel[41, :], file = result_env42)
        # print(' last_dof_vel :', self.last_dof_vel[41, :], file = result_env42)
        # print(' joint_accel :', vel_diff[41] / self.dt,'\n', file = result_env42)

        # print(' self.current_xyz_hand_pos(kinematics):', self.current_xyz_hand_pos[41], file = result_env42)
        # print(' rigid_body_states xyz :',self.rigid_body_states[41, :3], file = result_env42)
        # print(' self.goal_target_pos[env_ids] :', self.goal_target_pos[41], '\n', file = result_env42)
        
        print(' ee_error_dist(d) :',torch.norm(ee_error[41], p=2, dim=-1).item(), file = result_env42)
        # print(' reward :', self.rew_buf[:][41], '\n============================\n', file = result_env42)
        # result_env42.close()


    def compute_observations(self):
        env_ids = np.arange(self.num_envs)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.current_joint_pos_list = self.ur10_dof_state[..., 0].tolist()
        # target_tolist = self.ur10_dof_targets.tolist()
        self.ur10_xyz_dof_pos_list =[]
        self.ur10_hand_pos_list =[]
        for i in env_ids:
            # self.ur10_xyz_dof_pos_list.append(self._forwardkinematics(target_tolist[i]))
            self.ur10_hand_pos_list.append(self._forwardkinematics(self.current_joint_pos_list[i]))

        self.current_xyz_hand_pos = to_torch(self.ur10_hand_pos_list) # 현재 xyz position
        ee_error = self.goal_target_pos - self.current_xyz_hand_pos # self.rigid_body_states[self.hand_idxs, :3]

        self.obs_buf = torch.cat((self.current_xyz_hand_pos[env_ids], self.ur10_dof_vel[env_ids], ee_error[env_ids]), dim= -1) # 3,6,3
        return self.obs_buf


    def reset_idx(self, env_ids):   
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.env_ids_int32 = env_ids.to(dtype=torch.int32)     
        # reset ur10 -> 랜덤한 위치, 속도로 초기화 시키는듯하다. 위치는 limit사이로 하는듯
        pos = tensor_clamp(
            self.ur10_default_dof_pos.unsqueeze(0) + 0.785 / 0.5 * (torch.rand((len(env_ids), self.num_ur10_dofs), device=self.device) - 0.5),
            self.ur10_dof_lower_limits, self.ur10_dof_upper_limits)
        # pos = self.ur10_default_dof_pos # ur10 default로 초기화 

        self.ur10_dof_input_velocity[env_ids, :self.num_ur10_dofs] = torch.zeros_like(self.ur10_dof_vel[env_ids]) # 0,0,0,0,0,0 으로 일단 넣어줌
        self.ur10_dof_pos[env_ids, :] = pos     # 랜덤한 6개의 각도값
        self.ur10_dof_targets[env_ids, :self.num_ur10_dofs] = pos # 6개의 각도 아마도 radian값 env에 따라 랜덤하게 나옴
        # self.ur10_dof_vel[env_ids, :] = tensor_clamp(
        #      self.ur10_dof_vel[env_ids, :], self.ur10_dof_lower_limits, self.ur10_dof_upper_limits)
        
        # multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur10_dof_targets),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # DOF position targets를 주어진 actor indices로 맞춘다. Full DOF position targets buffer가 필요하다.
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # # Me addition
        
        # self.gym.set_dof_velocity_target_tensor_indexed(self.sim, 
        #                                                 gymtorch.unwrap_tensor(self.ur10_dof_input_velocity),
        #                                                 gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.axes_geom = gymutil.AxesGeometry(0.2)
        # Create an wireframe sphere
        self.sphere_rot = gymapi.Quat.from_euler_zyx(0, 0, math.pi)
        self.sphere_pos = gymapi.Transform(r=self.sphere_rot) #r=self.sphere_rot
        self.sphere_pos.p = gymapi.Vec3(0,0,0)
        self._set_goal_pos(env_ids) # reset 할때 goal pos 새로 넣어줌
        self.gym.clear_lines(self.viewer)
        # # Create an wireframe axis
        for i in range(self.num_envs):
            self.sphere_pos.p = gymapi.Vec3(self.goal_target_pos[i][0], self.goal_target_pos[i][1], self.goal_target_pos[i][2]) #공위치1 target1
            self.sphere_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, self.sphere_pos, color=(1, 0, 0))
            self.sphere_pos.p.x = 0.0 
            self.sphere_pos.p.y = 0.0 
            self.sphere_pos.p.z = 1.0 
            gymutil.draw_lines(self.sphere_geom, self.gym, self.viewer, self.envs[i], self.sphere_pos)
            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], self.sphere_pos)    
        
        # self.goal_target_pos[env_ids, :] = to_torch([0.3, 0.6, 0.9], dtype=torch.float, device=self.device)
        # 타겟위치1 target1

        self.last_dof_vel[env_ids] = 0 
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        env_ids = np.arange(self.num_envs) 
        self.actions = actions.clone().to(self.device)
        targets_velocity = self.ur10_dof_input_velocity[:, :self.num_ur10_dofs] + self.ur10_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.ur10_dof_input_velocity[:, :self.num_ur10_dofs] = tensor_clamp(targets_velocity, self.ur10_dof_lower_limits, self.ur10_dof_upper_limits)
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(self.ur10_dof_input_velocity))
        # print('===========shape1=============',self.dof_state.view(self.num_envs, -1,2).shape)
        # self.ur10_dof_input_velocity.type(float)
        # self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur10_dofs]
        
        # self.x0_dt_v = self.ur10_dof_pos.view(-1)+self.dt * self.ur10_dof_vel.view(-1)+0.5*self.dt * (self.ur10_dof_input_velocity.view(-1)-self.ur10_dof_vel.view(-1)) * (self.ur10_dof_input_velocity.view(-1)-self.ur10_dof_vel.view(-1)) ##
        # self.in_tens = torch.stack((self.x0_dt_v,self.ur10_dof_input_velocity.view(-1)),dim=-1)
        # print('===========intens view=============',self.in_tens)
        # print('======self.dof_state=====',self.dof_state)
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.in_tens)) ##
        # print('============ur10_dof_input_velocity=====',self.ur10_dof_input_velocity[0])
        dof_state_tensor_1 = self.gym.acquire_dof_state_tensor(self.sim) 
        self.dof_state_1 = gymtorch.wrap_tensor(dof_state_tensor_1) 
        
        self.ur10_dof_state_1 = self.dof_state_1.view(self.num_envs, -1, 2)[:, :self.num_ur10_dofs]
    
        self.ur10_dof_vel_1 = self.ur10_dof_state_1[..., 1]
        

    def post_physics_step(self):  
        self.progress_buf += 1 
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.last_dof_vel[:] = self.ur10_dof_vel[:]
        self.compute_observations()
        self.compute_reward()
        
# collision 무시해도될듯? self-collision check link-box or robot-box collision check.
# return에 속도 받음. output limit
# 위치차이, - 속도 명령 : set함수 - 리워드계산, 
# current pos, target pos, joint vel, joint pos, => state (reward계산)
# output => joint vel, setjoint로 입력 -> 움직임
# xyz 위치차이가 얼마 이하면 reset
# reset했을때 target, current pos 고정, current pos 고정 target랜덤 .... 바꿔서 학습가능
# tensorboard - parameters, reward - cartpole같은데 한번 적용해서 익히기

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_ur10_reward(reset_buf, progress_buf, actions, current_xyz_hand_pos, ee_error, joint_accel,
               num_envs, max_episode_length,  lambda_err, lambda_eff):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tuple[Tensor, Tensor]

    # 변수 이것저것
    d = torch.norm(ee_error, p=2, dim=-1) # 유클리드 거리를 계산해줌 xyz에러 -> 거리에러
    ee_error_dist = d
    reward = math.exp(-lambda_err * (d ** 2)) - lambda_eff * joint_accel 
    reward = torch.where(current_xyz_hand_pos[:,2]<0, reward-0.01, reward)

    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(ee_error_dist<=0.05, torch.ones_like(reset_buf), reset_buf) #일정거리 이하이면 리셋
    
    reset_buf = torch.where(ee_error_dist>=1.5, torch.ones_like(reset_buf), reset_buf)
    return reward, reset_buf
