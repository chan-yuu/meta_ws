import numpy as np
import gymnasium as gym
from typing import Dict, Any, Tuple
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.manager.agent_manager import VehicleAgentManager
from metadrive.component.vehicle.vehicle_type import vehicle_type, vehicle_class_to_type
from metadrive.utils import Config
from frenet_converter import FrenetConverter, ReferenceLineGenerator
import matplotlib.pyplot as plt

# 导入ROS发布器
try:
    from ros_publisher import TrajectoryPlanningROSPublisher
    ROS_AVAILABLE = True
except ImportError as e:
    print(f"ROS not available: {e}")
    ROS_AVAILABLE = False
    TrajectoryPlanningROSPublisher = None


class TrajectoryPlanningVehicle(BaseVehicle):
    """
    自定义车辆类
    """
    # 车辆物理参数 (基于 DefaultVehicle)
    TIRE_RADIUS = 0.313
    TIRE_WIDTH = 0.25
    MASS = 1100
    LATERAL_TIRE_TO_CENTER = 0.815
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    path = ('ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0))  # asset path, scale, offset, HPR

    # MAX_STEERING = 60

    DEFAULT_LENGTH = 4.515  # meters
    DEFAULT_HEIGHT = 1.19  # meters
    DEFAULT_WIDTH = 1.852  # meters

    @property
    def LENGTH(self):
        return self.DEFAULT_LENGTH

    # @property
    # def MAX_STEERING(self):
    #     return self.MAX_STEERING

    @property
    def WIDTH(self):
        return self.DEFAULT_WIDTH

    @property
    def HEIGHT(self):
        return self.DEFAULT_HEIGHT
    
    # def __init__(self, vehicle_config=None, name=None, random_seed=None, position=None, heading=None, _calling_reset=True):
    def __init__(self, vehicle_config=None, name=None, random_seed=None, position=None, heading=None, _calling_reset=True, ros_publisher=None):
        super().__init__(vehicle_config, name, random_seed, position, heading, _calling_reset)
        
        # 轨迹规划参数
        self.current_trajectory = None
        self.trajectory_time = 0.0
        self.dt = 0.1  # 时间步长
        
        # Frenet坐标系参数
        self.reference_path = None
        # 初始化
        self.ros_publisher = ros_publisher
        self._debug_counter = 0  # 调试计数器
        
    def step(self, actions):
        # 更新调试计数器
        self._debug_counter = getattr(self, '_debug_counter', 0) + 1
        
        # 获取当前轨迹点
        match_point = self._get_target_trajectory_point()
        
        # 执行轨迹跟踪控制
        if match_point is not None:
            steering, throttle_brake = self._trajectory_tracking_control(match_point)
            actions = [steering, throttle_brake]
        
        # 调用父类的step方法
        return super(TrajectoryPlanningVehicle, self).step(actions)

        
    def set_trajectory(self, trajectory_points, trajectory_time_horizon):
        """
        设置轨迹点
        Args:
            trajectory_points: 轨迹点列表 [(x, y, heading, velocity), ...]
            trajectory_time_horizon: 轨迹时间范围
        """
        self.current_trajectory = trajectory_points
        self.trajectory_time = 0.0
        self.trajectory_time_horizon = trajectory_time_horizon
        
    def _set_action(self, action):
        """
        动作设置方法，使用轨迹跟踪控制
        """
        if self.current_trajectory is None or len(self.current_trajectory) == 0:
            # 如果没有轨迹，使用默认控制
            super()._set_action(action)
            return
            
        # 根据当前时间获取目标轨迹点
        target_point = self._get_target_trajectory_point()
        if target_point is None:
            # 轨迹结束，保持当前状态
            super()._set_action([0, 0.1])  # 轻微油门保持运动
            return
            
        # 计算控制指令
        steering, throttle_brake = self._trajectory_tracking_control(target_point)
        
        # 调试输出
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
            
        # if self._debug_counter % 50 == 0:  # 每50步输出一次
        # 轨迹跟踪打印已移动到step方法中
        
        # 应用控制指令
        self.throttle_brake = throttle_brake
        self.steering = steering
        
        # 确保控制指令被正确应用
        if hasattr(self, 'system') and self.system is not None:
            self.system.setSteeringValue(self.steering * self.max_steering, 0)
            self.system.setSteeringValue(self.steering * self.max_steering, 1)
            self._apply_throttle_brake(throttle_brake)
        
        # 更新轨迹时间
        self.trajectory_time += self.dt
        
    def _get_target_trajectory_point(self):
        """
        根据统一前瞻机制获取目标轨迹点
        """
        if self.current_trajectory is None or len(self.current_trajectory) == 0:
            return None

        # 获取车辆当前位置
        vehicle_x, vehicle_y = self.position[0], self.position[1]
        
        # 在轨迹上找到最近点
        min_distance = float('inf')
        closest_index = 0
        
        for i, (x, y, _, _) in enumerate(self.current_trajectory):
            distance = np.sqrt((x - vehicle_x)**2 + (y - vehicle_y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 计算统一的前瞻距离,基于车速 动态前瞻
        # TODO 控制的调参
        # vehicle_speed = max(0.1, self.speed)  # 避免除零
        vehicle_speed = self.speed
        base_lookahead = 2.0  # 基础前瞻距离
        speed_factor = 1.0    # 速度系数
        unified_lookahead_distance = base_lookahead + speed_factor * vehicle_speed
        
        # 在轨迹上寻找前瞻点
        target_index = closest_index
        accumulated_distance = 0.0
        
        for i in range(closest_index, len(self.current_trajectory) - 1):
            x1, y1, _, _ = self.current_trajectory[i]
            x2, y2, _, _ = self.current_trajectory[i + 1]
            segment_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            if accumulated_distance + segment_distance >= unified_lookahead_distance:
                target_index = i + 1
                break
            accumulated_distance += segment_distance
            target_index = i + 1
        
        # 确保索引在有效范围内
        target_index = min(target_index, len(self.current_trajectory) - 1)
        # 打印前瞻距离和索引
        print("lookahead_distance: ", unified_lookahead_distance)
        print("target_index: ", target_index)
        x, y, yaw, match_vel = self.current_trajectory[target_index]
        
        # 发布匹配点的箭头标记
        if hasattr(self, 'ros_publisher') and self.ros_publisher is not None:
            try:
                self.ros_publisher.update_target_point_marker(x, y, yaw, match_vel)
            except Exception as e:
                print(f"Failed to publish target point marker: {e}")

        return self.current_trajectory[target_index]
        
    def _trajectory_tracking_control(self, target_point):
        """
        轨迹跟踪控制算法 - 纯跟踪控制器 (Pure Pursuit Controller)
        Args:
            target_point: (x, y, heading, velocity)
        Returns:
            steering, throttle_brake
        """
        target_x, target_y, target_heading, target_velocity = target_point
        
        # 获取当前状态
        current_x, current_y = self.position[0], self.position[1]
        current_heading = self.heading_theta
        # current_velocity = max(self.speed, 0.1)  # 避免除零
        current_velocity = self.speed
        
        # 车辆轴距参数
        wheelbase = 2.5  # 车辆轴距 (m)
        
        # 直接使用目标点作为前瞻点（已经通过统一前瞻机制计算）
        lookahead_x = target_x
        lookahead_y = target_y
        
        # 计算到目标点的距离
        dx = target_x - current_x
        dy = target_y - current_y
        lookahead_distance = np.sqrt(dx**2 + dy**2)
        lookahead_distance = max(lookahead_distance, 0.1)  # 避免除零
        
        print("unified_lookahead_distance: ", lookahead_distance)
        
        # 将前视点转换到车辆坐标系
        # 车辆坐标系：x轴指向前方，y轴指向左侧
        cos_heading = np.cos(current_heading)
        sin_heading = np.sin(current_heading)
        
        # 全局坐标到车辆坐标的转换
        local_x = (lookahead_x - current_x) * cos_heading + (lookahead_y - current_y) * sin_heading
        local_y = -(lookahead_x - current_x) * sin_heading + (lookahead_y - current_y) * cos_heading
        
        # 计算转向角 (纯跟踪公式)
        if abs(local_x) < 1e-6:  # 避免除零
            steering = 0.0
        else:
            # 纯跟踪转向角公式: δ = arctan(2 * L * sin(α) / ld)
            # 其中 α 是前视点相对于车辆的角度，L是轴距，ld是前视距离
            alpha = np.arctan2(local_y, local_x)
            steering = np.arctan2(2.0 * wheelbase * np.sin(alpha), lookahead_distance)
        
        steering_deg = np.clip(steering * 180 / np.pi, -60, 60)
        # self.MAX_STEERING = 60
        s_steering_deg = steering_deg / self.max_steering

        # 限制转向角范围
        steering = np.clip(s_steering_deg, -1.0, 1.0)
        
        # 速度控制（保持不变）
        velocity_error = target_velocity - current_velocity
        
        # 简单的比例控制
        kp_v = 0.5
        throttle_brake = kp_v * velocity_error
        
        # 限制油门/刹车范围
        throttle_brake = np.clip(throttle_brake, -1.0, 1.0)
        
        # 调试
        # if hasattr(self, '_debug_counter') and self._debug_counter % 50 == 0:
        #     print(f"Pure Pursuit Debug:")
        #     print(f"  Current pos: ({current_x:.2f}, {current_y:.2f})")
        #     print(f"  Target pos: ({target_x:.2f}, {target_y:.2f})")
        #     print(f"  Lookahead pos: ({lookahead_x:.2f}, {lookahead_y:.2f})")
        #     print(f"  Lookahead distance: {lookahead_distance:.2f}")
        #     print(f"  Local coords: ({local_x:.2f}, {local_y:.2f})")
        #     print(f"  Steering: {steering:.3f}")
        
        return steering, throttle_brake


# 注册 TrajectoryPlanningVehicle 到车辆类型字典
vehicle_type["trajectory_planning"] = TrajectoryPlanningVehicle
vehicle_class_to_type[TrajectoryPlanningVehicle] = "trajectory_planning"


# 全局变量存储当前环境实例
_current_trajectory_env = None

def get_current_trajectory_env():
    """获取当前轨迹规划环境实例"""
    return _current_trajectory_env

class TrajectoryPlanningEnv(MetaDriveEnv):
    """
    基于轨迹规划的MetaDrive强化学习环境
    智能体输出4维动作向量，环境将其转换为轨迹规划参数
    """
    
    @classmethod
    def default_config(cls) -> Config:
        config = super().default_config()
        config.update({
            # 动作空间配置
            "discrete_action": False,
            "use_render": True,
            
            # 轨迹规划参数
            "alpha_1": 0.5,  # 起点横向偏移校正系数
            "alpha_2": 3.5,  # 终点横向偏移系数
            "v_min": 5.0,    # 最小目标速度 (m/s)
            "v_max": 25.0,   # 最大目标速度 (m/s)
            "T_min": 2.0,    # 最小规划时域 (s)
            "T_max": 8.0,    # 最大规划时域 (s)
            
            # 轨迹生成参数
            "trajectory_dt": 0.1,  # 轨迹时间步长
            "trajectory_points": 40,  # 轨迹点数量
            "use_extended_reference_line": False,  # 是否使用扩展参考线（连接所有车道段）
        })
        return config
        
    def __init__(self, config=None):
        super().__init__(config)
        
        # 存储当前轨迹点（世界坐标系）
        self.current_trajectory = []
        
        # 初始化ROS发布器
        self.ros_publisher = None
        if ROS_AVAILABLE:
            try:
                self.ros_publisher = TrajectoryPlanningROSPublisher(
                    node_name="metadrive_trajectory_planning",
                    publish_rate=20.0  # 20Hz发布频率
                )
                self.ros_publisher.start_publishing()
                print("ROS Publisher initialized successfully")
            except Exception as e:
                print(f"Failed to initialize ROS Publisher: {e}")
                self.ros_publisher = None
        
        # 初始化轨迹规划器（在ros_publisher之后）
        self.trajectory_planner = TrajectoryPlanner(self.config, self.ros_publisher)
        
        # 初始化步数计数器
        self.step_count = 0
        
        # 设置全局环境引用
        global _current_trajectory_env
        _current_trajectory_env = self
        
        # # 设置engine的current_env属性（在engine初始化后）
        # if hasattr(self, 'engine') and self.engine is not None:
        #     self.engine.current_env = self

    def setup_engine(self):
        """重写setup_engine方法，在engine创建后设置current_env属性"""
        super().setup_engine()
        # 设置engine的current_env属性，供车辆创建时使用
        if hasattr(self, 'engine') and self.engine is not None:
            self.engine.current_env = self

        
    def _get_agent_manager(self):
        """
        重写 agent manager 创建方法，确保使用 TrajectoryPlanningVehicle
        """
        from metadrive.manager.agent_manager import VehicleAgentManager
        
        class TrajectoryVehicleAgentManager(VehicleAgentManager):
            def _create_agents(self, config_dict: dict):
                ret = {}
                for agent_id, v_config in config_dict.items():
                    # 直接使用 TrajectoryPlanningVehicle 类
                    obj_name = agent_id if self.engine.global_config["force_reuse_object_name"] else None
                    # obj = self.spawn_object(TrajectoryPlanningVehicle, vehicle_config=v_config, name=obj_name)
                    # 获取环境实例的ros_publisher
                    env_instance = self.engine.current_env if hasattr(self.engine, 'current_env') else None
                    ros_publisher = env_instance.ros_publisher if env_instance and hasattr(env_instance, 'ros_publisher') else None
                    obj = self.spawn_object(TrajectoryPlanningVehicle, vehicle_config=v_config, name=obj_name, ros_publisher=ros_publisher)

                    ret[agent_id] = obj
                    policy_cls = self.agent_policy
                    args = [obj, self.generate_seed()]
                    if hasattr(policy_cls, '__name__') and 'TrajectoryIDM' in policy_cls.__name__:
                        args.append(self.engine.map_manager.current_sdc_route)
                    self.add_policy(obj.id, policy_cls, *args)
                return ret
        
        return TrajectoryVehicleAgentManager(init_observations=self._get_observations())
    
    def get_current_trajectory(self):
        """
        获取当前存储的轨迹点（世界坐标系）
        Returns:
            list: 轨迹点列表 [(x, y, heading, velocity), ...]
        """
        return self.current_trajectory
        
    @property
    def action_space(self):
        """
        定义4维连续动作空间 [a1, a2, a3, a4] ∈ [-1, 1]^4
        """
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    
    @property
    def observation_space(self):
        """
        观测空间：包含车辆状态、轨迹信息和环境信息
        
        观测向量包含：
        - 车辆状态 (10维): [x, y, heading, speed, steering, throttle, brake, angular_vel, lateral_vel, longitudinal_vel]
        - 目标轨迹点 (4维): [target_x, target_y, target_heading, target_speed]
        - Frenet坐标 (6维): [s, d, ds_dt, dd_dt, d2s_dt2, d2d_dt2]
        - 环境信息 (5维): [lane_width, curvature, distance_to_left, distance_to_right, relative_heading]
        
        总计：25维观测空间
        """
        return gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(25,), 
            dtype=np.float32
        )
    
    def _get_observation(self):
        """
        获取当前观测向量
        
        Returns:
            np.ndarray: 25维观测向量
        """
        if self.agent is None:
            return np.zeros(25, dtype=np.float32)
        
        obs = []
        
        # 1. 车辆状态 (10维)
        vehicle_state = [
            self.agent.position[0],  # x
            self.agent.position[1],  # y
            self.agent.heading_theta,  # heading
            self.agent.speed,  # speed
            self.agent.steering,  # steering
            self.agent.throttle_brake,  # throttle
            0.0,  # brake (简化)
            self.agent.angular_velocity[2] if hasattr(self.agent, 'angular_velocity') else 0.0,  # angular_vel
            0.0,  # lateral_vel (简化)
            self.agent.velocity[0] if hasattr(self.agent, 'velocity') else self.agent.speed  # longitudinal_vel
        ]
        obs.extend(vehicle_state)
        
        # 2. 目标轨迹点 (4维)
        target_point = self.agent._get_target_trajectory_point() if hasattr(self.agent, '_get_target_trajectory_point') else None
        if target_point is not None:
            target_state = list(target_point)  # [target_x, target_y, target_heading, target_speed]
        else:
            target_state = [0.0, 0.0, 0.0, 0.0]
        obs.extend(target_state)
        
        # 3. Frenet坐标 (6维)
        frenet_state = self.trajectory_planner._get_current_frenet_state(self.agent)
        frenet_obs = [
            frenet_state['s0'],
            frenet_state['d0'],
            frenet_state['ds_dt0'],
            frenet_state['dd_dt0'],
            frenet_state['d2s_dt2_0'],
            frenet_state['d2d_dt2_0']
        ]
        obs.extend(frenet_obs)
        
        # 4. 环境信息 (5维)
        env_info = self._get_environment_info()
        obs.extend(env_info)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_environment_info(self):
        """
        获取环境信息
        
        Returns:
            list: 5维环境信息 [lane_width, curvature, distance_to_left, distance_to_right, relative_heading]
        """
        if self.agent is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # 默认值
        lane_width = 3.5
        curvature = 0.0
        distance_to_left = 1.75
        distance_to_right = 1.75
        relative_heading = 0.0
        
        # 尝试获取车道信息
        try:
            if hasattr(self.agent, 'lane') and self.agent.lane is not None:
                lane = self.agent.lane
                lane_width = lane.width
                
                # 获取车辆在车道中的位置
                s, d = lane.local_coordinates(self.agent.position)
                distance_to_left = lane_width / 2 + d
                distance_to_right = lane_width / 2 - d
                
                # 计算相对航向角
                lane_heading = lane.heading_theta_at(s)
                relative_heading = self.agent.heading_theta - lane_heading
                
                # 简化的曲率计算
                if hasattr(lane, 'curvature_at'):
                    curvature = lane.curvature_at(s)
                    
        except Exception as e:
            # 如果获取车道信息失败，使用默认值
            pass
        
        return [lane_width, curvature, distance_to_left, distance_to_right, relative_heading]
        
    def reset(self, seed=None, **kwargs):
        """
        重置环境到初始状态
        
        Args:
            seed: 随机种子
            **kwargs: 其他参数
            
        Returns:
            observation: 初始观测
            info: 环境信息
        """
        # 调用父类reset方法
        obs, info = super().reset(seed=seed, **kwargs)
        
        # 重置轨迹规划器
        if hasattr(self, 'trajectory_planner') and hasattr(self, 'agent') and self.agent is not None:
            self.trajectory_planner.reset(self.agent)
            
        # 重置ROS发布器（如果有reset方法）
        if hasattr(self, 'ros_publisher') and self.ros_publisher is not None:
            if hasattr(self.ros_publisher, 'reset'):
                self.ros_publisher.reset()
            
        # 重置步数计数器
        self.step_count = 0
        
        return obs, info
    
    def _get_reset_return(self, reset_info):
        """
        重置环境并返回初始观测
        """
        ret = super()._get_reset_return(reset_info)
        
        # 重新设置全局环境引用
        global _current_trajectory_env
        _current_trajectory_env = self
        
        # 设置engine的current_env属性
        if hasattr(self, 'engine') and self.engine is not None:
            self.engine.current_env = self
        
        # 初始化轨迹规划器
        if hasattr(self, 'agent') and self.agent is not None:
            self.trajectory_planner.reset(self.agent)
            
        # 重置后发布初始ROS数据
        if self.ros_publisher is not None and hasattr(self, 'agent') and self.agent is not None:
            try:
                self.ros_publisher.update_all_data(self, self.agent, [])
            except Exception as e:
                print(f"重置时ROS发布错误: {e}")
            
        return ret
        
    def step(self, actions):
        """
        执行一步环境交互
        Args:
            actions: 4维动作向量 [a1, a2, a3, a4]
        """
        # 更新步数计数器
        self.step_count += 1
        # 将4维动作转换为轨迹规划参数
        planning_params = self._action_to_planning_params(actions)
        
        # 生成轨迹
        trajectory = self.trajectory_planner.generate_trajectory(
            self.agent, planning_params
        )
        
        # 存储当前轨迹
        self.current_trajectory = trajectory
        
        # 调试 - 每n步打印一次
        if self.step_count % 1 == 0:
            # print(f"Step: {self.step_count} 生成轨迹: {len(trajectory)} 个点, 参数: {planning_params}")
            if len(trajectory) > 0:
                pass
                # print(f"轨迹起点: ({trajectory[0][0]:.2f}, {trajectory[0][1]:.2f})")
                # print(f"轨迹终点: ({trajectory[-1][0]:.2f}, {trajectory[-1][1]:.2f})")
                # 轨迹设置信息打印
                # print(f"设置轨迹: {len(trajectory)} 个点, 时间范围: {planning_params['T']:.2f}s")
            # 轨迹跟踪信息打印（每步都打印）
            if hasattr(self.agent, 'current_trajectory') and self.agent.current_trajectory is not None:
                target_point = self.agent._get_target_trajectory_point()
                if target_point is not None and hasattr(self.agent, 'trajectory_time'):
                    steering = getattr(self.agent, 'steering', 0.0)
                    throttle_brake = getattr(self.agent, 'throttle_brake', 0.0)
                    # print(f"轨迹跟踪: 时间={self.agent.trajectory_time:.2f}, 目标点=({target_point[0]:.1f},{target_point[1]:.1f}), 转向={steering:.3f}, 油门={throttle_brake:.3f})")
                    
        # 设置车辆轨迹
        if hasattr(self.agent, 'set_trajectory') and len(trajectory) > 0:
            self.agent.set_trajectory(trajectory, planning_params['T'])
        else:
            print("警告: 无法设置轨迹到车辆")
        
        # 发布ROS数据
        if self.ros_publisher is not None:
            try:
                self.ros_publisher.update_all_data(self, self.agent, trajectory)
            except Exception as e:
                print(f"ROS发布错误: {e}")
        
        # 执行环境步进（使用轨迹跟踪控制）
        obs, reward, terminated, truncated, info = super().step([0, 0])  # 传入dummy action，实际控制由轨迹跟踪完成
        
        # 计算自定义奖励
        custom_reward = self._get_reward(trajectory, planning_params)
        
        # 检查终止条件
        custom_terminated, termination_reason = self._check_termination()
        
        # 更新info信息
        info.update({
            'trajectory_length': len(trajectory),
            'planning_params': planning_params,
            'custom_reward': custom_reward,
            'termination_reason': termination_reason if custom_terminated else None
        })
        
        return obs, custom_reward, terminated or custom_terminated, truncated, info
        
    def _action_to_planning_params(self, actions):
        """
        将4维动作向量转换为规划参数
        Args:
            actions: [a1, a2, a3, a4] ∈ [-1, 1]^4
        Returns:
            dict: 规划参数字典
        """
        a1, a2, a3, a4 = actions
        
        # 起点横向偏移校正
        delta_d0 = self.config["alpha_1"] * a1
        
        # 终点横向偏移
        delta_dT = self.config["alpha_2"] * a2
        
        # 终端目标速度
        vT = self.config["v_min"] + (a3 + 1) / 2 * (self.config["v_max"] - self.config["v_min"])
        
        # 规划时域
        T = self.config["T_min"] + (a4 + 1) / 2 * (self.config["T_max"] - self.config["T_min"])
        
        return {
            'delta_d0': delta_d0,
            'delta_dT': delta_dT,
            'vT': vT,
            'T': T
        }
    
    def _get_reward(self, trajectory, planning_params):
        """
        计算自定义奖励函数
        Args:
            trajectory: 生成的轨迹点列表
            planning_params: 规划参数
        Returns:
            float: 奖励值
        """
        if not trajectory or self.agent is None:
            return -10.0  # 无轨迹或无车辆的惩罚
        
        reward = 0.0
        
        # 1. 轨迹跟踪精度奖励
        tracking_reward = self._calculate_tracking_reward()
        reward += tracking_reward * 0.4
        
        # 2. 速度控制奖励
        speed_reward = self._calculate_speed_reward(planning_params)
        reward += speed_reward * 0.3
        
        # 3. 安全性奖励
        safety_reward = self._calculate_safety_reward()
        reward += safety_reward * 0.2
        
        # 4. 轨迹平滑性奖励
        smoothness_reward = self._calculate_smoothness_reward(trajectory)
        reward += smoothness_reward * 0.1
        
        return reward
    
    def _calculate_tracking_reward(self):
        """
        计算轨迹跟踪精度奖励
        """
        if not hasattr(self.agent, 'current_trajectory') or self.agent.current_trajectory is None:
            return 0.0
        
        # 获取当前目标点
        target_point = self.agent._get_target_trajectory_point()
        if target_point is None:
            return 0.0
        
        target_x, target_y, target_heading, target_velocity = target_point
        current_x, current_y = self.agent.position[0], self.agent.position[1]
        current_heading = self.agent.heading_theta
        current_velocity = self.agent.speed
        
        # 位置误差
        position_error = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        position_reward = np.exp(-position_error / 2.0)  # 位置越接近奖励越高
        
        # 航向误差
        heading_error = abs(target_heading - current_heading)
        heading_error = min(heading_error, 2*np.pi - heading_error)  # 处理角度环绕
        heading_reward = np.exp(-heading_error / 0.5)  # 航向越接近奖励越高
        
        # 速度误差
        velocity_error = abs(target_velocity - current_velocity)
        velocity_reward = np.exp(-velocity_error / 5.0)  # 速度越接近奖励越高
        
        return (position_reward + heading_reward + velocity_reward) / 3.0
    
    def _calculate_speed_reward(self, planning_params):
        """
        计算速度控制奖励
        """
        current_speed = self.agent.speed
        target_speed = planning_params['vT']
        
        # 鼓励合理的速度范围
        if 10.0 <= current_speed <= 20.0:  # 理想速度范围
            speed_range_reward = 1.0
        elif 5.0 <= current_speed <= 25.0:  # 可接受速度范围
            speed_range_reward = 0.5
        else:
            speed_range_reward = -0.5  # 速度过低或过高的惩罚
        
        # 速度变化平滑性
        if hasattr(self, '_last_speed'):
            speed_change = abs(current_speed - self._last_speed)
            smoothness_reward = np.exp(-speed_change / 2.0)
        else:
            smoothness_reward = 1.0
        
        self._last_speed = current_speed
        
        return (speed_range_reward + smoothness_reward) / 2.0
    
    def _calculate_safety_reward(self):
        """
        计算安全性奖励
        """
        reward = 0.0
        
        # 检查是否在车道内
        if hasattr(self.agent, 'lane') and self.agent.lane is not None:
            try:
                s, d = self.agent.lane.local_coordinates(self.agent.position)
                lane_width = self.agent.lane.width
                
                # 车道中心奖励
                if abs(d) < lane_width / 4:  # 在车道中心1/4范围内
                    reward += 1.0
                elif abs(d) < lane_width / 2:  # 在车道内但偏离中心
                    reward += 0.5
                else:  # 偏离车道
                    reward -= 1.0
            except:
                reward -= 0.5  # 无法获取车道信息的轻微惩罚
        
        # 检查与其他车辆的距离
        min_distance = float('inf')
        for vehicle_id, vehicle in self.engine.agent_manager.active_agents.items():
            if vehicle_id != self.agent.id:
                distance = np.linalg.norm(np.array(self.agent.position) - np.array(vehicle.position))
                min_distance = min(min_distance, distance)
        
        if min_distance != float('inf'):
            if min_distance > 20.0:  # 安全距离
                reward += 0.5
            elif min_distance > 10.0:  # 较近但安全
                reward += 0.2
            elif min_distance > 5.0:  # 过近
                reward -= 0.5
            else:  # 非常危险
                reward -= 2.0
        
        return reward
    
    def _calculate_smoothness_reward(self, trajectory):
        """
        计算轨迹平滑性奖励
        """
        if len(trajectory) < 3:
            return 0.0
        
        # 计算轨迹曲率变化
        curvature_changes = []
        for i in range(1, len(trajectory) - 1):
            x1, y1, h1, v1 = trajectory[i-1]
            x2, y2, h2, v2 = trajectory[i]
            x3, y3, h3, v3 = trajectory[i+1]
            
            # 简化的曲率计算
            angle_change1 = abs(h2 - h1)
            angle_change2 = abs(h3 - h2)
            curvature_change = abs(angle_change2 - angle_change1)
            curvature_changes.append(curvature_change)
        
        if curvature_changes:
            avg_curvature_change = np.mean(curvature_changes)
            smoothness_reward = np.exp(-avg_curvature_change / 0.1)
        else:
            smoothness_reward = 1.0
        
        return smoothness_reward
    
    def _check_termination(self):
        """
        检查环境终止条件
        Returns:
            tuple: (terminated: bool, reason: str)
        """
        if self.agent is None:
            return True, "no_agent"
        
        # 1. 检查碰撞
        if self.agent.crash_vehicle or self.agent.crash_object:
            return True, "collision"
        
        # 2. 检查是否严重偏离道路
        if hasattr(self.agent, 'lane') and self.agent.lane is not None:
            try:
                s, d = self.agent.lane.local_coordinates(self.agent.position)
                lane_width = self.agent.lane.width
                
                # 如果横向偏移超过车道宽度，认为偏离道路
                if abs(d) > lane_width:
                    return True, "off_road"
            except:
                pass
        
        # 3. 检查速度异常
        if self.agent.speed < 0.5:  # 速度过低，可能卡住
            if not hasattr(self, '_low_speed_counter'):
                self._low_speed_counter = 0
            self._low_speed_counter += 1
            if self._low_speed_counter > 100:  # 连续100步速度过低
                return True, "stuck"
        else:
            self._low_speed_counter = 0
        
        if self.agent.speed > 50.0:  # 速度过高
            return True, "overspeed"
        
        return False, None
        
    def _get_vehicle_class(self):
        """
        返回自定义车辆类
        """
        return TrajectoryPlanningVehicle
    
    def close(self):
        """
        关闭环境，清理ROS资源
        """
        if self.ros_publisher is not None:
            try:
                self.ros_publisher.stop_publishing()
                print("ROS Publisher stopped")
            except Exception as e:
                print(f"Error stopping ROS Publisher: {e}")
        
        # 调用父类的close方法
        super().close()
    
    def __del__(self):
        """
        析构函数
        """
        if hasattr(self, 'ros_publisher') and self.ros_publisher is not None:
            try:
                self.ros_publisher.stop_publishing()
            except:
                pass


class TrajectoryPlanner:
    """
    轨迹规划器类，实现横向五次多项式和纵向三次多项式轨迹生成
    """
    
    def __init__(self, config, ros_publisher=None):
        self.config = config
        self.dt = config["trajectory_dt"]
        self.num_points = config["trajectory_points"]
        self.ros_publisher = ros_publisher
        # 是否使用扩展参考线（连接所有车道段）
        self.use_extended_reference_line = config.get("use_extended_reference_line", False)
        
        # 轨迹规划配置
        self.trajectory_planning_config = {
            "dt": 0.1,  # 时间步长
            "num_points": 50,  # 轨迹点数量
            "use_extended_reference_line": False,  # 是否使用扩展参考线（连接所有车道段）
        }
        
    def reset(self, vehicle):
        """
        重置规划器状态
        """
        self.vehicle = vehicle
        
    def generate_trajectory(self, vehicle, planning_params):
        """
        生成轨迹
        Args:
            vehicle: 车辆对象
            planning_params: 规划参数字典
        Returns:
            list: 轨迹点列表 [(x, y, heading, velocity), ...]
        """
        # 获取当前车辆状态
        current_state = self._get_current_frenet_state(vehicle)
        
        # 生成Frenet坐标系下的轨迹
        lateral_traj = self._generate_lateral_trajectory(current_state, planning_params)
        longitudinal_traj = self._generate_longitudinal_trajectory(current_state, planning_params)

        # print("lateral_traj: ", lateral_traj)
        # print("longitudinal_traj: ", longitudinal_traj)
        
        # 获取参考线长度进行边界检查
        reference_line_length = self._get_reference_line_length(vehicle)
        
        # 检查并截断超出参考线范围的轨迹
        lateral_traj, longitudinal_traj = self._truncate_trajectory_by_s_limit(
            lateral_traj, longitudinal_traj, reference_line_length
        )
        
        # 转换为世界坐标系
        world_trajectory = self._frenet_to_world(lateral_traj, longitudinal_traj, vehicle)
        # print("world_trajectory: ", world_trajectory)
        
        # 存储轨迹用于可视化和分析
        self.current_trajectory = world_trajectory
        
        return world_trajectory
        
    def _get_reference_line_length(self, vehicle):
        """
        获取参考线的最大s长度
        Args:
            vehicle: 车辆对象
        Returns:
            float: 参考线长度
        """
        try:
            current_lane = vehicle.lane
            if current_lane is not None:
                if self.use_extended_reference_line:
                    # 使用扩展参考线时，计算所有路段长度之和
                    from metadrive.component.lane.abs_lane import AbstractLane
                    
                    # 获取当前车道索引
                    current_lane_index = None
                    if hasattr(vehicle, 'navigation') and hasattr(vehicle.navigation, 'current_ref_lanes'):
                        current_ref_lanes = vehicle.navigation.current_ref_lanes
                        if current_ref_lanes:
                            current_lane_index = current_ref_lanes[0].index
                    
                    if current_lane_index is None:
                        # 回退到估算方法
                        return current_lane.length * 3.0
                    
                    # 获取车道序列并计算总长度
                    try:
                        road_network = vehicle.engine.current_map.road_network
                        lane_sequence = self._get_lane_sequence_for_length(
                            current_lane, road_network, current_lane_index, 500.0
                        )
                        
                        total_length = sum(lane_length for _, lane_length in lane_sequence)
                        return total_length
                    except Exception as e:
                        print(f"计算扩展参考线长度时出错: {e}")
                        return current_lane.length * 3.0
                else:
                    # 使用当前车道段长度
                    return current_lane.length
            else:
                # 默认直线参考线长度（增加到500m以支持更长的规划距离）
                return 500.0
        except:
            # 异常情况下返回默认长度
            return 500.0
            
    def _truncate_trajectory_by_s_limit(self, lateral_traj, longitudinal_traj, max_s_length):
        """
        根据参考线s坐标限制截断轨迹
        Args:
            lateral_traj: 横向轨迹
            longitudinal_traj: 纵向轨迹
            max_s_length: 参考线最大s长度
        Returns:
            tuple: 截断后的(lateral_traj, longitudinal_traj)
        """
        truncated_lateral = []
        truncated_longitudinal = []
        
        for i, (lat_point, lon_point) in enumerate(zip(lateral_traj, longitudinal_traj)):
            t, s, ds_dt, d2s_dt2 = lon_point
            
            # 检查s坐标是否超出参考线范围
            if s > max_s_length:
                # 发出警告并截断轨迹
                print(f"Warning: Trajectory s-coordinate ({s:.2f}) exceeds reference line length ({max_s_length:.2f}). Truncating trajectory at point {i}.")
                break
                
            truncated_lateral.append(lat_point)
            truncated_longitudinal.append(lon_point)
            
        # 如果轨迹被截断，确保至少有一个点
        if not truncated_lateral:
            truncated_lateral = [lateral_traj[0]] if lateral_traj else []
            truncated_longitudinal = [longitudinal_traj[0]] if longitudinal_traj else []
            print("Warning: All trajectory points exceeded reference line. Using only the first point.")
            
        return truncated_lateral, truncated_longitudinal
    
    def _get_lane_sequence_for_length(self, start_lane, road_network, start_lane_index, target_length):
        """
        获取车道序列，搜索所有连续的同ID车道段（不限制长度）
        
        Returns:
            List of (lane, lane_length) tuples
        """
        lane_sequence = []
        accumulated_length = 0.0
        visited_indices = set()  # 防止循环
        
        current_lane = start_lane
        current_index = start_lane_index
        
        # 添加当前车道
        lane_sequence.append((current_lane, current_lane.length))
        accumulated_length += current_lane.length
        visited_indices.add(current_index)
        
        # 继续寻找后续车道，移除长度和迭代限制
        while True:
            next_lane, next_index = self._get_next_lane_for_length(
                road_network, current_index
            )
            
            if next_lane is None or next_index in visited_indices:
                break
            
            lane_sequence.append((next_lane, next_lane.length))
            accumulated_length += next_lane.length
            visited_indices.add(next_index)
            
            current_lane = next_lane
            current_index = next_index
            
            print(f"添加车道 {next_index}, 长度: {next_lane.length:.2f}m, 总计: {accumulated_length:.2f}m")
        
        return lane_sequence
    
    def _get_next_lane_for_length(self, road_network, current_index):
        """
        获取下一个车道（用于长度计算），优先选择相同lane_id的车道
        
        Returns:
            Tuple of (next_lane, next_lane_index) or (None, None) if no next lane
        """
        start_node, end_node, lane_id = current_index
        
        # 查找从end_node出发的连接
        if not hasattr(road_network, 'graph') or end_node not in road_network.graph:
            return None, None
        
        # 获取所有可能的下一个节点
        next_connections = road_network.graph[end_node]
        
        if not next_connections:
            return None, None
        
        # 优先选择相同lane_id的车道
        for next_end_node, lanes in next_connections.items():
            # 处理不同的数据结构
            if isinstance(lanes, dict):
                # 如果是字典，尝试获取车道列表
                if 'lanes' in lanes:
                    lanes = lanes['lanes']
                else:
                    lanes = list(lanes.values())
            elif not isinstance(lanes, (list, tuple)):
                # 单个车道对象
                lanes = [lanes]
            
            # 优先选择相同lane_id的车道
            if len(lanes) > lane_id:
                next_lane = lanes[lane_id]
                next_index = (end_node, next_end_node, lane_id)
                print(f"找到相同lane_id={lane_id}的下一车道: {next_index}")
                return next_lane, next_index
            elif len(lanes) > 0:
                # 如果相同lane_id不存在，使用第一个可用车道
                next_lane = lanes[0]
                next_index = (end_node, next_end_node, 0)
                print(f"未找到相同lane_id={lane_id}，使用lane_id=0: {next_index}")
                return next_lane, next_index
        
        return None, None
        
    def _get_current_frenet_state(self, vehicle):
        """
        获取当前车辆的Frenet坐标系状态
        """
        # 使用车辆实际所在的车道，而不是导航参考车道
        if hasattr(vehicle, 'lane') and vehicle.lane is not None:
            actual_lane = vehicle.lane
            # print("actual lane: ", actual_lane.id)
            
            # WARN:对比实际车道和导航参考车道
            # LANE 定位为： ['左', '右', id]
            if hasattr(vehicle, 'navigation') and vehicle.navigation is not None and vehicle.navigation.current_ref_lanes:
                ref_lane = vehicle.navigation.current_ref_lanes[0]
                # print("reference lane: ", ref_lane.id)
                if actual_lane.id != ref_lane.id:
                    # print("WARNING: Vehicle actual lane differs from navigation reference lane!")
                    # 计算参考车道的坐标用于对比
                    ref_s, ref_d = ref_lane.local_coordinates(vehicle.position)
                    # print(f"Reference lane coordinates: s={ref_s:.3f}, d={ref_d:.3f}")
            
            # 使用实际车道计算Frenet坐标
            s, d = actual_lane.local_coordinates(vehicle.position)
            print(f"Actual lane coordinates: s={s:.3f}, d={d:.3f}")
            
            # 计算Frenet坐标系下的速度和加速度（简化）
            ds_dt = vehicle.speed  # 纵向速度
            dd_dt = 0.0  # 横向速度（简化为0）
            d2s_dt2 = 0.0  # 纵向加速度（简化为0）
            d2d_dt2 = 0.0  # 横向加速度（简化为0）
            
            return {
                's0': s, 'd0': -d,
                'ds_dt0': ds_dt, 'dd_dt0': dd_dt,
                'd2s_dt2_0': d2s_dt2, 'd2d_dt2_0': d2d_dt2
            }
        
        # 回退到导航参考车道（兼容性处理）
        elif hasattr(vehicle, 'navigation') and vehicle.navigation is not None:
            current_lane = vehicle.navigation.current_ref_lanes[0] if vehicle.navigation.current_ref_lanes else None
            if current_lane is not None:
                print("fallback to reference lane: ", current_lane.id)
                s, d = current_lane.local_coordinates(vehicle.position)
                print(f"Fallback coordinates: s={s:.3f}, d={d:.3f}")
                
                # 计算Frenet坐标系下的速度和加速度（简化）
                ds_dt = vehicle.speed  # 纵向速度
                dd_dt = 0.0  # 横向速度（简化为0）
                d2s_dt2 = 0.0  # 纵向加速度（简化为0）
                d2d_dt2 = 0.0  # 横向加速度（简化为0）
                
                return {
                    's0': s, 'd0': -d,
                    'ds_dt0': ds_dt, 'dd_dt0': dd_dt,
                    'd2s_dt2_0': d2s_dt2, 'd2d_dt2_0': d2d_dt2
                }
        
        # 默认状态
        return {
            's0': 0.0, 'd0': 0.0,
            'ds_dt0': vehicle.speed, 'dd_dt0': 0.0,
            'd2s_dt2_0': 0.0, 'd2d_dt2_0': 0.0
        }
        
    def _generate_lateral_trajectory(self, current_state, planning_params):
        """
        生成横向五次多项式轨迹
        """
        # 边界条件
        d0 = current_state['d0'] + planning_params['delta_d0']
        # print("d0: ", d0)
        dd_dt0 = current_state['dd_dt0']
        d2d_dt2_0 = current_state['d2d_dt2_0']
        
        dT = current_state['d0'] + planning_params['delta_dT']
        dd_dtT = 0.0  # 终端横向速度为0
        d2d_dt2_T = 0.0  # 终端横向加速度为0
        
        T = planning_params['T']
        
        # 计算五次多项式系数
        coeffs = self._solve_quintic_polynomial(
            d0, dd_dt0, d2d_dt2_0, dT, dd_dtT, d2d_dt2_T, T
        )
        
        # 生成轨迹点
        trajectory = []
        for i in range(self.num_points):
            t = i * T / (self.num_points - 1)
            d = self._evaluate_polynomial(coeffs, t)
            dd_dt = self._evaluate_polynomial_derivative(coeffs, t, order=1)
            d2d_dt2 = self._evaluate_polynomial_derivative(coeffs, t, order=2)
            trajectory.append((t, d, dd_dt, d2d_dt2))
        
        # print("trajectory: ", trajectory)

        return trajectory
        
    def _generate_longitudinal_trajectory(self, current_state, planning_params):
        """
        生成纵向三次多项式轨迹
        """
        # 边界条件
        s0 = current_state['s0']
        v0 = current_state['ds_dt0']
        vT = planning_params['vT']
        T = planning_params['T']
        
        # 三次多项式系数 s(t) = b0 + b1*t + b2*t^2 + b3*t^3
        # 边界条件: s(0)=s0, s'(0)=v0, s'(T)=vT, s''(0)=0
        b0 = s0
        b1 = v0
        b2 = 0.0  # 初始加速度为0
        
        # 正确计算b3系数：根据s'(T)=vT的边界条件
        # s'(t) = b1 + 2*b2*t + 3*b3*t^2
        # s'(T) = v0 + 2*0*T + 3*b3*T^2 = v0 + 3*b3*T^2 = vT
        # 因此: b3 = (vT - v0) / (3*T^2)
        # 但这样会导致位移不连续，需要用四次多项式或调整边界条件
        
        # 使用更合理的方法：假设平均速度来计算终点位置
        avg_velocity = (v0 + vT) / 2.0
        sT = s0 + avg_velocity * T
        
        # 重新计算系数以满足位置和速度边界条件
        # s(T) = b0 + b1*T + b2*T^2 + b3*T^3 = sT
        # s'(T) = b1 + 2*b2*T + 3*b3*T^2 = vT
        # 由于b2=0，简化为：
        # sT = s0 + v0*T + b3*T^3
        # vT = v0 + 3*b3*T^2
        
        # 从速度边界条件求解b3
        b3 = (vT - v0) / (3 * T**2)
        
        # 验证位置边界条件并调整b2
        predicted_sT = s0 + v0 * T + b3 * T**3
        position_error = sT - predicted_sT
        b2 = position_error / (T**2)
        
        coeffs = [b0, b1, b2, b3]
        
        # 生成轨迹点
        trajectory = []
        for i in range(self.num_points):
            t = i * T / (self.num_points - 1)
            s = self._evaluate_polynomial(coeffs, t)
            ds_dt = self._evaluate_polynomial_derivative(coeffs, t, order=1)
            d2s_dt2 = self._evaluate_polynomial_derivative(coeffs, t, order=2)
            trajectory.append((t, s, ds_dt, d2s_dt2))
            
        return trajectory
        
    def _solve_quintic_polynomial(self, d0, dd_dt0, d2d_dt2_0, dT, dd_dtT, d2d_dt2_T, T):
        """
        求解五次多项式系数
        """
        # 根据边界条件计算系数
        a0 = d0
        a1 = dd_dt0
        a2 = d2d_dt2_0 / 2
        
        # 计算Δd
        delta_d = dT - d0 - dd_dt0 * T - 0.5 * d2d_dt2_0 * T**2
        
        # 计算剩余系数
        a3 = (10 * delta_d - 4 * d2d_dt2_0 * T**2 - 6 * dd_dt0 * T) / (T**3)
        a4 = (-15 * delta_d + 7 * d2d_dt2_0 * T**2 + 8 * dd_dt0 * T) / (T**4)
        a5 = (6 * delta_d - 3 * d2d_dt2_0 * T**2 - 3 * dd_dt0 * T) / (T**5)
        
        return [a0, a1, a2, a3, a4, a5]
        
    def _evaluate_polynomial(self, coeffs, t):
        """
        计算多项式值
        """
        result = 0.0
        for i, coeff in enumerate(coeffs):
            result += coeff * (t ** i)
        return result
        
    def _evaluate_polynomial_derivative(self, coeffs, t, order=1):
        """
        计算多项式导数
        """
        if order == 0:
            return self._evaluate_polynomial(coeffs, t)
            
        # 计算导数系数
        deriv_coeffs = []
        for i in range(order, len(coeffs)):
            coeff = coeffs[i]
            for j in range(order):
                coeff *= (i - j)
            deriv_coeffs.append(coeff)
            
        if not deriv_coeffs:
            return 0.0
            
        result = 0.0
        for i, coeff in enumerate(deriv_coeffs):
            result += coeff * (t ** i)
        return result
        
    def _frenet_to_world(self, lateral_traj, longitudinal_traj, vehicle):
        """
        将Frenet坐标系轨迹转换为世界坐标系
        """
        world_trajectory = []
        
        # 获取当前车道信息用于Frenet坐标转换
        try:
            current_lane = vehicle.lane
            if current_lane is not None:
                # 根据标志位选择参考线生成方式
                if self.use_extended_reference_line:
                    # 使用扩展参考线（连接所有车道段）
                    # 使用车辆实际所在车道的索引，而不是导航参考车道
                    current_lane_index = None
                    if hasattr(current_lane, 'index'):
                        current_lane_index = current_lane.index
                        print(f"使用实际车道索引生成扩展参考线: {current_lane_index}")
                    
                    if current_lane_index is not None:
                        try:
                            road_network = vehicle.engine.current_map.road_network
                            reference_line, actual_length = ReferenceLineGenerator.generate_extended_from_lane(
                                current_lane, road_network, current_lane_index
                            )
                            print(f"扩展参考线生成成功，实际长度: {actual_length:.2f}m")
                        except Exception as e:
                            print(f"生成扩展参考线失败，回退到单车道: {e}")
                            reference_line = ReferenceLineGenerator.generate_from_lane(current_lane)
                    else:
                        # 如果无法获取车道索引，回退到单车道
                        print("无法获取车道索引，回退到单车道参考线")
                        reference_line = ReferenceLineGenerator.generate_from_lane(current_lane)
                else:
                    # 使用当前车道段参考线
                    reference_line = ReferenceLineGenerator.generate_from_lane(current_lane)
            else:
                reference_line = ReferenceLineGenerator.generate_straight_line(
                    start_x=vehicle.position[0], start_y=vehicle.position[1], 
                    heading=vehicle.heading_theta, length=500.0, num_points=100
                )
        except:
            # 如果获取车道信息失败，生成默认直线参考线
            reference_line = ReferenceLineGenerator.generate_straight_line(
                start_x=vehicle.position[0], start_y=vehicle.position[1], 
                heading=vehicle.heading_theta, length=500.0, num_points=100
            )
            
        # 发布参考线到ROS
        if hasattr(self, 'ros_publisher') and self.ros_publisher is not None:
            try:
                # 将参考线转换为ROS发布格式 [(x, y, heading), ...]
                reference_points = [(point[0], point[1], point[2]) for point in reference_line]
                self.ros_publisher.update_reference_path(reference_points)
            except Exception as e:
                print(f"Warning: Failed to publish reference line: {e}")
            
        # 创建Frenet转换器
        converter = FrenetConverter(reference_line)
            
        for i in range(len(lateral_traj)):
            t, d, dd_dt, d2d_dt2 = lateral_traj[i]
            _, s, ds_dt, d2s_dt2 = longitudinal_traj[i]
            
            # 使用FrenetConverter进行坐标转换
            x, y, heading = converter.frenet_to_cartesian(s, d)
            velocity = ds_dt
                
            world_trajectory.append((x, y, heading, velocity))
            
        return world_trajectory