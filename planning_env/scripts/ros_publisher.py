#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Publisher for MetaDrive Trajectory Planning Environment
实时发布车辆状态、轨迹和环境信息
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Point, Quaternion, TransformStamped, PointStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA, Float64
from tf.transformations import quaternion_from_euler
import tf2_ros
import threading
import time
import math


class TrajectoryPlanningROSPublisher:
    """
    轨迹规划环境的ROS发布器
    发布车辆状态、轨迹和环境信息
    """
    
    def __init__(self, node_name="trajectory_planning_publisher", publish_rate=10.0, env_id=None):
        """
        初始化ROS发布器
        
        Args:
            node_name: ROS节点名称
            publish_rate: 发布频率(Hz)
            env_id: 环境ID，用于多环境话题匿名处理
        """
        self.node_name = node_name
        self.publish_rate = publish_rate
        self.env_id = env_id
        self.is_initialized = False
        self.is_running = False
        
        # 发布器
        self.odom_pub = None
        self.path_pub = None
        self.marker_pub = None
        self.tf_broadcaster = None
        
        # 导航点相关发布器
        self.navigation_markers_pub = None
        self.navigation_path_pub = None
        self.current_checkpoint_pub = None
        self.next_checkpoint_pub = None
        self.destination_pub = None
        

        
        # 数据缓存
        self.current_odom = None
        self.current_path = None
        self.current_markers = None
        self.current_reference_path = None
        self.current_velocity_markers = None
        self.current_vehicle_status_markers = None
        self.current_lqr_trajectory = None
        
        # 导航点数据缓存
        self.current_navigation_markers = None
        self.current_navigation_path = None
        self.current_checkpoint = None
        self.next_checkpoint = None
        self.destination_point = None

        self.target_v = None
        self.current_velocity = None
        
        # 线程锁
        self.data_lock = threading.Lock()
        self.publish_thread = None
        
        # 初始化ROS
        self._init_ros()
        
    def _init_ros(self):
        """
        初始化ROS节点和发布器
        """
        try:
            # 检查ROS是否已经初始化
            if not rospy.get_node_uri():
                rospy.init_node(self.node_name, anonymous=True)
            
            
            # 创建发布器
            self.odom_pub = rospy.Publisher(f'/vehicle/odometry', Odometry, queue_size=10)
            self.path_pub = rospy.Publisher(f'/vehicle/trajectory', Path, queue_size=10)
            self.marker_pub = rospy.Publisher(f'/environment/markers', MarkerArray, queue_size=10)
            self.reference_path_pub = rospy.Publisher(f'/trajectory/reference_path', Path, queue_size=10)
            self.target_vel = rospy.Publisher(f'/target_vel', Float64, queue_size=10)
            self.current_vel = rospy.Publisher(f'/current_vel', Float64, queue_size=10)
            self.velocity_marker_pub = rospy.Publisher(f'/velocity/markers', MarkerArray, queue_size=10)
            self.vehicle_status_marker_pub = rospy.Publisher(f'/vehicle/status_markers', MarkerArray, queue_size=10)
            self.lqr_trajectory_pub = rospy.Publisher(f'/trajectory/lqr_processed', Path, queue_size=10)
            
            # 创建导航点相关发布器
            self.navigation_markers_pub = rospy.Publisher(f'/navigation/markers', MarkerArray, queue_size=10)
            self.navigation_path_pub = rospy.Publisher(f'/navigation/path', Path, queue_size=10)
            self.current_checkpoint_pub = rospy.Publisher(f'/navigation/current_checkpoint', PointStamped, queue_size=10)
            self.next_checkpoint_pub = rospy.Publisher(f'/navigation/next_checkpoint', PointStamped, queue_size=10)
            self.destination_pub = rospy.Publisher(f'/navigation/destination', PointStamped, queue_size=10)

            
            # 创建TF广播器
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
            
            rospy.loginfo(f"ROS Publisher initialized: {self.node_name}, env_id: {self.env_id}")
            self.is_initialized = True
            
        except Exception as e:
            rospy.logwarn(f"Failed to initialize ROS: {e}")
            self.is_initialized = False

    def start_publishing(self):
        """
        开始发布数据
        """
        if not self.is_initialized:
            rospy.logwarn("ROS not initialized, cannot start publishing")
            return
            
        if self.is_running:
            return
            
        self.is_running = True
        self.publish_thread = threading.Thread(target=self._publish_loop)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        rospy.loginfo("Started ROS publishing thread")
    
    def stop_publishing(self):
        """
        停止发布数据
        """
        self.is_running = False
        if self.publish_thread and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)
        rospy.loginfo("Stopped ROS publishing")
    
    def _publish_loop(self):
        """
        发布循环
        """
        rate = rospy.Rate(self.publish_rate)
        
        while self.is_running and not rospy.is_shutdown():
            try:
                with self.data_lock:
                    # 发布里程计
                    if self.current_odom is not None:
                        self.odom_pub.publish(self.current_odom)
                    
                    # 发布轨迹
                    if self.current_path is not None:
                        self.path_pub.publish(self.current_path)
                    
                    # 发布参考线路径
                    if self.current_reference_path is not None:
                        self.reference_path_pub.publish(self.current_reference_path)
                    
                    # 发布速度的值
                    if self.target_v is not None and self.current_velocity is not None:

                        self.target_vel.publish(self.target_v)
                        self.current_vel.publish(self.current_velocity)

                    # 发布速度可视化标记
                    if self.current_velocity_markers is not None:
                        self.velocity_marker_pub.publish(self.current_velocity_markers)
                    
                    # 发布车辆状态标记（转角和油门刹车量）
                    if self.current_vehicle_status_markers is not None:
                        self.vehicle_status_marker_pub.publish(self.current_vehicle_status_markers)
                    
                    # 发布LQR处理后的轨迹
                    if self.current_lqr_trajectory is not None:
                        self.lqr_trajectory_pub.publish(self.current_lqr_trajectory)

                    # 发布标记
                    if self.current_markers is not None:
                        self.marker_pub.publish(self.current_markers)
                    
                    # 发布导航点相关数据
                    if self.current_navigation_markers is not None:
                        self.navigation_markers_pub.publish(self.current_navigation_markers)
                    
                    if self.current_navigation_path is not None:
                        self.navigation_path_pub.publish(self.current_navigation_path)
                    
                    if self.current_checkpoint is not None:
                        self.current_checkpoint_pub.publish(self.current_checkpoint)
                    
                    if self.next_checkpoint is not None:
                        self.next_checkpoint_pub.publish(self.next_checkpoint)
                    
                    if self.destination_point is not None:
                        self.destination_pub.publish(self.destination_point)
                
                rate.sleep()
                
            except Exception as e:
                rospy.logwarn(f"Error in publish loop: {e}")
                time.sleep(0.1)
    
    def update_velocity(self, target_v, current_velocity):
        if not self.is_initialized:
            return
        try:
            self.target_v = target_v
            self.current_velocity = current_velocity
        
        except Exception as e:
            rospy.logwarn(f"Error updating velocity: {e}")

    def update_vehicle_odometry(self, vehicle):
        """
        更新车辆里程计信息
        
        Args:
            vehicle: MetaDrive车辆对象
        """
        if not self.is_initialized or vehicle is None:
            return
            
        try:
            # 获取车辆状态
            position = vehicle.position  # [x, y, z]
            heading = vehicle.heading_theta  # 航向角(弧度)
            velocity = vehicle.velocity  # [vx, vy, vz]
            angular_velocity = getattr(vehicle, 'angular_velocity', [0, 0, 0])
            
            # 创建Odometry消息
            odom = Odometry()
            odom.header = Header()
            odom.header.stamp = rospy.Time.now()
            odom.header.frame_id = "map"
            odom.child_frame_id = "base_footprint"
            
            # 位置和姿态
            odom.pose.pose.position.x = position[0]
            odom.pose.pose.position.y = position[1]
            odom.pose.pose.position.z = position[2] if len(position) > 2 else 0.0
            
            # 将航向角转换为四元数
            quat = quaternion_from_euler(0, 0, heading)
            odom.pose.pose.orientation.x = quat[0]
            odom.pose.pose.orientation.y = quat[1]
            odom.pose.pose.orientation.z = quat[2]
            odom.pose.pose.orientation.w = quat[3]
            
            # 速度
            odom.twist.twist.linear.x = velocity[0] if len(velocity) > 0 else 0.0
            odom.twist.twist.linear.y = velocity[1] if len(velocity) > 1 else 0.0
            odom.twist.twist.linear.z = velocity[2] if len(velocity) > 2 else 0.0
            
            # 角速度
            if hasattr(vehicle, 'angular_velocity') and len(angular_velocity) > 2:
                odom.twist.twist.angular.z = angular_velocity[2]
            else:
                # 估算角速度
                speed = vehicle.speed
                if speed > 0.1:  # 避免除零
                    # 简单估算：角速度 = 横向速度 / 速度
                    lateral_vel = velocity[1] if len(velocity) > 1 else 0.0
                    odom.twist.twist.angular.z = lateral_vel / speed
            
            with self.data_lock:
                self.current_odom = odom
            
            # 同时发布TF变换
            self._publish_tf_transform(position, heading)
                
        except Exception as e:
            rospy.logwarn(f"Error updating vehicle odometry: {e}")
    
    def _publish_tf_transform(self, position, heading):
        """
        发布map到base_footprint的TF变换
        
        Args:
            position: 车辆位置 [x, y, z]
            heading: 车辆航向角(弧度)
        """
        if not self.is_initialized or self.tf_broadcaster is None:
            return
            
        try:
            # 创建TF变换消息
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = "map"
            transform.child_frame_id = "base_footprint"
            
            # 设置位置
            transform.transform.translation.x = position[0]
            transform.transform.translation.y = position[1]
            transform.transform.translation.z = position[2] if len(position) > 2 else 0.0
            
            # 设置姿态(四元数)
            quat = quaternion_from_euler(0, 0, heading)
            transform.transform.rotation.x = quat[0]
            transform.transform.rotation.y = quat[1]
            transform.transform.rotation.z = quat[2]
            transform.transform.rotation.w = quat[3]
            
            # 发布TF变换
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            rospy.logwarn(f"Error publishing TF transform: {e}")
    
    def update_trajectory_path(self, trajectory_points):
        """
        更新轨迹路径
        
        Args:
            trajectory_points: 轨迹点列表 [(x, y, heading, velocity), ...]
        """
        if not self.is_initialized or not trajectory_points:
            return
            
        try:
            # 创建Path消息
            path = Path()
            path.header = Header()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "map"
            
            # 添加轨迹点
            for point in trajectory_points:
                if len(point) >= 3:  # 至少包含x, y, heading
                    x, y, heading = point[0], point[1], point[2]
                    
                    pose_stamped = PoseStamped()
                    pose_stamped.header = path.header
                    
                    # 位置
                    pose_stamped.pose.position.x = x
                    pose_stamped.pose.position.y = y
                    pose_stamped.pose.position.z = 0.0
                    
                    # 姿态
                    quat = quaternion_from_euler(0, 0, heading)
                    pose_stamped.pose.orientation.x = quat[0]
                    pose_stamped.pose.orientation.y = quat[1]
                    pose_stamped.pose.orientation.z = quat[2]
                    pose_stamped.pose.orientation.w = quat[3]
                    
                    path.poses.append(pose_stamped)
            
            with self.data_lock:
                self.current_path = path
                
        except Exception as e:
            rospy.logwarn(f"Error updating trajectory path: {e}")
    
    def update_environment_markers(self, env):
        """
        更新环境标记(车道线、车辆等)
        发布详细的地图信息，包括所有车道线和周围车辆
        
        Args:
            env: MetaDrive环境对象
        """
        if not self.is_initialized or env is None:
            return
            
        try:
            marker_array = MarkerArray()
            marker_id = 0
            
            # 清除之前的标记
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)
            marker_id += 1
            
            # 获取当前地图和车辆信息
            if hasattr(env, 'engine') and env.engine is not None:
                map_manager = env.engine.map_manager
                
                # 添加详细的车道线标记
                if hasattr(map_manager, 'current_map') and map_manager.current_map is not None:
                    marker_id = self._add_lane_markers(map_manager.current_map, marker_array, marker_id)
                
                # 添加增强的车辆标记
                if hasattr(env.engine, 'agent_manager'):
                    marker_id = self._add_vehicle_markers(env.engine.agent_manager, env.agent, marker_array, marker_id)
                
                # 添加交通标志和其他道路设施标记
                marker_id = self._add_traffic_elements(env, marker_array, marker_id)
            
            with self.data_lock:
                self.current_markers = marker_array
                
        except Exception as e:
            rospy.logwarn(f"Error updating environment markers: {e}")
    
    def _add_lane_markers(self, current_map, marker_array, marker_id):
        """
        添加详细的车道线标记，包括车道边界线、中心线等
        
        Args:
            current_map: 当前地图对象
            marker_array: 标记数组
            marker_id: 当前标记ID
            
        Returns:
            int: 更新后的标记ID
        """
        try:
            # 获取所有车道 - 使用正确的MetaDrive API
            all_lanes = []
            road_network = current_map.road_network
            
            # 根据road_network类型选择正确的访问方式
            if hasattr(road_network, 'get_all_lanes'):
                # NodeRoadNetwork类型
                all_lanes = road_network.get_all_lanes()
            elif hasattr(road_network, 'graph'):
                # EdgeRoadNetwork类型
                for lane_index, lane_info in road_network.graph.items():
                    if hasattr(lane_info, 'lane'):
                        all_lanes.append(lane_info.lane)
                    else:
                        all_lanes.append(lane_info)  # 直接是lane对象
            
            # 遍历所有车道
            for lane in all_lanes:
                if lane is None:
                    continue
                    
                try:
                    # 获取车道中心线
                    if hasattr(lane, 'get_polyline'):
                        center_line = lane.get_polyline()
                        if len(center_line) > 1:
                            lane_id = getattr(lane, 'index', f'lane_{marker_id}')
                            marker = self._create_lane_center_marker(center_line, marker_id, lane_id)
                            marker_array.markers.append(marker)
                            marker_id += 1
                        
                        # 获取车道边界线
                        left_boundary = self._get_lane_boundary_points(lane, 'left')
                        if len(left_boundary) > 1:
                            marker = self._create_lane_boundary_marker(left_boundary, marker_id, 'left', lane_id)
                            marker_array.markers.append(marker)
                            marker_id += 1
                        
                        right_boundary = self._get_lane_boundary_points(lane, 'right')
                        if len(right_boundary) > 1:
                            marker = self._create_lane_boundary_marker(right_boundary, marker_id, 'right', lane_id)
                            marker_array.markers.append(marker)
                            marker_id += 1
                        
                        # 添加车道方向箭头
                        if len(center_line) > 10:  # 确保有足够的点来计算方向
                            arrow_marker = self._create_lane_direction_marker(center_line, marker_id, lane_id)
                            if arrow_marker is not None:
                                marker_array.markers.append(arrow_marker)
                                marker_id += 1
                                
                except Exception as e:
                    rospy.logdebug(f"Error processing lane {getattr(lane, 'index', 'unknown')}: {e}")
                    continue
                            
        except Exception as e:
            rospy.logwarn(f"Error adding lane markers: {e}")
            
        return marker_id
    
    def _create_lane_center_marker(self, polyline, marker_id, lane_id):
        """
        创建车道中心线标记
        
        Args:
            polyline: 车道中心线点列表
            marker_id: 标记ID
            lane_id: 车道ID
            
        Returns:
            Marker: 车道中心线标记
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "lane_center_lines"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 初始化四元数为单位四元数
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # 设置中心线属性 - 黄色虚线
        marker.scale.x = 0.15  # 线宽
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        
        # 添加点
        for point in polyline:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.05  # 稍微抬高避免与地面重叠
            marker.points.append(p)
            
        return marker
    
    def _create_lane_boundary_marker(self, polyline, marker_id, boundary_type, lane_id):
        """
        创建车道边界线标记
        
        Args:
            polyline: 边界线点列表
            marker_id: 标记ID
            boundary_type: 边界类型 ('left' 或 'right')
            lane_id: 车道ID
            
        Returns:
            Marker: 车道边界线标记
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"lane_{boundary_type}_boundaries"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # 初始化四元数为单位四元数
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # 设置边界线属性 - 白色实线
        marker.scale.x = 0.12  # 线宽
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        # 添加点
        for point in polyline:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.02  # 稍微抬高避免与地面重叠
            marker.points.append(p)
            
        return marker
    
    def _create_lane_direction_marker(self, center_line, marker_id, lane_id):
        """
        创建车道方向箭头标记
        
        Args:
            center_line: 车道中心线点列表
            marker_id: 标记ID
            lane_id: 车道ID
            
        Returns:
            Marker: 方向箭头标记
        """
        try:
             # 在车道中间位置放置箭头
            mid_idx = len(center_line) // 2
            if mid_idx < len(center_line) - 1:
                current_point = center_line[mid_idx]
                next_point = center_line[mid_idx + 1]
                
                # 计算方向
                dx = next_point[0] - current_point[0]
                dy = next_point[1] - current_point[1]
                yaw = math.atan2(dy, dx)
                
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "lane_directions"
                marker.id = marker_id
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                
                # 设置箭头位置和方向
                marker.pose.position.x = float(current_point[0])
                marker.pose.position.y = float(current_point[1])
                marker.pose.position.z = 0.3
                
                # 设置箭头方向（四元数）
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = math.sin(yaw / 2.0)
                marker.pose.orientation.w = math.cos(yaw / 2.0)
                
                # 设置箭头大小和颜色
                marker.scale.x = 2.0  # 长度
                marker.scale.y = 0.5  # 宽度
                marker.scale.z = 0.3  # 高度
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.7
                
                return marker
                
        except Exception as e:
            rospy.logdebug(f"Error creating direction marker for lane {lane_id}: {e}")
            
        return None
    
    def _get_lane_boundary_points(self, lane, boundary_type):
        """
        获取车道边界点
        
        Args:
            lane: 车道对象
            boundary_type: 边界类型 ('left' 或 'right')
            
        Returns:
            list: 边界点列表
        """
        try:
             # 获取车道中心线
            center_line = lane.get_polyline()
            boundary_points = []
            
            # 获取车道宽度（如果可用）
            lane_width = getattr(lane, 'width', 3.5)  # 默认车道宽度3.5米
            offset = lane_width / 2.0
            
            if boundary_type == 'left':
                offset = -offset  # 左边界使用负偏移
            
            # 计算边界点
            for i in range(len(center_line)):
                if i < len(center_line) - 1:
                    # 计算当前点到下一点的方向
                    dx = center_line[i + 1][0] - center_line[i][0]
                    dy = center_line[i + 1][1] - center_line[i][1]
                    length = math.sqrt(dx * dx + dy * dy)
                    
                    if length > 0:
                        # 单位方向向量
                        ux = dx / length
                        uy = dy / length
                        
                        # 垂直方向向量（左转90度）
                        nx = -uy
                        ny = ux
                        
                        # 计算边界点
                        boundary_x = center_line[i][0] + offset * nx
                        boundary_y = center_line[i][1] + offset * ny
                        boundary_points.append([boundary_x, boundary_y])
                else:
                    # 最后一个点，使用前一个点的方向
                    if len(boundary_points) > 0:
                        # 使用与前一个边界点相同的偏移
                        if i > 0:
                            dx = center_line[i][0] - center_line[i - 1][0]
                            dy = center_line[i][1] - center_line[i - 1][1]
                            length = math.sqrt(dx * dx + dy * dy)
                            
                            if length > 0:
                                ux = dx / length
                                uy = dy / length
                                nx = -uy
                                ny = ux
                                
                                boundary_x = center_line[i][0] + offset * nx
                                boundary_y = center_line[i][1] + offset * ny
                                boundary_points.append([boundary_x, boundary_y])
            
            return boundary_points
            
        except Exception as e:
            rospy.logdebug(f"Error getting {boundary_type} boundary points: {e}")
            return []
    
    def _add_vehicle_markers(self, agent_manager, ego_vehicle, marker_array, marker_id):
        """
        添加增强的车辆标记，包括详细的车辆信息和颜色区分
        
        Args:
            agent_manager: 智能体管理器
            ego_vehicle: 主车对象
            marker_array: 标记数组
            marker_id: 当前标记ID
            
        Returns:
            int: 更新后的标记ID
        """
        try:
            if hasattr(agent_manager, 'active_agents'):
                for agent_id, vehicle in agent_manager.active_agents.items():
                    if vehicle != ego_vehicle:  # 排除主车
                        # 创建车辆主体标记
                        vehicle_marker = self._create_enhanced_vehicle_marker(vehicle, marker_id, agent_id)
                        if vehicle_marker is not None:
                            marker_array.markers.append(vehicle_marker)
                            marker_id += 1
                        
                        # 创建车辆速度向量标记
                        velocity_marker = self._create_vehicle_velocity_marker(vehicle, marker_id, agent_id)
                        if velocity_marker is not None:
                            marker_array.markers.append(velocity_marker)
                            marker_id += 1
                        
                        # 创建车辆ID文本标记
                        text_marker = self._create_vehicle_text_marker(vehicle, marker_id, agent_id)
                        if text_marker is not None:
                            marker_array.markers.append(text_marker)
                            marker_id += 1
                        
                        # 创建车辆边界框标记
                        bbox_marker = self._create_vehicle_bbox_marker(vehicle, marker_id, agent_id)
                        if bbox_marker is not None:
                            marker_array.markers.append(bbox_marker)
                            marker_id += 1
                            
        except Exception as e:
            rospy.logwarn(f"Error adding vehicle markers: {e}")
            
        return marker_id
    
    def _create_enhanced_vehicle_marker(self, vehicle, marker_id, agent_id):
        """
        创建增强的车辆标记
        
        Args:
            vehicle: 车辆对象
            marker_id: 标记ID
            agent_id: 智能体ID
            
        Returns:
            Marker: 车辆标记
        """
        try:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vehicles"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # 获取车辆位置和方向
            position = vehicle.position
            heading = getattr(vehicle, 'heading', getattr(vehicle, 'heading_theta', 0.0))
            
            # 设置位置
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = 1.0  # 车辆高度
            
            # 设置方向（四元数）
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(heading / 2.0)
            marker.pose.orientation.w = math.cos(heading / 2.0)
            
            # 设置车辆尺寸
            marker.scale.x = getattr(vehicle, 'LENGTH', 4.5)
            marker.scale.y = getattr(vehicle, 'WIDTH', 1.8)
            marker.scale.z = getattr(vehicle, 'HEIGHT', 1.2)
            
            # 根据车辆类型设置不同颜色
            vehicle_type = getattr(vehicle, 'vehicle_type', 'unknown')
            if 'traffic' in str(vehicle_type).lower():
                # 交通车辆 - 蓝色
                marker.color.r = 0.0
                marker.color.g = 0.5
                marker.color.b = 1.0
                marker.color.a = 0.8
            elif 'ego' in str(agent_id).lower():
                # 主车 - 绿色
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.8
            else:
                # 其他车辆 - 红色
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
            
            return marker
            
        except Exception as e:
            rospy.logwarn(f"Error creating enhanced vehicle marker: {e}")
            return None
    
    def _create_vehicle_velocity_marker(self, vehicle, marker_id, agent_id):
        """
        创建车辆速度向量标记
        
        Args:
            vehicle: 车辆对象
            marker_id: 标记ID
            agent_id: 智能体ID
            
        Returns:
            Marker: 速度向量标记
        """
        try:
            # 获取车辆速度
            velocity = getattr(vehicle, 'velocity', [0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            
            if speed < 0.1:  # 速度太小时不显示
                return None
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vehicle_velocities"
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # 设置起点位置
            position = vehicle.position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = 2.0  # 在车辆上方
            
            # 计算速度方向
            velocity_angle = math.atan2(velocity[1], velocity[0])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = math.sin(velocity_angle / 2.0)
            marker.pose.orientation.w = math.cos(velocity_angle / 2.0)
            
            # 根据速度设置箭头大小
            scale_factor = min(speed / 10.0, 3.0)  # 最大3倍缩放
            marker.scale.x = 1.0 + scale_factor  # 长度
            marker.scale.y = 0.3  # 宽度
            marker.scale.z = 0.3  # 高度
            
            # 设置颜色（黄色）
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            return marker
            
        except Exception as e:
            rospy.logdebug(f"Error creating velocity marker for vehicle {agent_id}: {e}")
            return None
    
    def update_target_point_marker(self, x, y, yaw, velocity):
        """
        更新目标匹配点的箭头标记
        
        Args:
            x: 目标点x坐标
            y: 目标点y坐标
            yaw: 目标点航向角(弧度)
            velocity: 目标点速度
        """
        if not self.is_initialized:
            return
            
        try:
            # 创建箭头标记
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "target_point"
            marker.id = 999  # 使用固定ID，便于更新
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # 设置箭头位置和方向
            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.5  # 抬高以便可见
            
            # 设置箭头方向（四元数）
            quat = quaternion_from_euler(0, 0, yaw)
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            
            # 设置箭头大小和颜色（红色）
            marker.scale.x = 3.0  # 长度
            marker.scale.y = 0.8  # 宽度
            marker.scale.z = 0.5  # 高度
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.9
            
            # 创建包含单个标记的MarkerArray
            marker_array = MarkerArray()
            marker_array.markers.append(marker)
            
            # 发布标记
            if self.marker_pub is not None:
                self.marker_pub.publish(marker_array)
                
        except Exception as e:
            rospy.logwarn(f"Error updating target point marker: {e}")

    def _create_vehicle_text_marker(self, vehicle, marker_id, agent_id):
        """
        创建车辆ID文本标记
        
        Args:
            vehicle: 车辆对象
            marker_id: 标记ID
            agent_id: 智能体ID
            
        Returns:
            Marker: 文本标记
        """
        try:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vehicle_texts"
            marker.id = marker_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # 设置文本位置（车辆上方）
            position = vehicle.position
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = 3.0
            
            # 初始化四元数为单位四元数
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 设置文本内容
            velocity = getattr(vehicle, 'velocity', [0, 0])
            speed = math.sqrt(velocity[0]**2 + velocity[1]**2)
            marker.text = f"ID: {agent_id}\nSpeed: {speed:.1f} m/s"
            
            # 设置文本大小和颜色
            marker.scale.z = 0.8  # 文本大小
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            
            return marker
            
        except Exception as e:
            rospy.logdebug(f"Error creating text marker for vehicle {agent_id}: {e}")
            return None
    
    def _create_vehicle_bbox_marker(self, vehicle, marker_id, agent_id):
        """
        创建车辆边界框标记
        
        Args:
            vehicle: 车辆对象
            marker_id: 标记ID
            agent_id: 智能体ID
            
        Returns:
            Marker: 边界框标记
        """
        try:
            import math
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vehicle_bboxes"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # 初始化四元数为单位四元数
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # 获取车辆位置和方向
            position = vehicle.position
            heading = getattr(vehicle, 'heading', getattr(vehicle, 'heading_theta', 0.0))
            
            # 车辆尺寸
            length = getattr(vehicle, 'LENGTH', 4.5)
            width = getattr(vehicle, 'WIDTH', 1.8)
            
            # 计算边界框的四个角点
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            
            # 相对于车辆中心的四个角点
            corners = [
                [-length/2, -width/2],  # 后左
                [length/2, -width/2],   # 前左
                [length/2, width/2],    # 前右
                [-length/2, width/2],   # 后右
                [-length/2, -width/2]   # 闭合
            ]
            
            # 转换到世界坐标系
            marker.scale.x = 0.1  # 线宽
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 0.5
            
            for corner in corners:
                # 旋转和平移
                x = corner[0] * cos_h - corner[1] * sin_h + position[0]
                y = corner[0] * sin_h + corner[1] * cos_h + position[1]
                
                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.1
                marker.points.append(p)
            
            return marker
            
        except Exception as e:
            rospy.logdebug(f"Error creating bbox marker for vehicle {agent_id}: {e}")
            return None
    
    def _add_traffic_elements(self, env, marker_array, marker_id):
        """
        添加交通标志和其他道路设施标记
        
        Args:
            env: MetaDrive环境对象
            marker_array: 标记数组
            marker_id: 当前标记ID
            
        Returns:
            int: 更新后的标记ID
        """
        try:
            # 这里可以添加交通信号灯、标志牌等的可视化
            # 由于MetaDrive的具体API可能不同，这里提供一个框架
            pass
            
        except Exception as e:
            rospy.logwarn(f"Error adding traffic elements: {e}")
            
        return marker_id
    
    def update_reference_path(self, reference_points):
        """
        更新参考线路径
        
        Args:
            reference_points: 参考线点列表 [(x, y, heading), ...] 或 [(x, y), ...]
        """
        if not self.is_initialized:
            rospy.logwarn("ROS Publisher not initialized, cannot update reference path")
            return
            
        if not reference_points:
            rospy.logwarn("Empty reference points, cannot update reference path")
            return
            
        try:
            rospy.logdebug(f"Updating reference path with {len(reference_points)} points")
            
            # 创建Path消息
            path = Path()
            path.header = Header()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "map"
            
            # 添加参考线点
            for i, point in enumerate(reference_points):
                if len(point) >= 2:  # 至少包含x, y
                    x, y = point[0], point[1]
                    heading = point[2] if len(point) > 2 else 0.0
                    
                    pose_stamped = PoseStamped()
                    pose_stamped.header = path.header
                    
                    # 位置
                    pose_stamped.pose.position.x = float(x)
                    pose_stamped.pose.position.y = float(y)
                    pose_stamped.pose.position.z = 0.0
                    
                    # 姿态(四元数)
                    quat = quaternion_from_euler(0, 0, heading)
                    pose_stamped.pose.orientation.x = quat[0]
                    pose_stamped.pose.orientation.y = quat[1]
                    pose_stamped.pose.orientation.z = quat[2]
                    pose_stamped.pose.orientation.w = quat[3]
                    
                    path.poses.append(pose_stamped)
            
            with self.data_lock:
                self.current_reference_path = path
                rospy.logdebug(f"Successfully updated reference path with {len(path.poses)} poses")
                
        except Exception as e:
            rospy.logwarn(f"Error updating reference path: {e}")
            import traceback
            rospy.logwarn(f"Traceback: {traceback.format_exc()}")
    
    def update_all_data(self, env, vehicle, trajectory_points, reference_points=None):
        """
        一次性更新所有数据
        
        Args:
            env: MetaDrive环境对象
            vehicle: 车辆对象
            trajectory_points: 轨迹点列表
            reference_points: 参考线点列表(可选)
        """
        self.update_vehicle_odometry(vehicle)
        self.update_trajectory_path(trajectory_points)
        if reference_points is not None:
            self.update_reference_path(reference_points)
        self.update_environment_markers(env)
    
    def update_velocity_markers(self, current_velocity, target_velocity, vehicle_position=None, vehicle_heading=None):
        """
        更新速度可视化标记（跟随车辆移动但保持一定距离）
        
        Args:
            current_velocity: 当前速度 (m/s)
            target_velocity: 目标速度 (m/s)
            vehicle_position: 车辆位置 [x, y]
            vehicle_heading: 车辆航向角 (弧度)
        """
        if not self.is_initialized:
            return
            
        try:
            marker_array = MarkerArray()
            
            # 如果没有提供车辆位置，使用默认位置
            if vehicle_position is None:
                vehicle_x, vehicle_y = 0.0, 0.0
            else:
                vehicle_x, vehicle_y = vehicle_position[0], vehicle_position[1]
            
            # 如果没有提供航向角，使用默认值
            if vehicle_heading is None:
                vehicle_heading = 0.0
            
            # 在车辆坐标系中定义相对位置 (x为前，y为左)
            # 速度信息显示在车辆左前方
            relative_positions = [
                (-2.0, 4.0),  # 当前速度：左前方
                (-2.0, 2.0),  # 目标速度：左前方稍后
            ]
            
            # 使用旋转矩阵将车辆坐标系转换到世界坐标系
            cos_heading = math.cos(vehicle_heading)
            sin_heading = math.sin(vehicle_heading)
            
            markers_info = [
                ("Cur Speed", current_velocity, (0.0, 1.0, 0.0)),  # 绿色
                ("Tar Speed", target_velocity, (1.0, 0.0, 0.0)),   # 红色
            ]
            
            for i, ((rel_x, rel_y), (label, value, color)) in enumerate(zip(relative_positions, markers_info)):
                # 坐标变换：从车辆坐标系到世界坐标系
                world_x = vehicle_x + rel_x * cos_heading - rel_y * sin_heading
                world_y = vehicle_y + rel_x * sin_heading + rel_y * cos_heading
                
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "velocity_display"
                marker.id = i
                marker.type = Marker.TEXT_VIEW_FACING
                marker.action = Marker.ADD
                
                marker.pose.position.x = world_x
                marker.pose.position.y = world_y
                marker.pose.position.z = 3.0  # 车辆上方3米
                
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                marker.text = f"{label}: {value:.2f} m/s"
                marker.scale.z = 1.5  # 字体大小
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                
                marker_array.markers.append(marker)
            
            with self.data_lock:
                self.current_velocity_markers = marker_array
                
        except Exception as e:
            rospy.logwarn(f"Error updating velocity markers: {e}")
    
    def update_vehicle_status_display(self, steering_angle, throttle_brake_percent, vehicle_position=None, vehicle_heading=None):
        """
        更新车辆状态显示（转角和油门刹车量）
        
        Args:
            steering_angle: 当前转角 (度)
            throttle_brake_percent: 油门刹车量百分比 (-100到100，负值为刹车，正值为油门)
            vehicle_position: 车辆位置 [x, y]
            vehicle_heading: 车辆航向角 (弧度)
        """
        if not self.is_initialized:
            return
            
        try:
            marker_array = MarkerArray()
            
            # 如果没有提供车辆位置，使用默认位置
            if vehicle_position is None:
                vehicle_x, vehicle_y = 0.0, 0.0
            else:
                vehicle_x, vehicle_y = vehicle_position[0], vehicle_position[1]
            
            # 如果没有提供航向角，使用默认值
            if vehicle_heading is None:
                vehicle_heading = 0.0
            
            # 在车辆坐标系中定义相对位置 (x为前，y为左)
            # 车辆状态信息显示在车辆右前方
            relative_positions = [
                (-2.0, -4.0),  # 转角：右前方
                (-2.0, -6.0),  # 油门刹车：右前方稍后
            ]
            
            # 使用旋转矩阵将车辆坐标系转换到世界坐标系
            cos_heading = math.cos(vehicle_heading)
            sin_heading = math.sin(vehicle_heading)
            
            # 转角显示标记
            rel_x, rel_y = relative_positions[0]
            world_x = vehicle_x + rel_x * cos_heading - rel_y * sin_heading
            world_y = vehicle_y + rel_x * sin_heading + rel_y * cos_heading
            
            steering_marker = Marker()
            steering_marker.header.frame_id = "map"
            steering_marker.header.stamp = rospy.Time.now()
            steering_marker.ns = "vehicle_status_display"
            steering_marker.id = 0
            steering_marker.type = Marker.TEXT_VIEW_FACING
            steering_marker.action = Marker.ADD
            
            steering_marker.pose.position.x = world_x
            steering_marker.pose.position.y = world_y
            steering_marker.pose.position.z = 3.0  # 车辆上方3米
            
            steering_marker.pose.orientation.x = 0.0
            steering_marker.pose.orientation.y = 0.0
            steering_marker.pose.orientation.z = 0.0
            steering_marker.pose.orientation.w = 1.0
            
            steering_marker.text = f"Steering: {steering_angle:.1f}°"
            steering_marker.scale.z = 1.5
            steering_marker.color.r = 0.0
            steering_marker.color.g = 0.0
            steering_marker.color.b = 1.0
            steering_marker.color.a = 1.0
            
            marker_array.markers.append(steering_marker)
            
            # 油门刹车量显示标记
            rel_x, rel_y = relative_positions[1]
            world_x = vehicle_x + rel_x * cos_heading - rel_y * sin_heading
            world_y = vehicle_y + rel_x * sin_heading + rel_y * cos_heading
            
            throttle_brake_marker = Marker()
            throttle_brake_marker.header.frame_id = "map"
            throttle_brake_marker.header.stamp = rospy.Time.now()
            throttle_brake_marker.ns = "vehicle_status_display"
            throttle_brake_marker.id = 1
            throttle_brake_marker.type = Marker.TEXT_VIEW_FACING
            throttle_brake_marker.action = Marker.ADD
            
            throttle_brake_marker.pose.position.x = world_x
            throttle_brake_marker.pose.position.y = world_y
            throttle_brake_marker.pose.position.z = 3.0  # 车辆上方3米
            
            throttle_brake_marker.pose.orientation.x = 0.0
            throttle_brake_marker.pose.orientation.y = 0.0
            throttle_brake_marker.pose.orientation.z = 0.0
            throttle_brake_marker.pose.orientation.w = 1.0
            
            # 根据正负值显示油门或刹车
            if throttle_brake_percent >= 0:
                throttle_brake_marker.text = f"Throttle: {throttle_brake_percent:.1f}%"
                throttle_brake_marker.color.r = 0.0
                throttle_brake_marker.color.g = 1.0
                throttle_brake_marker.color.b = 1.0
            else:
                throttle_brake_marker.text = f"Brake: {abs(throttle_brake_percent):.1f}%"
                throttle_brake_marker.color.r = 1.0
                throttle_brake_marker.color.g = 0.5
                throttle_brake_marker.color.b = 0.0
            
            throttle_brake_marker.scale.z = 1.5
            throttle_brake_marker.color.a = 1.0
            
            marker_array.markers.append(throttle_brake_marker)
            
            with self.data_lock:
                self.current_vehicle_status_markers = marker_array
                
        except Exception as e:
            rospy.logwarn(f"Error updating vehicle status display: {e}")
    
    def update_lqr_trajectory(self, lqr_trajectory_points):
        """
        更新LQR处理后的轨迹
        
        Args:
            lqr_trajectory_points: LQR轨迹点列表 [(x, y, yaw), ...]
        """
        if not self.is_initialized or not lqr_trajectory_points:
            return
            
        try:
            path = Path()
            path.header = Header()
            path.header.stamp = rospy.Time.now()
            path.header.frame_id = "map"
            
            for point in lqr_trajectory_points:
                if len(point) >= 3:  # x, y, yaw
                    x, y, yaw = point[0], point[1], point[2]
                    
                    pose_stamped = PoseStamped()
                    pose_stamped.header = path.header
                    
                    pose_stamped.pose.position.x = float(x)
                    pose_stamped.pose.position.y = float(y)
                    pose_stamped.pose.position.z = 0.0
                    
                    # 将yaw角转换为四元数
                    quat = quaternion_from_euler(0, 0, yaw)
                    pose_stamped.pose.orientation.x = quat[0]
                    pose_stamped.pose.orientation.y = quat[1]
                    pose_stamped.pose.orientation.z = quat[2]
                    pose_stamped.pose.orientation.w = quat[3]
                    
                    path.poses.append(pose_stamped)
            
            with self.data_lock:
                self.current_lqr_trajectory = path
                
        except Exception as e:
            rospy.logwarn(f"Error updating LQR trajectory: {e}")

    def update_navigation_data(self, vehicle):
        """
        更新导航点数据
        
        Args:
            vehicle: MetaDrive车辆对象
        """
        if not self.is_initialized or vehicle is None or not hasattr(vehicle, 'navigation'):
            return
            
        try:
            navigation = vehicle.navigation
            current_time = rospy.Time.now()
            
            # 获取当前和下一个检查点
            checkpoints = navigation.get_checkpoints()
            if checkpoints and len(checkpoints) >= 2:
                current_checkpoint_pos = checkpoints[0]
                next_checkpoint_pos = checkpoints[1]
                
                # 创建当前检查点消息
                current_checkpoint = PointStamped()
                current_checkpoint.header.stamp = current_time
                current_checkpoint.header.frame_id = "map"
                current_checkpoint.point.x = current_checkpoint_pos[0]
                current_checkpoint.point.y = current_checkpoint_pos[1]
                current_checkpoint.point.z = 0.0
                
                # 创建下一个检查点消息
                next_checkpoint = PointStamped()
                next_checkpoint.header.stamp = current_time
                next_checkpoint.header.frame_id = "map"
                next_checkpoint.point.x = next_checkpoint_pos[0]
                next_checkpoint.point.y = next_checkpoint_pos[1]
                next_checkpoint.point.z = 0.0
                
                with self.data_lock:
                    self.current_checkpoint = current_checkpoint
                    self.next_checkpoint = next_checkpoint
            
            # 获取最终目标点
            if hasattr(navigation, '_dest_node_path') and navigation._dest_node_path:
                dest_pos = navigation._dest_node_path.getPos()
                destination = PointStamped()
                destination.header.stamp = current_time
                destination.header.frame_id = "map"
                destination.point.x = dest_pos[0]
                destination.point.y = dest_pos[1]
                destination.point.z = 0.0
                
                with self.data_lock:
                    self.destination_point = destination
            
            # 创建导航路径
            if hasattr(navigation, 'checkpoints') and navigation.checkpoints:
                nav_path = Path()
                nav_path.header.stamp = current_time
                nav_path.header.frame_id = "map"
                
                for checkpoint in navigation.checkpoints:
                    if len(checkpoint) >= 2:
                        pose_stamped = PoseStamped()
                        pose_stamped.header = nav_path.header
                        pose_stamped.pose.position.x = checkpoint[0]
                        pose_stamped.pose.position.y = checkpoint[1]
                        pose_stamped.pose.position.z = 0.0
                        
                        # 设置默认姿态
                        pose_stamped.pose.orientation.w = 1.0
                        nav_path.poses.append(pose_stamped)
                
                with self.data_lock:
                    self.current_navigation_path = nav_path
            
            # 创建导航标记
            self._create_navigation_markers(navigation, current_time)
                
        except Exception as e:
            rospy.logwarn(f"Error updating navigation data: {e}")
    
    def _create_navigation_markers(self, navigation, current_time):
        """
        创建导航标记
        
        Args:
            navigation: 导航模块对象
            current_time: 当前时间戳
        """
        try:
            marker_array = MarkerArray()
            marker_id = 0
            
            # 当前检查点标记
            checkpoints = navigation.get_checkpoints()
            if checkpoints and len(checkpoints) >= 1:
                current_marker = Marker()
                current_marker.header.stamp = current_time
                current_marker.header.frame_id = "map"
                current_marker.ns = "navigation"
                current_marker.id = marker_id
                marker_id += 1
                current_marker.type = Marker.SPHERE
                current_marker.action = Marker.ADD
                # print("checkpoints[0][0]: ", checkpoints[0][0])
                current_marker.pose.position.x = checkpoints[0][0]
                current_marker.pose.position.y = checkpoints[0][1]
                current_marker.pose.position.z = 1.0
                current_marker.pose.orientation.w = 1.0
                
                current_marker.scale.x = 2.0
                current_marker.scale.y = 2.0
                current_marker.scale.z = 2.0
                
                current_marker.color.r = 0.0
                current_marker.color.g = 1.0
                current_marker.color.b = 0.0
                current_marker.color.a = 0.8
                
                marker_array.markers.append(current_marker)
            
            # 下一个检查点标记
            if checkpoints and len(checkpoints) >= 2:
                next_marker = Marker()
                next_marker.header.stamp = current_time
                next_marker.header.frame_id = "map"
                next_marker.ns = "navigation"
                next_marker.id = marker_id
                marker_id += 1
                next_marker.type = Marker.SPHERE
                next_marker.action = Marker.ADD
                # print("checkpoints[1][0]: ", checkpoints[1][0])
                next_marker.pose.position.x = checkpoints[1][0]
                next_marker.pose.position.y = checkpoints[1][1]
                next_marker.pose.position.z = 1.0
                next_marker.pose.orientation.w = 1.0
                
                next_marker.scale.x = 1.5
                next_marker.scale.y = 1.5
                next_marker.scale.z = 1.5
                
                next_marker.color.r = 0.0
                next_marker.color.g = 0.0
                next_marker.color.b = 1.0
                next_marker.color.a = 0.8
                
                marker_array.markers.append(next_marker)
            
            # 最终目标点标记
            if hasattr(navigation, '_dest_node_path') and navigation._dest_node_path:
                dest_pos = navigation._dest_node_path.getPos()
                dest_marker = Marker()
                dest_marker.header.stamp = current_time
                dest_marker.header.frame_id = "map"
                dest_marker.ns = "navigation"
                dest_marker.id = marker_id
                marker_id += 1
                dest_marker.type = Marker.CYLINDER
                dest_marker.action = Marker.ADD
                # print("dest_pos[0]: ", dest_pos[0])
                dest_marker.pose.position.x = dest_pos[0]
                dest_marker.pose.position.y = dest_pos[1]
                dest_marker.pose.position.z = 1.0
                dest_marker.pose.orientation.w = 1.0
                
                dest_marker.scale.x = 3.0
                dest_marker.scale.y = 3.0
                dest_marker.scale.z = 2.0
                
                dest_marker.color.r = 1.0
                dest_marker.color.g = 0.0
                dest_marker.color.b = 0.0
                dest_marker.color.a = 0.8
                
                marker_array.markers.append(dest_marker)
            
            with self.data_lock:
                self.current_navigation_markers = marker_array
                
        except Exception as e:
            rospy.logwarn(f"Error creating navigation markers: {e}")

    def __del__(self):
        """
        析构函数
        """
        self.stop_publishing()