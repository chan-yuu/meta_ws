import numpy as np
from typing import Tuple, List

class FrenetConverter:
    """Frenet坐标系与世界坐标系之间的转换工具类"""
    
    def __init__(self, reference_line: List[Tuple[float, float, float]]):
        """
        初始化Frenet转换器
        
        Args:
            reference_line: 参考线点列表，每个点包含(x, y, heading)
        """
        self.reference_line = np.array(reference_line)
        self.s_values = self._compute_s_values()
        
    def _compute_s_values(self) -> np.ndarray:
        """计算参考线上每个点的累积弧长s值"""
        if len(self.reference_line) < 2:
            return np.array([0.0])
            
        s_values = [0.0]
        for i in range(1, len(self.reference_line)):
            dx = self.reference_line[i][0] - self.reference_line[i-1][0]
            dy = self.reference_line[i][1] - self.reference_line[i-1][1]
            ds = np.sqrt(dx**2 + dy**2)
            s_values.append(s_values[-1] + ds)
            
        return np.array(s_values)
    
    def _find_closest_point(self, s: float) -> Tuple[int, float]:
        """找到给定s值对应的最近参考点索引和插值权重"""
        if s <= self.s_values[0]:
            return 0, 0.0
        if s >= self.s_values[-1]:
            return len(self.s_values) - 2, 1.0
            
        # 二分查找
        left, right = 0, len(self.s_values) - 1
        while right - left > 1:
            mid = (left + right) // 2
            if self.s_values[mid] <= s:
                left = mid
            else:
                right = mid
                
        # 计算插值权重
        ds = self.s_values[right] - self.s_values[left]
        if ds < 1e-6:
            weight = 0.0
        else:
            weight = (s - self.s_values[left]) / ds
            
        return left, weight
    
    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """
        将Frenet坐标(s, d)转换为世界坐标(x, y, heading)
        
        Args:
            s: 纵向位置（沿参考线的弧长）
            d: 横向位置（相对于参考线的偏移）
            
        Returns:
            (x, y, heading): 世界坐标系下的位置和航向角
        """
        if len(self.reference_line) == 0:
            return 0.0, 0.0, 0.0
            
        # 找到对应的参考点
        idx, weight = self._find_closest_point(s)
        
        # 插值计算参考点的位置和航向
        if idx >= len(self.reference_line) - 1:
            ref_x, ref_y, ref_heading = self.reference_line[-1]
        else:
            ref_x1, ref_y1, ref_heading1 = self.reference_line[idx]
            ref_x2, ref_y2, ref_heading2 = self.reference_line[idx + 1]
            
            ref_x = ref_x1 + weight * (ref_x2 - ref_x1)
            ref_y = ref_y1 + weight * (ref_y2 - ref_y1)
            
            # 航向角插值需要考虑角度的连续性
            angle_diff = ref_heading2 - ref_heading1
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            ref_heading = ref_heading1 + weight * angle_diff
        
        # 计算垂直于参考线的单位向量
        normal_x = -np.sin(ref_heading)
        normal_y = np.cos(ref_heading)
        
        # 计算世界坐标
        x = ref_x + d * normal_x
        y = ref_y + d * normal_y
        heading = ref_heading
        
        return x, y, heading
    
    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        """
        将世界坐标(x, y)转换为Frenet坐标(s, d)
        
        Args:
            x, y: 世界坐标系下的位置
            
        Returns:
            (s, d): Frenet坐标系下的纵向和横向位置
        """
        if len(self.reference_line) == 0:
            return 0.0, 0.0
            
        min_distance = float('inf')
        closest_s = 0.0
        closest_d = 0.0
        
        # 遍历参考线找到最近点
        for i in range(len(self.reference_line)):
            ref_x, ref_y, ref_heading = self.reference_line[i]
            
            # 计算到参考点的距离
            dx = x - ref_x
            dy = y - ref_y
            distance = np.sqrt(dx**2 + dy**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_s = self.s_values[i]
                
                # 计算横向偏移（带符号）
                normal_x = -np.sin(ref_heading)
                normal_y = np.cos(ref_heading)
                closest_d = dx * normal_x + dy * normal_y
        
        return closest_s, closest_d
    
    def get_reference_line_length(self) -> float:
        """获取参考线总长度"""
        return self.s_values[-1] if len(self.s_values) > 0 else 0.0
    
    def get_curvature_at_s(self, s: float) -> float:
        """
        计算给定s位置处的曲率
        
        Args:
            s: 纵向位置
            
        Returns:
            曲率值
        """
        idx, weight = self._find_closest_point(s)
        
        if idx >= len(self.reference_line) - 2:
            return 0.0
            
        # 使用三点法计算曲率
        p1 = self.reference_line[max(0, idx-1)]
        p2 = self.reference_line[idx]
        p3 = self.reference_line[min(len(self.reference_line)-1, idx+1)]
        
        # 计算曲率 k = |det(v1, v2)| / |v1|^3
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]
        v1_norm = np.linalg.norm(v1)
        
        if v1_norm < 1e-6:
            return 0.0
            
        curvature = abs(cross_product) / (v1_norm ** 3)
        return curvature

class ReferenceLineGenerator:
    """参考线生成器，用于从MetaDrive的车道信息生成参考线"""
    
    @staticmethod
    def generate_from_lane(lane, num_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        从MetaDrive的车道对象生成参考线（原始方法，只使用单个车道）
        
        Args:
            lane: MetaDrive的车道对象
            num_points: 参考线点数
            
        Returns:
            参考线点列表，每个点包含(x, y, heading)
        """
        reference_line = []
        
        try:
            # print("lane", lane, "lane.length", lane.length)
            # 获取车道长度
            lane_length = lane.length
            
            # 沿车道中心线采样点
            for i in range(num_points):
                s = (i / (num_points - 1)) * lane_length
                
                # 获取车道中心线上的点
                position = lane.position(s, 0)  # s位置，0横向偏移
                heading = lane.heading_theta_at(s)
                
                reference_line.append((position[0], position[1], heading))
                
        except Exception as e:
            print(f"生成参考线时出错: {e}")
            # 如果出错，生成一条简单的直线参考线
            for i in range(num_points):
                x = i * 2.0  # 每2米一个点
                y = 0.0
                heading = 0.0
                reference_line.append((x, y, heading))
        
        return reference_line
    
    @staticmethod
    def generate_extended_from_lane(lane, road_network, lane_index, 
                                  target_length: float = 500.0, 
                                  num_points: int = 100) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        从MetaDrive的车道对象生成扩展参考线，通过连接多个车道段达到目标长度
        
        Args:
            lane: 当前MetaDrive车道对象
            road_network: 道路网络对象
            lane_index: 当前车道索引 (start_node, end_node, lane_id)
            target_length: 目标参考线长度（米）
            num_points: 参考线点数
            
        Returns:
            Tuple[参考线点列表, 实际长度]
        """
        reference_line = []
        actual_length = 0.0
        
        try:
            # 获取车道序列
            lane_sequence = ReferenceLineGenerator._get_lane_sequence(
                lane, road_network, lane_index, target_length
            )
            
            # print(f"找到 {len(lane_sequence)} 个车道段用于生成参考线")
            
            accumulated_length = 0.0
            total_points_generated = 0
            
            for i, (current_lane, lane_length) in enumerate(lane_sequence):
                # 计算这个车道段的有效长度
                remaining_length = target_length - accumulated_length
                if remaining_length <= 0:
                    break
                    
                segment_length = min(lane_length, remaining_length)
                segment_ratio = segment_length / lane_length
                
                # 为这个车道段分配点数
                if i == len(lane_sequence) - 1:  # 最后一个段
                    segment_points = num_points - total_points_generated
                else:
                    segment_points = max(1, int(num_points * (segment_length / target_length)))
                
                segment_points = max(1, min(segment_points, num_points - total_points_generated))
                
                # 生成这个车道段的点
                for j in range(segment_points):
                    if total_points_generated >= num_points:
                        break
                        
                    s = (j / max(1, segment_points - 1)) * segment_length
                    
                    if accumulated_length + s >= target_length:
                        break
                    
                    position = current_lane.position(s, 0)
                    heading = current_lane.heading_theta_at(s)
                    
                    reference_line.append((position[0], position[1], heading))
                    total_points_generated += 1
                
                accumulated_length += segment_length
                actual_length = accumulated_length
                
                if accumulated_length >= target_length or total_points_generated >= num_points:
                    break
            
            print(f"生成了 {len(reference_line)} 个参考点，总长度: {actual_length:.2f}m")
                
        except Exception as e:
            print(f"生成扩展参考线时出错: {e}")
            # 回退到原始方法
            reference_line = ReferenceLineGenerator.generate_from_lane(lane, num_points)
            actual_length = lane.length
        
        return reference_line, actual_length
    
    @staticmethod
    def _get_lane_sequence(start_lane, road_network, start_lane_index, target_length):
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
            next_lane, next_index = ReferenceLineGenerator._get_next_lane(
                road_network, current_index
            )
            
            if next_lane is None or next_index in visited_indices:
                if next_lane is None:
                    print(f"在 {current_index} 后没有找到更多车道")
                break
            
            lane_sequence.append((next_lane, next_lane.length))
            accumulated_length += next_lane.length
            visited_indices.add(next_index)
            
            current_lane = next_lane
            current_index = next_index
            
            print(f"添加车道 {next_index}, 长度: {next_lane.length:.2f}m, 总计: {accumulated_length:.2f}m")
        
        return lane_sequence
    
    @staticmethod
    def _get_next_lane(road_network, current_index):
        """
        获取下一个车道
        
        Returns:
            Tuple of (next_lane, next_lane_index) or (None, None) if no next lane
        """
        try:
            start_node, end_node, lane_id = current_index
            
            # 尝试多种方式访问道路网络
            graph = None
            if hasattr(road_network, 'graph'):
                graph = road_network.graph
            elif hasattr(road_network, 'road_network_graph'):
                graph = road_network.road_network_graph
            elif hasattr(road_network, '_graph'):
                graph = road_network._graph
            
            if graph is None or end_node not in graph:
                # print(f"无法访问道路网络图或节点 {end_node} 不存在")
                return None, None
            
            # 获取所有可能的下一个节点
            next_connections = graph[end_node]
            
            if not next_connections:
                # print(f"节点 {end_node} 没有后续连接")
                return None, None
            
            # 优先选择相同lane_id的车道
            for next_end_node, lanes_data in next_connections.items():
                # 处理不同的数据结构
                lanes = None
                if isinstance(lanes_data, dict):
                    # 如果是字典，可能包含车道信息
                    if 'lanes' in lanes_data:
                        lanes = lanes_data['lanes']
                    else:
                        # 尝试直接使用字典值
                        lanes = list(lanes_data.values())
                elif isinstance(lanes_data, (list, tuple)):
                    lanes = lanes_data
                else:
                    # 单个车道对象
                    lanes = [lanes_data]
                
                if lanes and len(lanes) > 0:
                    # 优先选择相同lane_id的车道
                    if len(lanes) > lane_id:
                        next_lane = lanes[lane_id]
                        next_index = (end_node, next_end_node, lane_id)
                        # print(f"找到相同lane_id={lane_id}的下一车道: {next_index}")
                        return next_lane, next_index
                    else:
                        # 如果相同lane_id不存在，使用第一个可用车道
                        next_lane = lanes[0]
                        next_index = (end_node, next_end_node, 0)
                        # print(f"未找到相同lane_id={lane_id}，使用lane_id=0: {next_index}")
                        return next_lane, next_index
            
            # print(f"在节点 {end_node} 后没有找到可用的车道")
            return None, None
            
        except Exception as e:
            # print(f"获取下一个车道时出错: {e}")
            return None, None
    
    @staticmethod
    def generate_straight_line(start_x: float = 0.0, start_y: float = 0.0, 
                             heading: float = 0.0, length: float = 200.0, 
                             num_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        生成直线参考线
        
        Args:
            start_x, start_y: 起点坐标
            heading: 航向角
            length: 参考线长度
            num_points: 点数
            
        Returns:
            参考线点列表
        """
        reference_line = []
        
        for i in range(num_points):
            s = (i / (num_points - 1)) * length
            x = start_x + s * np.cos(heading)
            y = start_y + s * np.sin(heading)
            reference_line.append((x, y, heading))
            
        return reference_line