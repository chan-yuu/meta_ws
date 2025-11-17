#!/usr/bin/env python3
"""
测试导航点发布功能
"""

import time
import numpy as np
from trajectory_planning_env import TrajectoryPlanningEnv

def test_navigation_publisher():
    """
    测试导航点发布功能
    """
    print("开始测试导航点发布功能...")
    
    # 创建环境配置
    config = {
        'use_render': False,  # 不使用渲染以加快测试
        'manual_control': False,
        'traffic_density': 0.0,  # 无交通
        'map': 4,  # 使用简单地图
        'vehicle_config': {
            'show_lidar': True, 
            'show_navi_mark': True, 
            'show_line_to_navi_mark': True
        },
    }
    
    try:
        # 创建环境
        env = TrajectoryPlanningEnv(config)
        print("环境创建成功")
        
        # 重置环境
        obs = env.reset()
        print("环境重置成功")
        
        # 检查ROS发布器是否初始化
        if env.ros_publisher is not None:
            print("ROS发布器初始化成功")
        else:
            print("警告: ROS发布器未初始化")
            return False
        
        # 检查车辆是否有导航模块
        if hasattr(env, 'agent') and env.agent is not None:
            if hasattr(env.agent, 'navigation'):
                print("车辆导航模块检测成功")
                
                # 获取导航点信息
                navigation = env.agent.navigation
                checkpoints = navigation.get_checkpoints()
                print(f"获取到 {len(checkpoints)} 个检查点")
                
                if len(checkpoints) >= 2:
                    print(f"当前检查点: ({checkpoints[0][0]:.2f}, {checkpoints[0][1]:.2f})")
                    print(f"下一个检查点: ({checkpoints[1][0]:.2f}, {checkpoints[1][1]:.2f})")
                
                # 检查目标点
                if hasattr(navigation, '_dest_node_path') and navigation._dest_node_path:
                    dest_pos = navigation._dest_node_path.getPos()
                    print(f"目标点: ({dest_pos[0]:.2f}, {dest_pos[1]:.2f})")
                
            else:
                print("错误: 车辆没有导航模块")
                return False
        else:
            print("错误: 没有检测到车辆")
            return False
        
        # 运行几步来测试导航点发布
        print("\n开始运行环境步骤...")
        for step in range(5):
            # 随机动作
            action = np.random.uniform(-0.5, 0.5, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"步骤 {step+1}: 车辆位置 ({env.agent.position[0]:.2f}, {env.agent.position[1]:.2f})")
            
            # 检查导航点数据是否更新
            if env.ros_publisher is not None:
                try:
                    # 手动调用导航点更新
                    env.ros_publisher.update_navigation_data(env.agent)
                    print(f"  导航点数据更新成功")
                except Exception as e:
                    print(f"  导航点数据更新失败: {e}")
            
            time.sleep(0.1)  # 短暂延迟
            
            if terminated or truncated:
                print("环境提前终止")
                break
        
        print("\n测试完成!")
        print("请在Rviz中检查以下话题:")
        print("- /metadrive_trajectory_planning/navigation_markers (visualization_msgs/MarkerArray)")
        print("- /metadrive_trajectory_planning/navigation_path (nav_msgs/Path)")
        print("- /metadrive_trajectory_planning/current_checkpoint (geometry_msgs/PointStamped)")
        print("- /metadrive_trajectory_planning/next_checkpoint (geometry_msgs/PointStamped)")
        print("- /metadrive_trajectory_planning/destination (geometry_msgs/PointStamped)")
        
        # 清理
        env.close()
        return True
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_navigation_publisher()
    if success:
        print("\n✅ 导航点发布功能测试通过")
    else:
        print("\n❌ 导航点发布功能测试失败")