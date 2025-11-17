#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from trajectory_planning_env import TrajectoryPlanningEnv
import time
import gymnasium as gym
from collections import defaultdict
import json
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei"]

def test_action_mapping():
    print("=== 测试动作映射功能 ===")
    
    # 创建环境
    config = {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.0,
        # "alpha_1": 2.0,  # 去除起点横向偏移校正
        "alpha_2": 3.5,
        "v_min": 5.0,
        "v_max": 30.0,
        "T_min": 3.0,
        "T_max": 8.0,
        "use_extended_reference_line": True
    }
    
    env = TrajectoryPlanningEnv(config)
    
    # 测试不同的动作向量 (3维: [a2, a3, a4])
    test_actions = [
        [-1.0, -1.0, -1.0],  # 最小值
        [0.0, 0.0, 0.0],     # 中间值
        [1.0, 1.0, 1.0],     # 最大值
        [-0.3, 0.8, -0.2]    # 随机值
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n测试动作 {i+1}: {action}")
        
        # 映射动作到规划参数 (3维: [a2, a3, a4])
        a2, a3, a4 = action
        delta_d0 = 0.0  # 不再使用起点横向偏移校正
        delta_dT = config["alpha_2"] * a2
        vT = config["v_min"] + (a3 + 1) / 2 * (config["v_max"] - config["v_min"])
        T = config["T_min"] + (a4 + 1) / 2 * (config["T_max"] - config["T_min"])
        
        print(f"  起点横向偏移校正: {delta_d0:.2f} m (固定为0)")
        print(f"  终点横向偏移: {delta_dT:.2f} m")
        print(f"  终端目标速度: {vT:.2f} m/s")
        print(f"  规划时域: {T:.2f} s")
    
    env.close()
    print("动作映射测试完成！")

def test_trajectory_generation():
    print("\n=== 测试轨迹生成功能 ===")
    
    from trajectory_planning_env import TrajectoryPlanner
    
    # 创建轨迹规划器
    config = {
        'trajectory_dt': 0.1,
        'trajectory_points': 50,
        'lookahead_distance': 5.0
    }
    planner = TrajectoryPlanner(config)
    
    # 设置初始状态
    initial_state = {
        'd0': 0.0, 'dd0': 0.0, 'd2d0': 0.0,  # 横向初始状态
        's0': 0.0, 'v0': 15.0                # 纵向初始状态
    }
    
    # 设置规划参数
    planning_params = {
        'delta_d0': 0.5,   # 起点横向偏移校正
        'delta_dT': 2.0,   # 终点横向偏移
        'vT': 20.0,        # 终端目标速度
        'T': 5.0           # 规划时域
    }
    
    print(f"初始状态: {initial_state}")
    print(f"规划参数: {planning_params}")
    
    # 创建模拟的vehicle对象
    class MockVehicle:
        def __init__(self, state):
            self.speed = state['v0']
            self.position = [state['s0'], 0]  # [x, y]
            self.heading = 0.0
            self.heading_theta = 0.0
            self.lane = None
            self.navigation = None
    
    mock_vehicle = MockVehicle(initial_state)
    
    # 生成轨迹
    world_trajectory = planner.generate_trajectory(
        vehicle=mock_vehicle,
        planning_params=planning_params
    )
    
    print(f"\n生成轨迹点数: {len(world_trajectory)}")
    print(f"世界坐标轨迹前3个点: {world_trajectory[:3]}")
    
    # 验证边界条件
    print("\n=== 验证边界条件 ===")
    
    if world_trajectory:
        print(f"轨迹首点: x={world_trajectory[0][0]:.3f}, y={world_trajectory[0][1]:.3f}, heading={world_trajectory[0][2]:.3f}, v={world_trajectory[0][3]:.3f}")
        print(f"轨迹末点: x={world_trajectory[-1][0]:.3f}, y={world_trajectory[-1][1]:.3f}, heading={world_trajectory[-1][2]:.3f}, v={world_trajectory[-1][3]:.3f}")
        print(f"初始速度: 期望={initial_state['v0']:.3f}, 实际={world_trajectory[0][3]:.3f}")
        print(f"终端速度: 期望={planning_params['vT']:.3f}, 实际={world_trajectory[-1][3]:.3f}")
    
    print("轨迹生成测试完成！")
    
    return world_trajectory

def test_coordinate_conversion():
    """
    测试坐标转换功能
    """
    print("\n=== 测试坐标转换功能 ===")
    
    from frenet_converter import FrenetConverter, ReferenceLineGenerator
    
    # 生成直线参考线
    reference_line = ReferenceLineGenerator.generate_straight_line(
        start_x=0.0, start_y=0.0, heading=0.0, length=100.0, num_points=50
    )
    
    print(f"参考线点数: {len(reference_line)}")
    print(f"参考线前3个点: {reference_line[:3]}")
    
    # 创建转换器
    converter = FrenetConverter(reference_line)
    
    # 测试Frenet到世界坐标转换
    test_points = [
        (0.0, 0.0),    # 起点
        (50.0, 0.0),   # 中点
        (100.0, 0.0),  # 终点
        (25.0, 2.0),   # 偏移点
        (75.0, -1.5)   # 负偏移点
    ]
    
    print("\nFrenet到世界坐标转换测试:")
    for s, d in test_points:
        x, y, heading = converter.frenet_to_cartesian(s, d)
        print(f"  Frenet({s:5.1f}, {d:5.1f}) -> World({x:6.2f}, {y:6.2f}, {heading:6.3f})")
        
        # 反向转换验证
        s_back, d_back = converter.cartesian_to_frenet(x, y)
        print(f"    反向转换: World({x:6.2f}, {y:6.2f}) -> Frenet({s_back:5.1f}, {d_back:5.1f})")
    
    print("坐标转换测试完成！")

class EpisodeStatistics:
    """
    Episode统计信息类
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_reward = 0.0
        self.steps = 0
        self.termination_reason = "unknown"
        self.trajectory_points = []
        self.rewards = []
        self.actions = []
        self.start_time = time.time()
    
    def add_step(self, action, reward, obs=None):
        self.steps += 1
        self.total_reward += reward
        self.rewards.append(reward)
        self.actions.append(action.copy() if isinstance(action, list) else action)
        
        # 记录车辆位置（如果观测中包含位置信息）
        if obs is not None and hasattr(obs, '__len__') and len(obs) >= 2:
            self.trajectory_points.append([obs[0], obs[1]])
    
    def set_termination(self, reason):
        self.termination_reason = reason
        self.duration = time.time() - self.start_time
    
    def get_summary(self):
        return {
            'total_reward': self.total_reward,
            'steps': self.steps,
            'avg_reward': self.total_reward / max(1, self.steps),
            'termination_reason': self.termination_reason,
            'duration': getattr(self, 'duration', 0),
            'trajectory_length': len(self.trajectory_points)
        }

def run_single_episode(env, episode_id, max_steps=500, action_policy='random', render=False):
    """    
    Args:
        env: 环境实例
        episode_id: episode编号
        max_steps: 最大步数
        action_policy: 动作策略 ('random', 'zero', 'custom')
        render: 是否渲染
    Returns:
        EpisodeStatistics: episode统计信息
    """
    stats = EpisodeStatistics()
    
    try:
        # 重置环境
        obs, info = env.reset()
        print(f"\nEpisode {episode_id}: 开始")
        
        if render:
            print(f"  初始观测维度: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
        
        # 运行episode
        for step in range(max_steps):
            # 根据策略生成动作 (3维: [a2, a3, a4])
            if action_policy == 'random':
                action = env.action_space.sample()
            elif action_policy == 'zero':
                if step < 200:
                    action = [0.0, 0.5, -0.5]  # [终点偏移, 速度, 时域]
                elif step < 300:
                    action = [0.0, -0.75, -0.5]  # [终点偏移, 速度, 时域]
                elif step < 500:
                    action = [0.0, 0.2, -0.5]  # [终点偏移, 速度, 时域]
            elif action_policy == 'custom':
                # 自定义策略：轻微的随机扰动
                action = [np.random.uniform(-0.2, 0.2) for _ in range(3)]
            else:
                action = [0.0, 0.0, 0.0]
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            # print("obs: ", obs)
            # 记录统计信息
            stats.add_step(action, reward, obs)
            
            # 渲染（可选）
            if render and step % 10 == 0:  # 每10步渲染一次以提高性能
                env.render(
                    mode="topdown",
                    window=True,
                    screen_record=False,
                    screen_size=(700, 700)
                )
            
            # 检查终止条件
            done = terminated or truncated
            
            if done:
                # 确定终止原因
                if terminated:
                    if 'termination_reason' in info:
                        reason = info['termination_reason']
                    else:
                        reason = "environment_terminated"
                else:
                    reason = "max_steps_reached"
                
                stats.set_termination(reason)
                print(f"  Episode {episode_id}: 在第{step+1}步结束，原因: {reason}")
                break
        else:
            # 达到最大步数
            stats.set_termination("max_steps_reached")
            print(f"  Episode {episode_id}: 达到最大步数{max_steps}")
        
        # 输出episode摘要
        summary = stats.get_summary()
        print(f"  总奖励: {summary['total_reward']:.3f}, 平均奖励: {summary['avg_reward']:.3f}, 步数: {summary['steps']}")
        
    except Exception as e:
        print(f"  Episode {episode_id} 出错: {e}")
        import traceback
        traceback.print_exc()
        stats.set_termination("error")
    
    return stats

def run_multiple_episodes(num_episodes=10, max_steps=500, action_policy='random', render_episodes=None):
    """    
    Args:
        num_episodes: episode数量
        max_steps: 每个episode最大步数
        action_policy: 动作策略
        render_episodes: 需要渲染的episode列表，None表示不渲染
    Returns:
        dict: 包含所有episode统计信息的字典
    """
    print(f"\n=== 开始运行{num_episodes}个Episodes ===")
    print(f"每个Episode最大步数: {max_steps}")
    print(f"动作策略: {action_policy}")
    
    # 创建环境
    config = {
        "use_render": True,
        "manual_control": False,
        "traffic_density": 0.0,
        # "alpha_1": 2.0,  # 已移除起点横向偏移校正
        "alpha_2": 3.5,
        "v_min": -0.1,
        "v_max": 15.0,
        "T_min": 2.0,
        "T_max": 8.0,
        "use_extended_reference_line": True,
        "map": 4
    }
    
    env = TrajectoryPlanningEnv(config)
    # 测试
    # env.reset()
    # env.step()
    
    # 收集所有episode的统计信息
    all_episodes = []
    termination_counts = defaultdict(int)
    
    try:
        print(f"环境信息:")
        print(f"  动作空间: {env.action_space}")
        print(f"  观测空间: {env.observation_space}")
        
        for episode_id in range(1, num_episodes + 1):
            # 判断是否需要渲染
            should_render = render_episodes is not None and episode_id in render_episodes
            
            # 运行单个episode
            episode_stats = run_single_episode(
                env, episode_id, max_steps, action_policy, should_render
            )
            
            all_episodes.append(episode_stats)
            termination_counts[episode_stats.termination_reason] += 1
            
            # 每5个episode输出一次进度
            if episode_id % 5 == 0:
                print(f"\n进度: {episode_id}/{num_episodes} episodes完成")
    
    except KeyboardInterrupt:
        print("\n用户中断测试")
    
    finally:
        env.close()
    
    # 分析统计结果
    results = analyze_episode_statistics(all_episodes, termination_counts)
    
    # 将episode数据添加到结果中以便后续可视化
    results['episodes'] = all_episodes
    
    return results

def analyze_episode_statistics(all_episodes, termination_counts):
    """
    分析episode统计信息
    
    Args:
        all_episodes: 所有episode的统计信息列表
        termination_counts: 终止原因计数
    
    Returns:
        dict: 分析结果
    """
    if not all_episodes:
        return {"error": "没有episode数据"}
    
    print(f"\n=== Episode统计分析 ===")
    
    # 基本统计
    total_rewards = [ep.total_reward for ep in all_episodes]
    avg_rewards = [ep.get_summary()['avg_reward'] for ep in all_episodes]
    steps_counts = [ep.steps for ep in all_episodes]
    durations = [ep.get_summary()['duration'] for ep in all_episodes]
    
    # 计算统计指标
    stats = {
        'num_episodes': len(all_episodes),
        'total_reward': {
            'mean': np.mean(total_rewards),
            'std': np.std(total_rewards),
            'min': np.min(total_rewards),
            'max': np.max(total_rewards)
        },
        'avg_reward_per_step': {
            'mean': np.mean(avg_rewards),
            'std': np.std(avg_rewards),
            'min': np.min(avg_rewards),
            'max': np.max(avg_rewards)
        },
        'steps': {
            'mean': np.mean(steps_counts),
            'std': np.std(steps_counts),
            'min': np.min(steps_counts),
            'max': np.max(steps_counts)
        },
        'duration': {
            'mean': np.mean(durations),
            'std': np.std(durations),
            'total': np.sum(durations)
        },
        'termination_reasons': dict(termination_counts)
    }
    
    # 输出统计结果
    print(f"总Episodes数: {stats['num_episodes']}")
    print(f"\n奖励统计:")
    print(f"  总奖励 - 均值: {stats['total_reward']['mean']:.3f}, 标准差: {stats['total_reward']['std']:.3f}")
    print(f"  总奖励 - 最小: {stats['total_reward']['min']:.3f}, 最大: {stats['total_reward']['max']:.3f}")
    print(f"  平均奖励/步 - 均值: {stats['avg_reward_per_step']['mean']:.3f}, 标准差: {stats['avg_reward_per_step']['std']:.3f}")
    
    print(f"\n步数统计:")
    print(f"  均值: {stats['steps']['mean']:.1f}, 标准差: {stats['steps']['std']:.1f}")
    print(f"  最小: {stats['steps']['min']}, 最大: {stats['steps']['max']}")
    
    print(f"\n时间统计:")
    print(f"  平均每episode时长: {stats['duration']['mean']:.2f}秒")
    print(f"  总测试时长: {stats['duration']['total']:.2f}秒")
    
    print(f"\n终止原因统计:")
    for reason, count in termination_counts.items():
        percentage = (count / stats['num_episodes']) * 100
        print(f"  {reason}: {count}次 ({percentage:.1f}%)")
    
    # 保存详细统计到文件
    save_statistics_to_file(stats, all_episodes)
    
    return stats

def save_statistics_to_file(stats, all_episodes):
    """
    保存统计信息到文件
    """
    try:
        # 准备保存的数据
        save_data = {
            'summary_statistics': stats,
            'episode_details': []
        }
        
        # 添加每个episode的详细信息
        for i, episode in enumerate(all_episodes):
            episode_data = episode.get_summary()
            episode_data['episode_id'] = i + 1
            episode_data['rewards_history'] = episode.rewards
            episode_data['actions_history'] = episode.actions
            save_data['episode_details'].append(episode_data)
        
        # 保存到JSON文件
        filename = f"/home/cyun/9.23/metadrive/planing_env/episode_statistics_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细统计信息已保存到: {filename}")
        
    except Exception as e:
        print(f"保存统计信息时出错: {e}")


def visualize_trajectory(world_trajectory):
    """
    可视化生成的轨迹
    """
    print("\n=== 轨迹可视化 ===")
    
    try:
        # 提取轨迹数据
        x_values = [point[0] for point in world_trajectory]
        y_values = [point[1] for point in world_trajectory]
        headings = [point[2] for point in world_trajectory]
        velocities = [point[3] for point in world_trajectory]
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # X位置
        ax1.plot(range(len(x_values)), x_values, 'b-', linewidth=2)
        ax1.set_xlabel('轨迹点索引')
        ax1.set_ylabel('X位置 (m)')
        ax1.set_title('X轨迹')
        ax1.grid(True)
        
        # Y位置
        ax2.plot(range(len(y_values)), y_values, 'r-', linewidth=2)
        ax2.set_xlabel('轨迹点索引')
        ax2.set_ylabel('Y位置 (m)')
        ax2.set_title('Y轨迹')
        ax2.grid(True)
        
        # 速度曲线
        ax3.plot(range(len(velocities)), velocities, 'g-', linewidth=2)
        ax3.set_xlabel('轨迹点索引')
        ax3.set_ylabel('速度 (m/s)')
        ax3.set_title('速度轨迹')
        ax3.grid(True)
        
        # XY轨迹图
        ax4.plot(x_values, y_values, 'purple', linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('X位置 (m)')
        ax4.set_ylabel('Y位置 (m)')
        ax4.set_title('世界坐标轨迹')
        ax4.grid(True)
        ax4.axis('equal')
        
        plt.tight_layout()
        plt.savefig('/home/cyun/9.23/metadrive/planing_env/trajectory_visualization.png', dpi=150)
        print("轨迹图已保存到: trajectory_visualization.png")
        
    except Exception as e:
        print(f"可视化出错: {e}")
        print("跳过可视化步骤")

def test_basic_functions():
    """
    测试基本功能（可选）
    """
    print("\n=== 基本功能测试 ===")
    
    try:
        # 1. 测试动作映射
        test_action_mapping()
        
        # 2. 测试轨迹生成
        world_trajectory = test_trajectory_generation()
        
        # 3. 测试坐标转换
        test_coordinate_conversion()
        
        # 4. 可视化轨迹
        # if world_trajectory:
        #     visualize_trajectory(world_trajectory)
        
        return True
        
    except Exception as e:
        print(f"基本功能测试出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试函数 - 基于Episode的测试
    """
    print("开始轨迹规划环境Episode测试...")
    print("=" * 60)
    
    try:
        # 配置测试参数
        test_configs = [
            {
                'name': '零动作策略测试',
                'num_episodes': 10,
                'max_steps': 500,
                'action_policy': 'zero',
                'render_episodes': None  # 不渲染 只渲染第1个episode
            },
            {
                'name': '随机动作策略测试',
                'num_episodes': 0,
                'max_steps': 500,
                'action_policy': 'random',
                'render_episodes': None  # 不渲染
            },
            {
                'name': '自定义动作策略测试',
                'num_episodes': 0,
                'max_steps': 500,
                'action_policy': 'custom',
                'render_episodes': None  # 渲染第1和第4个episode
            }
        ]
        
        # 运行不同配置的测试
        all_results = {}
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n{'='*20} 测试配置 {i}: {config['name']} {'='*20}")
            
            # 运行多个episodes
            results = run_multiple_episodes(
                num_episodes=config['num_episodes'],
                max_steps=config['max_steps'],
                action_policy=config['action_policy'],
                render_episodes=config['render_episodes']
            )
            
            all_results[config['name']] = results
            
        
        # 输出总结
        print(f"\n{'='*60}")
        print("测试总结:")
        for test_name, results in all_results.items():
            if 'error' not in results:
                print(f"\n{test_name}:")
                print(f"  Episodes数: {results.get('num_episodes', 0)}")
                if 'total_reward' in results:
                    print(f"  平均总奖励: {results['total_reward']['mean']:.3f}")
                    print(f"  平均步数: {results['steps']['mean']:.1f}")
        
        # # 可选：运行基本功能测试
        # user_input = input("\n是否运行基本功能测试? (y/n): ").lower().strip()
        # if user_input == 'y':
        #     test_basic_functions()
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("轨迹规划环境Episode测试完成！")

if __name__ == "__main__":
    main()