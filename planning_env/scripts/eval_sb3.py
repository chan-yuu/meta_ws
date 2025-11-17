#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SB3 模型评估
    python eval_sb3.py --model-path path/to/model.zip --num-episodes 100 [--render]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from trajectory_planning_env_no_print import TrajectoryPlanningEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif
import os
import time
from typing import Dict, List, Tuple


def create_env(need_monitor=False, render=False):
    """
    创建 TrajectoryPlanningEnv 环境
    
    Args:
        need_monitor (bool): 是否使用 Monitor 包装环境
    
    Returns:
        TrajectoryPlanningEnv: 配置好的环境实例
    """
    env = TrajectoryPlanningEnv(dict(
        map="S",
        # 连续动作空间
        # discrete_action=False,
        # horizon=500,
        # 场景设置
        # random_spawn_lane_index=False,
        num_scenarios=10,
        start_seed=5,
        # accident_prob=0,
        # log_level=50,
        use_render=True,
        traffic_density=0.0,
        alpha_2=3.5,
        v_min=5.0,
        v_max=15.0,
        T_min=2.0,
        T_max=8.0,
        use_extended_reference_line=True,
    ))
    
    if need_monitor:
        env = Monitor(env)
    
    return env


def evaluate_model(model_path: str, num_episodes: int = 100, render: bool = False, 
                  save_gif: bool = False) -> Dict[str, float]:
    """
    评估训练好的模型
    
    Args:
        model_path (str): 模型文件路径
        num_episodes (int): 评估回合数
        render (bool): 是否启用可视化
        save_gif (bool): 是否保存 GIF 动画
    
    Returns:
        Dict[str, float]: 评估结果统计
    """
    print(f"正在加载模型: {model_path}")
    
    # 加载训练好的模型
    try:
        model = PPO.load(model_path)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return {}
    
    # 创建评估环境
    env = create_env(need_monitor=False, render=render)
    
    # 评估指标收集
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    crash_count = 0
    out_of_road_count = 0
    max_step_count = 0
    
    print(f"开始评估，共 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0.0
        done = False
        
        # 如果需要保存 GIF，只在第一个回合记录
        if save_gif and episode == 0:
            frames = []
        
        while not done:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            action = list(action[:2]) + [-0.9]
            print("action: ", action)

            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # 记录渲染帧（如果需要）
            if save_gif and episode == 0:
                frame = env.render(mode="topdown", 
                                 window=False,
                                 screen_size=(600, 600), 
                                 camera_position=(50, 50))
                frames.append(frame)
            
            # 检查终止条件
            if done or truncated:
                # 统计终止原因
                if 'success' in info and info['success']:
                    success_count += 1
                elif 'crash' in info and info['crash']:
                    crash_count += 1
                elif 'out_of_road' in info and info['out_of_road']:
                    out_of_road_count += 1
                elif 'max_step' in info and info['max_step']:
                    max_step_count += 1
                
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"已完成 {episode + 1}/{num_episodes} 回合")
        
        # 保存第一个回合的 GIF
        if save_gif and episode == 0 and frames:
            print("正在生成 GIF 动画...")
            try:
                env.top_down_renderer.generate_gif()
                print("GIF 动画已保存")
            except Exception as e:
                print(f"GIF 生成失败: {e}")
    
    env.close()
    
    # 计算统计指标
    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes * 100,
        'crash_rate': crash_count / num_episodes * 100,
        'out_of_road_rate': out_of_road_count / num_episodes * 100,
        'max_step_rate': max_step_count / num_episodes * 100,
        'total_episodes': num_episodes
    }
    
    return results


def print_evaluation_results(results: Dict[str, float]):
    """
    打印评估结果
    
    Args:
        results (Dict[str, float]): 评估结果字典
    """
    if not results:
        print("评估失败，无结果可显示")
        return
    
    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    
    print(f"总回合数: {results['total_episodes']}")
    print("\n奖励统计:")
    print(f"  平均奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  最小奖励: {results['min_reward']:.2f}")
    print(f"  最大奖励: {results['max_reward']:.2f}")
    
    print("\n回合长度统计:")
    print(f"  平均长度: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    
    print("\n终止状态统计:")
    print(f"  成功率: {results['success_rate']:.2f}%")
    print(f"  碰撞率: {results['crash_rate']:.2f}%")
    print(f"  出路率: {results['out_of_road_rate']:.2f}%")
    print(f"  超时率: {results['max_step_rate']:.2f}%")
    
    print("="*50)


def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="评估 Stable Baselines3 训练的 TrajectoryPlanningEnv 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="训练好的模型文件路径 (.zip 文件)"
    )
    
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="评估回合数"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="是否启用可视化渲染"
    )
    
    parser.add_argument(
        "--save_gif",
        action="store_true",
        help="是否保存第一个回合的 GIF 动画"
    )
    
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help="随机种子"
    # )
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # # 设置随机种子
    # np.random.seed(args.seed)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    print(f"模型路径: {args.model_path}")
    print(f"评估回合数: {args.num_episodes}")
    # print(f"启用渲染: {args.render}")
    print(f"保存 GIF: {args.save_gif}")
    # print(f"随机种子: {args.seed}")
    
    # 开始评估
    start_time = time.time()
    results = evaluate_model(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        render=args.render,
        save_gif=args.save_gif
    )
    end_time = time.time()
    
    # 打印结果
    print_evaluation_results(results)
    print(f"\n评估耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()