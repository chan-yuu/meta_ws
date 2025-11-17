#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轨迹规划环境模型评估脚本（适配Ray 2.2版本）
"""

import logging
import os
import time
from typing import Dict

import matplotlib
import numpy as np
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPO, PPOConfig  # Ray 2.x中PPO的导入方式
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

# 导入轨迹规划环境和Gym包装器
from trajectory_planning_env import TrajectoryPlanningEnv
from metadrive.envs.gym_wrapper import createGymWrapper

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# 驾驶回调类，用于收集和记录训练/评估过程中的各种指标
class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], 
        episode: Episode, env_index: int, **kwargs
    ):
        # 初始化每个episode的数据收集列表
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: Episode, 
        env_index: int,** kwargs
    ):
        # 每步收集车辆状态和奖励信息
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], 
        episode: Episode, **kwargs
    ):
        # 获取episode结束时的终止状态信息
        last_info = episode.last_info_for()
        arrive_dest = last_info["arrive_dest"]
        crash = last_info["crash"]
        out_of_road = last_info["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        
        # 记录各种成功率和失败率指标
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        
        # 记录速度相关统计指标
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        
        # 记录转向相关统计指标
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        
        # 记录加速度相关统计指标
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        
        # 记录奖励相关统计指标
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        
        # 记录总成本
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # 初始化训练结果中的关键指标
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result.get("episode_len_mean", np.nan)
        result["cost"] = np.nan
        
        # 如果存在自定义指标，则更新结果
        if "custom_metrics" not in result:              
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]


def initialize_ray(local_mode=False, num_gpus=None, test_mode=False,** kwargs):
    os.environ['OMP_NUM_THREADS'] = '1'

    # Ray 2.x 初始化方式
    ray.init(
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
        local_mode=local_mode,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        **kwargs
    )
    print("Successfully initialize Ray!")
    try:
        print("Available resources: ", ray.available_resources())
    except Exception:
        pass


def get_trainer(checkpoint_path=None, extra_config=None, num_workers=0):
    """获取PPO训练器用于评估轨迹规划模型（适配Ray 2.x的Config API）"""
    # 使用Ray 2.x推荐的Config API配置参数
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.constants import DEFAULT_AGENT, TerminationState

    config = (
        PPOConfig()
        .environment(
            env=createGymWrapper(TrajectoryPlanningEnv),
            env_config=dict(
                # 轨迹规划特定参数（与训练脚本保持一致）
                alpha_2=3.5,                    # 轨迹规划权重参数
                v_min=-0.1,                     # 最小速度限制 (m/s)
                v_max=15.0,                     # 最大速度限制 (m/s)
                T_min=2.0,                      # 最小规划时间范围 (秒)
                T_max=8.0,                      # 最大规划时间范围 (秒)
                trajectory_dt=0.1,              # 轨迹时间步长 (秒)
                trajectory_points=50,           # 轨迹点数量
                
                # MetaDrive基础环境参数
                num_scenarios=10,                # 评估使用1个场景
                start_seed=42,                  # 评估的固定随机种子
                random_traffic=False,           # 不启用随机交通
                traffic_density=0.1,            # 交通密度
                use_render=True,                # 评估时启用渲染
                map="S",                           # 使用地图编号1
                # random_spawn_lane_index=True,
                # agent_configs={
                #     DEFAULT_AGENT: dict(
                #         use_special_color=True,
                #         spawn_lane_index=(('>', '>>', 2)),
                #     )
                # },
            )
        )
        .framework(framework="torch")
        .rollouts(
            num_rollout_workers=num_workers,
            batch_mode="complete_episodes",
        )
        .training(
            # horizon=1000,
            lr=0.0,  # 评估时学习率设为0
        )
        .callbacks(DrivingCallbacks)
        .exploration(
            explore=False,  # 评估时不进行探索
        )
        .resources(
            num_gpus=0,
            num_cpus_per_worker=1,
        )
    )
    
    if extra_config:
        config.update_from_dict(extra_config)
    
    # 创建PPO算法实例
    trainer = PPO(config=config)
    
    if checkpoint_path is not None:
        trainer.restore(os.path.expanduser(checkpoint_path))
        
    return trainer


def evaluate(trainer, num_episodes=20):
    """评估训练好的轨迹规划模型（适配Ray 2.x，使用worker.sample()替代ParallelRollouts）"""
    # 初始化结果收集列表
    ret_reward = []
    ret_length = []
    ret_success_rate = []
    ret_out_rate = []
    ret_crash_rate = []
    ret_max_step_rate = []
    ret_route_completion = []
    ret_velocity_mean = []
    
    start = time.time()
    episode_count = 0
    # 获取本地工作器用于采样
    local_worker = trainer.workers.local_worker()
    
    print(f"开始评估模型，目标episode数量: {num_episodes}")
    
    while episode_count < num_episodes:
        # Ray 2.x中使用worker.sample()获取采样数据
        rollout_batch = local_worker.sample()
        episodes = rollout_batch.split_by_episode()

        # 收集基本指标
        ret_reward.extend([e["rewards"].sum() for e in episodes])
        ret_length.extend([e.count for e in episodes])
        
        # 收集轨迹规划环境特定的终止状态信息
        for e in episodes:
            last_info = e["infos"][-1]
            ret_success_rate.append(last_info.get("arrive_dest", False))
            ret_out_rate.append(last_info.get("out_of_road", False))
            ret_crash_rate.append(last_info.get("crash", False))
            
            # 计算是否因为达到最大步数而结束
            arrive_dest = last_info.get("arrive_dest", False)
            crash = last_info.get("crash", False)
            out_of_road = last_info.get("out_of_road", False)
            max_step = not (arrive_dest or crash or out_of_road)
            ret_max_step_rate.append(max_step)
            
            # 收集其他有用的指标
            ret_route_completion.append(last_info.get("route_completion", 0.0))
            ret_velocity_mean.append(last_info.get("velocity", 0.0))

        episode_count += len(episodes)
        print(f"完成 {episode_count} 个episodes")

    # 计算最终统计结果
    ret = dict(
        reward=np.mean(ret_reward),
        length=np.mean(ret_length),
        success_rate=np.mean(ret_success_rate),
        out_of_road_rate=np.mean(ret_out_rate),
        crash_rate=np.mean(ret_crash_rate),
        max_step_rate=np.mean(ret_max_step_rate),
        route_completion=np.mean(ret_route_completion),
        velocity_mean=np.mean(ret_velocity_mean),
        episode_count=episode_count,
        time=time.time() - start,
    )
    
    print(
        f"评估完成！收集了 {episode_count} 个episodes，耗时: {time.time() - start:.3f} 秒\n"
        f"评估结果: {{{', '.join([f'{k}: {round(v, 3)}' for k, v in ret.items()])}}}"
    )
    return ret


if __name__ == '__main__':
    # 初始化Ray环境
    initialize_ray()
    
    # 设置训练好的模型检查点路径
    checkpoint_path = "/home/cyun/Disk/ubuntu20/meta_ws/src/planning_env/scripts/trajectory_experiment/PPO_GymEnvWrapper_93de9_00000_0_num_scenarios=1,start_seed=5000,traffic_density=0.1000_2025-09-28_01-15-34/checkpoint_000245"
    
    # 获取训练器并加载模型
    trainer = get_trainer(checkpoint_path)
    
    # 开始评估模型，设置评估episode数量
    evaluate(trainer, num_episodes=100)
