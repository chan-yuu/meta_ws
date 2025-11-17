#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Planning Environment Training Script
完全仿照train_generalization_experiment.py的结构来训练trajectory_planning_env.py环境
"""

import argparse
import copy
import logging
import numpy as np
import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import CLIReporter
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from typing import Dict, Optional

# 导入轨迹规划环境
from trajectory_planning_env import TrajectoryPlanningEnv
from metadrive.envs.gym_wrapper import createGymWrapper


class TrajectoryPlanningCallbacks(DefaultCallbacks):
    """
    轨迹规划环境的自定义回调函数，用于记录训练指标
    完全仿照DrivingCallbacks的结构
    """
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy],
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        # 初始化episode级别的指标
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["trajectory_length"] = []
        episode.user_data["tracking_error"] = []
        episode.user_data["cost"] = []
        
    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        # 获取环境信息
        info = episode.last_info_for()
        if info is not None:
            # 记录轨迹规划相关指标
            if "velocity" in info:
                episode.user_data["velocity"].append(info["velocity"])
            if "steering" in info:
                episode.user_data["steering"].append(info["steering"])
            if "acceleration" in info:
                episode.user_data["acceleration"].append(info["acceleration"])
            if "trajectory_length" in info:
                episode.user_data["trajectory_length"].append(info["trajectory_length"])
            if "tracking_error" in info:
                episode.user_data["tracking_error"].append(info["tracking_error"])
            if "cost" in info:
                episode.user_data["cost"].append(info["cost"])
                
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                      policies: Dict[str, Policy],
                      episode: MultiAgentEpisode, env_index: int, **kwargs):
        # 计算episode级别的统计指标
        info = episode.last_info_for()
        
        # 速度统计
        if len(episode.user_data["velocity"]) > 0:
            episode.custom_metrics["velocity_mean"] = np.mean(episode.user_data["velocity"])
            episode.custom_metrics["velocity_std"] = np.std(episode.user_data["velocity"])
        
        # 转向统计
        if len(episode.user_data["steering"]) > 0:
            episode.custom_metrics["steering_mean"] = np.mean(np.abs(episode.user_data["steering"]))
            episode.custom_metrics["steering_std"] = np.std(episode.user_data["steering"])
        
        # 加速度统计
        if len(episode.user_data["acceleration"]) > 0:
            episode.custom_metrics["acceleration_mean"] = np.mean(np.abs(episode.user_data["acceleration"]))
            episode.custom_metrics["acceleration_std"] = np.std(episode.user_data["acceleration"])
        
        # 轨迹长度统计
        if len(episode.user_data["trajectory_length"]) > 0:
            episode.custom_metrics["trajectory_length_mean"] = np.mean(episode.user_data["trajectory_length"])
        
        # 跟踪误差统计
        if len(episode.user_data["tracking_error"]) > 0:
            episode.custom_metrics["tracking_error_mean"] = np.mean(episode.user_data["tracking_error"])
            episode.custom_metrics["tracking_error_std"] = np.std(episode.user_data["tracking_error"])
        
        # 成本统计
        if len(episode.user_data["cost"]) > 0:
            episode.custom_metrics["cost_mean"] = np.mean(episode.user_data["cost"])
            episode.custom_metrics["cost_std"] = np.std(episode.user_data["cost"])
        
        # 成功率和其他终止条件统计（仿照原始结构）
        if info is not None:
            episode.custom_metrics["success_rate"] = float(info.get("arrive_dest", False))
            episode.custom_metrics["crash_rate"] = float(info.get("crash", False))
            episode.custom_metrics["out_of_road_rate"] = float(info.get("out_of_road", False))
            episode.custom_metrics["max_step_rate"] = float(info.get("max_step", False))
        else:
            episode.custom_metrics["success_rate"] = 0.0
            episode.custom_metrics["crash_rate"] = 0.0
            episode.custom_metrics["out_of_road_rate"] = 0.0
            episode.custom_metrics["max_step_rate"] = 0.0
            
    def on_train_result(self, *, trainer, result: dict, **kwargs):
        # 训练结果回调，记录训练级别的指标
        result["callback_ok"] = True


def train(
    trainer,
    config,
    stop,
    exp_name,
    num_gpus=0,
    test_mode=False,
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    custom_callback=None,
    max_failures=5,
    **kwargs
):
    ray.init(
        num_gpus=num_gpus,
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
    )
    used_config = {
        "callbacks": custom_callback if custom_callback else TrajectoryPlanningCallbacks,  # Must Have!
        "log_level": "DEBUG" if test_mode else "WARN",
    }
    used_config.update(config)
    config = copy.deepcopy(used_config)

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns=metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    kwargs["progress_reporter"] = progress_reporter

    if "verbose" not in kwargs:
        # kwargs["verbose"] = 1 if not test_mode else 2
        kwargs["verbose"] = 2

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=".",
        **kwargs
    )
    return analysis


def get_train_parser():
    """
    命令行参数解析器，完全仿照原始结构
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="trajectory_planning_ppo")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-cpus-per-worker", type=float, default=0.5)
    parser.add_argument("--num-cpus-for-driver", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with fewer timesteps")
    parser.add_argument("--local-mode", action="store_true", help="Run Ray in local mode for debugging")
    return parser


if __name__ == "__main__":
    # 解析命令行参数
    args = get_train_parser().parse_args()
    
    # 配置PPO算法，完全仿照原始结构
    config = {
        # 环境配置
        "env": createGymWrapper(TrajectoryPlanningEnv),
        "env_config": {
            # 轨迹规划环境的特定配置
            "use_render": False,
            "manual_control": False,
            "traffic_density": 0.1,
            "accident_prob": 0.0,
            "map": "S",  # 使用简单地图
            "start_seed": 0,
            "random_traffic": False,
            "random_lane_width": False,
            "random_lane_num": False,
            "driving_reward": 1.0,
            "speed_reward": 0.1,
            "use_lateral_reward": False,
            "crash_vehicle_penalty": 40,
            "crash_object_penalty": 40,
            "out_of_road_penalty": 40,
            "success_reward": 10,
            # 轨迹规划特定参数
            "alpha_2": 3.0,
            "v_min": 5.0,
            "v_max": 15.0,
            "T_min": 3.0,
            "T_max": 8.0,
            "trajectory_dt": 0.1,
            "trajectory_points": 50,
        },
        
        # 框架配置
        "framework": "torch",
        
        # 评估配置
        "evaluation_interval": 10,
        "evaluation_duration": 10,
        "evaluation_config": {
            "env_config": {
                "start_seed": 1000,
            }
        },
        
        # PPO算法配置
        "lr": 3e-4,
        "gamma": 0.99,
        "lambda": 0.95,
        "kl_coeff": 0.2,
        "clip_rewards": True,
        "clip_param": 0.2,
        "vf_clip_param": 10.0,
        "entropy_coeff": 0.01,
        "train_batch_size": 4000,
        "sgd_minibatch_size": 128,
        "num_sgd_iter": 10,
        "batch_mode": "truncate_episodes",
        
        # 资源配置
        "num_gpus": args.num_gpus,
        "num_workers": args.num_workers,
        "num_cpus_per_worker": args.num_cpus_per_worker,
        "num_cpus_for_driver": args.num_cpus_for_driver,
        "num_envs_per_worker": 1,
        
        # 回调函数
        "callbacks": TrajectoryPlanningCallbacks,
        
        # 模型配置
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
    }
    
    # 更新配置中的timesteps
    config["timesteps"] = args.timesteps
    
    # 开始训练
    print(f"开始训练轨迹规划环境...")
    print(f"实验名称: {args.exp_name}")
    print(f"GPU数量: {args.num_gpus}")
    print(f"Worker数量: {args.num_workers}")
    print(f"总时间步数: {args.timesteps}")
    print(f"测试模式: {args.test_mode}")
    
    train(
        "PPO",
        exp_name=args.exp_name,
        keep_checkpoints_num=5,
        stop=args.timesteps,
        config=config,
        num_gpus=args.num_gpus,
        test_mode=args.test_mode
    )
    
    print("训练完成！")