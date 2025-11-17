#!/usr/bin/env python3
"""
轨迹规划环境训练
"""

import argparse
import copy
import logging
from typing import Dict

import numpy as np

# 导入轨迹规划环境
from trajectory_planning_env import TrajectoryPlanningEnv
from metadrive.envs.gym_wrapper import createGymWrapper

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.env import BaseEnv
    from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
    from ray.rllib.policy import Policy
except ImportError:
    ray = None
    raise ValueError("请通过 'pip install ray[rllib]==2.2.0' 安装ray")


class TrajectoryPlanningCallbacks(DefaultCallbacks):
    """轨迹规划环境的自定义回调函数"""
    
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], 
        episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        """episode开始时初始化记录变量"""
        # 基础指标
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []
        
        # 轨迹规划相关指标
        episode.user_data["trajectory_length"] = []
        episode.user_data["planning_time_horizon"] = []
        episode.user_data["target_velocity"] = []
        episode.user_data["lateral_offset"] = []
        episode.user_data["tracking_error"] = []

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, 
        env_index: int, **kwargs
    ):
        """每步记录相关指标"""
        info = episode.last_info_for()
        if info is not None:
            # 基础指标
            episode.user_data["velocity"].append(info.get("velocity", 0.0))
            episode.user_data["steering"].append(info.get("steering", 0.0))
            episode.user_data["step_reward"].append(info.get("step_reward", 0.0))
            episode.user_data["acceleration"].append(info.get("acceleration", 0.0))
            episode.user_data["cost"].append(info.get("cost", 0.0))
            
            # 轨迹规划相关指标
            episode.user_data["trajectory_length"].append(info.get("trajectory_length", 0))
            episode.user_data["planning_time_horizon"].append(info.get("planning_time_horizon", 0.0))
            episode.user_data["target_velocity"].append(info.get("target_velocity", 0.0))
            episode.user_data["lateral_offset"].append(info.get("lateral_offset", 0.0))
            episode.user_data["tracking_error"].append(info.get("tracking_error", 0.0))

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], 
        episode: MultiAgentEpisode, **kwargs
    ):
        """episode结束时计算统计指标"""
        # 获取终止状态
        info = episode.last_info_for()
        arrive_dest = info.get("arrive_dest", False)
        crash = info.get("crash", False)
        out_of_road = info.get("out_of_road", False)
        max_step_rate = not (arrive_dest or crash or out_of_road)
        
        # 基础成功率指标
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        
        # 速度相关指标
        if episode.user_data["velocity"]:
            episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
            episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
            episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        
        # 控制相关指标
        if episode.user_data["steering"]:
            episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
            episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
            episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        
        if episode.user_data["acceleration"]:
            episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
            episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
            episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        
        # 奖励相关指标
        if episode.user_data["step_reward"]:
            episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
            episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
            episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        
        # 成本指标
        if episode.user_data["cost"]:
            episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))
        
        # 轨迹规划相关指标
        if episode.user_data["trajectory_length"]:
            episode.custom_metrics["trajectory_length_mean"] = float(np.mean(episode.user_data["trajectory_length"]))
        
        if episode.user_data["planning_time_horizon"]:
            episode.custom_metrics["planning_time_horizon_mean"] = float(np.mean(episode.user_data["planning_time_horizon"]))
        
        if episode.user_data["target_velocity"]:
            episode.custom_metrics["target_velocity_mean"] = float(np.mean(episode.user_data["target_velocity"]))
        
        if episode.user_data["lateral_offset"]:
            episode.custom_metrics["lateral_offset_mean"] = float(np.mean(episode.user_data["lateral_offset"]))
            episode.custom_metrics["lateral_offset_std"] = float(np.std(episode.user_data["lateral_offset"]))
        
        if episode.user_data["tracking_error"]:
            episode.custom_metrics["tracking_error_mean"] = float(np.mean(episode.user_data["tracking_error"]))
            episode.custom_metrics["tracking_error_std"] = float(np.std(episode.user_data["tracking_error"]))

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """训练结果处理"""
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        result["tracking_error"] = np.nan
        
        if "custom_metrics" not in result:
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]
        
        if "tracking_error_mean_mean" in result["custom_metrics"]:
            result["tracking_error"] = result["custom_metrics"]["tracking_error_mean_mean"]


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
    """训练函数"""
    ray.init(
        num_gpus=num_gpus,
        logging_level=logging.ERROR if not test_mode else logging.DEBUG,
        log_to_driver=test_mode,
    )
    
    used_config = {
        "callbacks": custom_callback if custom_callback else TrajectoryPlanningCallbacks,
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

    # 设置进度报告器 - 使用Ray Tune标准指标
    progress_reporter = CLIReporter(
        metric_columns=[
            "trial_id",
            "status", 
            "timesteps_total",  # 当前训练步数
            "episode_reward_mean",  # 平均奖励
            "episode_len_mean",  # 平均episode长度
            "time_this_iter_s",  # 每次迭代时间
            "training_iteration",  # 训练迭代次数
        ],
        max_progress_rows=10,  # 减少显示行数
        max_error_rows=3,
        max_column_length=12,  # 减少列宽度
        print_intermediate_tables=True
    )
    # 添加自定义指标列（如果存在）
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    progress_reporter.add_metric_column("tracking_error")
    kwargs["progress_reporter"] = progress_reporter

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # 开始训练
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir="./ray_results",
        **kwargs
    )
    return analysis


def get_train_parser():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(description="轨迹规划环境训练脚本")
    parser.add_argument("--exp-name", type=str, default="trajectory_planning_experiment", 
                       help="实验名称")
    parser.add_argument("--num-gpus", type=int, default=1, 
                       help="使用的GPU数量")
    parser.add_argument("--num-workers", type=int, default=2, 
                       help="并行worker数量")
    parser.add_argument("--timesteps", type=int, default=1000000, 
                       help="总训练步数")
    parser.add_argument("--test-mode", action="store_true", 
                       help="测试模式")
    parser.add_argument("--checkpoint-freq", type=int, default=10, 
                       help="检查点保存频率")
    parser.add_argument("--evaluation-interval", type=int, default=5, 
                       help="评估间隔")
    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    
    exp_name = args.exp_name
    stop = int(args.timesteps)
    
    # 训练配置
    config = dict(
        # ===== 环境配置 =====
        env=createGymWrapper(TrajectoryPlanningEnv),
        env_config=dict(
            # 轨迹规划参数
            alpha_2=3.5,
            v_min=-0.1,
            v_max=15.0,
            T_min=2.0,
            T_max=8.0,
            
            # MetaDrive环境参数
            map=1,  # 使用地图4
            traffic_density=0.0,
            random_traffic=False,
            use_render=False,  # 训练时关闭渲染
            
            # 轨迹生成参数
            trajectory_dt=0.1,
            trajectory_points=40,
            use_extended_reference_line=True,
        ),

        # ===== 框架配置 =====
        framework="torch",

        # ===== 评估配置 =====
        evaluation_interval=args.evaluation_interval,
        evaluation_num_episodes=20,
        metrics_smoothing_episodes=100,
        evaluation_config=dict(
            env_config=dict(
                map=tune.choice([1]),  # , 2, 3, 4, 5 评估时使用不同地图
                traffic_density=0.0,
                use_render=False,
            )
        ),
        evaluation_num_workers=2,

        # ===== PPO算法参数 =====
        # 适配3维动作空间的PPO配置
        horizon=1000,  # episode最大长度
        rollout_fragment_length=200,  # rollout片段长度
        sgd_minibatch_size=128,  # SGD mini-batch大小
        train_batch_size=4000,  # 训练batch大小
        num_sgd_iter=10,  # SGD迭代次数
        lr=3e-4,  # 学习率
        
        # PPO特定参数
        **{"lambda": 0.95},  # GAE lambda
        gamma=0.99,  # 折扣因子
        clip_param=0.2,  # PPO clip参数
        vf_clip_param=10.0,  # 价值函数clip参数
        entropy_coeff=0.01,  # 熵系数
        
        # 网络架构
        model=dict(
            fcnet_hiddens=[256, 256],  # 全连接层隐藏单元
            fcnet_activation="relu",   # 激活函数
        ),

        # ===== 资源配置 =====
        num_workers=min(args.num_workers, 2),  # 限制最大worker数量为2
        num_gpus=0.5 if args.num_gpus != 0 else 0,  # 减少GPU使用
        num_cpus_per_worker=0.25,  # 减少每个worker的CPU使用
        num_cpus_for_driver=0.5,   # 减少driver的CPU使用
    )
    
    print(f"开始训练轨迹规划环境...")
    print(f"实验名称: {exp_name}")
    print(f"总训练步数: {stop}")
    print(f"使用GPU数量: {args.num_gpus}")
    print(f"并行worker数量: {args.num_workers}")
    
    # 开始训练
    analysis = train(
        "PPO",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        test_mode=args.test_mode,
        checkpoint_freq=args.checkpoint_freq
    )
    
    print("训练完成!")
    print(f"最佳检查点: {analysis.best_checkpoint}")
    print(f"最佳奖励: {analysis.best_result['episode_reward_mean']}")