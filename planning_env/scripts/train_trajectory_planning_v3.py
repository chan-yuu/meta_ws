"""
This script demonstrates how to train a set of policies under different number of training scenarios and test them
in the same test set using rllib.

We verified this script with ray==2.2.0. Please report to use if you find newer version of ray is not compatible with
this script. Installation guide:

    pip install ray[rllib]==2.2.0
    pip install tensorflow_probability==0.24.0
    pip install torch

"""
import argparse
import copy
import logging
from typing import Dict

import numpy as np

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
    raise ValueError("Please install ray through 'pip install ray'.")


class DrivingCallbacks(DefaultCallbacks):
    def on_episode_start(
        self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        env_index: int, **kwargs
    ):
        episode.user_data["velocity"] = []
        episode.user_data["steering"] = []
        episode.user_data["step_reward"] = []
        episode.user_data["acceleration"] = []
        episode.user_data["cost"] = []

    def on_episode_step(
        self, *, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, env_index: int, **kwargs
    ):
        info = episode.last_info_for()
        if info is not None:
            episode.user_data["velocity"].append(info["velocity"])
            episode.user_data["steering"].append(info["steering"])
            episode.user_data["step_reward"].append(info["step_reward"])
            episode.user_data["acceleration"].append(info["acceleration"])
            episode.user_data["cost"].append(info["cost"])

    def on_episode_end(
        self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode,
        **kwargs
    ):
        arrive_dest = episode.last_info_for()["arrive_dest"]
        crash = episode.last_info_for()["crash"]
        out_of_road = episode.last_info_for()["out_of_road"]
        max_step_rate = not (arrive_dest or crash or out_of_road)
        episode.custom_metrics["success_rate"] = float(arrive_dest)
        episode.custom_metrics["crash_rate"] = float(crash)
        episode.custom_metrics["out_of_road_rate"] = float(out_of_road)
        episode.custom_metrics["max_step_rate"] = float(max_step_rate)
        episode.custom_metrics["velocity_max"] = float(np.max(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_mean"] = float(np.mean(episode.user_data["velocity"]))
        episode.custom_metrics["velocity_min"] = float(np.min(episode.user_data["velocity"]))
        episode.custom_metrics["steering_max"] = float(np.max(episode.user_data["steering"]))
        episode.custom_metrics["steering_mean"] = float(np.mean(episode.user_data["steering"]))
        episode.custom_metrics["steering_min"] = float(np.min(episode.user_data["steering"]))
        episode.custom_metrics["acceleration_min"] = float(np.min(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_mean"] = float(np.mean(episode.user_data["acceleration"]))
        episode.custom_metrics["acceleration_max"] = float(np.max(episode.user_data["acceleration"]))
        episode.custom_metrics["step_reward_max"] = float(np.max(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_mean"] = float(np.mean(episode.user_data["step_reward"]))
        episode.custom_metrics["step_reward_min"] = float(np.min(episode.user_data["step_reward"]))
        episode.custom_metrics["cost"] = float(sum(episode.user_data["cost"]))

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        result["success"] = np.nan
        result["crash"] = np.nan
        result["out"] = np.nan
        result["max_step"] = np.nan
        result["length"] = result["episode_len_mean"]
        result["cost"] = np.nan
        if "custom_metrics" not in result:              
            return

        if "success_rate_mean" in result["custom_metrics"]:
            result["success"] = result["custom_metrics"]["success_rate_mean"]
            result["crash"] = result["custom_metrics"]["crash_rate_mean"]
            result["out"] = result["custom_metrics"]["out_of_road_rate_mean"]
            result["max_step"] = result["custom_metrics"]["max_step_rate_mean"]
        if "cost_mean" in result["custom_metrics"]:
            result["cost"] = result["custom_metrics"]["cost_mean"]


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
        "callbacks": custom_callback if custom_callback else DrivingCallbacks,  # Must Have!
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
        kwargs["verbose"] = 1 if not test_mode else 2
        # kwargs["verbose"] = 2
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="trajectory_experiment")
    parser.add_argument("--num-gpus", type=int, default=0)
    return parser


if __name__ == '__main__':
    args = get_train_parser().parse_args()
    exp_name = args.exp_name
    stop = int(100_0000)
    config = dict(

        # ===== 训练环境配置 =====
        # 为轨迹规划环境训练策略
        env=createGymWrapper(TrajectoryPlanningEnv),  # 使用Gym包装器包装轨迹规划环境
        env_config=dict(
            # 轨迹规划特定参数
            alpha_2=3.5,                    # 轨迹规划中的权重参数，影响轨迹平滑度
            v_min=-0.1,                     # 最小速度限制 (m/s)，允许轻微倒车
            v_max=15.0,                     # 最大速度限制 (m/s)，约54km/h
            T_min=2.0,                      # 最小规划时间范围 (秒)
            T_max=8.0,                      # 最大规划时间范围 (秒)
            trajectory_dt=0.1,              # 轨迹时间步长 (秒)，影响轨迹精度
            trajectory_points=50,           # 轨迹点数量，决定轨迹分辨率
            # MetaDrive基础环境参数
            num_scenarios=tune.grid_search([1]),  # , 3 训练场景数量，网格搜索1和3个场景
            start_seed=tune.grid_search([5000]),     # 随机种子，确保实验可重复性
            random_traffic=False,           # 是否启用随机交通，False表示固定交通模式
            traffic_density=tune.grid_search([0.1]), #, 0.3           # 交通密度 (0.0-1.0)，0表示无其他车辆
            use_render=False,               # 是否启用可视化渲染，训练时通常关闭以提高速度
            # 地图配置
            map="C"                           # 使用地图编号1
        ),

        # ===== 深度学习框架 =====
        framework="torch",               # 使用PyTorch作为深度学习后端

        # ===== 评估配置 =====
        # 在未见过的场景中评估训练好的策略
        evaluation_interval=2,            # 每2个训练迭代进行一次评估
        evaluation_num_episodes=40,       # 每次评估运行40个回合
        metrics_smoothing_episodes=200,   # 指标平滑化的回合数，用于减少噪声
        evaluation_config=dict(env_config=dict(
            # 评估环境配置（与训练环境类似但参数可能不同）
            alpha_2=3.5,                  # 评估时的轨迹规划权重参数
            v_min=-0.1,                   # 评估时的最小速度限制
            v_max=15.0,                   # 评估时的最大速度限制
            T_min=2.0,                    # 评估时的最小规划时间
            T_max=8.0,                    # 评估时的最大规划时间
            trajectory_dt=0.1,            # 评估时的轨迹时间步长
            trajectory_points=50,         # 评估时的轨迹点数量
            num_scenarios=1,              # 评估使用1个场景（固定）
            start_seed=42,                # 评估的固定随机种子，确保评估一致性
            use_render=False,             # 评估时不启用渲染
            # 评估地图配置
            map="C"                         # 评估使用地图编号1
        )),
        evaluation_num_workers=1,         # 评估时使用的并行工作进程数

        # ===== PPO训练超参数 =====
        # PPO (Proximal Policy Optimization) 算法的超参数
        horizon=1000,                     # 每个rollout的最大步数，影响经验收集长度
        rollout_fragment_length="auto",      # 每个rollout片段的长度，影响并行化效率
        sgd_minibatch_size=256,          # SGD小批次大小，影响梯度更新的稳定性
        train_batch_size=4096,          # 训练批次大小，每次更新使用的总样本数
        num_sgd_iter=10,                 # 每个训练批次的SGD迭代次数
        lr=3e-4,                         # 学习率，控制参数更新步长
        num_workers=1,                   # 并行采样的工作进程数，影响数据收集速度
        **{"lambda": 0.95},             # GAE (Generalized Advantage Estimation) 的λ参数，平衡偏差和方差

        # ===== 计算资源分配 =====
        num_gpus=0.7 if args.num_gpus != 0 else 0,  # GPU使用比例，0.7表示使用70%的GPU资源
        num_cpus_per_worker=0.2,         # 每个工作进程分配的CPU核心数
        num_cpus_for_driver=0.5,         # 主驱动进程分配的CPU核心数
    )

    train(
        "PPO",
        exp_name=exp_name,
        keep_checkpoints_num=5,
        stop=stop,
        config=config,
        num_gpus=args.num_gpus,
        test_mode=False
    )
