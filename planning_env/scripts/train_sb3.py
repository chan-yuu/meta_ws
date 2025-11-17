# from metadrive.envs import MetaDriveEnv
from trajectory_planning_env_no_print import TrajectoryPlanningEnv
from metadrive.policy.lange_change_policy import LaneChangePolicy
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from metadrive.component.map.base_map import BaseMap
from metadrive.utils.doc_utils import generate_gif
from IPython.display import Image
from stable_baselines3.common.callbacks import CheckpointCallback  # 新增回调函数导入
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from IPython.display import clear_output
import os


def create_env(need_monitor=False):
    env = TrajectoryPlanningEnv(dict(
                      map="S",
                      # 此处为连续空间                 
                    #   discrete_action=False,
                    #   horizon=500,
                      # scenario setting
                    #   random_spawn_lane_index=False,
                      num_scenarios=10,
                      start_seed=5,
                    #   accident_prob=0,
                    #   log_level=50,
                      use_render=False,
                      traffic_density=0.0,
                      alpha_2=3.5,
                      v_min=5.0,
                      v_max=15.0,
                      T_min=2.0,
                      T_max=8.0,
                      use_extended_reference_line=True
                      )
                    )
    if need_monitor:
        env = Monitor(env)
    return env

# env=create_env()
# env.reset()
# ret = env.render(mode="topdown", 
#                  window=False,
#                  screen_size=(600, 600), 
#                  camera_position=(50, 50))
# env.close()
# plt.axis("off")
# plt.imshow(ret)
# 
if __name__ == '__main__':

    set_random_seed(0)
    # 4 subprocess to rollout
    train_env=SubprocVecEnv([partial(create_env, True) for _ in range(1)])  # env: 4
    tensorboard_dir = "./sb3_tensorboard_logs"
    model = PPO("MlpPolicy", 
                train_env,
                n_steps=4096,
                verbose=1,
                tensorboard_log=tensorboard_dir
                )
    
    # 创建模型保存目录
    model_save_dir = "./saved_models"
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 设置定期保存回调（每10000步保存一次）
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_dir,
        name_prefix="metadrive_ppo"
    )

    model.learn(total_timesteps=1000 if os.getenv('TEST_DOC') else 100_0000,
                log_interval=1,
                tb_log_name="metadrive_ppo",
                # 进度条
                progress_bar=True,
                )
    
    clear_output()

    from datetime import datetime
    # 保存最终训练好的模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(model_save_dir, f"metadrive_ppo_final_{timestamp}")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    print("Training is finished! Generate gif ...")

    # evaluation
    total_reward = 0
    env=create_env()
    obs, _ = env.reset()
    try:
        for i in range(1000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            ret = env.render(mode="topdown", 
                            screen_record=True,
                            window=False,
                            screen_size=(600, 600), 
                            camera_position=(50, 50))
            if done:
                print("episode_reward", total_reward)
                break
                
        env.top_down_renderer.generate_gif()
    finally:
        env.close()
    print("gif generation is finished ...")