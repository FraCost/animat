import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from plants import SequentialReacher
from networks import LowLevelController
from environments import SequentialReachingEnv
from loggers import EpisodeLoggerCallback
from environment_factory import make_env


def main(mode, 
         base_path, 
         timesteps=1_000_000, 
         checkpoint_freq=250_000, 
         print_freq=250, 
         save_freq=1000, 
         num_envs=4):
    
    # ---------------------------------------------
    # Initialize parallel environments
    # ---------------------------------------------
    plant_config = {"class": SequentialReacher, "kwargs": {"plant_xml_file": "arm.xml"}}
    llctrl_config = {"class": LowLevelController, "kwargs": {"num_units": 25, "weights_inhibit": 0.2, "k_l": 0.05, "k_v": 0.025, "k_g": 1.0}}
    env_config = {"class": SequentialReachingEnv, "kwargs": {"target_duration": {"mean": 3.0, "min": 1.0, "max": 6.0}, "num_targets": 1, "reward_params": {"l0_weight": 0.01, "bonus": 1.0}}}

    vec_env = SubprocVecEnv([
        make_env(mode, plant_config, env_config, llctrl_config, seed=i) for i in range(num_envs)
    ])
    vec_env = VecMonitor(vec_env)

    # ---------------------------------------------
    # PPO model setup
    # ---------------------------------------------
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        gamma=0.99,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=8,
        ent_coef=0.01,
        policy_kwargs = dict(net_arch=[256, 256]),
        tensorboard_log=os.path.join(base_path, f"tensorboard_ppo_{mode}")
    )

    # ---------------------------------------------
    # Checkpoint & logging
    # ---------------------------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=os.path.join(base_path, "checkpoints"),
        name_prefix=f"ppo_arm_{mode}"
    )
        
    episode_logger = EpisodeLoggerCallback(
        print_freq=print_freq,
        save_freq=save_freq,
        log_actions=(mode == "discrete_llctrl"),
        save_path=os.path.join(os.path.join(base_path, "logs"), f"episode_stats_{mode}.npz")
    )

    # ---------------------------------------------
    # Train controller
    # ---------------------------------------------
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, episode_logger]
    )


if __name__ == "__main__":
    #mode = "continuous_llctrl"
    #mode = "discrete_llctrl"
    #mode = "direct"
    mode = "target_llctrl"
    base_path = "/Users/teachinglab/Documents/code/paton_lab/animat/src/mujoco_rnn_cmaes_mj/hand-wired-net/models"
    main(mode, base_path)
