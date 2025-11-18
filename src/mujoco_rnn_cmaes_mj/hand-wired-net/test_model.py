import os
import gymnasium as gym
from stable_baselines3 import PPO
from plants import SequentialReacher
from networks import LowLevelController
from environments import SequentialReachingEnv
from environment_factory import make_env


def test_model(model_path, env, mode, num_episodes=10):

    # Load trained PPO model
    model = PPO.load(model_path, env=env)

    for ep in range(num_episodes):
        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
            info = {}

        done = False
        frames = []

        while not done:
            frame = env.render(mode="human")

            # Model prediction
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
    

if __name__ == "__main__":

    # ----------------------------
    #  Environment Configurations
    # ----------------------------
    plant_config = {
        "class": SequentialReacher,
        "kwargs": {"plant_xml_file": "arm.xml"}
    }

    llctrl_config = {
        "class": LowLevelController,
        "kwargs": {
            "num_units": 25,
            "weights_inhibit": 0.2,
            "k_l": 0.05,
            "k_v": 0.025,
            "k_g": 1.0
        }
    }

    env_config = {
        "class": SequentialReachingEnv,
        "kwargs": {
            "target_duration": {"mean": 3.0, "min": 1.0, "max": 6.0},
            "num_targets": 1,
            "reward_params": {"l0_weight": 0.02, "bonus": 1.0},
        }
    }

    # Create environment
    mode = "discrete_llctrl"
    env = make_env(mode, plant_config, env_config, llctrl_config, seed=23)()

    # Path to trained model
    model_path = (
        "/Users/teachinglab/Documents/code/paton_lab/animat/src/"
        "mujoco_rnn_cmaes_mj/hand-wired-net/models/checkpoints/"
        f"ppo_arm_{mode}_1000000_steps.zip"
    )

    # Run testing and record video
    test_model(model_path, env, mode, num_episodes=20)
