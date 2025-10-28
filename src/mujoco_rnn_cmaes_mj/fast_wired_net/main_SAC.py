import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from plants import SequentialReacher
from networks import LowLevelController
from environments_SAC import SequentialReachingEnv
from stable_baselines3.common.callbacks import BaseCallback

class PrintEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_num = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]  # infos from rollout
        for info in infos:
            if "episode" in info:
                self.episode_num += 1
                print(
                    f"Episode {self.episode_num} finished | "
                    f"Reward: {info['episode']['r']:.2f} | "
                    f"Length: {info['episode']['l']}"
                )
        return True

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # ------------------------
    # 1) Plant
    # ------------------------
    plant = SequentialReacher(plant_xml_file="arm.xml")

    # ------------------------
    # 2) Low-level controller
    # ------------------------
    low_lv_ctrl = LowLevelController(
        num_units=25,
        num_actuators=plant.num_actuators,
        weigths_inhib=0.5
    )

    # ------------------------
    # 3) Gymnasium environment
    # ------------------------
    env = SequentialReachingEnv(
        plant=plant,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        low_lv_ctrl=low_lv_ctrl,
        render_mode=None  # "human" to visualize
    )

    # Wrap environment with Monitor to track episode reward & length
    env = Monitor(env)

    # ------------------------
    # 4) SAC agent
    # ------------------------
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        buffer_size=100000,
        tensorboard_log="./sac_tensorboard/"
    )

    # ------------------------
    # 5) Checkpoint callback
    # ------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="sac_model"
    )

    # ------------------------
    # 6) Train
    # ------------------------
    model.learn(
        total_timesteps=500000,  
        callback=[checkpoint_callback, PrintEpisodeCallback()]
    )

    # ------------------------
    # 7) Evaluation loop with printing
    # ------------------------
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Print per-episode stats
        if "episode" in info:
            ep_reward = info["episode"]["r"]
            ep_length = info["episode"]["l"]
            print(f"Episode finished | Reward: {ep_reward:.2f} | Length: {ep_length}")
