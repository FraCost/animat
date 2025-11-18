from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os


class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, print_freq=100, save_path=None, save_freq=1000, log_actions=False):
        super().__init__()
        self.print_freq = print_freq
        self.save_path = save_path
        self.save_freq = save_freq
        self.log_actions = log_actions

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_success = []
        self.episode_actions = []  # List of lists for actions per episode
        self.current_actions = []  # Temp buffer for current episode actions
        self.episode_count = 0

        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        actions = self.locals.get("actions", None)

        # If discrete_ctrl, log actions
        if self.log_actions and actions is not None:
            # Store as int (assuming actions are discrete)
            self.current_actions.extend([int(a) for a in actions])

        for info in infos:
            if "episode" in info.keys():
                self.episode_count += 1
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                success = info.get("hit_target", None)

                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                self.episode_success.append(success)

                if self.log_actions:
                    # Store current episode actions
                    self.episode_actions.append(self.current_actions.copy())
                    self.current_actions = []  # reset buffer for next episode

                if self.episode_count % self.print_freq == 0:
                    print(
                        f"Episode {self.episode_count}: "
                        f"reward={ep_reward:.3f}, length={ep_length}, success={success}, timesteps={self.num_timesteps}"
                    )

                if self.save_path is not None and self.episode_count % self.save_freq == 0:
                    self._save_logs()

        return True

    def _on_training_end(self) -> None:
        if self.save_path is not None:
            self._save_logs()

    def _save_logs(self):
        save_dict = {
            "rewards": np.array(self.episode_rewards),
            "lengths": np.array(self.episode_lengths),
            "success": np.array(self.episode_success, dtype=object)
        }
        if self.log_actions:
            save_dict["actions"] = np.array(self.episode_actions, dtype=object)

        np.savez(self.save_path, **save_dict)
