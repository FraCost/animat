import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import truncated_exponential

class SequentialReachingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, plant, target_duration, num_targets, low_lv_ctrl=None, render_mode=None):
        super().__init__()
        self.plant = plant
        self.low_lv_ctrl = low_lv_ctrl
        self.render_mode = render_mode
        self.num_targets = num_targets
        self.target_duration = target_duration

        # Observation space: normalized target + sensors
        obs_dim = 3 + self.plant.num_sensors
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action space: either low-level controller units or direct muscle activations
        self.action_dim = low_lv_ctrl.num_units if low_lv_ctrl else self.plant.num_actuators
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

        self.current_target_idx = 0
        self.target_positions = None
        self.target_offset_times = None
        self.trial_duration = 0.0
        self.elapsed_time = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.plant.reset()
        self.current_target_idx = 0

        # Sample targets
        self.target_positions = self.plant.sample_targets(self.num_targets)
        durations = truncated_exponential(
            mu=self.target_duration["mean"],
            a=self.target_duration["min"],
            b=self.target_duration["max"],
            size=self.num_targets
        )
        self.target_offset_times = durations.cumsum()
        self.trial_duration = durations.sum()
        self.elapsed_time = 0.0

        self.plant.update_target(self.target_positions[0])

        context, feedback = self.plant.get_obs()
        obs = np.concatenate([context, feedback]).astype(np.float32)
        return obs, {}

    def step(self, action):
        # Compute muscle activations
        if self.low_lv_ctrl:
            context, feedback = self.plant.get_obs()
            activation = self.low_lv_ctrl.step(action, feedback[:self.plant.num_actuators])
        else:
            activation = action

        self.plant.step(activation)

        if self.render_mode == "human":
            self.plant.render()

        context, feedback = self.plant.get_obs()
        obs = np.concatenate([context, feedback]).astype(np.float32)

        # Compute reward: negative Euclidean distance
        hand_pos = self.plant.get_hand_pos()
        target_pos = self.target_positions[self.current_target_idx]
        reward = -np.linalg.norm(hand_pos - target_pos)

        self.elapsed_time += self.plant.model.opt.timestep
        done = False
        if self.elapsed_time > self.target_offset_times[self.current_target_idx]:
            self.current_target_idx += 1
            if self.current_target_idx >= self.num_targets:
                done = True
            else:
                self.plant.update_target(self.target_positions[self.current_target_idx])

        return obs, reward, done, False, {}
