import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import truncated_exponential


class SequentialReachingEnv(gym.Env):
        
    def __init__(self,
                 plant,
                 target_duration, 
                 num_targets, 
                 action_range=(0.0, 1.0),
                 reward_params=None,
                 discrete_ctrl=False,
                 target_based_ctrl=False, 
                 low_lv_ctrl=None,
                 render_mode='human'):
        
        super().__init__()
        self.plant = plant
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.low_lv_ctrl = low_lv_ctrl
        self.discrete_ctrl = discrete_ctrl
        self.target_based_ctrl = target_based_ctrl
        self.min_action = action_range[0]
        self.max_action = action_range[1]
        self.reward_params = reward_params if reward_params is not None else {}
        self.render_mode = render_mode

        if low_lv_ctrl is not None:
            if self.discrete_ctrl:
                self.action_space = spaces.Discrete(low_lv_ctrl.num_units)
            else:
                self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(low_lv_ctrl.num_units,), dtype=np.float32)
            self.low_lv_ctrl.get_max_alpha_activation(max_input=self.max_action, discrete=self.discrete_ctrl)
            # Debugging
            #print(low_lv_ctrl.max_alpha_activation)
        else:
            self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(plant.num_actuators,), dtype=np.float32)

        obs_dim = plant.num_sensors + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._setup_targets()

    def _setup_targets(self):
        td = self.target_duration
        if isinstance(td, dict):
            self.target_durations = truncated_exponential(
                mu=td["mean"],
                a=td["min"],
                b=td["max"],
                size=self.num_targets
            )
        else:
            self.target_durations = np.full(self.num_targets, float(td))
        self.target_offset_times = self.target_durations.cumsum()
        self.current_target_idx = 0
        self.plant.reset()
        self.target_positions = self.plant.sample_targets(self.num_targets)
        self.plant.update_target(self.target_positions[0])

    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._setup_targets()
        if self.low_lv_ctrl is not None:
            self.low_lv_ctrl.reset_state()  
        context, feedback = self.plant.get_obs()
        return np.concatenate([context, feedback]), {}

    def step(self, action):
        if self.low_lv_ctrl is not None:
            _, feedback = self.plant.get_obs()
            feedback = feedback[:self.plant.num_actuators*2]

            if self.discrete_ctrl:
                primitives = np.zeros(self.low_lv_ctrl.num_units)
                primitives[int(action)] = self.max_action
                muscle_activations = self.low_lv_ctrl.step(primitives, feedback)
            else:
                muscle_activations = self.low_lv_ctrl.step(action, feedback)
        else:
            muscle_activations = action

        self.plant.step(muscle_activations)

        context, feedback = self.plant.get_obs()
        obs = np.concatenate([context, feedback])

        reward, hit_target = self.compute_reward(
            self.plant.get_hand_pos(),
            self.target_positions[self.current_target_idx],
            action,
            **self.reward_params
        )

        time_exceeded = self.plant.data.time > self.target_offset_times[self.current_target_idx]
        if hit_target or time_exceeded:
            self.current_target_idx += 1
            if self.current_target_idx < self.num_targets:
                self.plant.update_target(self.target_positions[self.current_target_idx])

        terminated = self.current_target_idx >= self.num_targets
        truncated = False

        info = {"hit_target": hit_target}

        return obs, reward, terminated, truncated, info
            
    def compute_reward(self, hand_position, target_position, activation, 
                   dist_threshold=0.04, bonus=0.0, l0_threshold=0.01,
                   l0_weight=0.0, l1_weight=0.0):
        # Euclidean distance
        euclidean_distance = np.linalg.norm(target_position - hand_position)
        reward = -euclidean_distance

        # L0 regularization
        l0_penalty = l0_weight * np.sum(np.abs(activation) > l0_threshold)
        reward -= l0_penalty

        # L1 regularization
        l1_penalty = l1_weight * np.sum(np.abs(activation))
        reward -= l1_penalty

        # Bonus for hitting target
        hit_target = euclidean_distance <= dist_threshold
        if hit_target:
            reward += bonus

        return reward, hit_target

    def render(self, mode=None, fps=30):
        if mode is None:
            mode = self.render_mode

        if mode == "human":
            self.plant.render(mode="human", fps=fps)

        elif mode == "rgb_array":
            pass
        
        else:
            raise ValueError(f"Unknown render mode: {mode}")