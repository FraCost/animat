import copy
from utils import *
import math


# ----------------------------------------------------------
# Low-level controller
# ----------------------------------------------------------
class LowLevelController:
    def __init__(self, num_units, plant, weights_inhibit=0.2, k_l=0.05, k_v=0.025, k_g=1.0, mode="multi_joint"):
        self.num_units = num_units
        self.plant = plant
        self.num_actuators = self.plant.num_actuators
        self.weights_inhibit = weights_inhibit
        self.mode = mode
        self.weights = self.init_weights(self.mode)
        self.max_alpha_activation = None
        self.dt = self.plant.control_timestep
        self.alpha_state = np.zeros(self.num_actuators)
        self.k_l = k_l
        self.k_v = k_v
        self.k_g = k_g
        self.unit_target_positions = None

    def init_weights(self, mode="single_joint"):
        if self.num_actuators % 2 != 0:
            raise ValueError("Number of actuators must be even")
        
        if self.num_actuators == 2:
            W0 = np.linspace(1.0, 0.0, self.num_units)
            W1 = np.linspace(0.0, 1.0, self.num_units)
            W = np.vstack([W0, W1])
        
        elif self.num_actuators == 4:
            if mode == "multi_joint":
                resolution = math.isqrt(self.num_units)
                if resolution * resolution != self.num_units:
                    raise ValueError(f"num_units ({self.num_units}) must be a perfect square.")
                gradient = np.linspace(1.0, 0.0, resolution)
                W0 = np.repeat(gradient, resolution)
                W1 = np.repeat(gradient[::-1], resolution)
                W2 = np.tile(gradient, resolution)
                W3 = np.tile(gradient[::-1], resolution)
                W = np.vstack([W0, W1, W2, W3])
            
            elif mode == "single_joint":
                half_units = self.num_units // 2
                # First joint: first half of neurons active, second half zero
                grad = np.linspace(1.0, 0.0, half_units)
                W0 = np.concatenate([grad, np.zeros(half_units)])
                W1 = np.concatenate([grad[::-1], np.zeros(half_units)])
                # Second joint: first half zero, second half active
                W2 = np.concatenate([np.zeros(half_units), grad])
                W3 = np.concatenate([np.zeros(half_units), grad[::-1]])
                W = np.vstack([W0, W1, W2, W3])
            
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        return W

    def step(self, activation, feedback, rescale=True, tau=0.03):
        gamma_activation = self.weights @ activation
        
        spindle_lengths = np.array(feedback[:self.num_actuators])
        spindle_velocities = np.array(feedback[self.num_actuators:]) 

        spindle_activation = (self.k_g * gamma_activation) + (self.k_l * spindle_lengths) + (self.k_v * spindle_velocities)
        spindle_activation = np.maximum(0.0, spindle_activation)

        antagonists = [(2 * i, 2 * i + 1) for i in range(self.num_actuators // 2)]
        alpha_raw = np.zeros_like(spindle_activation)
        for f, e in antagonists:
            alpha_raw[f] = max(0.0, spindle_activation[f] - self.weights_inhibit * spindle_activation[e])
            alpha_raw[e] = max(0.0, spindle_activation[e] - self.weights_inhibit * spindle_activation[f])

        # First-order dynamics: τ dα/dt = α_raw - α
        self.alpha_state += (self.dt / tau) * (alpha_raw - self.alpha_state)
        alpha_activation = self.alpha_state

        if rescale and self.max_alpha_activation is not None:
            alpha_activation = np.clip(alpha_activation / self.max_alpha_activation, 0, 1)

        return alpha_activation
        
    def get_max_alpha_activation(self, max_input=1.0, n_samples=100000, seed=23, discrete=False):
        
        if discrete:
            _, max_obs = self.plant.get_feedback_range()
            max_spindle_len = max_obs[:self.num_actuators]
            max_spindle_vel = max_obs[self.num_actuators:self.num_actuators*2]
            max_spindle_activation = (self.k_g * max_input) + (self.k_l * max_spindle_len) + (self.k_v * max_spindle_vel)
            self.max_alpha_activation = max_spindle_activation * (1 - self.weights_inhibit)
        
        else:
            if seed is not None:
                np.random.seed(seed)
            
            max_alpha = np.zeros(self.num_actuators)

            for _ in range(n_samples):
                activation = np.random.uniform(0, max_input, self.num_units) 
                _, feedback = self.plant.get_obs()
                feedback = feedback[:self.num_actuators*2]
                alpha = self.step(activation, feedback, rescale=False)
                max_alpha = np.maximum(max_alpha, alpha)
                self.plant.step(alpha)
                self.reset_state()

            self.max_alpha_activation = max_alpha
            self.plant.reset()

    def reset_state(self):
        self.alpha_state = np.zeros(self.num_actuators)
        
    def get_unit_targets(self, steps=100, activation=1.0):
        unit_targets = np.zeros((self.num_units, 3))

        for u in range(self.num_units):
            self.plant.reset()
            self.reset_state()

            act = np.zeros(self.num_units)
            act[u] = activation

            for _ in range(steps):
                _, feedback = self.plant.get_obs()
                feedback = feedback[:self.num_actuators * 2]

                alpha = self.step(act, feedback)
                self.plant.step(alpha)

            unit_targets[u] = self.plant.get_hand_pos()
        
        normalized_targets = np.zeros_like(unit_targets)
        mean = self.plant.hand_position_stats["mean"].values
        std = self.plant.hand_position_stats["std"].values

        for u in range(self.num_units):
            normalized_targets[u] = zscore(unit_targets[u], mean, std)
            
        self.unit_target_positions = normalized_targets
        return self.unit_target_positions

