import pickle
import math
import numpy as np
from utils import *


class SequentialReachingEnv:
    # -----------------------------------------
    # Initialization
    # -----------------------------------------
    def __init__(
        self, 
        plant, 
        target_duration, 
        num_targets, 
        loss_weights, 
        low_lv_ctrl=None
        ):
        
        self.plant = plant
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.loss_weights = loss_weights
        self.logger = None
        self.low_lv_ctrl = low_lv_ctrl
        
    # -----------------------------------------
    # Logging
    # -----------------------------------------
    def log(self, time, sensors, target_position, hand_position,
            manhattan_distance, euclidean_distance, energy, entropy, reward, fitness):
        if self.logger is None:
            self.logger = {}
            self.logger["time"] = []
            self.logger["sensors"] = {k: [] for k in sensors.keys()}
            self.logger["target_position"] = []
            self.logger["hand_position"] = []
            self.logger["manhattan_distance"] = []
            self.logger["euclidean_distance"] = []
            self.logger["energy"] = []
            self.logger["entropy"] = []
            self.logger["reward"] = []
            self.logger["fitness"] = []

        self.logger["time"].append(time)
        for k in sensors.keys():
            self.logger["sensors"][k].append(sensors[k])
        self.logger["target_position"].append(target_position)
        self.logger["hand_position"].append(hand_position)
        self.logger["manhattan_distance"].append(manhattan_distance)
        self.logger["euclidean_distance"].append(euclidean_distance)
        self.logger["energy"].append(energy)
        self.logger["entropy"].append(entropy)
        self.logger["reward"].append(reward)
        self.logger["fitness"].append(fitness)

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------
    def evaluate(self, rnn, seed=0, render=False, log=False):
        np.random.seed(seed)
        rnn.init_state()
        antagonists = [(2*i, 2*i + 1) for i in range(self.plant.num_actuators // 2)]
        self.plant.reset()

        target_positions = self.plant.sample_targets(self.num_targets)
        target_durations = truncated_exponential(
            mu=self.target_duration["mean"],
            a=self.target_duration["min"],
            b=self.target_duration["max"],
            size=self.num_targets
        )
        target_offset_times = target_durations.cumsum()
        trial_duration = target_durations.sum()
        total_reward = 0
        target_idx = 0

        self.plant.update_target(target_positions[target_idx])
        hand_position = self.plant.get_hand_pos()

        while target_idx < self.num_targets:
            if render:
                self.plant.render()

            # Get observation
            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])                

            # Send control signal to MuJoCo
            if self.low_lv_ctrl is not None:
                rnn_output = rnn.step(obs)
                activation = self.low_lv_ctrl.step(rnn_output, feedback[:self.num_actuators])
            else:
                activation = rnn.step(obs)
            self.plant.step(activation)

            # Compute loss
            previous_hand_position = hand_position
            hand_position = self.plant.get_hand_pos()
            target_position = target_positions[target_idx]
            manhattan_distance = l1_norm(target_position - hand_position)
            euclidean_distance = l2_norm(target_position - hand_position)
            energy = np.mean(np.abs(activation))
            entropy = action_entropy(activation)

            reward = -(
                euclidean_distance * self.loss_weights["euclidean"]
                + manhattan_distance * self.loss_weights["manhattan"]
                + energy * entropy * self.loss_weights["energy"]
                + l1_norm(rnn.get_params() * self.loss_weights["ridge"])
                + l2_norm(rnn.get_params() * self.loss_weights["lasso"])
            )

            total_reward += reward

            # Log data
            if log:
                sensors_dict = {}
                n = self.plant.num_actuators
                for i, name in enumerate(self.plant.actuator_names):
                    sensors_dict[f"{name}_len"] = feedback[i]
                    sensors_dict[f"{name}_vel"] = feedback[n + i]
                    sensors_dict[f"{name}_frc"] = feedback[2*n + i]
                    
                self.log(
                    time=self.plant.data.time,
                    sensors=sensors_dict,
                    target_position=target_position,
                    hand_position=hand_position,
                    manhattan_distance=manhattan_distance,
                    euclidean_distance=euclidean_distance,
                    energy=energy,
                    entropy=entropy,
                    reward=reward,
                    fitness=total_reward / trial_duration
                )

            if self.plant.data.time > target_offset_times[target_idx]:
                target_idx += 1
                if target_idx < self.num_targets:
                    self.plant.update_target(target_positions[target_idx])

        self.plant.close()
        return total_reward / trial_duration