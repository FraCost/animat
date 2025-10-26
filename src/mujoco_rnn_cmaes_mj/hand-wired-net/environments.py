import pickle
import math
import matplotlib.pyplot as plt
import numpy as np
from plants import SequentialReacher
from networks import RNN
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
        num_interneurons=25
        ):
        
        self.plant = plant
        self.target_duration = target_duration
        self.num_targets = num_targets
        self.loss_weights = loss_weights
        self.logger = None
        self.num_interneurons = num_interneurons
        self.num_actuators = plant.num_actuators
        self.W_inter_to_gamma = self.init_W_inter_to_gamma()
        self.actuator_names = self.plant.actuator_names
    
    def init_W_inter_to_gamma(self):
        if self.num_actuators % 2 != 0:
            raise ValueError("Number of actuators must be even")
        
        if self.num_actuators == 2:
            W0 = np.linspace(1.0, 0.0, self.num_interneurons)
            W1 = np.linspace(0.0, 1.0, self.num_interneurons)
            W = np.vstack([W0, W1])
        
        elif self.num_actuators == 4:
            resolution = math.isqrt(self.num_interneurons)
            if resolution * resolution != self.num_interneurons:
                raise ValueError(f"num_interneurons ({self.num_interneurons}) must be a perfect square.")
            
            gradient = np.linspace(1.0, 0.0, resolution)
            W0 = np.repeat(gradient, resolution)
            W1 = np.repeat(gradient[::-1], resolution)
            W2 = np.tile(gradient, resolution)
            W3 = np.tile(gradient[::-1], resolution)
            W = np.vstack([W0, W1, W2, W3]) 
                
        return W
        
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

            context, feedback = self.plant.get_obs()
            obs = np.concatenate([context, feedback])                

            # RNN → interneurons
            interneurons = rnn.step(obs)

            # Interneurons → gamma MN offsets
            gamma_offsets = self.W_inter_to_gamma @ interneurons  

            # Compute alpha activations
            spindle_lengths = np.array(feedback[:self.num_actuators])
            alpha_act = spindle_lengths + gamma_offsets 
            
            # Send final control signal to MuJoCo
            self.plant.step(alpha_act)

            previous_hand_position = hand_position
            hand_position = self.plant.get_hand_pos()
            target_position = target_positions[target_idx]
            manhattan_distance = l1_norm(target_position - hand_position)
            euclidean_distance = l2_norm(target_position - hand_position)
            energy = np.mean(np.abs(alpha_act))
            entropy = action_entropy(alpha_act)

            reward = -(
                euclidean_distance * self.loss_weights["euclidean"]
                + manhattan_distance * self.loss_weights["manhattan"]
                + energy * entropy * self.loss_weights["energy"]
                + l1_norm(rnn.get_params() * self.loss_weights["ridge"])
                + l2_norm(rnn.get_params() * self.loss_weights["lasso"])
            )

            total_reward += reward

            if log:
                sensors_dict = {}
                n = self.num_actuators
                for i, name in enumerate(self.actuator_names):
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

    # -----------------------------------------
    # Stimulation 
    # -----------------------------------------
    def stimulate(self, units, delay=1, seed=0, render=False):
        np.random.seed(seed)
        self.plant.reset()

        if render:
            self.plant.render()

        # Turn on the "nail"
        self.plant.model.eq_active0 = 1

        force_data = {"position": [], "force": []}
        total_delay = 0

        grid_positions = np.array(self.plant.grid_positions.copy())
        grid_pos_idx = 0
        self.plant.update_nail(grid_positions[grid_pos_idx])

        # Dummy target, mainly for plant mechanics
        target_position = self.plant.sample_targets(1)
        self.plant.update_target(target_position)

        while grid_pos_idx < len(grid_positions) - 1:
            if render:
                self.plant.render()

            _, feedback = self.plant.get_obs()
            spindle_lengths = np.array(feedback[:self.num_actuators])

            # Directly stimulate specified interneurons
            interneurons = np.zeros(self.num_interneurons)
            if self.plant.data.time > total_delay - delay / 2:
                interneurons[units] = 1.0  

            # Interneurons → gamma offsets → alpha activations
            gamma_offsets = self.W_inter_to_gamma @ interneurons
            alpha_act_pre = spindle_lengths + gamma_offsets
            #alpha_act_pre = logistic(alpha_act_pre, k=1.0)
            
            W_inhib = 1.0
            alpha_act_post = np.zeros_like(alpha_act_pre)
            
            antagonists = [(2*i, 2*i + 1) for i in range(self.num_actuators // 2)]

            for i_f, i_e in antagonists:
                alpha_act_post[i_f] = max(0.0, alpha_act_pre[i_f] - W_inhib * alpha_act_pre[i_e])
                alpha_act_post[i_e] = max(0.0, alpha_act_pre[i_e] - W_inhib * alpha_act_pre[i_f])

            # Step the plant
            self.plant.step(alpha_act_post)

            # Log forces
            force = self.plant.data.efc_force.copy()
            if force.shape != (3,):
                force = np.full(3, np.nan)
            force_data["position"].append(grid_positions[grid_pos_idx])
            force_data["force"].append(force)

            # Update nail position
            if self.plant.data.time > total_delay:
                grid_pos_idx += 1
                if grid_pos_idx < len(grid_positions):
                    self.plant.update_nail(grid_positions[grid_pos_idx])
                total_delay += delay

        self.plant.close()
        return force_data

    # -----------------------------------------
    # Plot
    # -----------------------------------------
    def plot(self):
        log = self.logger
        n_muscles = self.num_actuators

        _, axes = plt.subplots(3, 2, figsize=(10, 10))

        # Targets
        target_onset_idcs = np.where(
            np.any(np.diff(np.array(log["target_position"]), axis=0) != 0, axis=1)
        )[0]
        target_onset_idcs = np.insert(target_onset_idcs, 0, 0)
        target_onset_times = np.array([log["time"][idx] for idx in target_onset_idcs])
        for t in target_onset_times:
            for ax in axes.flat:
                ax.axvline(x=t, color="gray", linestyle="--", linewidth=0.5)

        linewidth = 1

        # Length
        length_keys = [f"{name}_len" for name in self.actuator_names]
        for k in length_keys:
            axes[0,0].plot(log["time"], log["sensors"][k], label=k)
        axes[0,0].set_title("Length")

        # Velocity
        vel_keys = [f"{name}_vel" for name in self.actuator_names]
        for k in vel_keys:
            axes[0,1].plot(log["time"], log["sensors"][k], label=k)
        axes[0,1].set_title("Velocity")
        axes[0,1].legend(loc="center left", bbox_to_anchor=(1,0.5))

        # Force
        frc_keys = [f"{name}_frc" for name in self.actuator_names]
        for k in frc_keys:
            axes[1,0].plot(log["time"], log["sensors"][k], label=k)
        axes[1,0].set_title("Force")

        # Distance
        axes[1,1].plot(log["time"], log["manhattan_distance"], label="Manhattan", linewidth=linewidth)
        axes[1,1].plot(log["time"], log["euclidean_distance"], label="Euclidean", linewidth=linewidth)
        axes[1,1].set_title("Distance")
        axes[1,1].set_ylim([-0.05,2.05])
        axes[1,1].legend()

        # Energy
        axes[2,0].plot(log["time"], log["entropy"], linewidth=0.1, label="Entropy")
        axes[2,0].plot(log["time"], log["energy"], linewidth=0.1, label="Energy")
        axes[2,0].set_title("Energy")
        axes[2,0].set_ylim([-0.05,2.05])
        axes[2,0].legend()

        # Reward / Fitness
        axes[2,1].plot(log["time"], log["reward"], linewidth=linewidth, label="Reward")
        axes[2,1].set_title("Loss")
        axes[2,1].set_ylim([-2.05,0.05])
        ax_right = axes[2,1].twinx()
        ax_right.plot(log["time"], log["fitness"], color=(0.25,0.25,0.25))
        ax_right.set_ylabel("Cumulative Reward", color=(0.25,0.25,0.25))
        ax_right.tick_params(axis="y", labelcolor=(0.25,0.25,0.25))

        for ax in axes.flat:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Arb.")

        plt.tight_layout()
        plt.show()

        # Hand Velocity
        plt.figure(figsize=(10,1))
        for idx in target_onset_idcs:
            plt.axvline(x=log["time"][idx], color="blue", linestyle="--", linewidth=0.5,
                        label="Target Change" if idx==target_onset_idcs[0] else None)
        hand_positions = np.array(log["hand_position"])
        hand_velocities = np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)
        time = np.array(log["time"][:-1])
        plt.plot(time, hand_velocities, color="black", label="Hand Velocity", linewidth=linewidth)
        plt.xlabel("Time (s)")
        plt.ylabel("Hand velocity (a.u.)")
        ax_right = plt.gca().twinx()
        ax_right.plot(time, log["euclidean_distance"][:-1], color="red", label="Euclidean Distance", linewidth=linewidth)
        ax_right.set_ylabel("Euclidean Distance", color="red")
        ax_right.tick_params(axis="y", labelcolor="red")

        self.logger = None
