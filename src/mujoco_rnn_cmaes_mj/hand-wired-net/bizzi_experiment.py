#%%
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plants import SequentialReacher
from networks import LowLevelController
import seaborn as sns
from utils import analyze_damped_oscillation, color_gradient


#%%
plant = SequentialReacher(plant_xml_file="one_joint_arm.xml")

low_lv_ctrl = LowLevelController(
    num_units=10, 
    plant=plant,
    weights_inhibit=0.2,
    mode='multi_joint'
    )

W = low_lv_ctrl.weights

fontsize = 18
ax = sns.heatmap(W, cmap="viridis", cbar=True)
ax.set_xticks([])
ax.set_yticks([])
plt.xlabel("Interneurons", fontsize=fontsize)
plt.ylabel("Gamma-MNs", fontsize=fontsize)
cbar = ax.collections[0].colorbar
cbar.set_label("Weights", fontsize=fontsize)
cbar.ax.tick_params(labelsize=14) 
plt.tight_layout()
plt.show()


# %%
plant = SequentialReacher(plant_xml_file="one_joint_arm.xml")

low_lv_ctrl = LowLevelController(
    num_units=10, 
    plant=plant,
    weights_inhibit=0.2,
    )

stim_on = 25
timesteps = 300 + stim_on
activation = low_lv_ctrl.action_range[1]

low_lv_ctrl.get_max_alpha_activation(max_input=activation)
print(low_lv_ctrl.max_alpha_activation)

logger = {
    "alpha": np.zeros((timesteps, low_lv_ctrl.num_units, plant.num_actuators)),
    "spindle_length": np.zeros((timesteps, low_lv_ctrl.num_units, plant.num_actuators)),
    "spindle_velocity": np.zeros((timesteps, low_lv_ctrl.num_units, plant.num_actuators)),
    "joint_angle": np.zeros((timesteps, low_lv_ctrl.num_units))
}

for i in range(low_lv_ctrl.num_units):
    action = np.zeros(low_lv_ctrl.num_units)

    for t in range(timesteps):
        _, feedback = plant.get_obs()
        feedback = feedback[:plant.num_actuators*2]
        
        if t < stim_on:
            action[i] = low_lv_ctrl.action_range[0]
        else:
            action[i] = activation
                    
        alpha_activation = low_lv_ctrl.step(action, feedback)        
            
        plant.step(alpha_activation)
        #plant.render()
        
        low_lv_ctrl.reset_state()
        
        logger["joint_angle"][t, i] = plant.get_joint_angles_deg()['shoulder']
        logger["alpha"][t, i] = alpha_activation 
        logger["spindle_length"][t, i] = feedback[:low_lv_ctrl.num_actuators]
        logger["spindle_velocity"][t, i] = feedback[low_lv_ctrl.num_actuators:]
        
    plant.reset()


# Plot
# Joint angles
joint_angles = np.array([max(logger['joint_angle'][stim_on:, i], key=abs) for i in range(low_lv_ctrl.num_units)])
fontsize = 18
plt.figure(figsize=(5, 5))
theta = np.radians(joint_angles)  
r = np.ones_like(theta)
colors = plt.cm.RdBu(np.linspace(0, 1, len(theta)))
ax = plt.subplot(111, polar=True)
for a, rad, c in zip(theta, r, colors):
    ax.plot(a, rad, marker='.', markersize=15, color=c, linestyle='None')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)  
ax.set_thetamin(-65)
ax.set_thetamax(65)
ax.set_rticks([])
ax.tick_params(labelsize=14)
ax.set_xlabel("Joint angle (deg)", fontsize=fontsize, labelpad=0)
plt.show()

# Alpha dynamics
e_col = color_gradient((1, 0.9, 0.8), (1, 0.4, 0), low_lv_ctrl.num_units)
f_col = color_gradient((0.8, 0.9, 1), (0, 0, 0.8), low_lv_ctrl.num_units)

t = np.linspace(-(plant.control_timestep * stim_on), (plant.control_timestep * timesteps), timesteps)

'''
plt.figure(figsize=(10, 4))
for i in range(low_lv_ctrl.num_units):
    plt.plot(t, logger['alpha'][:, i, 0], c=f_col[i])
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Alpha-MN activation', fontsize=14)
plt.tight_layout()
plt.show()
'''

plt.figure(figsize=(10, 4))
plt.plot(t, logger['alpha'][:, i, 0], c=f_col[i])
plt.plot(t, logger['alpha'][:, i, 1], c=e_col[i])
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Alpha-MN activation', fontsize=14)
plt.tight_layout()
plt.show()

# Spindle dynamics
plt.figure(figsize=(10, 4))
plt.plot(t, logger['spindle_length'][:, i, 0], c=f_col[i])
plt.plot(t, logger['spindle_length'][:, i, 1], c=e_col[i])
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Spindle length', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(t, logger['spindle_velocity'][:, i, 0], c=f_col[i])
plt.plot(t, logger['spindle_velocity'][:, i, 1], c=e_col[i])
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Spindle velocity', fontsize=14)
plt.tight_layout()
plt.show()

# Joint angle dynamics
plt.figure(figsize=(10, 4))
plt.plot(t, logger['joint_angle'][:, i], c='k')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Joint angle (deg)', fontsize=14)
plt.tight_layout()
plt.show()


# %%
plant = SequentialReacher(plant_xml_file="one_joint_arm.xml")

low_lv_ctrl = LowLevelController(
    num_units=10, 
    plant=plant,
    weights_inhibit=0.2,
    mode='multi_joint',
    k_v=0.0
    )

stim_on = 25
timesteps = 600 + stim_on
time = np.linspace(-(plant.control_timestep * stim_on), 
                (plant.control_timestep * timesteps), timesteps)

unit_idx = 0

activations = [0.25, 0.5, 1, 2, 4, 8]
weights_inhibit = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
num_activations = len(activations)
num_weights = len(weights_inhibit)

joint_angles = np.zeros((num_weights, num_activations, timesteps))
log_decrement = np.zeros((num_weights, num_activations))
settling_time = np.zeros((num_weights, num_activations))
settling_amplitude = np.zeros((num_weights, num_activations))
peak_amplitude =np.zeros((num_weights, num_activations))
offset = np.zeros((num_weights, num_activations))

for w, weight in enumerate(weights_inhibit):
    print(f"Weight {w+1}/{num_weights}")
    low_lv_ctrl.weights_inhibit = weight
    
    for a, activation in enumerate(activations):
        print(f"Activation {a+1}/{num_activations}")
        
        low_lv_ctrl.get_max_alpha_activation(max_input=activation)

        action = np.zeros(low_lv_ctrl.num_units)

        for t in range(timesteps):
            _, feedback = plant.get_obs()
            feedback = feedback[:plant.num_actuators*2]
            
            if t < stim_on:
                action[unit_idx] = 0.0
            else:
                action[unit_idx] = activation
                        
            alpha_activation = low_lv_ctrl.step(action, feedback)        
                
            plant.step(alpha_activation)
            
            joint_angles[w, a, t] = plant.get_joint_angles_deg()['shoulder']
            
        metrics = analyze_damped_oscillation(time, 
                                             joint_angles[w, a], 
                                             settle_threshold=0.02, 
                                             plot=False)
        log_decrement[w, a] = metrics['damping_ratio']
        settling_time[w, a] = metrics['settling_time']
        settling_amplitude[w, a] = metrics['settling_amplitude']
        peak_amplitude[w, a] = metrics['peak_amplitude']
        offset[w, a] = metrics['steady_state_offset']
            
        plant.reset()
        low_lv_ctrl.reset_state()


# Plot
red_colors = color_gradient((1, 0.9, 0.9), (0.5, 0, 0), num_weights)
blue_colors = color_gradient((0.9, 0.9, 1), (0, 0, 0.5), num_weights)
orange_colors = color_gradient((1, 0.95, 0.9), (1, 0.4, 0), num_weights)
green_colors = color_gradient((0.9, 1, 0.9), (0, 0.5, 0), num_weights)

# --- Damping Ratio ---
plt.figure(figsize=(6, 4))
for w, weight in enumerate(weights_inhibit):
    plt.plot(activations, log_decrement[w], 'o-', color=green_colors[w], linewidth=1.5, label=f"{weight:.2f}")
plt.xlabel("Interneuron activation (a.u)", fontsize=18)
plt.ylabel("Damping ratio", fontsize=18)
plt.legend(fontsize=12, title="$w_{{inhib}}$", title_fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# --- Peak and final angle ---
plt.figure(figsize=(6, 4))
for w, weight in enumerate(weights_inhibit):
    plt.plot(activations, peak_amplitude[w], 'o-', color=red_colors[w], linewidth=1.5)
    plt.plot(activations, settling_amplitude[w], 'o-', color=blue_colors[w], linewidth=1.5)
plt.xlabel("Interneuron activation (a.u)", fontsize=18)
plt.ylabel("Angle (deg)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()


# %%
plant = SequentialReacher(plant_xml_file="one_joint_arm.xml")

low_lv_ctrl = LowLevelController(
    num_units=10, 
    plant=plant,
    weights_inhibit=0.2,
    mode='multi_joint',
    )

stim_on = 25
timesteps = 600 + stim_on
time = np.linspace(-(plant.control_timestep * stim_on), 
                (plant.control_timestep * timesteps), timesteps)

unit_idx = 0

k_v = [0.0, 0.25, 0.5, 1.0]
num_k = len(k_v)

joint_angles = np.zeros((num_k, timesteps))

for g, gain in enumerate(k_v):
    low_lv_ctrl.k_v = gain
    
    low_lv_ctrl.get_max_alpha_activation(max_input=1.0)

    action = np.zeros(low_lv_ctrl.num_units)

    for t in range(timesteps):
        _, feedback = plant.get_obs()
        feedback = feedback[:plant.num_actuators*2]
        
        if t < stim_on:
            action[unit_idx] = 0.0
        else:
            action[unit_idx] = 1.0
                    
        alpha_activation = low_lv_ctrl.step(action, feedback)        
            
        plant.step(alpha_activation)
        
        joint_angles[g, t] = plant.get_joint_angles_deg()['shoulder']
        
    plant.reset()
    low_lv_ctrl.reset_state()


# Plot
plt.figure(figsize=(6, 4))
for g, gain in enumerate(k_v):
    plt.plot(time, joint_angles[g], linewidth=1, label=f"{gain:.2f}")
plt.xlabel("Time (s)", fontsize=18)
plt.ylabel("Joint angle (deg)", fontsize=18)
plt.legend(fontsize=12, title="$k_{{g}}$", title_fontsize=13)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# %%
