#%%
import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
import seaborn as sns


#%%
reacher = SequentialReacher(plant_xml_file="one_joint_arm.xml")
env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=20,
    num_interneurons=10,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)


# %%
W = env.W_inter_to_gamma

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

#%%
'''
joint_idx = 0
muscle1_idx = 2*joint_idx     # flexor
muscle2_idx = 2*joint_idx + 1 # extensor

num_samples = 20
timesteps = 500
spindle_range = np.linspace(0, 1, num_samples)

# Arrays to store equilibrium joint angles
joint_angles_flexor = np.zeros(num_samples)
joint_angles_extensor = np.zeros(num_samples)

# Sweep flexor
for i, s in enumerate(spindle_range):
    spindle_lengths = np.zeros(env.num_actuators)
    spindle_lengths[muscle1_idx] = s
    spindle_lengths[muscle2_idx] = 0.5  # keep extensor constant
    alpha_act = spindle_lengths
    
    for t in range(timesteps):
        env.plant.step(alpha_act)
        joint_angle = env.plant.get_joint_angles_deg()
    joint_angles_flexor[i] = joint_angle['shoulder']  # equilibrium
    
    env.plant.reset()

# Sweep extensor
for i, s in enumerate(spindle_range):
    spindle_lengths = np.zeros(env.num_actuators)
    spindle_lengths[muscle1_idx] = 0.5  # keep flexor constant
    spindle_lengths[muscle2_idx] = s
    alpha_act = spindle_lengths
    
    for t in range(timesteps):
        env.plant.step(alpha_act)
        joint_angle = env.plant.get_joint_angles_deg()
    joint_angles_extensor[i] = joint_angle['shoulder']  # equilibrium
    
    env.plant.reset()

plt.figure(figsize=(6,4))
plt.plot(spindle_range, joint_angles_flexor, label="Flexor", linewidth=2.5)
plt.plot(spindle_range, joint_angles_extensor, label="Extensor", linewidth=2.5)
plt.xlabel("Spindle activity", fontsize=18)
plt.ylabel("Joint angle (deg)", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
'''

# %%
timesteps = 500
joint_angles = np.zeros(env.num_interneurons)
thetas = np.zeros(timesteps)
for i in range(env.num_interneurons):
    interneurons = np.zeros(env.num_interneurons)
    interneurons[i] = 1.0  
    gamma_offsets = env.W_inter_to_gamma @ interneurons

    for t in range(timesteps):
        _, feedback = env.plant.get_obs()
        spindle_lengths = np.array(feedback[:env.num_actuators])
        alpha_act = spindle_lengths + gamma_offsets
        env.plant.step(alpha_act)
        #env.plant.render()
        thetas[t] = env.plant.get_joint_angles_deg()['shoulder']
    joint_angles[i] = max(thetas, key=abs) 
    env.plant.reset()


fontsize = 18
plt.figure(figsize=(5, 5))
theta = np.radians(joint_angles)  
r = np.ones_like(theta)
colors = plt.cm.RdBu(np.linspace(0, 1, len(theta)))
ax = plt.subplot(111, polar=True)
for t, rad, c in zip(theta, r, colors):
    ax.plot(t, rad, marker='.', markersize=15, color=c, linestyle='None')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)  
ax.set_thetamin(-60)
ax.set_thetamax(60)
ax.set_rticks([])
ax.tick_params(labelsize=14)
ax.set_xlabel("Joint angle (deg)", fontsize=fontsize, labelpad=0)
plt.show()
# %%
