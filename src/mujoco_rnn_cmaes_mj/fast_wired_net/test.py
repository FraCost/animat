#%%
import numpy as np
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plants import SequentialReacher
from environments import SequentialReachingEnv
import seaborn as sns
from utils import logistic


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


# %%
timesteps = 2000
stim_on = 250
stim_off = 1250
W_inhib = 0.5
interneuron_activation = 2.0
antagonists = [(2*i, 2*i + 1) for i in range(env.num_actuators // 2)]

logger = {
    "alpha": np.zeros((timesteps, env.num_interneurons, 2)),
    "spindle": np.zeros((timesteps, env.num_interneurons, 2)),
    "joint_angle": np.zeros((timesteps, env.num_interneurons))
}

for i in range(env.num_interneurons):
    interneurons = np.zeros(env.num_interneurons)
    interneurons[i] = interneuron_activation
    gamma_activation = env.W_inter_to_gamma @ interneurons

    for t in range(timesteps):
        _, feedback = env.plant.get_obs()
        spindle_lengths = np.array(feedback[:env.num_actuators])
        
        if t < stim_on or t > stim_off:
            spindle_activation = spindle_lengths        
        else:
            spindle_activation = spindle_lengths + gamma_activation
        #TODO: modle gamma dynamics (memory/decay) & re-adjustment
        #TODO: normalize spindle length to (0, 1) ??
            
        alpha_activation = np.zeros_like(spindle_activation)
        for f, e in antagonists:
            alpha_activation[f] = max(0.0, spindle_activation[f] - W_inhib * spindle_activation[e])
            alpha_activation[e] = max(0.0, spindle_activation[e] - W_inhib * spindle_activation[f])

        env.plant.step(alpha_activation)
        #env.plant.render()
        
        logger["joint_angle"][t, i] = env.plant.get_joint_angles_deg()['shoulder']
        logger["alpha"][t, i] = alpha_activation 
        logger["spindle"][t, i] = spindle_activation
        
    env.plant.reset()


# %% Plot
from matplotlib.colors import LinearSegmentedColormap

def color_gradient(min, max, N):
    cmap = LinearSegmentedColormap.from_list("blue_gradient", [min, max], N=N)
    return [cmap(i/(N-1)) for i in range(N)]

# Joint angles
joint_angles = np.array([max(logger['joint_angle'][stim_on : stim_off, i], key=abs) for i in range(env.num_interneurons)])
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
ax.set_thetamin(-65)
ax.set_thetamax(65)
ax.set_rticks([])
ax.tick_params(labelsize=14)
ax.set_xlabel("Joint angle (deg)", fontsize=fontsize, labelpad=0)
plt.show()

# Alpha dynamics
e_col = color_gradient((1, 0.9, 0.8), (1, 0.4, 0), env.num_interneurons)
f_col = color_gradient((0.8, 0.9, 1), (0, 0, 0.8), env.num_interneurons)

plt.figure(figsize=(10, 4))
for i in range(env.num_interneurons):
    plt.plot(logger['alpha'][:, i, 0], c=f_col[i])
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Alpha drive', fontsize=14)
plt.tight_layout()
plt.show()
    
plt.figure(figsize=(10, 4))
plt.plot(logger['alpha'][:, i, 0], c=f_col[i])
plt.plot(logger['alpha'][:, i, 1], c=e_col[i])
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Alpha drive', fontsize=14)
plt.tight_layout()
plt.show()

# Spindle dynamics
plt.figure(figsize=(10, 4))
plt.plot(logger['spindle'][:, i, 0], c=f_col[i])
plt.plot(logger['spindle'][:, i, 1], c=e_col[i])
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Spindle activation', fontsize=14)
plt.tight_layout()
plt.show()

# Joint angle dynamics
plt.figure(figsize=(10, 4))
plt.plot(logger['joint_angle'][:, i], c='k')
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Spindle activation', fontsize=14)
plt.tight_layout()
plt.show()
