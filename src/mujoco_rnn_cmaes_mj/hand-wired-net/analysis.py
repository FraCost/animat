#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define your datasets
datasets = {
    "direct": "/Users/teachinglab/Documents/code/paton_lab/animat/src/mujoco_rnn_cmaes_mj/hand-wired-net/models/logs/episode_stats_direct.npz",
    "discrete_llctrl": "/Users/teachinglab/Documents/code/paton_lab/animat/src/mujoco_rnn_cmaes_mj/hand-wired-net/models/logs/episode_stats_discrete_llctrl.npz",
    "continuous_llctrl": "/Users/teachinglab/Documents/code/paton_lab/animat/src/mujoco_rnn_cmaes_mj/hand-wired-net/models/logs/episode_stats_continuous_llctrl.npz",
}

# Function to load and smooth rewards
def moving_average(x, win=10):
    return np.convolve(x, np.ones(win)/win, mode='same')

dt = 0.02


#%%
colors = {
    "direct": "blue",
    "discrete_llctrl": "darkorange",
    "continuous_llctrl": "darkgreen",
}

# Load rewards and durations
rewards_data = {}
durations_data = {}
min_episodes = np.inf

for label, path in datasets.items():
    data = np.load(path)
    rewards = data["rewards"]
    durations = data.get("lengths", data.get("durations", None))
    rewards_data[label] = rewards
    durations_data[label] = durations
    min_episodes = min(min_episodes, len(rewards), len(durations))

# --------------------
# Figure 1: Rewards
# --------------------
plt.figure(figsize=(8, 6))
for label, rewards in rewards_data.items():
    trimmed = rewards[:min_episodes]
    smoothed = moving_average(trimmed, win=100)
    plt.plot(smoothed, label=label, color=colors[label])
plt.xlabel("Episode", fontsize=18)
plt.ylabel("Reward", fontsize=18)
plt.legend(['MLP', 'MLP + CFFs (1-hot)', 'MLP + CFFs'], fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# --------------------
# Figure 2: Episode Durations
# --------------------
plt.figure(figsize=(8, 6))
for label, durations in durations_data.items():
    trimmed = durations[:min_episodes] * dt
    smoothed = moving_average(trimmed, win=100)
    plt.plot(smoothed, label=label, color=colors[label])
plt.xlabel("Episode", fontsize=18)
plt.ylabel("Episode duration (s)", fontsize=18)
plt.legend(['MLP', 'MLP + CFFs (1-hot)', 'MLP + CFFs'], fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()


# %%
# Load actions
data = np.load(datasets["discrete_llctrl"], allow_pickle=True)
actions = data["actions"]  # list of lists, one per episode

num_active_units = [
    len(np.unique(ep)) / len(ep) if len(ep) > 0 else 0  
    for ep in actions
]

num_active_units = moving_average(num_active_units, win=250)

plt.figure(figsize=(8, 6))
plt.plot(num_active_units, alpha=0.7)
plt.xlabel("Episode", fontsize=18)
plt.ylabel("Active interneurons \n (normalized by episode length)", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(0.25, 0.5)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# %%
def plot_activation_heatmap(actions_list, start_idx, end_idx, title, cmap='RdBu'):
    # Select the episodes
    selected = actions_list[start_idx:end_idx]
    num_eps = len(selected)
    max_len = max(len(ep) for ep in selected)  # pad to longest episode

    # Create array and fill with NaN
    heatmap = np.full((num_eps, max_len), np.nan)
    for i, ep in enumerate(selected):
        heatmap[i, :len(ep)] = ep
        
    num_actions = int(np.nanmax(heatmap)) + 1

    cmap = plt.get_cmap('RdBu', num_actions)
    cmap = ListedColormap(cmap(np.arange(num_actions)))  
    cmap.set_bad(color='white')  

    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap, aspect='auto', cmap=cmap, interpolation='none', vmin=-0.5, vmax=num_actions-0.5)
    cbar = plt.colorbar(im)
    cbar.set_label("Active interneuron ID", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("Timestep", fontsize=18)
    plt.ylabel("Episode", fontsize=18)  
    plt.yticks([])  
    plt.xticks(fontsize=16)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

# Last num_episodes
num_episodes = 20
plot_activation_heatmap(actions, max(0, len(actions)-num_episodes), len(actions), f"Last {num_episodes} episodes", cmap='RdBu')

# %%
