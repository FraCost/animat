# %%
"""
Sequential Reacher Analysis (Compatible with new alpha+gamma plant)
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *

# %%
# Initialize the reacher and RNN
reacher = SequentialReacher(plant_xml_file="arm.xml")
print("Number of sensors:", reacher.num_sensors)
print("Number of actuators:", reacher.num_actuators)

rnn = RNN(
    input_size=3 + reacher.num_sensors,
    hidden_size=25,
    output_size=reacher.num_actuators,
    activation=tanh,
    alpha=1,
)

env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=10,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)

# Load the trained RNN from optimizer
models_dir = "../../models"
gen_idx = 9999
model_file = f"optimizer_gen_{gen_idx}_cmaesv2.pkl"
with open(os.path.join(models_dir, model_file), "rb") as f:
    optimizer = pickle.load(f)
best_rnn = rnn.from_params(optimizer.mean)

# Evaluate the RNN
env.evaluate(best_rnn, seed=0, render=True, log=True)
env.plot()

# %%
# Extract RNN weights and biases
weights_input = best_rnn.W_in
weights_hidden = best_rnn.W_h
weights_output = best_rnn.W_out.T
bias_hidden = best_rnn.b_h
bias_output = best_rnn.b_out

# Plot weights and biases
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
sns.heatmap(weights_input, cmap="viridis", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title("Input Weights")
axes[0, 0].set_xlabel("Input Features")
axes[0, 0].set_ylabel("Hidden Units")

sns.heatmap(weights_hidden, cmap="viridis", cbar=True, ax=axes[0, 1])
axes[0, 1].set_title("Hidden Weights")
axes[0, 1].set_xlabel("Hidden Units")
axes[0, 1].set_ylabel("Hidden Units")

sns.heatmap(weights_output, cmap="viridis", cbar=True, ax=axes[0, 2])
axes[0, 2].set_title("Output Weights")
axes[0, 2].set_xlabel("Hidden Units")
axes[0, 2].set_ylabel("Output Units")

sns.heatmap(bias_hidden.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 1])
axes[1, 1].set_title("Hidden Biases")
axes[1, 1].set_xlabel("Hidden Units")
axes[1, 1].set_yticks([])

sns.heatmap(bias_output.reshape(1, -1), cmap="viridis", cbar=True, annot=False, ax=axes[1, 2])
axes[1, 2].set_title("Output Biases")
axes[1, 2].set_xlabel("Output Units")
axes[1, 2].set_yticks([])

axes[1, 0].axis("off")
plt.tight_layout()
plt.show()

# %%
# Sample targets and plot x/y coordinates
num_targets = 25
sampled_targets = reacher.sample_targets(num_targets)
x_coords = sampled_targets[:, 0]
y_coords = sampled_targets[:, 1]

plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c="red", s=25, alpha=0.7)
plt.title("Sampled Targets: X vs Y Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

# %%
# Project targets through RNN input weights and PCA
input_weights_targets = weights_input[:, :3]  # first 3 are spatial inputs
projections = np.dot(sampled_targets, input_weights_targets.T)

pca = PCA(n_components=3)
pca_projections = pca.fit_transform(projections)

# 3D Scatter for selected units
selected_units = [6, 19, 24]
selected_weights = input_weights_targets[selected_units, :]
selected_projections = np.dot(sampled_targets, selected_weights.T)

fig = go.Figure(data=[
    go.Scatter3d(
        x=selected_projections[:, 0],
        y=selected_projections[:, 1],
        z=selected_projections[:, 2],
        mode="markers",
        marker=dict(size=2, color="black", opacity=0.8)
    )
])
fig.update_layout(
    title="Selected Projections of Sampled Targets Through Input Weights",
    scene=dict(xaxis_title=f"Unit {selected_units[0]}", 
               yaxis_title=f"Unit {selected_units[1]}", 
               zaxis_title=f"Unit {selected_units[2]}")
)
fig.show()

# PCA 3D visualization
fig = go.Figure(data=[
    go.Scatter3d(
        x=pca_projections[:, 0],
        y=pca_projections[:, 1],
        z=pca_projections[:, 2],
        mode="markers",
        marker=dict(size=2, color=pca_projections[:, 0], colorscale="Viridis", opacity=0.8)
    )
])
fig.update_layout(title="PCA Projections of Sampled Targets", scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"))
fig.show()

# %%
# Analyze weight distributions
total_abs_output_weights = np.sum(np.abs(weights_output), axis=1)
total_abs_input_weights = np.sum(np.abs(weights_input), axis=1)
total_abs_hidden_weights = np.sum(np.abs(weights_hidden), axis=1)

plt.figure(figsize=(10, 6))
sns.histplot(total_abs_output_weights, kde=True, bins=20, color="blue")
plt.title("Total Absolute Output Weights")
plt.xlabel("Sum of Absolute Weights")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(total_abs_input_weights, kde=True, bins=20, color="green")
plt.title("Total Absolute Input Weights")
plt.xlabel("Sum of Absolute Weights")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(total_abs_hidden_weights, kde=True, bins=20, color="orange")
plt.title("Total Absolute Hidden Weights")
plt.xlabel("Sum of Absolute Weights")
plt.ylabel("Frequency")
plt.show()

# %%
# 2D joint distributions of weights
def plot_joint(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    joint_plot = sns.jointplot(
        x=x, y=y, kind="scatter", cmap="viridis", marginal_kws=dict(bins=20, fill=True)
    )
    for i, (xi, yi) in enumerate(zip(x, y)):
        joint_plot.ax_joint.text(xi, yi + 0.1, str(i), fontsize=8, color="black", ha="center", va="center")
    plt.suptitle(title, y=1.02)
    joint_plot.set_axis_labels(xlabel, ylabel)
    plt.show()

plot_joint(total_abs_input_weights, total_abs_output_weights, "Input vs Output Weights", "Input", "Output")
plot_joint(total_abs_input_weights, total_abs_hidden_weights, "Input vs Hidden Weights", "Input", "Hidden")
plot_joint(total_abs_hidden_weights, total_abs_output_weights, "Hidden vs Output Weights", "Hidden", "Output")

# %%
# Stimulate each hidden unit and plot force fields
for unit_idx in range(rnn.hidden_size):
    force_data = env.stimulate(
        best_rnn,
        units=np.array([unit_idx]),
        action_modifier=1,
        delay=1,
        seed=0,
        render=True if unit_idx == 0 else False
    )

    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
    time = np.linspace(0, reacher.data.time, len(force_vecs))

    # Average forces over 100ms windows
    avg_positions, rest_forces, stim_forces = [], [], []
    time_window = 0.1
    for t in range(1, int(reacher.data.time) + 1):
        # Rest forces
        indices_rest = (time > t - 0.5 - time_window) & (time <= t - 0.5)
        rest_forces.append(np.mean(force_vecs[indices_rest], axis=0))

        # Stim forces
        indices_stim = (time > t - time_window) & (time <= t)
        stim_forces.append(np.mean(force_vecs[indices_stim], axis=0))
        avg_positions.append(np.mean(position_vecs[indices_stim], axis=0))

    avg_positions = np.array(avg_positions)
    rest_forces = np.array(rest_forces)
    stim_forces = np.array(stim_forces)

    plt.figure(figsize=(8, 8))
    # Stimulated forces
    plt.quiver(avg_positions[:, 0], avg_positions[:, 1], stim_forces[:, 0], stim_forces[:, 1],
               angles="xy", scale_units="xy", scale=500, color="red", label="Stim")
    # Rest forces
    plt.quiver(avg_positions[:, 0], avg_positions[:, 1], rest_forces[:, 0], rest_forces[:, 1],
               angles="xy", scale_units="xy", scale=500, color="black", label="Rest")

    plt.title(f"Convergence Force Field (Unit {unit_idx})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0])
    plt.ylim(reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1])
    plt.legend()
    plt.show()
