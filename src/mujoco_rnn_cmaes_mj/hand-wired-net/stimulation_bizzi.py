
# %%
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN
from utils import *
from sklearn.decomposition import PCA

# %%
reacher = SequentialReacher(plant_xml_file="arm.xml")
print("Number of sensors:", reacher.num_sensors)
print("Number of actuators:", reacher.num_actuators)

env = SequentialReachingEnv(
    plant=reacher,
    target_duration={"mean": 3, "min": 1, "max": 6},
    num_targets=10,
    num_interneurons=20,
    loss_weights={
        "euclidean": 1,
        "manhattan": 0,
        "energy": 0,
        "ridge": 0.001,
        "lasso": 0,
    },
)

# %%
# Extract weights
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
num_targets = 25
sampled_targets = reacher.sample_targets(num_targets)
x_coords = sampled_targets[:, 0]
y_coords = sampled_targets[:, 1]

# Plot the x and z coordinates
plt.figure(figsize=(6, 6))
plt.scatter(x_coords, y_coords, c="red", s=25, alpha=0.7)
plt.title("Sampled Targets: X vs Z Coordinates")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.grid(True)
plt.show()

# %%
for unit_idx in range(0, env.num_interneurons):

    force_data = env.stimulate(
        units=np.array([unit_idx]),
        delay=1,
        seed=0,
        render=False,
    )

    # Plot force vectors over time
    position_vecs = np.nan_to_num(np.array(force_data["position"]))
    force_vecs = np.nan_to_num(np.array(force_data["force"]))
    time = np.linspace(0, reacher.data.time, len(force_vecs))

    if unit_idx == 0:
        plt.figure(figsize=(25, 5))
        for i in range(force_vecs.shape[1] - 1):
            plt.plot(time, force_vecs[:, i], label=f"Force Component {i+1}")
        for t in np.arange(0.5, reacher.data.time, 1.0):
            plt.axvline(x=t, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        lower_percentile = np.percentile(force_vecs, 1)
        upper_percentile = np.percentile(force_vecs, 99)
        plt.ylim([lower_percentile, upper_percentile])
        plt.xlabel("Time (s)")
        plt.ylabel("Force (a.u.)")
        plt.title("Force Vectors Over Time")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    # Define the time window for averaging (100 ms)
    time_window = 0.1  # 100 ms
    
    # Initialize a list to store average force vectors
    average_positions = []
    rest_average_forces = []
    stim_average_forces = []

    # Iterate second by second
    for t in range(1, int(reacher.data.time) + 1):

        # Find indices corresponding to the last 100 ms of the current second
        start_time = t - .5 - time_window
        stop_time = t - .5
        indices = (time > start_time) & (time <= stop_time)

        # Compute the average force vector within the 100-ms period
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        rest_average_forces.append(avg_force_vec)

        # Find indices corresponding to the last 100 ms of the current second
        start_time = t - time_window
        stop_time = t
        indices = (time > start_time) & (time <= stop_time)

        # Compute the average position vector within the 100-ms period
        avg_position_vec = np.mean(position_vecs[indices], axis=0)
        average_positions.append(avg_position_vec)

        # Compute the average force vector within the 100-ms period
        avg_force_vec = np.mean(force_vecs[indices], axis=0)
        stim_average_forces.append(avg_force_vec)

    # Convert the list to a numpy array for further analysis
    average_positions = np.array(average_positions)
    rest_average_forces = np.array(rest_average_forces)
    stim_average_forces = np.array(stim_average_forces)

    print(average_positions.shape)
    print(rest_average_forces.shape)
    print(stim_average_forces.shape)

    plt.figure(figsize=(8, 8))

    # Extract x and y components from average positions and forces
    x_positions = [pos[0] for pos in average_positions]
    y_positions = [pos[1] for pos in average_positions]
    x_forces = [force[0] for force in stim_average_forces]
    y_forces = [force[1] for force in stim_average_forces]

    # Plot the 2D vector field
    plt.quiver(
        x_positions,
        y_positions,
        x_forces,
        y_forces,
        angles="xy",
        scale_units="xy",
        scale=500,
        linewidth=1,
        color="red",
        edgecolor="red",
        facecolor='none',
        label="Stimulated",
    )

    # Extract x and y components from average positions and forces
    x_positions = [pos[0] for pos in average_positions]
    y_positions = [pos[1] for pos in average_positions]
    x_forces = [force[0] for force in rest_average_forces]
    y_forces = [force[1] for force in rest_average_forces]

    # Compute the weighted average of positions using the magnitude of stim forces
    force_magnitudes = np.linalg.norm(stim_average_forces, axis=1)  # ||F||
    convergence_point_weighted = np.average(average_positions, axis=0, weights=force_magnitudes)
    plt.scatter(
    convergence_point_weighted[0],
    convergence_point_weighted[1],
    color="red",
    s=200,
    edgecolor="black",
)
    # Plot the 2D vector field
    plt.quiver(
        x_positions,
        y_positions,
        x_forces,
        y_forces,
        angles="xy",
        scale_units="xy",
        scale=500,
        linewidth=1,
        color="black",
        edgecolor="black",
        facecolor='none',
        label="Rest",
    )

    plt.legend()

    plt.title(f"Convergence force field (CFF) stimulating unit {unit_idx}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.axis("equal")
    plt.xlim(
        reacher.hand_position_stats["min"][0], reacher.hand_position_stats["max"][0]
    )
    plt.ylim(
        reacher.hand_position_stats["min"][1], reacher.hand_position_stats["max"][1]
    )
    plt.show()