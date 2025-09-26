# %%
"""
Data collection and preprocessing for SequentialReacher with alpha/gamma muscles
"""

import os
import mujoco
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import binary_fill_holes

# %%
# Change working directory to script location
os.chdir(os.path.dirname(__file__))

# Load model
MODEL_XML_PATH = "../../mujoco/arm_model.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

num_actuators = model.nu  # total actuators (alpha + gamma)
hand_id = model.geom("hand").id

# Simulation parameters
dur2run = 3600  # seconds
ctrl_increment = 0.05

# Data containers
time_data = []
hand_position_data = {"x": [], "y": [], "z": []}

# Sensor data now includes alpha muscle lengths and gamma offsets
sensor_data = {
    "deltoid_alpha_length": [],
    "deltoid_alpha_velocity": [],
    "deltoid_alpha_force": [],
    "deltoid_gamma_offset": [],
    "latissimus_alpha_length": [],
    "latissimus_alpha_velocity": [],
    "latissimus_alpha_force": [],
    "latissimus_gamma_offset": [],
    "biceps_alpha_length": [],
    "biceps_alpha_velocity": [],
    "biceps_alpha_force": [],
    "biceps_gamma_offset": [],
    "triceps_alpha_length": [],
    "triceps_alpha_velocity": [],
    "triceps_alpha_force": [],
    "triceps_gamma_offset": [],
}

# %%
# Reset simulation
mujoco.mj_resetData(model, data)

while data.time < dur2run:
    # Random actuator control (alpha + gamma)
    data.ctrl[:] = np.clip(data.ctrl + np.random.randn(num_actuators) * ctrl_increment, 0, 1)

    # Step simulation
    mujoco.mj_step(model, data)

    # Record time
    time_data.append(data.time)

    # Record hand position
    hand_pos = data.geom_xpos[hand_id].copy()
    for i, key in enumerate(hand_position_data.keys()):
        hand_position_data[key].append(hand_pos[i])

    # Record sensor data
    # Assuming sensors are in order: deltoid_alpha, deltoid_gamma, latissimus_alpha, latissimus_gamma, etc.
    for i, key in enumerate(sensor_data.keys()):
        sensor_data[key].append(data.sensordata[i])

# %%
# Convert to pandas DataFrames
sensor_df = pd.DataFrame(sensor_data)
hand_position_df = pd.DataFrame(hand_position_data)

# Compute stats
sensor_stats_df = pd.DataFrame({
    "min": sensor_df.min(),
    "max": sensor_df.max(),
    "mean": sensor_df.mean(),
    "std": sensor_df.std(),
})

hand_position_stats_df = pd.DataFrame({
    "min": hand_position_df.min(),
    "max": hand_position_df.max(),
    "mean": hand_position_df.mean(),
    "std": hand_position_df.std(),
})

print(sensor_stats_df)
print(hand_position_stats_df)

# %%
# Hand workspace 2D histogram
x, y = hand_position_data["x"], hand_position_data["y"]
x_min, x_max = min(x), max(x)
x_min -= 0.05 * (x_max - x_min)
x_max += 0.05 * (x_max - x_min)
y_min, y_max = min(y), max(y)
y_min -= 0.05 * (y_max - y_min)
y_max += 0.05 * (y_max - y_min)

counts2d, x_edges, y_edges = np.histogram2d(
    x, y, bins=500, range=[[x_min, x_max], [y_min, y_max]]
)

reachable_image = counts2d > 0
reachable_image = binary_fill_holes(reachable_image)

# %%
# Generate candidate targets
contours = measure.find_contours(reachable_image, level=0.5)
contours_image = np.zeros_like(reachable_image, dtype=bool)
for contour in contours:
    rr, cc = contour[:, 0].astype(int), contour[:, 1].astype(int)
    rr = np.clip(rr, 0, contours_image.shape[0] - 1)
    cc = np.clip(cc, 0, contours_image.shape[1] - 1)
    contours_image[rr, cc] = True

# Zero out fraction of interior pixels
num_countour_pixels = contours_image.astype(int).sum()
num_reachable_pixels = reachable_image.astype(int).sum()
fraction_to_zero_out = 1 - num_countour_pixels / (num_reachable_pixels - num_countour_pixels)

zeroed_image = reachable_image.copy()
num_pixels = zeroed_image.size
num_zeroed = int(fraction_to_zero_out * num_pixels)
flat_image = zeroed_image.flatten()
zero_indices = np.random.choice(num_pixels, num_zeroed, replace=False)
flat_image[zero_indices] = 0
zeroed_image = flat_image.reshape(zeroed_image.shape)

candidate_targets_image = np.logical_or(zeroed_image, contours_image)

# Compute candidate targets coordinates
candidate_idcs = np.argwhere(candidate_targets_image)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
candidate_targets = [(x_centers[i], y_centers[j], 0) for i, j in candidate_idcs]
candidate_targets_df = pd.DataFrame(candidate_targets, columns=["x", "y", "z"])

# %%
# Generate 2D grid positions
xy_resolution = 0.1
x_grid_points = int((x_max - x_min) / xy_resolution) + 1
y_grid_points = int((y_max - y_min) / xy_resolution) + 1
x_grid = np.linspace(x_min, x_max, x_grid_points)
y_grid = np.linspace(y_min, y_max, y_grid_points)

grid_image = np.zeros_like(reachable_image, dtype=bool)
for x_val in x_grid:
    for y_val in y_grid:
        x_idx = np.searchsorted(x_edges, x_val) - 1
        y_idx = np.searchsorted(y_edges, y_val) - 1
        if 0 <= x_idx < grid_image.shape[1] and 0 <= y_idx < grid_image.shape[0]:
            grid_image[y_idx, x_idx] = True

reachable_grid_image = grid_image & reachable_image
grid_idcs = np.argwhere(reachable_grid_image)
grid_positions = [(x_centers[i], y_centers[j], 0) for i, j in grid_idcs]
grid_positions_df = pd.DataFrame(grid_positions, columns=["x", "y", "z"])

# %%
# Save all statistics and positions
save_dir = "../../mujoco"
sensor_stats_df.to_pickle(f"{save_dir}/sensor_stats.pkl")
hand_position_stats_df.to_pickle(f"{save_dir}/hand_position_stats.pkl")
grid_positions_df.to_pickle(f"{save_dir}/grid_positions.pkl")
candidate_targets_df.to_pickle(f"{save_dir}/candidate_targets.pkl")
reachable_positions_df = pd.DataFrame([(x_centers[i], y_centers[j], 0) 
                                        for i,j in np.argwhere(reachable_image)],
                                       columns=["x","y","z"])
reachable_positions_df.to_pickle(f"{save_dir}/reachable_positions.pkl")
