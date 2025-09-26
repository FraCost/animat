# %%
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load the model
model_path = "/Users/teachinglab/Documents/code/paton_lab/animat/mujoco/one_joint_arm.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Launch the viewer
viewer = mujoco.viewer.launch(model, data)

# Simulation loop
while viewer.is_alive():
    # Apply random control inputs (flexor and extensor between 0 and 1)
    data.ctrl[:] = np.random.rand(model.nu)

    # Step the simulation
    mujoco.mj_step(model, data)

    # Update the viewer
    viewer.sync()
    
    # Optional: slow down simulation to real time
    time.sleep(model.opt.timestep)

# %%
