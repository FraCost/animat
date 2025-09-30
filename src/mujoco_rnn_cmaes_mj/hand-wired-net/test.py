# %% 
import mujoco
import os
from utils import *

# Load your model
mj_dir = os.path.join(get_root_path(), "mujoco")
xml_path = os.path.join(mj_dir, "arm.xml")
model = mujoco.MjModel.from_xml_path(xml_path)

# %% 
num_actuators = model.nu
actuator_names = [model.actuator(i).name for i in range(num_actuators)]
print("Actutor names:", actuator_names)

print("Joint names:", [model.joint(i).name for i in range(model.njnt)])
# %%
