#%%
import numpy as np
from plants import SequentialReacher
from networks import LowLevelController

#%%
plant = SequentialReacher(plant_xml_file="arm.xml")

low_lv_ctrl = LowLevelController(
    num_units=25, 
    plant=plant,
    weights_inhibit=0.2,
    mode='multi_joint'
    )


# %%
stim_on = 0
timesteps = 100 + stim_on
activation = 4.0

low_lv_ctrl.get_max_alpha_activation(max_input=activation)

for i in range(low_lv_ctrl.num_units):
    action = np.zeros(low_lv_ctrl.num_units)

    for t in range(timesteps):
        _, feedback = plant.get_obs()
        feedback = feedback[:plant.num_actuators*2]
        
        if t < stim_on:
            action[i] = 0.0
        else:
            action[i] = activation
                    
        alpha_activation = low_lv_ctrl.step(action, feedback)        
            
        plant.step(alpha_activation)
        
        plant.render()
        low_lv_ctrl.reset_state()

        
    plant.reset()