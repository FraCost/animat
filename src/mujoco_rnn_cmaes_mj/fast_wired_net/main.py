from plants import SequentialReacher
from environments import SequentialReachingEnv
from networks import RNN, LowLevelController
from utils import tanh


if __name__ == "__main__":

    # ----------------------------------------------------------
    # 1) Initialize the Mujoco plant
    # ----------------------------------------------------------
    plant = SequentialReacher(plant_xml_file="arm.xml")

    # ----------------------------------------------------------
    # 2) Define controller
    # ----------------------------------------------------------
    low_lv_ctrl = LowLevelController(
        num_units=25, 
        num_actuators=plant.num_actuators, 
        weigths_inhib=0.5
        )
    
    rnn = RNN(
        input_size=3 + plant.num_sensors,
        hidden_size=62,
        output_size=low_lv_ctrl.num_units,
        activation=tanh,
        alpha=plant.model.opt.timestep / 0.01
    )

    # ----------------------------------------------------------
    # 3) Initialize task
    # ----------------------------------------------------------
    env = SequentialReachingEnv(
        plant=plant,
        target_duration={"mean": 3, "min": 1, "max": 6},
        num_targets=10,
        low_lv_ctrl=low_lv_ctrl,
        loss_weights={
            "euclidean": 1,
            "manhattan": 0,
            "energy": 0,
            "ridge": 0,
            "lasso": 0
        }
    )
    
 