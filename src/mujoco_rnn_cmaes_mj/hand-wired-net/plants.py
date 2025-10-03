import os
import time
import pickle
import numpy as np
import mujoco
import mujoco.viewer
from utils import *


class SequentialReacher:
    def __init__(self, plant_xml_file="arm.xml"):
        """Initialize Mujoco simulation"""
        mj_dir = os.path.join(get_root_path(), "mujoco")
        xml_path = os.path.join(mj_dir, plant_xml_file)
        
        self.model = mujoco.MjModel.from_xml_path(xml_path) 
        self.data = mujoco.MjData(self.model)
        self.num_sensors = self.model.nsensor
        self.num_actuators = self.model.nu
        self.num_joints = self.model.njnt
        self.viewer = None
        
        try:
            self.actuator_names = [self.model.actuator(i).name for i in range(self.num_actuators)]
        except AttributeError:
            # fallback if not available
            self.actuator_names = [f"muscle{i}" for i in range(self.num_actuators)]

        # End-effector and sensor
        self.hand_id = self.model.geom("hand").id
        self.hand_force_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "hand_force"
        )
        self.hand_default_mass = self.model.body_mass[self.hand_id]

        
        # Load precomputed stats
        model_name = os.path.splitext(plant_xml_file)[0]
        with open(os.path.join(mj_dir, f"sensor_stats_{model_name}.pkl"), "rb") as f:
            self.sensor_stats = pickle.load(f)
        with open(os.path.join(mj_dir, f"hand_position_stats_{model_name}.pkl"), "rb") as f:
            self.hand_position_stats = pickle.load(f)
        with open(os.path.join(mj_dir, f"candidate_targets_{model_name}.pkl"), "rb") as f:
            self.candidate_targets = pickle.load(f)
        with open(os.path.join(mj_dir, f"grid_positions_{model_name}.pkl"), "rb") as f:
            self.grid_positions = pickle.load(f)
        
        '''
        with open(os.path.join(mj_dir, f"sensor_stats.pkl"), "rb") as f:
            self.sensor_stats = pickle.load(f)
        with open(os.path.join(mj_dir, f"hand_position_stats.pkl"), "rb") as f:
            self.hand_position_stats = pickle.load(f)
        with open(os.path.join(mj_dir, f"candidate_targets.pkl"), "rb") as f:
            self.candidate_targets = pickle.load(f)
        with open(os.path.join(mj_dir, f"grid_positions.pkl"), "rb") as f:
            self.grid_positions = pickle.load(f)
        '''
        
    
    def randomize_configuration(self):
        """Randomize the configuration of all joints"""
        for i in range(self.model.nq):
            self.data.qpos[i] = np.random.uniform(np.deg2rad(-60), np.deg2rad(60))
        mujoco.mj_forward(self.model, self.data)

    def solve_ik(self, target_pos, max_iters=100, tol=1e-4, alpha=0.5):
        """Solve inverse kinematics for any number of joints"""
        dof_idxs = [self.model.jnt_dofadr[j] for j in range(self.num_joints)]

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[self.hand_id].copy()
            error = target_pos - current_pos

            if np.linalg.norm(error) < tol:
                break

            # Compute full Jacobian
            J = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, J, None, self.hand_id)
            J_reduced = J[:, dof_idxs]  # shape (3, num_joints)

            # Least-squares update
            dq = alpha * np.linalg.pinv(J_reduced) @ error

            # Apply update and clip to joint limits
            for i, dof in enumerate(dof_idxs):
                self.data.qpos[dof] = np.clip(self.data.qpos[dof] + dq[i],
                                              np.deg2rad(-60), np.deg2rad(60))
        mujoco.mj_forward(self.model, self.data)

    def sample_targets(self, num_samples=10):
        return self.candidate_targets.sample(num_samples).values

    def update_target(self, position):
        self.data.mocap_pos[0] = position
        mujoco.mj_forward(self.model, self.data)

    def update_nail(self, position):
        self.data.eq_active[0] = 0
        self.data.mocap_pos[1] = position
        mujoco.mj_forward(self.model, self.data)
        self.data.eq_active[0] = 1
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def get_obs(self):
        target_position = self.data.mocap_pos[0].copy()
        sensor_data = self.data.sensordata.copy()
        norm_target_position = zscore(
            target_position,
            self.hand_position_stats["mean"].values,
            self.hand_position_stats["std"].values,
        )
        norm_sensor_data = zscore(
            sensor_data,
            self.sensor_stats["mean"].values,
            self.sensor_stats["std"].values,
        )
        return norm_target_position, norm_sensor_data

    def get_hand_pos(self):
        return self.data.geom_xpos[self.hand_id].copy()

    def step(self, muscle_activations):
        self.data.ctrl[:] = muscle_activations
        mujoco.mj_step(self.model, self.data)

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] = True
            self.viewer.cam.lookat[:] = [0, -0.25, 0]
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -90
        else:
            if self.viewer.is_running():
                self.viewer.sync()
                time.sleep(self.model.opt.timestep)

    def get_force_at_eq(self, eq_name):
        eq_id = None
        for i in range(self.model.neq):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
            if name == eq_name:
                eq_id = i
                break
        if eq_id is None:
            raise RuntimeError(f"Equality constraint '{eq_name}' not found.")

        eq_type = self.model.eq_type[eq_id]
        eq_sizes = {
            mujoco.mjtEq.mjEQ_CONNECT: 3,
            mujoco.mjtEq.mjEQ_WELD: 6,
            mujoco.mjtEq.mjEQ_JOINT: 1,
            mujoco.mjtEq.mjEQ_TENDON: 1,
            mujoco.mjtEq.mjEQ_DISTANCE: 1,
        }
        constraint_dim = eq_sizes[eq_type]

        efc_start = 0
        for i in range(eq_id):
            prev_type = self.model.eq_type[i]
            efc_start += eq_sizes[prev_type]

        force_vec = self.data.efc_force[efc_start : efc_start + constraint_dim]
        return force_vec
    
    def get_joint_angles_deg(self):
        joint_angles = {}
        for j in range(self.num_joints):
            joint_name = self.model.joint(j).name
            qpos_index = self.model.jnt_dofadr[j]  # index in qpos
            joint_angles[joint_name] = np.rad2deg(self.data.qpos[qpos_index])
        return joint_angles

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
