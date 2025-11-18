import gymnasium as gym

def make_env(mode, plant_config, env_config, llc_config=None, seed=None):
    valid_modes = ['direct', 'continuous_llctrl', 'discrete_llctrl']
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes are: {valid_modes}")
    
    def _init():
        # Initialize plant
        plant_cls = plant_config["class"]
        plant_kwargs = plant_config.get("kwargs", {})
        plant = plant_cls(**plant_kwargs)

        # Initialize low-level controller if provided
        if mode == 'direct' or llc_config is None:
            low_lv_ctrl = None
        else:
            llc_cls = llc_config["class"]
            llc_kwargs = llc_config.get("kwargs", {})
            low_lv_ctrl = llc_cls(plant=plant, **llc_kwargs)

        # Initialize environment
        env_cls = env_config["class"]
        env_kwargs = env_config.get("kwargs", {}).copy()
        env_kwargs["low_lv_ctrl"] = low_lv_ctrl
        env_kwargs["discrete_ctrl"] = True if mode == 'discrete_llctrl' else False
        if mode == 'discrete_llctrl':
            env_kwargs["reward_params"]["l0_weight"] = 0.0
        env = env_cls(plant=plant, **env_kwargs)

        # Reset environment with seed
        env.reset(seed=seed)
        return env

    return _init
