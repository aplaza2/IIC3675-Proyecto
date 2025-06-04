setup = {
    "DQN": {
        # https://huggingface.co/sb3/dqn-MountainCar-v0
        "env": "MountainCar-v0",
        "total_timesteps": 120000,
        "params": {
            "policy": "MlpPolicy",
            'buffer_size': 10000,
            'exploration_final_eps': 0.07,
            'exploration_fraction': 0.2,
            'gamma': 0.98,
            'gradient_steps': 8,
            'learning_rate': 0.004,
            'learning_starts': 1000,
            'policy_kwargs': {"net_arch": [256, 256]},
            'target_update_interval': 600,
            'train_freq': 16,
            # 'normalize': False
        }
    },
    "DDPG": {
        # https://huggingface.co/sb3/ddpg-MountainCarContinuous-v0
        "env": "MountainCarContinuous-v0",
        "total_timesteps": 300000,
        "params": {
            'policy': "MlpPolicy",
            'noise_std': 0.5,
            'noise_type': "ornstein-uhlenbeck", 
            'normalize': False
        },
    },
    "PPO_Continuous": {
        # https://huggingface.co/sb3/ppo-MountainCarContinuous-v0
        "env": "MountainCarContinuous-v0",
        "total_timesteps": 20000,
        "params": {
            'policy': "MlpPolicy",
            'batch_size': 256,
            'clip_range': 0.1,
            'ent_coef': 0.00429,
            'gae_lambda': 0.9,
            'gamma': 0.9999,
            'learning_rate': 7.77e-05,
            'max_grad_norm': 5,
            'n_envs': 1,
            'n_epochs': 10,
            'n_steps': 8,
            'normalize': True,
            'policy_kwargs': {"log_std_init": -3.29, "ortho_init": False},
            'use_sde': True,
            'vf_coef': 0.19,
            'normalize_kwargs': {'norm_obs': True, 'norm_reward': False}
        }
    },
    "PPO_Discrete": {
        # https://huggingface.co/sb3/ppo-MountainCar-v0
        "env": "MountainCar-v0",
        "total_timesteps": 1000000,
        "params": {
            'policy': "MlpPolicy",
            'ent_coef': 0.0,
            'gae_lambda': 0.98,
            'gamma': 0.99,
            # 'n_envs': 16,
            'n_epochs': 4,
            'n_steps': 16,
            # 'normalize': True,
            'normalize_kwargs': {'norm_obs': True, 'norm_reward': False}
        }
    }
}