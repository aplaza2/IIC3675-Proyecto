from stable_baselines3 import DQN, DDPG, PPO
from functions import run_experiment


if __name__ == "__main__":
    run_experiment(DQN, device="cpu", env_type=None)
