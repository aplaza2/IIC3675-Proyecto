from codecarbon import track_emissions
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import pandas as pd
import constants as consts
from agentsSetup import setup
from collections import defaultdict


@track_emissions()
def learn_and_track(model, total_timesteps):
    model.learn(total_timesteps=total_timesteps, progress_bar=True)


def save_results(results, model_name):
    for i in range(0, len(results)):
        results[i] /= consts.N_EXPERIMENTS
    
    results = pd.DataFrame(results.items(), columns=["Episode", "Average Length"])
    filename = f"{consts.RESULTS_DIR}/{model_name}.xlsx"
    results.to_excel(filename, index=False)


def get_model_name(agent, env_type):
    model_name = agent.__name__
    if env_type is not None:
        model_name += f"_{env_type}"
    return model_name


def verify_experiment_params(agent, device, env_type):
    if device not in consts.ALLOWED_DEVICES:
        raise ValueError(f"device debe ser uno de {consts.ALLOWED_DEVICES}, no '{device}'.")

    if env_type not in consts.ALLOWED_ENV_TYPES:
        raise ValueError(f"env_type debe ser uno de {consts.ALLOWED_ENV_TYPES}, no '{env_type}'.")
    
    model_name = get_model_name(agent, env_type)
    if model_name not in setup:
        raise ValueError(f"El modelo '{model_name}' no está configurado en 'setup'.")


def run_experiment(agent, device: consts.DeviceType = "cpu", env_type: consts.EnvType = None):
    verify_experiment_params(agent, device, env_type)

    results = defaultdict(int)
    model_name = get_model_name(agent, env_type)
    
    for n_experiment in range(consts.N_EXPERIMENTS):
        model_setup = setup[model_name]
        filename = f"{consts.MONITORS_DIR}/{model_name}{n_experiment}_results"

        env = gym.make(model_setup["env"])
        env = Monitor(env, filename=filename)
        policy = model_setup["params"].pop("policy")
        agent = agent(policy, env, verbose=1, device=device, **model_setup["params"])

        learn_and_track(model=agent, total_timesteps=model_setup["total_timesteps"])

        for n, episode_length in enumerate(env.get_episode_lengths()):
            results[n] += episode_length

    save_results(results, model_name)
