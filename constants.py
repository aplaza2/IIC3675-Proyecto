import os
from typing import Literal

N_EXPERIMENTS = 1

_actual_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_actual_dir, "results")
MONITORS_DIR = os.path.join(_actual_dir, "results/monitors")

ALLOWED_DEVICES = ["cpu", "cuda"]
ALLOWED_ENV_TYPES = [None, "Discrete", "Continuous"]

DeviceType = Literal["cpu", "cuda"]
EnvType = Literal[None, "Discrete", "Continuous"]