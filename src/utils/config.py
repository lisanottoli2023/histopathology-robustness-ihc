from omegaconf import OmegaConf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_config(model_override=None, experiment_override=None):
    config = OmegaConf.load(CONFIG_DIR / "base.yaml")

    if model_override:
        model_config_path = CONFIG_DIR / "models" / f"{model_override}.yaml"
        model_config = OmegaConf.load(model_config_path)
        config = OmegaConf.merge(config, model_config)

    if experiment_override:
        experiment_config_path = CONFIG_DIR / "experiments" / f"{experiment_override}.yaml"
        experiment_config = OmegaConf.load(experiment_config_path)
        config = OmegaConf.merge(config, experiment_config)
        
    return config

