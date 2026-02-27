
import mlflow
from omegaconf import OmegaConf



def flatten_dict(d, parent=""):
    result = {}

    for key, value in d.items():

        new_key = f"{parent}.{key}" if parent else key

        if isinstance(value, dict):
            result.update(flatten_dict(value, new_key))
        else:
            result[new_key] = value

    return result

def setup_mlflow(config):
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    return mlflow.start_run(run_name=None)

def log_config(config):
    container = OmegaConf.to_container(config, resolve=True)
    flat_config = flatten_dict(container)
    mlflow.log_params(flat_config)

def log_metrics(metrics, step, prefix=""):
    for key, value in metrics.items():
        new_key = f"{prefix}.{key}" if prefix else key
        mlflow.log_metric(new_key, value, step=step)        
    

