import os
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from hydra import initialize, compose
from omegaconf import OmegaConf
from train import train_ae, train_classifier

def reconstruct_nested_dict(flat_dict):
    nested_dict = {}

    for key, value in flat_dict.items():
        parts = key.split(".")  # Split by dots
        d = nested_dict

        for part in parts[:-1]:  # Iterate through keys except last one
            d = d.setdefault(part, {})  # Create dict if not exists
        
        d[parts[-1]] = value  # Assign final value

    return nested_dict

def merge_dicts(original, updates):
    """Recursively merges `updates` into `original` without overwriting nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            merge_dicts(original[key], value)  # Recursively merge nested dictionaries
        else:
            original[key] = value  # Overwrite or add new key-value pair

def sweep():
    for i in range(1):
        wandb.init()
        sweep_overloads = dict(wandb.config)
        print(f"Sweep overloads {sweep_overloads}")
        with initialize(config_path=sweep_overloads["config_path"],version_base=None):
            cfg = compose(config_name=sweep_overloads["config_name"])
        config = OmegaConf.to_container(cfg)
        nested_dict = reconstruct_nested_dict(sweep_overloads)
        merge_dicts(config, nested_dict)
        print(f"Full config: {config}")
        global logger
        logger = WandbLogger(log_model=False, name=config["name"] ,
                             project=config['project_name'],
                             entity="milad-research",
                             save_dir=os.path.join(os.getcwd(), config['logs_save_dir'] ))
        
        logger.log_hyperparams(config)

        print(f"Running experiment with name: {config['name']}, iteration {i+1}/{config['repeat_exp']}")
        if not config["student"]:
            train_ae(config,logger)
        else:
            train_classifier(config,logger)
        wandb.finish()
        
if __name__ == "__main__":
    sweep()