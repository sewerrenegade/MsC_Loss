import os
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from hydra import initialize, compose
from omegaconf import OmegaConf

from train import train_ae, train_classifier
def sweep1():
    wandb.init()
    print(f"wandb config:{dict(wandb.config)}")
    wandb.finish()
    # config = OmegaConf.to_container(config)
    # print(f"hydra config:{config}")
    # train_experiment(config)
    

def sweep():
    for i in range(1):
        wandb.init()
        sweep_overloads = dict(wandb.config)
        print(f"Sweep overloads {sweep_overloads}")
        with initialize(config_path=sweep_overloads["config_path"],version_base=None):
            cfg = compose(config_name=sweep_overloads["config_name"])
        config = OmegaConf.to_container(cfg)
        config.update(sweep_overloads)
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