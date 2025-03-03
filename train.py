import os
import torch
import wandb
import yaml
from datasets.acevedo_dataloader import DualAcevedoImageDataModule
from experiments.classifier_experiment import ClassifierExperiment
from experiments.vae_experiment import VAE_exp
from models.beta_variational_autoencoder import BetaVAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from models.image_classifier import ImageClassifier
import hydra
from omegaconf import DictConfig,OmegaConf
    
from configs.default_config import DefaultExperimentValues
DEV = DefaultExperimentValues

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_ae(config,logger):
    datamodule = DualAcevedoImageDataModule(dataset_target="ae",stratify=config["stratify_dataset"],batch_size=config["ae_batch_size"])
    model = BetaVAE(latent_dim=config["ae_latent_dim"])
    experiment = VAE_exp(model=model, beta=config['vae_beta'],lr=config["ae_lr"],weight_decay=config['ae_weight_decay'])
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best_model")
    trainer = pl.Trainer(max_epochs=config["ae_max_epoch"], callbacks=[checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)
    print(result)

def train_classifier(config,logger):
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier",stratify=config["stratify_dataset"],batch_size=config["classifier_batch_size"])
    from models.DinoBloom.dinobloom_hematology_feature_extractor import get_dino_bloom_w_resize
 
    model = ImageClassifier(num_classes=config['num_classes'],latent_dim=config['classifier_backbone_out_dim'],dino_bloom_backbone=get_dino_bloom_w_resize() if config["use_dinobloom_as_backbone"] else None)
    experiment = ClassifierExperiment(model=model,
                                      ae_encoder_path=config['teacher_model_path'],
                                      lr=config["classifier_lr"],
                                      weight_decay=config["classifier_weight_decay"],
                                      distillation_weight=config["distillation_loss_weight"],
                                      dist_fnc_name=config["distance_fn_name"],
                                      distillation_loss_name=config["distillation_loss_name"],
                                      distillation_loss_config = config["distillation_loss_config"],
                                      teacher_model_type = config["teacher_model_type"],
                                      kill_classification_loss=config["kill_classification_loss"],
                                      classifier_checkpoint_path=config["load_classifier_checkpoint_path"],
                                      freeze_classifier_backbone = config["freeze_classifier_backbone"])
    if config["take_last_checkpoint"]:
        classifier_checkpoint_callback = ModelCheckpoint(
            save_last=True,  # Ensures last epoch is always saved
            save_top_k=0  # Prevents saving based on validation loss
        )
    else:
        classifier_checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", save_top_k=1, mode="min",filename="{epoch}-{val_accuracy:.4f}-{val_loss_epoch}")
    
    trainer = pl.Trainer(max_epochs=config["classifier_max_epoch"], callbacks=[classifier_checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.last_model_path if config["take_last_checkpoint"] else classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)

def train_experiment(config):
    for i in range(config["repeat_exp"]):
        logger = WandbLogger(log_model=False, name=config["name"] ,project=config['project_name'], entity="milad-research",save_dir=os.path.join(os.getcwd(), config['logs_save_dir'] ))
        logger.log_hyperparams(config)

        print(f"Running experiment with name: {config['name']}, iteration {i+1}/{config['repeat_exp']}")
        if not config["student"]:
            train_ae(config,logger)
        else:
            train_classifier(config,logger)
        wandb.finish()


@hydra.main(version_base=None)
def main(config: DictConfig):
    config = OmegaConf.to_container(config)
    print(f"hydra config:{config}")
    train_experiment(config)

    
    
if __name__ == "__main__":
    main()
#python your_script.py --config-dir /path/to/configs
