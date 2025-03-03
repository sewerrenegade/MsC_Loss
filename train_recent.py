import torch
import wandb
import yaml
from datasets.acevedo_dataloader import DualAcevedoImageDataModule
from experiments.classifier_experiment import ClassifierExperiment
from experiments.vae_experiment import VAE_exp
from models.beta_variational_autoencoder import BetaVAE
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from models.image_classifier import ImageClassifier

from configs.default_config import DefaultExperimentValues
DEV = DefaultExperimentValues

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_ae(config):
    datamodule = DualAcevedoImageDataModule(dataset_target="ae")
    model = BetaVAE(latent_dim=config.get('AE_LATENT_DIM',DEV.DEFAULT_AE_LATENT_DIM))
    experiment = VAE_exp(model=model, beta=config.get('beta',DEV.DEFAULT_BETA),lr=config.get("LR",DEV.DEFAULT_CLASSIFIER_LR),weight_decay=config.get("WEIGHT_DECAY",DEV.DEFAULT_AE_WIEGHT_DECAY))
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best_model")
    trainer = pl.Trainer(max_epochs=config.get('AE_MAX_EPOCH',DEV.DEFAULT_AE_MAX_EPOCH), callbacks=[checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config.get("HPC",DEV.DEFAULT_HPC))
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)
    print(result)

def train_classifier(config):
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier")
    model = ImageClassifier(num_classes=config.get('NUM_CLASSES',DEV.DEFAULT_NUM_CLASSES))
    experiment = ClassifierExperiment(model=model,
                                      ae_encoder_path=config.get("AE_MODEL_PATH",DEV.DEFAULT_AE_MODEL_PATH),
                                      lr=config.get("LR",DEV.DEFAULT_CLASSIFIER_LR),
                                      weight_decay=config.get("WEIGHT_DECAY",DEV.DEFAULT_CLASSIFIER_WIEGHT_DECAY),
                                      distillation_weight=config.get("topo_weight",DEV.DEFAULT_TOPO_WEIGHT),
                                      dist_fnc_name=config.get("DISTANCE_FN_NAME",DEV.DEFAULT_DISTANCE_FN_NAME),
                                      distillation_loss_name=config.get("DISTILLATION_LOSS_FN_NAME",DEV.DEFAULT_DISTILLATION_LOSS_NAME),
                                      distillation_loss_config = config.get("DISTILLATION_LOSS_CONFIG",DEV.DEFAULT_DISTILLATION_LOSS_CONFIG))
    classifier_checkpoint_callback = ModelCheckpoint(monitor="val_accuracy", save_top_k=1, mode="max", filename="best_model")
    trainer = pl.Trainer(max_epochs=config.get('CLASSIFIER_MAX_EPOCH',DEV.DEFAULT_CLASSIFIER_MAX_EPOCH), callbacks=[classifier_checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config.get("HPC",DEV.DEFAULT_HPC))
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)

def train_experiment(config):
    mode = config.get("mode",DEV.DEFAULT_MODE)
    for i in range(config.get('REPEAT_EXPERIMENT',DEV.DEFAULT_REPEAT_EXPERIMENT)):
        global logger
        logger = WandbLogger(log_model=False, name=f"{config.get('mode',DEV.DEFAULT_MODE)}_{config.get('DISTILLATION_LOSS_FN_NAME',DEV.DEFAULT_MODE)}", project=config.get('PROJECT_NAME',DEV.DEFAULT_MODE), entity="milad-research",save_dir="wandb_logs")
        logger.log_hyperparams(config)

        print(f"Running {mode}, iteration {i+1}/{config.get('REPEAT_EXPERIMENT',DEV.DEFAULT_REPEAT_EXPERIMENT)}")
        if mode == 'train_ae':
            train_ae(config)
        elif mode == 'train_classifier':
            train_classifier(config)
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Select a mode of operation.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    config = load_config(args.config)
    train_experiment(config)

if __name__ == "__main__":
    main()
