print("Started Import")
import os
import torch
import wandb
import yaml



import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import hydra
from omegaconf import DictConfig,OmegaConf
    
def get_dataset(config):
    dataset_name = config["dataset_name"]
    if dataset_name == "Acevedo":
        from datasets.acevedo_dataloader import DualAcevedoImageDataModule
        dataset_class =  DualAcevedoImageDataModule
    elif dataset_name == "bone_marrow":
        from datasets.bone_marrow_dataloader import BoneMarrowImageDataModule
        dataset_class = BoneMarrowImageDataModule
    elif "imagenet_a" in dataset_name.lower():
        from datasets.imagenet_a import ImageNetADataModule
        dataset_class = ImageNetADataModule
    else:
        raise ValueError("Unrecognized Dataset")
    if config["student"]:
        dataset = dataset_class(dataset_target="classifier",stratify=config["stratify_dataset"],batch_size=config["classifier_batch_size"])
    else:
        dataset = dataset_class(dataset_target="ae",stratify=config["stratify_dataset"],batch_size=config["ae_batch_size"])
    return dataset
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_ae(config,logger):
    from experiments.vae_experiment import VAE_exp
    from models.beta_variational_autoencoder import BetaVAE
    datamodule = get_dataset(config=config)
    model = BetaVAE(latent_dim=config["ae_latent_dim"])
    experiment = VAE_exp(model=model, beta=config['vae_beta'],lr=config["ae_lr"],weight_decay=config['ae_weight_decay'])
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best_model")
    trainer = pl.Trainer(max_epochs=config["ae_max_epoch"], callbacks=[checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)
    print(result)

def train_classifier(config,logger):
    from experiments.classifier_experiment import ClassifierExperiment
    from models.image_classifier import ImageClassifier
    datamodule = get_dataset(config=config)
    model = ImageClassifier(num_classes=datamodule.nb_classes,encoder_name = config["image_encoder_name"],latent_dim=config['classifier_backbone_out_dim'],use_image_net_weights=config['use_image_net_weights'],dataset_type=datamodule.dataset_type)
    print(f"Distillation Loss config: {config['distillation_loss_config']}")
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
                                      kill_distillation_loss=config["kill_distillation_loss"],
                                      classifier_checkpoint_path=config["load_classifier_checkpoint_path"],
                                      freeze_classifier_backbone = config["freeze_classifier_backbone"],
                                      different_learning_rate_for_classifier= config["different_learning_rate_for_classifier"])
    if config["take_last_checkpoint"]:
        classifier_checkpoint_callback = ModelCheckpoint(
            save_last=True,  # Ensures last epoch is always saved
            save_top_k=0  # Prevents saving based on validation loss
        )
    else:
        if config["kill_classification_loss"]:
            classifier_checkpoint_callback = ModelCheckpoint(monitor="val_distillation_loss", save_top_k=1, mode="min",filename="{epoch}-{val_distillation_loss:.4f}-{val_loss_epoch}")
        else:
            classifier_checkpoint_callback = ModelCheckpoint(monitor="val_loss_epoch", save_top_k=1, mode="min",filename="{epoch}-{val_accuracy:.4f}-{val_loss_epoch}")
    
    trainer = pl.Trainer(max_epochs=config["classifier_max_epoch"], callbacks=[classifier_checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.last_model_path if config["take_last_checkpoint"] else classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path if experiment.best_checkpoint_path != "" else None)

def train_experiment(config):
    print("started WandBlogger import")
    from pytorch_lightning.loggers.wandb import WandbLogger
    print("finised WandBlogger import")
    for i in range(config["repeat_exp"]):
        logger = WandbLogger(log_model=False, name=config["name"] ,project=config['project_name'], entity="milad-research",save_dir=os.path.join(os.getcwd(), config['logs_save_dir'] ))
        logger.log_hyperparams(config)

        print(f"Running experiment with name: {config['name']}, iteration {i+1}/{config['repeat_exp']}")
        if not config["student"]:
            train_ae(config,logger)
        else:
            train_classifier(config,logger)
        wandb.finish()
def find_config_with_name(d):
    if isinstance(d, dict):
        if "name" in d:
            return d  # Found the desired config
        for key, value in d.items():
            result = find_config_with_name(value)
            if result:
                return result  # Return immediately once found
    return None  # If no matching dict is found

@hydra.main(version_base=None)
def main(config: DictConfig):
    og_config = OmegaConf.to_container(config)
    config = find_config_with_name(og_config)
    print(f"hydra config:{config}")
    train_experiment(config)

    
    
if __name__ == "__main__":
    print("Finished Import")
    main()
#python your_script.py --config-dir /path/to/configs
