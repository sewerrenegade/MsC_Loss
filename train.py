
import torch
from datasets.acevedo_dataloader import DualAcevedoImageDataModule
from experiments.classifier_experiment import ClassifierExperiment
from experiments.vae_experiment import VAE_exp
from models.beta_variational_autoencoder import BetaVAE
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from models.image_classifier import ImageClassifier

BEST_AE_ENCODER_MODEL = r"lightning_logs\version_13\checkpoints\best_model.ckpt"
MODES_OF_OPERATION =  ["train_ae","train_classifier","train_moor_topo_ae_classifier","train_prop_topo_ae_classifier","train_mds_ae_classifier"]#
AE_LATENT_DIM = 10 
NUM_OF_CLASSES = 8


logger = None


def train_ae():
    datamodule = DualAcevedoImageDataModule(dataset_target="ae")
    model = BetaVAE(latent_dim=AE_LATENT_DIM)
    experiment = VAE_exp(model= model,beta=0.0)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="best_model"
    )
    trainer = pl.Trainer(max_epochs=1,callbacks=[checkpoint_callback],accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule= datamodule)
    experiment.best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment,datamodule,ckpt_path=experiment.best_checkpoint_path)
    print(result)
    

CLASSIFER_MAX_EPOCH = 100
def train_classifier():
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier")
    model = ImageClassifier(num_classes=NUM_OF_CLASSES)
    experiment = ClassifierExperiment(model= model)
    classifier_checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy", save_top_k=1, mode="max", filename="best_model"
            )
    trainer = pl.Trainer(max_epochs=CLASSIFER_MAX_EPOCH,callbacks=[classifier_checkpoint_callback],accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule = datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment,datamodule,ckpt_path=experiment.best_checkpoint_path)
    
def train_classifier_with_distillation(path_to_ae = "",distillation_loss_name = 'prop_topo'):
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier")
    model = ImageClassifier(num_classes=NUM_OF_CLASSES)
    experiment = ClassifierExperiment(model= model,distillation_loss_name=distillation_loss_name,ae_encoder = get_ae_encoder_from_path(path_to_ae),topo_weight=0.01)
    classifier_checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy", save_top_k=1, mode="max", filename="best_model"
            )
    trainer = pl.Trainer(max_epochs=CLASSIFER_MAX_EPOCH,callbacks=[classifier_checkpoint_callback],accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule = datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment,datamodule,ckpt_path=experiment.best_checkpoint_path)

def get_ae_encoder_from_path(path_to_ae):
    ae_model = BetaVAE(latent_dim=AE_LATENT_DIM)
    ae_model.load_state_dict({k.replace("model.", ""): v for k,v in torch.load(path_to_ae, map_location=torch.device("cpu"))["state_dict"].items()})
    ae_model.eval()
    return ae_model.encoder
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a mode of operation.")
    parser.add_argument(
        "--mode", 
        choices=MODES_OF_OPERATION, 
        help="Choose a mode of operation from the predefined list.",
        default="train_ae"
    )
    args = parser.parse_args()
    mode = args.mode
    print(f"Running {mode}")
    wnb_logger = WandbLogger(log_model=False,
                        name = mode,
                        project="Connectivity_Distillation_Experiment",
                        entity = "milad-research")
    wnb_logger.experiment.summary["version_number_sum"] = "i <3 light"
    wnb_logger.experiment.summary["version_number_sum"] = wnb_logger.version
    wnb_logger.experiment.summary["experiment_type"] = mode

    
    logger = wnb_logger

    if mode == 'train_ae':
        train_ae()
    elif mode == 'train_classifier':
        train_classifier()
    elif mode == 'train_prop_topo_ae_classifier':
        train_classifier_with_distillation(distillation_loss_name='prop_topo')
    elif mode == 'train_moor_topo_ae_classifier':
        train_classifier_with_distillation(distillation_loss_name='moor_topo') 
    elif mode == 'train_mds_ae_classifier':
        train_classifier_with_distillation(distillation_loss_name='mds')     
    #train_classifier_with_distillation(r"lightning_logs\version_13\checkpoints\best_model.ckpt")

