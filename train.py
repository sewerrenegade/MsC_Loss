
import torch
from datasets.acevedo_dataloader import DualAcevedoImageDataModule
from experiments.classifier_experiment import ClassifierExperiment
from experiments.vae_experiment import VAE_exp
from models.beta_variational_autoencoder import BetaVAE
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.image_classifier import ImageClassifier

AE_LATENT_DIM = 10 
NUM_OF_CLASSES = 8
def train_vae():
    datamodule = DualAcevedoImageDataModule(dataset_target="vae")
    model = BetaVAE(latent_dim=AE_LATENT_DIM)
    experiment = VAE_exp(model= model)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", save_top_k=1, mode="min", filename="best_model"
    )
    trainer = pl.Trainer(max_epochs=1,callbacks=[checkpoint_callback],accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule= datamodule)
    best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment,datamodule,ckpt_path=best_checkpoint_path)
    print(result)
    
    
def train_classifier():
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier")
    model = ImageClassifier(num_classes=NUM_OF_CLASSES)
    experiment = ClassifierExperiment(model= model)
    trainer = pl.Trainer(max_epochs=50,accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule = datamodule)
    
def train_classifier_with_distillation(path_to_ae = ""):
    datamodule = DualAcevedoImageDataModule(dataset_target="classifier")
    model = ImageClassifier(num_classes=NUM_OF_CLASSES)
    experiment = ClassifierExperiment(model= model,ae_encoder = get_ae_encoder_from_path(path_to_ae),topo_weight=0.01)
    trainer = pl.Trainer(max_epochs=50,accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)  # Use gpus=1 if GPU is available
    trainer.fit(experiment, datamodule = datamodule)

def get_ae_encoder_from_path(path_to_ae):
    ae_model = BetaVAE(latent_dim=AE_LATENT_DIM)
    ae_model.load_state_dict({k.replace("model.", ""): v for k,v in torch.load(path_to_ae, map_location=torch.device("cpu"))["state_dict"].items()})
    ae_model.eval()
    return ae_model.encoder
    

if __name__ == "__main__":
    train_vae()
    #train_classifier_with_distillation(r"lightning_logs\version_13\checkpoints\best_model.ckpt")

