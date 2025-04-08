print("Started Import")
import os
import torch

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig,OmegaConf
    
def get_dataset(config, fold_number):
    dataset_name = config["dataset_name"].lower()
    if dataset_name == "acevedo":
        from datasets.acevedo_dataloader import DualAcevedoImageDataModule
        dataset_class =  DualAcevedoImageDataModule
    elif dataset_name == "bone_marrow":
        from datasets.bone_marrow_dataloader import BoneMarrowImageDataModule
        dataset_class = BoneMarrowImageDataModule
    elif "imagenet_a" in dataset_name:
        from datasets.imagenet_a import ImageNetADataModule
        dataset_class = ImageNetADataModule
    else:
        raise ValueError("Unrecognized Dataset")
    if config["student"]:
        dataset = dataset_class(dataset_target="classifier",base_data_dir=config["base_data_dir"],stratify=config["stratify_dataset"],batch_size=config["classifier_batch_size"],k_fold=config["k_fold"],fold_nb=fold_number)
    else:
        dataset = dataset_class(dataset_target="ae",base_data_dir=config["base_data_dir"],stratify=config["stratify_dataset"],batch_size=config["ae_batch_size"])
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
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best_model",dirpath=config["checkpoint_dir"])
    trainer = pl.Trainer(max_epochs=config["ae_max_epoch"], callbacks=[checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path)
    print(result)

def train_classifier(config,logger,fold_number = -1):
    from experiments.classifier_experiment import ClassifierExperiment
    from models.image_classifier import ImageClassifier
    import wandb
    datamodule = get_dataset(config=config,fold_number = fold_number)
    model = ImageClassifier(num_classes=datamodule.nb_classes,encoder_name = config["image_encoder_name"],latent_dim=config['classifier_backbone_out_dim'],use_image_net_weights=config['use_image_net_weights'],dataset_type=datamodule.dataset_type)
    print(f"Distillation Loss config: {config['distillation_loss_config']}")
    class_weighting = datamodule.class_weighting if config["class_weighting"] else None
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
                                      different_learning_rate_for_classifier= config["different_learning_rate_for_classifier"],
                                      class_weighting = class_weighting,
                                      normalize_dm=config["normalize_distance_matrix"],
                                      label_smoothing=config["label_smoothing"],
                                      optimizer_name = config["optimizer_name"])
    run_id = wandb.run.id  
    checkpoint_dir = os.path.join(config["checkpoint_dir"], run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if config["take_last_checkpoint"]:
        classifier_checkpoint_callback = ModelCheckpoint(
            save_last=True,  # Ensures last epoch is always saved
            save_top_k=0,  # Prevents saving based on validation loss
            dirpath=checkpoint_dir,
        )
    else:
        if config["kill_classification_loss"]:
            monitor,mode = ("val_dm_knn_acc","max") if experiment.calculate_val_knn_accuracy else ("val_distillation_loss","min")
            classifier_checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=1, mode=mode,dirpath=checkpoint_dir,filename="{epoch}-{val_distillation_loss:.4f}-{val_loss_epoch}")
        else:
            classifier_checkpoint_callback = ModelCheckpoint(monitor="val_classification_loss", save_top_k=1, mode="min",dirpath=checkpoint_dir,filename="{epoch}-{val_accuracy:.4f}-{val_loss_epoch}")
    
    trainer = pl.Trainer(max_epochs=config["classifier_max_epoch"], callbacks=[classifier_checkpoint_callback], accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=logger, enable_progress_bar= not config["hpc"])
    trainer.fit(experiment, datamodule=datamodule)
    experiment.best_checkpoint_path = classifier_checkpoint_callback.last_model_path if config["take_last_checkpoint"] else classifier_checkpoint_callback.best_model_path
    result = trainer.test(experiment, datamodule, ckpt_path=experiment.best_checkpoint_path if experiment.best_checkpoint_path != "" else None)

def train_experiment(config):
    print("started WandBlogger import")
    from pytorch_lightning.loggers.wandb import WandbLogger
    print("finised WandBlogger import")
    is_k_fold = config["k_fold"] >= 2
    repeats = range(config["k_fold"]) if is_k_fold else range(config["repeat_exp"])
    if is_k_fold and isinstance(config.get("fold_nb",None),int):
        repeats = [config.get("fold_nb",None)]
        
    for i in repeats:
        BASE_LOG_DIR = "/ictstr01/home/icb/milad.bassil/" # os.getcwd()
        logger = WandbLogger(offline = False,log_model=False, name=config["name"] ,project=config['project_name'], entity="milad-research",save_dir=os.path.join(BASE_LOG_DIR, config['logs_save_dir']),tags = config["experiment_tags"])
        logger.log_hyperparams(config)

        print(f"Running experiment with name: {config['name']}, iteration {i+1}/{len(repeats)}")
        if not config["student"]:
            train_ae(config,logger)
        else:
            train_classifier(config=config,logger=logger,fold_number = i if is_k_fold else -1)
        logger.experiment.finish()
def find_config_with_name(d):
    if isinstance(d, dict):
        if "name" in d:
            return d  # Found the desired config
        for key, value in d.items():
            result = find_config_with_name(value)
            if result:
                return result  # Return immediately once found
    if "hydra" in d:
        del d["hydra"]
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


# from pytorch_lightning.callbacks import ModelCheckpoint

# class CustomModelCheckpoint(ModelCheckpoint):
#     def __init__(self, *args, **kwargs):
#         self.save_top_k = kwargs.get("top_k_to_save", 3)
#         if "top_k_to_save" in kwargs:
#             del kwargs["top_k_to_save"]
#         super().__init__(*args, **kwargs)

#         assert isinstance(self.save_top_k, int)
#         self.best_models = []  # Store tuples of (val_correct_epoch, val_loss_epoch, filepath)

#     def get_best_path(self):
#         """Return the path of the best model (highest val_correct_epoch, lowest val_loss_epoch for ties)."""
#         return self.best_models[0][2] if self.best_models else None

#     def _compare_metrics(self, current_metrics, trainer, pl_module):
#         """
#         Compare the current metrics (val_correct_epoch, val_loss_epoch) with the saved models.
#         Keeps the top `save_top_k` models, prioritizing val_correct_epoch and resolving ties using val_loss_epoch.
#         """
#         current_correct = current_metrics["val_correct_epoch"]
#         current_loss = current_metrics["val_loss_epoch"]

#         # Save the current model first
#         current_model_path = self._save_model(trainer, pl_module)

#         # Insert in the correct position based on sorting criteria
#         inserted = False
#         for i, (correct, loss, _) in enumerate(self.best_models):
#             if current_correct > correct or (current_correct == correct and current_loss < loss):
#                 self.best_models.insert(i, (current_correct, current_loss, current_model_path))
#                 inserted = True
#                 break
        
#         if not inserted and len(self.best_models) < self.save_top_k:
#             self.best_models.append((current_correct, current_loss, current_model_path))

#         # Ensure we only keep the top `save_top_k` models
#         self.best_models = sorted(self.best_models, key=lambda x: (-x[0], x[1]))[:self.save_top_k]  # Sort by correct (desc), loss (asc)
        
#     def on_validation_end(self, trainer, pl_module):
#         """
#         Called at the end of validation. This determines whether to save the model based on metrics.
#         """
#         logs = trainer.callback_metrics
#         if "val_correct_epoch" not in logs or "val_loss_epoch" not in logs:
#             return  # Avoid errors if metrics are missing

#         current_metrics = {
#             "val_correct_epoch": logs["val_correct_epoch"].item(),
#             "val_loss_epoch": logs["val_loss_epoch"].item()
#         }

#         # Compare and update best models
#         if pl_module.current_epoch != 0:
#             self._compare_metrics(current_metrics, trainer, pl_module)
#             self.custom_best_model_path = self.best_models[0][2]  # Update best model path

#             # Debug print to verify sorting behavior
#             print(f"Best models (sorted): {self.best_models}")

#     def _save_model(self, trainer, pl_module):
#         """
#         Use ModelCheckpoint's internal save logic to save the model.
#         """
#         name_metrics = {"epoch": trainer.current_epoch}
#         name_metrics.update(trainer.callback_metrics)
#         filepath = self.format_checkpoint_name(name_metrics)
#         self._save_checkpoint(trainer, filepath)
#         return filepath  # Return saved model path

# custom_colors = [
#     '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', 
#     '#984ea3', '#999999', '#e41a1c', '#dede00', '#8dd3c7',
#     '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
#     '#bc80bd', '#ccebc5', '#ffed6f', '#1f78b4', '#33a02c',
#     '#ffb3b3', '#b3b3ff', '#ffd700', '#7fc97f', '#beaed4'
# ]
# markers = ['o', '^', 's', 'D', 'P', 'X', '*', 'h', 'v', '<']
# def plot_and_log_2D_embedding(embedding,labels,name,log_wandb = True):

#     assert len(embedding) == len(labels)
#     le = LabelEncoder()
#     y = le.fit_transform(labels)  # Convert labels to integers
#     label_names = le.classes_  # Get label names for the legend
#     reducers = {"UMAP":umap.UMAP(n_components=2),"PHATE": phate.PHATE(n_components=2)}
#     for reducer_name,reducer in reducers.items():
#         embedding_2D = reducer.fit_transform(embedding)

#         # Create a scatter plot with matplotlib
#         plt.figure(figsize=(10, 7))

#         # Generate color map
#         # colors = cm.rainbow(np.linspace(0, 1, len(np.unique(y))))
#         #plt.cm.tab20.colors + plt.cm.tab20b.colors[:5]
#         cmap = ListedColormap(custom_colors)

#         # Plot each label with a specific color
#         for i, label in enumerate(np.unique(y)):
#             plt.scatter(embedding_2D[y == label, 0], embedding_2D[y == label, 1], marker=markers[i % len(markers)],
#                         color=cmap(i), label=label_names[label], alpha=0.7, s=30)

#         # Add legend to map colors to labels
#         plt.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')  
#         plt.title(f"{name} Encoder {reducer_name} Embeddings")
#         plt.xlabel(f"{reducer_name} Dimension 1")
#         plt.ylabel(f"{reducer_name} Dimension 2")
#         plt.tight_layout()
#         if log_wandb:
#             wandb.log({f"{name}_{reducer_name}": wandb.Image(plt)})
#         else:
#             plt.savefig(f"{name}_{reducer_name}", bbox_inches='tight')
#         plt.close()
