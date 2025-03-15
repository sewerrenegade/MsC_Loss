import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import umap
import re
import os
import inspect
import wandb
from models.learnable_uncertainty_based_weighting import UncertWeighting
from models.beta_variational_autoencoder import BetaVAE
from MsC_Loss.multi_scale_connectivity_loss import TopologicalZeroOrderLoss
from MsC_Loss.mse_wrapper import MSELoss
from MsC_Loss.moor_topo_reg import TopologicalSignatureDistance
from MsC_Loss.mds_regularizer import MDSLoss
from MsC_Loss.static_distance_matrix_metrics import StaticDistanceMatrixMetricCalculator
from models.image_classifier import ImageClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#from models.basic_distance_fns import pairwise_euclidean_distance,pairwise_cosine_similarity
from MsC_Loss.distance_functions import EuclideanDistance,L1Distance,CosineDistance,UMAP_Distance, LpDistance
import numbers
from dect.ect_loss import ECT_Loss
import numpy as np

DISTILLATION_LOSS_NAMES = ['prop_topo','moor_topo','mds','random','l2','ect','',None]
class ClassifierExperiment(pl.LightningModule):
    def __init__(self, model: ImageClassifier,ae_encoder_path = None, lr=0.001,distillation_weight = 0.0,dist_fnc_name = "euclidean",distillation_loss_name = None,teacher_model_type = "",distillation_loss_config = {},weight_decay=0.00001,kill_classification_loss = False,kill_distillation_loss = False,classifier_checkpoint_path = "",freeze_classifier_backbone = False, different_learning_rate_for_classifier = False): 
        super(ClassifierExperiment, self).__init__()
        assert distillation_loss_name in DISTILLATION_LOSS_NAMES
        self.distillation_loss_name = distillation_loss_name
        self.classifier_model = model
        wandb.log({"backbone_out_dim":self.model.latent_dim,"number_of_parameters_in_classifier_head":self.model.number_of_parameters_in_classifier,"number_of_parameters_in_image_encoder":self.model.number_of_parameters_in_image_encoder,"total_number_of_parameters_in_classifier":self.model.number_of_parameters_in_classifier+self.model.number_of_parameters_in_image_encoder})
        self.dataset_type = self.classifier_model.dataset_type
        self.teacher_model_type = teacher_model_type
        self.kill_classification_loss = kill_classification_loss
        self.kill_distillation_loss = kill_distillation_loss
        self.topo_weight = self.parse_and_apply_distillation(distillation_weight)
        self.ae_encoder_path = ae_encoder_path
        self.freeze_classifier_backbone = freeze_classifier_backbone
        self.different_learning_rate_for_classifier = different_learning_rate_for_classifier
        self.dist_fn = ClassifierExperiment.get_distance_fn(dist_fnc_name)
        self.classifier_checkpoint_path = classifier_checkpoint_path
        self.load_from_checkpoint_file()
        if self.freeze_classifier_backbone:
            self.classifier_model.freeze_backbone()
        self.teacher = self.get_teacher()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.dist_loss_fn =  self.get_normalized_distillation_loss_fn(distillation_loss_config)  #get_distillation_loss_fnget_distillation_loss_fn
        self.loss_fn = self.combined_loss_fn
        self.test_phase_data = {"labels":[],"embeddings":[],"logits": []}
        self.lr = lr
        self.best_checkpoint_path = None
        self.weight_decay= weight_decay

    def parse_and_apply_distillation(self,topo_weight):
        if ClassifierExperiment.is_scalar(topo_weight):
            pass
        if topo_weight == "uncert_based":
            self.uncertainty_weighting_module = UncertWeighting(n_losses = 2)
        else:   
            self.uncertainty_weighting_module = None
        if isinstance(topo_weight,dict):
            if "on_off" in topo_weight:
                self.scheduler = OnOffScheduler(patience=topo_weight.get("patience",10),low_weight=topo_weight.get("low_weight",0.0))
                self.kill_classification_loss = self.scheduler.active
                return self.scheduler.get_distillation_weight()
            else:
                self.scheduler = None
        return topo_weight
            
    def on_validation_epoch_end(self):
        """Hook to execute code at the end of every validation epoch."""
        # Ensure the logger is a WandB logger
        if self.scheduler:
            try:
                # Retrieve the latest logged "val_distillation_loss"
                history = self.logger.experiment.history  # Get history dictionary
                if "val_distillation_loss" in history:
                    latest_val_distillation_loss = history["val_distillation_loss"][-1]  # Get last logged value

                    # Update some value based on latest val_distillation_loss
                    self.scheduler.update(latest_val_distillation_loss)
                    self.topo_weight = self.scheduler.get_distillation_weight()
                    self.kill_classification_loss =  self.scheduler.active
                    print(f"Updated some_value with latest val_distillation_loss: {self.some_value}")
                else:
                    raise KeyError("val_distillation_loss was never logged")

            except Exception as e:
                print(f"Failed to retrieve val_distillation_loss from WandB: {e}")
                
    def get_teacher(self):
        if self.teacher_model_type == "ae":
            return ClassifierExperiment.get_ae_encoder_from_path(self.ae_encoder_path) if self.distillation_loss_name !="" and not self.distillation_loss_name is None else None
        elif "dino" in self.teacher_model_type:
            from models.DinoV2.dinov2 import get_dino_w_resize_and_model_nb_of_params,get_dino_size
            size = get_dino_size(self.teacher_model_type)  
            model,nb_parameters,out_dim = get_dino_w_resize_and_model_nb_of_params(size=size,target_data=self.dataset_type)
            wandb.log({"nb_parameters_in_teacher_model":nb_parameters,"teacher_output_dim":out_dim})
            return model
        else:
            return None
        

    def load_from_checkpoint_file(self):
        if self.classifier_checkpoint_path and os.path.exists(self.classifier_checkpoint_path):
            print(f"Loading weights of classifier from {self.classifier_checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.classifier_checkpoint_path, map_location="cpu")
            
            # Extract state dict and remove "classifier_model." prefix
            state_dict = checkpoint.get("state_dict", checkpoint)
            new_state_dict = {k.replace("classifier_model.", ""): v for k, v in state_dict.items()}
            
            # Load model state dict
            missing, unexpected = self.classifier_model.load_state_dict(new_state_dict, strict=False)
            
            # Print warnings for missing/unexpected keys
            if missing:
                print("Warning: Missing keys during checkpoint loading:", missing)
            if unexpected:
                print("Warning: Unexpected keys in checkpoint:", unexpected)
        else:
            print(f"Checkpoint file {self.classifier_checkpoint_path} not found or path is empty.")

    @staticmethod
    def get_distance_fn(dist_fn_name):
        if dist_fn_name == "euclidean":
            return EuclideanDistance()
        elif dist_fn_name == "cosine":
            return CosineDistance()
        elif dist_fn_name == 'umap_similarity':
            return UMAP_Distance()
        elif dist_fn_name == 'manhattan':
            return L1Distance()
        elif 'lp=' in dist_fn_name:
            return LpDistance(p=LpDistance.extract_lp_value(dist_fn_name))
        elif dist_fn_name == "" or dist_fn_name is None:
            return None
        else:
            raise ValueError(f"Distance function {dist_fn_name} is unknown")
    @staticmethod
    def get_ae_encoder_from_path(path_to_ae):
        if path_to_ae is None or path_to_ae == "":
            return None
        else:
            ae_model = BetaVAE(latent_dim=64)
            ae_model.load_state_dict({k.replace("model.", ""): v for k,v in torch.load(path_to_ae, map_location=torch.device("cpu"))["state_dict"].items()})
            ae_model.eval()
            return ae_model.encoder
        
    @staticmethod
    def filter_valid_kwargs(cls, config):
        """Filter config to only include valid parameters of the given class constructor."""
        valid_params = inspect.signature(cls).parameters
        return {k: v for k, v in config.items() if k in valid_params}

    def get_distillation_loss_fn(self, config):
        if self.distillation_loss_name == DISTILLATION_LOSS_NAMES[0]:
            loss_class = TopologicalZeroOrderLoss
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[1]:
            loss_class = TopologicalSignatureDistance
            # Additional fixed arguments that should always be passed
            config["match_edges"] = "symmetric"
            config["to_gpu"] = torch.cuda.is_available()
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[2]:
            return MDSLoss()  # No config needed
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[4]:
            return MSELoss()  # No config needed
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[5]:
            return ECT_Loss()  # No config needed
        elif self.distillation_loss_name in (DISTILLATION_LOSS_NAMES[-1], DISTILLATION_LOSS_NAMES[-2]):
            return None
        else:
            raise NotImplementedError(f"{self.distillation_loss_name} is not implemented")

        # Filter out invalid keys
        filtered_config = ClassifierExperiment.filter_valid_kwargs(loss_class, config)
        print(f"Getting distillation loss: {self.distillation_loss_name}, with config {config}")
        # Instantiate the loss function with only valid kwargs
        return loss_class(**filtered_config)
    def get_normalized_distillation_loss_fn(self, config):
        loss_fn = self.get_distillation_loss_fn(config)
        scale_factors = {
            DISTILLATION_LOSS_NAMES[0]: 300,
            DISTILLATION_LOSS_NAMES[1]: 16,
            DISTILLATION_LOSS_NAMES[2]: 12,
            DISTILLATION_LOSS_NAMES[4]: 0.4,
            DISTILLATION_LOSS_NAMES[5]: 0.01
        }
        
        # Default scale is 1
        scale = scale_factors.get(self.distillation_loss_name, 1)

        # Wrapper function that passes any number of arguments to loss_fn
        def calc_loss_and_normalize(*args, **kwargs):
            loss = loss_fn(*args, **kwargs)  # Pass all args and kwargs to loss_fn
            if isinstance(loss, tuple):
                loss = (loss[0] * scale, *loss[1:])
            else:
                loss = loss * scale
            return loss

        return calc_loss_and_normalize
        
    def combined_loss_fn(self, logits, labels, student_lc = None, teacher_lc = None):
        if teacher_lc is None or self.topo_weight is None or self.topo_weight == 0.0 or self.dist_loss_fn is None:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            return c_e_loss,c_e_loss,torch.tensor(-0.0000001)
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[4] or self.distillation_loss_name == DISTILLATION_LOSS_NAMES[5]:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            dist_loss,_ = self.dist_loss_fn(student_lc,teacher_lc) #torch.sum(student_lc ** 2)#
        else:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            student_dm = self.dist_fn(student_lc)
            student_dm = student_dm/student_dm.mean().detach()
            teacher_dm = self.dist_fn(teacher_lc)
            teacher_dm = teacher_dm/teacher_dm.mean().detach()
            dist_loss,_ = self.dist_loss_fn(student_dm,teacher_dm) #torch.sum(student_lc ** 2)#
            
        if self.kill_classification_loss:
            total_loss = dist_loss
        elif self.kill_distillation_loss:
            total_loss = c_e_loss
        elif isinstance(self.topo_weight, (int, float)):
            total_loss = c_e_loss + self.topo_weight * dist_loss
        elif self.topo_weight == "uncert_based":
            comb_loss = torch.stack((c_e_loss, dist_loss))
            total_loss = self.uncertainty_weighting_module(comb_loss)[0]
            
                
        return total_loss, c_e_loss, dist_loss
        
    def forward(self, x):
        logits, latent_code = self.classifier_model(x)
        return  logits, latent_code

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.teacher(inputs) if self.teacher else None
        if self.teacher_model_type == 'ae':
            ae_lc = ae_lc.chunk(2, dim=-1)[0]
        loss,c_e_loss,dist_loss = self.loss_fn(logits, labels,student_lc = classifier_lc,teacher_lc = ae_lc)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_classification_loss", c_e_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_distillation_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)           
        return loss
    
    def on_train_epoch_end(self):
        if self.uncertainty_weighting_module:
            self.log_dict(self.uncertainty_weighting_module.get_weights())

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.teacher(inputs) if self.teacher else None
        if self.teacher_model_type == 'ae':
            ae_lc = ae_lc.chunk(2, dim=-1)[0]
        loss,c_e_loss,dist_loss = self.loss_fn(logits, labels,student_lc = classifier_lc,teacher_lc = ae_lc)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_classification_loss", c_e_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_distillation_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def calculate_test_classification_stats(self):
        test_phase_data = self.test_phase_data
        # Concatenate all labels, embeddings, and logits
        all_labels = torch.cat(test_phase_data['labels'], dim=0).cpu()
        all_logits = torch.cat(test_phase_data['logits'], dim=0).cpu()

        # Convert logits to predicted class labels
        pred_labels = all_logits.argmax(dim=1)

        # Calculate precision, recall, and F1 for macro (averaged over each class)
        precision_macro = precision_score(all_labels, pred_labels, average='macro')
        recall_macro = recall_score(all_labels, pred_labels, average='macro')
        f1_macro = f1_score(all_labels, pred_labels, average='macro')


        # Calculate precision, recall, and F1 for micro (aggregated across all classes)
        precision_micro = precision_score(all_labels, pred_labels, average='micro')
        recall_micro = recall_score(all_labels, pred_labels, average='micro')
        f1_micro = f1_score(all_labels, pred_labels, average='micro')
        accuracy_micro = accuracy_score(all_labels, pred_labels)

        return {
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "test_f1_macro": f1_macro,
            "test_precision_micro": precision_micro,
            "test_recall_micro": recall_micro,
            "test_f1_micro": f1_micro,
            "test_accuracy_micro": accuracy_micro
        }
        
    def test_step(self,batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.teacher(inputs) if self.teacher else None
        if self.teacher_model_type == 'ae':
            ae_lc = ae_lc.chunk(2, dim=-1)[0]
        loss,c_e_loss,dist_loss = self.loss_fn(logits, labels,student_lc = classifier_lc,teacher_lc = ae_lc)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_classification_loss", c_e_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_distillation_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_phase_data['labels'].append(labels.cpu() if torch.is_tensor(labels) else labels)
        self.test_phase_data['embeddings'].append(classifier_lc.cpu())
        self.test_phase_data['logits'].append(logits.cpu())

        
    def on_test_end(self):
        """Logs the best checkpoint path and visualizes latent codes at the end of testing."""
        wandb.log(self.calculate_test_classification_stats())
        self.visualize_classifier_latent_space()
        self.calculate_distance_matrix_metrics()
        if self.best_checkpoint_path:
            # self.logger.experiment.add_text("best_checkpoint", self.best_checkpoint_path)
            self.logger.experiment.log({"best_checkpoint": self.best_checkpoint_path})
            match = re.search(r"epoch=(\d+)", self.best_checkpoint_path)
            if match:
                epoch = int(match.group(1))
                print("Epoch:", epoch)
                self.logger.experiment.log({"best_epoch": epoch})
            else:
                print("Epoch not found in filename, assuming we will take last epoch.")
                self.logger.experiment.log({"best_epoch": "last epoch"})       

    def calculate_distance_matrix_metrics(self,max_n_samples=2000):
        if not self.test_phase_data['embeddings']:
            print("No embeddings collected during testing.")
            return
        dist_fn = self.dist_fn if self.dist_fn else EuclideanDistance() # in case distance function not defined over space
        latent_distance_matrix = dist_fn(ClassifierExperiment.take_first_n(torch.cat(self.test_phase_data['embeddings'], dim=0).cpu(),max_n_samples)).numpy()
        labels = ClassifierExperiment.take_first_n(torch.cat(self.test_phase_data['labels'], dim=0).cpu(),max_n_samples).numpy()
        metrics_dict = StaticDistanceMatrixMetricCalculator.calculate_distance_matrix_metrics(latent_distance_matrix,labels)
        metrics_dict = {
                f"dm_metrics_{key}": value
                for key, value in metrics_dict.items()
                if ClassifierExperiment.is_scalar(value)  # Only include if value is a scalar
            }
        self.logger.experiment.log(metrics_dict)
        
    def visualize_classifier_latent_space(self,max_n_samples=2000):
        if not self.test_phase_data['embeddings']:
            print("No embeddings collected during testing.")
            return

        # Stack embeddings and labels
        latents = ClassifierExperiment.take_first_n(torch.cat(self.test_phase_data['embeddings'], dim=0).cpu(),max_n_samples).numpy()
        labels = ClassifierExperiment.take_first_n(torch.cat(self.test_phase_data['labels'], dim=0).cpu(),max_n_samples).numpy()

        # Downproject using UMAP
        reducer = umap.UMAP(n_components=2, random_state=42)
        latents_2d = reducer.fit_transform(latents)

        # Create scatter plot
        fig = plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label="Class Labels")
        plt.title("UMAP Projection of Test Latent Codes")
        plt.xlabel("UMAP Dim 1")
        plt.ylabel("UMAP Dim 2")

        # Log figure to Weights & Biases
        if self.logger:
            wandb.log({"umap_test_projection": wandb.Image(fig)})
            #self.logger.experiment.add_figure({"umap_test_projection", fig})
        plt.close()
        
    def configure_optimizers(self):
        # Start with the classifier model's parameters
        if self.different_learning_rate_for_classifier:
            params = [
                {"params": filter(lambda p: p.requires_grad, self.classifier_model.backbone.parameters()), "lr": self.lr},
                {"params": filter(lambda p: p.requires_grad, self.classifier_model.classifier_head.parameters()), "lr": self.lr*3}
            ]
        else:
            params = [{"params":  filter(lambda p: p.requires_grad, self.classifier_model.parameters())}]
        
        # Append uncertainty weighting module parameters if it exists
        if self.uncertainty_weighting_module:
            uncertainty_lr = 0.001
            params.append({"params": self.uncertainty_weighting_module.parameters(), "lr": uncertainty_lr})
            self.logger.experiment.config["uncertainty_lr"] = uncertainty_lr
            print(f"Setting LR for uncertaintry weighting parameters as {uncertainty_lr}")
        
        # Return Adam optimizer
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    @staticmethod
    def is_scalar(x):
        # Check Python and NumPy scalar types
        if isinstance(x, (int, float, numbers.Integral, numbers.Real, np.integer, np.floating)):
            return True
        
        # Check NumPy array with size 1
        if isinstance(x, np.ndarray) and (x.shape == () or x.size == 1):
            return True
        
        # Check Torch Tensor with size 1
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return True
        
        return False
    
    @staticmethod
    def take_first_n(tensor, n):
        """
        Takes the first n samples from a torch tensor.
        
        - If m < n, returns the whole tensor.
        - If tensor has shape (m,), returns a (min(m, n),) shaped tensor.
        - If tensor has shape (m, ...), returns a (min(m, n), ...) shaped tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (m, ...) or (m,).
            n (int): Number of samples to take.

        Returns:
            torch.Tensor: A tensor with at most n samples.
        """
        return tensor[:min(n, tensor.shape[0])]


class OnOffScheduler:
    def __init__(self, patience: int, low_weight: float):
        """
        Initializes the OnOffScheduler.

        Args:
            patience (int): Number of epochs to wait before triggering the "off" state.
            low_weight (float): Weight for distillation loss when the scheduler is active.
        """
        self.patience = patience
        self.low_weight = low_weight
        self.best_val_loss = float('inf')  # Initially set to infinity
        self.epochs_since_improvement = 0
        self.active = True  # Starts active

    def get_distillation_weight(self):
        """
        Returns the current distillation weight based on whether the scheduler is active.

        Returns:
            float: distillation loss weight, either `low_weight` or 1.0.
        """
        if self.active:
            return self.low_weight
        return 1.0

    def update(self, val_loss):
        """
        Updates the scheduler based on the current validation loss.

        Args:
            val_loss (torch.Tensor): Current validation loss (will be moved to CPU for comparison).
        """
        if torch.is_tensor(val_loss):
            val_loss = val_loss.cpu().item()  # Move to CPU and get scalar value

        if val_loss < self.best_val_loss:
            # New best validation loss
            self.best_val_loss = val_loss
            self.epochs_since_improvement = 0
            self.active = True  # Stay active as long as there's improvement
        else:
            # No improvement
            self.epochs_since_improvement += 1
            if self.epochs_since_improvement >= self.patience:
                self.active = False  # Trigger "off" if patience period exceeded