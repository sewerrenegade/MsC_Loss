from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import umap
import re
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    def __init__(self, model: ImageClassifier,ae_encoder_path = None, lr=0.001,distillation_weight = 0.0,dist_fnc_name = "euclidean",distillation_loss_name = None,teacher_model_type = "",distillation_loss_config = {},weight_decay=0.00001,kill_classification_loss = False,kill_distillation_loss = False,classifier_checkpoint_path = "",freeze_classifier_backbone = False, different_learning_rate_for_classifier = False,class_weighting= None,normalize_dm = True,label_smoothing = None,optimizer_name = "adam"): 
        super(ClassifierExperiment, self).__init__()
        self.distillation_loss_name = distillation_loss_name
        self.classifier_model = model
        self.lr = lr
        self.class_weighting = class_weighting
        self.best_checkpoint_path = None
        self.weight_decay= weight_decay
        self.normalize_dm = normalize_dm
        wandb.log({"backbone_out_dim":self.classifier_model.latent_dim,"number_of_parameters_in_classifier_head":self.classifier_model.number_of_parameters_in_classifier,"number_of_parameters_in_image_encoder":self.classifier_model.number_of_parameters_in_image_encoder,"total_number_of_parameters_in_classifier":self.classifier_model.number_of_parameters_in_classifier+self.classifier_model.number_of_parameters_in_image_encoder})
        self.dataset_type = self.classifier_model.dataset_type
        self.teacher_model_type = teacher_model_type
        self.kill_classification_loss = kill_classification_loss
        self.kill_distillation_loss = kill_distillation_loss
        self.label_smoothing = label_smoothing
        self.ae_encoder_path = ae_encoder_path
        self.optimizer_name = optimizer_name
        self.freeze_classifier_backbone = freeze_classifier_backbone
        self.different_learning_rate_for_classifier = different_learning_rate_for_classifier
        self.dist_fn = ClassifierExperiment.get_distance_fn(dist_fnc_name)
        self.classifier_checkpoint_path = classifier_checkpoint_path
        self.load_from_checkpoint_file()
        if self.freeze_classifier_backbone:
            self.classifier_model.freeze_backbone()
        self.teacher = self.get_teacher()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weighting if self.class_weighting is not None else None,
                label_smoothing=self.label_smoothing if self.label_smoothing else 0.0
            )
        self.dist_loss_fn =  self.get_normalized_distillation_loss_fn(distillation_loss_config)  #get_distillation_loss_fnget_distillation_loss_fn
        self.topo_weight = self.parse_and_apply_distillation_weight(distillation_weight)
        self.loss_fn = self.combined_loss_fn
        self.test_phase_data = {"labels":[],"embeddings":[],"logits": []}
        self.calculate_val_knn_accuracy = True
        if self.calculate_val_knn_accuracy:
            self.val_phase_data = {"labels":[],"embeddings":[],"logits": []}
            
        

    def parse_and_apply_distillation_weight(self,topo_weight):
        self.scheduler = None
        self.uncertainty_weighting_module = None
        if self.distillation_loss_name is None or self.teacher is None:
            return None
        if ClassifierExperiment.is_scalar(topo_weight):
            pass
        if topo_weight == "uncert_based":
            self.uncertainty_weighting_module = UncertWeighting(n_losses = 2)            
        if isinstance(topo_weight,dict):
            if "on_off" in topo_weight:
                self.scheduler = OnOffScheduler(patience=topo_weight.get("patience",10),low_weight=topo_weight.get("low_weight",0.0),reset_optimizer_fnc=self.configure_optimizers)
                self.kill_classification_loss = self.scheduler.active
                return self.scheduler.get_distillation_weight()
        return topo_weight
        
    def on_validation_epoch_end(self):
        """Hook to execute code at the end of every validation epoch."""
        # Ensure the logger is a WandB logger
        if self.scheduler:
            if self.scheduler.active:
                try:
                    latest_val_distillation_loss = self.trainer.callback_metrics.get("val_distillation_loss", None)
                    # Retrieve the latest logged "val_distillation_loss" 
                    if latest_val_distillation_loss:
                        # Update some value based on latest val_distillation_loss
                        self.scheduler.update(latest_val_distillation_loss,epoch = self.trainer.current_epoch)
                        self.topo_weight = self.scheduler.get_distillation_weight()
                        self.kill_classification_loss =  self.scheduler.active
                        print(f"The scheduler is active: {self.scheduler.active}")
                        print(f"Updated some_value with latest val_distillation_loss: {latest_val_distillation_loss}")
                    else:
                        raise KeyError("val_distillation_loss was never logged")

                except Exception as e:
                    print(f"Failed to retrieve val_distillation_loss from WandB: {e}")
        if self.calculate_val_knn_accuracy:
            self.val_phase_data['embeddings'] = torch.cat(self.val_phase_data['embeddings'], dim=0).cpu()
            self.val_phase_data['labels'] = torch.cat(self.val_phase_data['labels'], dim=0).cpu()
            self.val_phase_data['logits'] = torch.cat(self.val_phase_data['logits'], dim=0).cpu()
            #self.sub_sampled_val_phase_data ={}
            #self.sub_sampled_val_phase_data['embeddings'] ,self.sub_sampled_val_phase_data['logits'],self.sub_sampled_val_phase_data['labels'],_,_,_ = ClassifierExperiment.stratified_take_n(tensor1= self.val_phase_data['embeddings'],tensor2 = self.val_phase_data['logits'],labels=self.val_phase_data['labels'],n=500)
            dm_metric = self.calculate_distance_matrix_knn_acc(self.val_phase_data)
            if dm_metric:
                self.log("val_dm_knn_acc", dm_metric, on_step=False, on_epoch=True)
            self.val_phase_data = {"labels":[],"embeddings":[],"logits": []}
                
    def get_teacher(self):
        if self.teacher_model_type == "ae":
            return ClassifierExperiment.get_ae_encoder_from_path(self.ae_encoder_path) if self.distillation_loss_name !="" and not self.distillation_loss_name is None else None
        elif "dino" in self.teacher_model_type.lower():
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
        assert self.distillation_loss_name in DISTILLATION_LOSS_NAMES
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
        if teacher_lc is None or self.topo_weight is None or  self.dist_loss_fn is None:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            return c_e_loss,c_e_loss,torch.tensor(-0.0000001)
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[4] or self.distillation_loss_name == DISTILLATION_LOSS_NAMES[5]:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            dist_loss,_ = self.dist_loss_fn(student_lc,teacher_lc) #torch.sum(student_lc ** 2)#
        else:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            student_dm = self.dist_fn(student_lc)
            teacher_dm = self.dist_fn(teacher_lc)
            if self.normalize_dm:
                student_dm = student_dm/student_dm.mean().detach()
                
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
        if self.calculate_val_knn_accuracy:
            self.val_phase_data['labels'].append(labels.cpu() if torch.is_tensor(labels) else labels)
            self.val_phase_data['embeddings'].append(classifier_lc.cpu())
            self.val_phase_data['logits'].append(logits.cpu())
        
    def calculate_test_classification_stats(self):
        test_phase_data = self.test_phase_data
        # Concatenate all labels, embeddings, and logits
        all_labels = test_phase_data['labels']
        all_logits = test_phase_data['logits']

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
        self.test_phase_data['embeddings'] = torch.cat(self.test_phase_data['embeddings'], dim=0).cpu()
        self.test_phase_data['labels'] = torch.cat(self.test_phase_data['labels'], dim=0).cpu()
        self.test_phase_data['logits'] = torch.cat(self.test_phase_data['logits'], dim=0).cpu()
        self.sub_sampled_test_phase_data ={}
        self.sub_sampled_test_phase_data['embeddings'] ,self.sub_sampled_test_phase_data['logits'],self.sub_sampled_test_phase_data['labels'],_,_,_ = ClassifierExperiment.stratified_take_n(tensor1= self.test_phase_data['embeddings'],tensor2 = self.test_phase_data['logits'],labels=self.test_phase_data['labels'],n=2000)
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

    def calculate_distance_matrix_metrics(self):
        if not self.test_phase_data:
            print("No embeddings collected during testing.")
            return
        test_data = self.sub_sampled_test_phase_data
        dist_fn = self.dist_fn if self.dist_fn else EuclideanDistance() # in case distance function not defined over space
        latent_distance_matrix = dist_fn(test_data['embeddings']).numpy()
        labels = test_data['labels'].numpy()   
        metrics_dict = StaticDistanceMatrixMetricCalculator.calculate_distance_matrix_metrics(latent_distance_matrix,labels)
        metrics_dict = {
                f"dm_metrics_{key}": value
                for key, value in metrics_dict.items()
                if ClassifierExperiment.is_scalar(value)  # Only include if value is a scalar
            }
        self.logger.experiment.log(metrics_dict)
        
    def calculate_distance_matrix_knn_acc(self,test_data):
        dist_fn = self.dist_fn if self.dist_fn else EuclideanDistance() # in case distance function not defined over space
        latent_distance_matrix = dist_fn(test_data['embeddings']).numpy()
        labels = test_data['labels'].numpy()   
        if np.isnan(latent_distance_matrix).any() or np.isnan(labels).any():
            knn_acc = None
            print(f"ERROR: either dist matrix or labels contain NaN. DMAT contains nan:{np.isnan(latent_distance_matrix).any()}; labels contain nan {np.isnan(labels).any()} ")
        else:
            
            knn_acc, _, _, _ = StaticDistanceMatrixMetricCalculator.evaluate_knn_classifier_from_distance_matrix(
                distance_matrix=latent_distance_matrix, labels=labels, k=3
            )
        return knn_acc
        
    def visualize_classifier_latent_space(self):
        if not self.test_phase_data:
            print("No embeddings collected during testing.")
            return
        test_data = self.sub_sampled_test_phase_data
        # Stack embeddings and labels
        latents = test_data['embeddings'].numpy()
        labels = test_data['labels'].numpy()
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
        print("Configuring Optimizers")
        lr = self.lr["lr"] if isinstance(self.lr, dict) else self.lr
        optimizer_name = self.optimizer_name.lower()
        
        # Prepare parameter groups
        params = []
        if self.different_learning_rate_for_classifier:
            params.append({
                "params": filter(lambda p: p.requires_grad, self.classifier_model.classifier_head.parameters()),
                "lr": lr * 3
            })
            if hasattr(self.classifier_model.backbone, "parameters"):
                params.append({
                    "params": filter(lambda p: p.requires_grad, self.classifier_model.backbone.parameters()),
                    "lr": lr
                })
        else:
            params.append({"params": filter(lambda p: p.requires_grad, self.classifier_model.parameters())})
        
        # Append uncertainty weighting module parameters if it exists
        if self.uncertainty_weighting_module:
            uncertainty_lr = 0.001
            params.append({"params": self.uncertainty_weighting_module.parameters(), "lr": uncertainty_lr})
            self.logger.experiment.config["uncertainty_lr"] = uncertainty_lr
            print(f"Setting LR for uncertainty weighting parameters as {uncertainty_lr}")
        
        # Choose optimizer
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=self.weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)
        elif optimizer_name == "adan":
            from adan_pytorch import Adan
            optimizer = Adan(params, lr=lr, betas=(0.02, 0.08, 0.01), weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        
        # Handle learning rate scheduler if lr is a dictionary
        if isinstance(self.lr, dict):
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=self.lr.get("patience", 3), factor=self.lr.get("factor", 0.5), verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.lr.get("monitor", "val_classification_loss"),
                    'interval': 'epoch'
                }
            }
        
        return optimizer


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
    @staticmethod
    def stratified_take_n(tensor1, tensor2, labels, n, random_state=42):
        """
        Takes the first `n` samples from three tensors in a stratified way, preserving class distribution.

        - If `n` is greater than the dataset size, returns the full dataset.
        - Otherwise, it selects `n` samples while maintaining the class distribution.

        Args:
            tensor1 (torch.Tensor): First input tensor of shape (m, ...).
            tensor2 (torch.Tensor): Second input tensor of shape (m, ...).
            labels (torch.Tensor): Label tensor of shape (m,).
            n (int): Number of samples to take.
            random_state (int, optional): Random seed for reproducibility.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            The stratified subset (tensor1_A, tensor1_B, tensor2_A, tensor2_B, labels_A, labels_B).
        """
        assert tensor1.shape[0] == tensor2.shape[0] == labels.shape[0]
        
        total_samples = labels.shape[0]
        
        # If requested size is greater than available samples, return all
        if n >= total_samples:
            return tensor1,tensor2, labels, tensor1[:0] , tensor2[:0], labels[:0]
        
        # Get stratified sample indices
        indices = torch.arange(total_samples)
        selected_idx, remaining_idx = train_test_split(indices.numpy(), stratify=labels.numpy(), 
                                                    train_size=n, random_state=random_state)

        # Split tensors using stratified indices
        tensor1_A, tensor1_B = tensor1[selected_idx], tensor1[remaining_idx]
        tensor2_A, tensor2_B = tensor2[selected_idx], tensor2[remaining_idx]
        labels_A, labels_B = labels[selected_idx], labels[remaining_idx]
        assert labels_A.shape[0] == tensor1_A.shape[0] == tensor2_A.shape[0]

        return tensor1_A,  tensor2_A, labels_A, tensor1_B, tensor2_B,  labels_B


class OnOffScheduler:
    def __init__(self, patience: int, low_weight: float, reset_optimizer_fnc: callable = None):
        """
        Initializes the OnOffScheduler.

        Args:
            patience (int): Number of epochs to wait before triggering the "off" state.
            low_weight (float): Weight for distillation loss when the scheduler is active.
            reset_optimizer_fnc (callable, optional): Function to reset the optimizer when switching phases.
        """
        self.patience = patience
        self.low_weight = low_weight
        self.best_val_loss = float('inf')  # Initially set to infinity
        self.epochs_since_improvement = 0
        self.active = True  # Starts active
        self.reset_optimizer_fnc = reset_optimizer_fnc

        print(f"[OnOffScheduler] Initialized with patience={self.patience}, low_weight={self.low_weight}")

    def get_distillation_weight(self):
        """
        Returns the current distillation weight based on whether the scheduler is active.

        Returns:
            float: distillation loss weight, either `low_weight` or 1.0.
        """
        weight = 1.0 if self.active else self.low_weight
        print(f"[OnOffScheduler] Current distillation weight: {weight} (Active: {self.active})")
        return weight

    def update(self, val_loss,epoch = -1):
        """
        Updates the scheduler based on the current validation loss.

        Args:
            val_loss (torch.Tensor): Current validation loss (will be moved to CPU for comparison).
        """
        if torch.is_tensor(val_loss):
            val_loss = val_loss.cpu().item()  # Move to CPU and get scalar value

        print(f"[OnOffScheduler] Received validation loss: {val_loss:.6f} at epoch:{epoch}")

        if val_loss < self.best_val_loss:
            # New best validation loss
            print(f"[OnOffScheduler] New best validation loss: {val_loss:.6f} (Previous: {self.best_val_loss:.6f})")
            self.best_val_loss = val_loss
            self.epochs_since_improvement = 0
            self.active = True  # Stay active as long as there's improvement
        else:
            # No improvement
            self.epochs_since_improvement += 1
            print(f"[OnOffScheduler] No improvement for {self.epochs_since_improvement}/{self.patience} epochs.")

            if self.epochs_since_improvement >= self.patience:
                if self.active:
                    print("[OnOffScheduler] Patience exceeded. Switching to OFF state.")
                    if self.reset_optimizer_fnc is not None:
                        print("[OnOffScheduler] Resetting optimizer.")
                        self.reset_optimizer_fnc()
                self.active = False