import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import umap
import re
from collections.abc import Iterable
import wandb
from models.learnable_uncertainty_based_weighting import UncertWeighting
from models.beta_variational_autoencoder import BetaVAE
from custom_topo_tools.connectivity_topo_regularizer import TopologicalZeroOrderLoss
from custom_topo_tools.moor_topo_reg import TopologicalSignatureDistance
from custom_topo_tools.mds_regularizer import MDSLoss
from models.image_classifier import ImageClassifier
from models.basic_distance_fns import pairwise_euclidean_distance,pairwise_cosine_similarity

DISTILLATION_LOSS_NAMES = ['prop_topo','moor_topo','mds','random','l2','',None]
class ClassifierExperiment(pl.LightningModule):
    def __init__(self, model: ImageClassifier,ae_encoder_path = None, lr=0.001,distillation_weight = 0.0,dist_fnc_name = "euclidean",distillation_loss_name = None,teacher_model_type = "",distillation_loss_config = {},weight_decay=0.00001,kill_classification_loss = False,classifier_checkpoint_path = "",freeze_classifier_backbone = False):
        super(ClassifierExperiment, self).__init__()
        assert distillation_loss_name in DISTILLATION_LOSS_NAMES
        self.distillation_loss_name = distillation_loss_name
        self.classifier_model = model
        self.teacher_model_type = teacher_model_type
        self.topo_weight = distillation_weight
        self.ae_encoder_path = ae_encoder_path
        self.freeze_classifier_backbone = freeze_classifier_backbone
        self.kill_classification_loss = kill_classification_loss
        self.dist_fn = ClassifierExperiment.get_distance_fn(dist_fnc_name)
        self.classifier_checkpoint_path = classifier_checkpoint_path
        self.load_from_checkpoint_file()
        if self.freeze_classifier_backbone:
            self.classifier_model.freeze_backbone()
        self.teacher = self.get_teacher()
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.dist_loss_fn =  self.get_normalized_distillation_loss_fn(distillation_loss_config)  #get_distillation_loss_fnget_distillation_loss_fn
        self.loss_fn = self.combined_loss_fn
        self.uncertainty_weighting_module = UncertWeighting(n_losses = 2) if self.topo_weight == "uncert_based" else None
        self.test_phase_data = {"labels":[],"embeddings":[]}
        self.lr = lr
        self.best_checkpoint_path = None
        self.weight_decay= weight_decay
    
    def get_teacher(self):
        if self.teacher_model_type == "ae":
            return ClassifierExperiment.get_ae_encoder_from_path(self.ae_encoder_path) if self.distillation_loss_name !="" and not self.distillation_loss_name is None else None
        elif self.teacher_model_type == "dinobloom":
            from models.DinoBloom.dinobloom_hematology_feature_extractor import get_dino_bloom_w_resize
            return get_dino_bloom_w_resize()
        else:
            return None
        

    def load_from_checkpoint_file(self):
        if self.classifier_checkpoint_path != "" and not self.classifier_checkpoint_path is None:
            print(f"Loading weights of classifier from {self.classifier_checkpoint_path}")
            checkpoint = torch.load(self.classifier_checkpoint_path, map_location=torch.device("cpu"))
            self.classifier_model.load_state_dict(checkpoint["state_dict"], strict=False)  # Set strict=True if you expect exact keys
          
    @staticmethod
    def get_distance_fn(dist_fn_name):
        if dist_fn_name == "euclidean":
            return pairwise_euclidean_distance
        elif dist_fn_name == "cosine":
            return pairwise_euclidean_distance
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
        
    def get_distillation_loss_fn(self,config):
        if self.distillation_loss_name == DISTILLATION_LOSS_NAMES[0]:
            return TopologicalZeroOrderLoss(**config)#method="deep",timeout=5, multithreading=True,importance_calculation_strat='min'
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[1]:
            return TopologicalSignatureDistance(match_edges='symmetric',to_gpu=True if torch.cuda.is_available() else False,**config)
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[2]:
            return MDSLoss()
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[4]:
            return nn.MSELoss(reduction="mean")
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[-1] or self.distillation_loss_name == DISTILLATION_LOSS_NAMES[-2]:
            return None            
        else:
            raise NotImplementedError(f"{self.distillation_loss_name} is not implemented")
    def get_normalized_distillation_loss_fn(self, config):
        loss_fn = self.get_distillation_loss_fn(config)
        scale_factors = {
            DISTILLATION_LOSS_NAMES[1]: 10,
            DISTILLATION_LOSS_NAMES[2]: 50,
            DISTILLATION_LOSS_NAMES[4]: 500
        }
        
        # Default scale is 1
        scale = scale_factors.get(self.distillation_loss_name, 1)

        # Wrapper function that passes any number of arguments to loss_fn
        def calc_loss_and_normalize(*args, **kwargs):
            loss = loss_fn(*args, **kwargs)  # Pass all args and kwargs to loss_fn
            if isinstance(loss, Iterable) and not isinstance(loss, torch.Tensor):
                loss = (loss[0] / scale,) + loss[1:]
            else:
                loss = loss / scale
            return loss

        return calc_loss_and_normalize
        
    def combined_loss_fn(self, logits, labels, student_lc = None, teacher_lc = None):
        if teacher_lc is None or self.topo_weight is None or self.topo_weight == 0.0 or self.dist_loss_fn is None:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            return c_e_loss,c_e_loss,torch.tensor(-1.0)
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[4]:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            print(f"teacher dim {teacher_lc.shape}")
            print(f"student dim {student_lc.shape}")
            dist_loss = self.dist_loss_fn(student_lc,teacher_lc) #torch.sum(student_lc ** 2)#
            if self.kill_classification_loss:
                total_loss = dist_loss
            elif isinstance(self.topo_weight, (int, float)):
                total_loss = c_e_loss + self.topo_weight * dist_loss
            elif self.topo_weight == "uncert_based":
                comb_loss = torch.stack((c_e_loss, dist_loss))
                total_loss = self.uncertainty_weighting_module(comb_loss)[0]     
            return total_loss, c_e_loss, dist_loss 
        else:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            student_dm = self.dist_fn(student_lc)
            student_dm = student_dm/student_dm.mean().detach()
            teacher_dm = self.dist_fn(teacher_lc)
            teacher_dm = teacher_dm/teacher_dm.mean().detach()
            dist_loss = self.dist_loss_fn(student_dm,teacher_dm)[0] #torch.sum(student_lc ** 2)#
            if self.kill_classification_loss:
                total_loss = dist_loss
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
        self.test_phase_data['labels'].append(labels)
        self.test_phase_data['embeddings'].append(classifier_lc)
    
    def on_test_end(self):
        """Logs the best checkpoint path and visualizes latent codes at the end of testing."""
        self.visualize_classifier_latent_space()
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

            
    def visualize_classifier_latent_space(self):
        if not self.test_phase_data['embeddings']:
            print("No embeddings collected during testing.")
            return

        # Stack embeddings and labels
        latents = torch.cat(self.test_phase_data['embeddings'], dim=0).cpu().numpy()
        labels = torch.cat(self.test_phase_data['labels'], dim=0).cpu().numpy()

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
        params = [{"params": self.classifier_model.parameters()}]
        
        # Append uncertainty weighting module parameters if it exists
        if self.uncertainty_weighting_module:
            uncertainty_lr = 0.003
            params.append({"params": self.uncertainty_weighting_module.parameters(), "lr": uncertainty_lr})
            self.logger.experiment.config["uncertainty_lr"] = uncertainty_lr
            print(f"Setting LR for uncertaintry weighting parameters as {uncertainty_lr}")
        
        # Return Adam optimizer
        return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
