import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import umap
import wandb
from custom_topo_tools.connectivity_topo_regularizer import TopologicalZeroOrderLoss
from custom_topo_tools.moor_topo_reg import TopologicalSignatureDistance
from custom_topo_tools.mds_regularizer import MDSLoss
from models.image_classifier import ImageClassifier
from models.basic_distance_fns import pairwise_euclidean_distance,pairwise_cosine_similarity

DISTILLATION_LOSS_NAMES = ['prop_topo','moor_topo','mds','random','',None]
class ClassifierExperiment(pl.LightningModule):
    def __init__(self, model: ImageClassifier,ae_encoder = None, learning_rate=0.001,topo_weight = 0.0,dist_fnc = pairwise_euclidean_distance,distillation_loss_name = None):
        super(ClassifierExperiment, self).__init__()
        assert distillation_loss_name in DISTILLATION_LOSS_NAMES
        self.distillation_loss_name = distillation_loss_name
        self.classifier_model = model
        self.topo_weight = topo_weight
        self.dist_fn = dist_fnc
        self.ae_encoder = ae_encoder
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss()
        self.dist_loss_fn =  self.get_distillation_loss_fn() #TopologicalZeroOrderLoss(method="deep",timeout=5, multithreading=False,importance_calculation_strat=None)
        self.loss_fn = self.combined_loss_fn
        self.test_phase_data = {"labels":[],"embeddings":[]}
        self.lr = learning_rate
        self.best_checkpoint_path = None
        
    def get_distillation_loss_fn(self):
        if self.distillation_loss_name == DISTILLATION_LOSS_NAMES[0]:
            return TopologicalZeroOrderLoss(method="deep",timeout=5, multithreading=True,importance_calculation_strat='min')
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[1]:
            return TopologicalSignatureDistance(match_edges='symmetric',to_gpu=True if torch.cuda.is_available() else False)
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[2]:
            return MDSLoss()
        elif self.distillation_loss_name == DISTILLATION_LOSS_NAMES[-1] or self.distillation_loss_name == DISTILLATION_LOSS_NAMES[-2]:
            return None            
        else:
            raise NotImplementedError(f"{self.distillation_loss_name} is not implemented")
            
        
    def combined_loss_fn(self, logits, labels, student_lc = None, teacher_lc = None):
        if teacher_lc is None or self.topo_weight is None or self.topo_weight == 0.0:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            return c_e_loss,c_e_loss,torch.tensor(-1.0)
        else:
            c_e_loss = self.cross_entropy_loss_fn(logits,labels)
            student_dm = self.dist_fn(student_lc)
            student_dm = student_dm/student_dm.mean().detach()
            teacher_dm = self.dist_fn(teacher_lc)
            teacher_dm = teacher_dm/teacher_dm.mean().detach()
            dist_loss = self.dist_loss_fn(student_dm,teacher_dm)[0] #torch.sum(student_lc ** 2)#
            total_loss = c_e_loss + self.topo_weight * dist_loss
            print(f"total:{total_loss}, distillation;{dist_loss}, class:{c_e_loss}")
            return total_loss, c_e_loss, dist_loss
        
    def forward(self, x):
        logits, latent_code = self.classifier_model(x)
        return  logits, latent_code

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.ae_encoder(inputs).chunk(2, dim=-1)[0] if self.ae_encoder else None
        loss,c_e_loss,dist_loss = self.loss_fn(logits, labels,student_lc = classifier_lc,teacher_lc = ae_lc)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_classification_loss", c_e_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_distillation_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.ae_encoder(inputs).chunk(2, dim=-1)[0] if self.ae_encoder else None
        loss,c_e_loss,dist_loss = self.loss_fn(logits, labels,student_lc = classifier_lc,teacher_lc = ae_lc)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_classification_loss", c_e_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_distillation_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True)
        
    def test_step(self,batch, batch_idx):
        inputs, labels = batch
        logits,classifier_lc  = self.classifier_model(inputs)
        ae_lc = self.ae_encoder(inputs).chunk(2, dim=-1)[0] if self.ae_encoder else None
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
            self.logger.experiment.add_text("best_checkpoint", self.best_checkpoint_path)
            
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
        return torch.optim.Adam(self.classifier_model.parameters(), lr=self.lr)





