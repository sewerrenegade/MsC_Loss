import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import umap
import wandb
import numpy as np
from models.beta_variational_autoencoder import BetaVAE

class VAE_exp(pl.LightningModule):
    def __init__(self, model:BetaVAE, beta=4, lr=1e-3,weight_decay = 0.0000005):
        """
        Beta-VAE implementation in PyTorch Lightning.
        
        Args:
            input_dim (tuple): Dimensions of the input image (C, H, W).
            latent_dim (int): Size of the latent space.
            beta (float): Weight of the KL divergence term.
            lr (float): Learning rate for the optimizer.
        """
        super(VAE_exp, self).__init__()
        self.beta = beta
        if self.beta == 0:
            model.normal_ae = True
            
        self.lr = lr
        self.model = model
        self.test_phase_data = {"labels":[],"embeddings":[]}
        self.best_checkpoint_path = None
        self.weight_decay = weight_decay

    def forward(self, x):
        """Forward pass through the VAE."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch  # Ignore labels
        x_recon, mu, logvar = self.forward(x)
        loss, recon_loss, kl_div = self.beta_vae_loss(x, x_recon, mu, logvar,beta=self.beta)
        self.log('train_loss', loss, prog_bar=True,on_step=False,on_epoch=True)
        self.log('train_recon_loss', recon_loss, prog_bar=False,on_step=True,on_epoch=True)
        self.log('train_kl_div', kl_div, prog_bar=False,on_step=False,on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_recon, mu, logvar = self.forward(x)
        loss, recon_loss, kl_div = self.beta_vae_loss(x, x_recon, mu, logvar,beta=self.beta)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, prog_bar=False)
        self.log('val_kl_div', kl_div, prog_bar=False)

        
        
    def test_step(self, batch, batch_idx):
        x, y = batch  # For testing, labels are used
        x_recon, mu, logvar = self.forward(x)
        loss, recon_loss, kl_div = self.beta_vae_loss(x, x_recon, mu, logvar,beta=self.beta)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_recon_loss', recon_loss, prog_bar=False)
        self.log('test_kl_div', kl_div, prog_bar=False)

        self.test_phase_data['labels'].append(y)
        self.test_phase_data['embeddings'].append(mu)
        self.log_image_comparison(x, x_recon, batch_idx)

        return {"latent": mu, "labels": y, "test_loss": loss}
        
    def on_test_end(self):
        """Logs the best checkpoint path and visualizes latent codes at the end of testing."""
        self.visualize_ae_latent_space()
        if self.best_checkpoint_path:
            self.logger.experiment.log({"best_checkpoint": self.best_checkpoint_path})
            
    def visualize_ae_latent_space(self):
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
        plt.close()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)

    def beta_vae_loss(self, x, recon_x, mean, log_var, beta=4):
        recon_loss = nn.MSELoss()(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss , recon_loss,kl_loss

    def log_image_comparison(self, x, x_recon, batch_idx, num_images=5):
        """Logs a comparison of original vs reconstructed images."""
        
        # Ensure x and x_recon have the same number of images
        num_available = min(len(x), len(x_recon), num_images)
        
        if num_available == 0:
            print(f"Warning: No images available for logging at batch {batch_idx}")
            return
        
        x = x[:num_available].cpu().detach()
        x_recon = x_recon[:num_available].cpu().detach()

        fig, axes = plt.subplots(2, num_available, figsize=(num_available * 2, 4))

        # Ensure axes is always iterable even when num_available == 1
        if num_available == 1:
            axes = np.expand_dims(axes, axis=1)  # Make it (2, num_available) for consistency

        for i in range(num_available):
            axes[0, i].imshow(x[i].permute(1, 2, 0).numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(x_recon[i].permute(1, 2, 0).numpy(), cmap='gray')
            axes[1, i].axis('off')

        plt.suptitle(f"Original vs Reconstructed (Batch {batch_idx})")

        # Log the image to wandb
        wandb.log({f"Reconstruction_test_imgs/batch_{batch_idx}": wandb.Image(fig)})
        plt.close(fig)
        
