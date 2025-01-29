import torch
import torch.nn as nn


class BetaVAE(nn.Module):
    def __init__(self, latent_dim, normal_ae = False):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.normal_ae = normal_ae
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 36 * 36, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # Latent mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 36 * 36),
            nn.ReLU(),
            nn.Unflatten(1, (64, 36, 36)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def encode(self, x):
        latent = self.encoder(x)
        mean, log_var = latent.chunk(2, dim=-1)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        if self.normal_ae:
            return mean
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var



