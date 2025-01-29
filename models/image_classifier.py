import torch
import torch.nn as nn
import torchvision.models as models
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim = 50):
        super(ImageClassifier, self).__init__()
        # Load pre-trained ResNet18
        self.backbone = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        # Remove the fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1],nn.Flatten(start_dim=1), nn.Linear(512, 256))  # Keep layers except the final fc layer
        # Define a classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(256, latent_dim),  # ResNet-18 outputs a 512-d feature vector
            nn.ReLU(),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, x):
        # Extract latent code
        latent_code = self.backbone(x)
        # latent_code = torch.flatten(latent_code, start_dim=1)  # Flatten from (N, 512, 1, 1) to (N, 512)
        logits = self.classifier_head(latent_code)
        return logits, latent_code
    
if __name__ == "__main__":
    classifier = ImageClassifier(num_classes= 8 , latent_dim= 50)
    pass