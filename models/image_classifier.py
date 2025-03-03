import torch
import torch.nn as nn
import torchvision.models as models
class ImageClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim = 384,dino_bloom_backbone = None):
        super(ImageClassifier, self).__init__()
        # Load pre-trained ResNet18
        if dino_bloom_backbone:
            print("Using Dinobloom v2 S as backbone")
            self.backbone = dino_bloom_backbone
            latent_dim = 384
        else:
            self.backbone = models.resnet18(weights = models.ResNet18_Weights.DEFAULT) 
            # Remove the fully connected layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1],nn.Flatten(start_dim=1), nn.Linear(512, latent_dim))  # Keep layers except the final fc layer# dinobloomS output dim 384
        # Define a classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, int(latent_dim/2)), 
            nn.ReLU(),
            nn.Linear(int(latent_dim/2), num_classes),
        )

    def freeze_backbone(self):
        print("Freezing classifier backbone")
        for param in self.backbone.parameters():
            param.requires_grad = False
    def unfreeze_backbone(self):
        print("Unfreezing classifier backbone")
        for param in self.backbone.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        # Extract latent code
        latent_code = self.backbone(x)
        # latent_code = torch.flatten(latent_code, start_dim=1)  # Flatten from (N, 512, 1, 1) to (N, 512)
        logits = self.classifier_head(latent_code)
        return logits, latent_code
    
if __name__ == "__main__":
    classifier = ImageClassifier(num_classes= 8 , latent_dim= 50)
    pass