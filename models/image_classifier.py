import torch.nn as nn
from models.image_encoders import create_feature_encoder

class ImageClassifier(nn.Module):
    def __init__(self, num_classes, latent_dim = 384, encoder_name = 'ResNet18', use_image_net_weights = False,dataset_type = "single_cell"):
        super(ImageClassifier, self).__init__()
        self.encoder_name = encoder_name
        self.dataset_type = dataset_type
        self.use_image_net_weights = use_image_net_weights
        self.num_classes = num_classes
        self.backbone,self.latent_dim = create_feature_encoder(model_name = encoder_name, output_dim = latent_dim, use_pretrained_weights = self.use_image_net_weights,dataset_type = self.dataset_type)
        self.number_of_parameters_in_image_encoder = sum(p.numel() for p in self.backbone.parameters())
        # Define a classification head
        self.classifier_head = nn.Sequential(
            nn.Linear(self.latent_dim, int(self.latent_dim/2)), 
            nn.ReLU(),
            nn.Linear(int(self.latent_dim/2), num_classes),
        )
        self.number_of_parameters_in_classifier = sum(p.numel() for p in self.classifier_head.parameters())

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