
# Define model feature output sizes


acceptable_models = [
    "SqueezeNet1.0",
    "SqueezeNet1.1",
    "ShuffleNetV2_x0.5",
    "ShuffleNetV2_x1.5",
    "ShuffleNetV2_x2.0",
    "MobileNetV2",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "DinoBloom-S"
]
modified_models_feature_output_size = {
    "SqueezeNet1.0": 512,
    "SqueezeNet1.1": 512,
    "ShuffleNetV2_x0.5": 1024,
    "ShuffleNetV2_x1.5": 1024,
    "ShuffleNetV2_x2.0": 2048,
    "MobileNetV2": 1280,
    "ResNet18":512,
    "ResNet34":512,
    "ResNet50":2048,
    "ResNet101":2048
}

# Define function to create feature encoder
def create_feature_encoder(model_name, output_dim,use_pretrained_weights,dataset_type):
    import torch.nn as nn
    from torchvision.models import squeezenet1_0, squeezenet1_1, shufflenet_v2_x0_5, shufflenet_v2_x1_5, shufflenet_v2_x2_0, mobilenet_v2, resnet18, resnet18, resnet34, resnet50
    
    if model_name not in modified_models_feature_output_size:
        raise ValueError(f"Model '{model_name}' not found in available models.")
    if "dinov2" in model_name.lower():
        from models.DinoV2.dinov2 import get_dino_w_resize_and_model_nb_of_params, get_dino_size
        size = get_dino_size(model_name)
        model,_,out_dim = get_dino_w_resize_and_model_nb_of_params(size=size, target_data=dataset_type)
        return model, out_dim
    feature_dim = modified_models_feature_output_size[model_name]
    hidden_dim = output_dim // 2  # Half the output dimension
    weights = "IMAGENET1K_V1" if use_pretrained_weights else None
    # Load base model
    if model_name == "SqueezeNet1.0":
        base_model = squeezenet1_0(weights=weights).features
    elif model_name == "SqueezeNet1.1":
        base_model = squeezenet1_1(weights=weights).features
    elif model_name == "ShuffleNetV2_x0.5":
        base_model = nn.Sequential(*list(shufflenet_v2_x0_5(weights=weights).children())[:-1])
    elif model_name == "ShuffleNetV2_x1.5":
        base_model = nn.Sequential(*list(shufflenet_v2_x1_5(weights=weights).children())[:-1])
    elif model_name == "ShuffleNetV2_x2.0":
        base_model = nn.Sequential(*list(shufflenet_v2_x2_0(weights=weights).children())[:-1])
    elif model_name == "ResNet18":
        base_model = nn.Sequential(*list(resnet18(weights=weights).children())[:-1])
    elif model_name == "ResNet34":
        base_model = nn.Sequential(*list(resnet34(weights=weights).children())[:-1])
    elif model_name == "ResNet50":
        base_model = nn.Sequential(*list(resnet50(weights=weights).children())[:-1])
    elif model_name == "ResNet101":
        base_model = nn.Sequential(*list(resnet50(weights=weights).children())[:-1])
    elif model_name == "MobileNetV2":
        base_model = mobilenet_v2(weights=weights).features
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Build the final model
    feature_encoder = nn.Sequential(
        base_model,
        nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
        nn.Flatten(),  # Flatten output to (batch, feature_dim)
        nn.Linear(feature_dim, hidden_dim),  # FC1: feature_dim -> hidden_dim
        nn.ReLU(),  # Non-linearity
        nn.Linear(hidden_dim, output_dim)  # FC2: hidden_dim -> output_dim
    )

    return feature_encoder, output_dim


if __name__ == "main":
    # Create a dummy input tensor
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    desired_dim = 384
    # Print model information
    for name, _ in modified_models_feature_output_size.items():
        model = create_feature_encoder(name,desired_dim)
        output_size = model(dummy_input).size()
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {name}, Parameters: {num_params:,}, Output Size: {output_size}")