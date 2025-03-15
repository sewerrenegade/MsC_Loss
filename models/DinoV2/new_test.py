import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import os

target_data = ['single_cell','image_net']
sizes = ['small','big','large','giant']
DINO_NETWORKS_INFOS = {
    "small": {"out_dim": 384, "single_cell_weights_filename": "DinoBloom-S.pth", "image_net_weights_filename": "dinov2_vits14_pretrain.pth","model_name": "dinov2_vits14"},
    "big": {"out_dim": 768, "single_cell_weights_filename": "DinoBloom-B.pth","image_net_weights_filename": "dinov2_vitb14_pretrain.pth", "model_name": "dinov2_vitb14"},
    "large": {"out_dim": 1024, "single_cell_weights_filename": "DinoBloom-L.pth","image_net_weights_filename": "dinov2_vitl14_pretrain.pth", "model_name": "dinov2_vitl14"},
    "giant": {"out_dim": 1536, "single_cell_weights_filename": "DinoBloom-G.pth","image_net_weights_filename": "dinov2_vitg14_pretrain.pth", "model_name": "dinov2_vitg14"}
}

def resize_224(tensor):
    """
    Resizes the input tensor to 224x224.
    Handles both single images [C, H, W] and batches [B, C, H, W].
    """
    if tensor.ndim == 4:  # Batch of images
        return F.resize(tensor, [224, 224])
    elif tensor.ndim == 3:  # Single image
        return F.resize(tensor.unsqueeze(0), [224, 224]).squeeze(0)
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")



def get_path_to_weights(target_dataset, network_weights_name):
    """Returns the full path to the pretrained weight file."""
    path = f"models/DinoV2/{target_dataset}_weights/{network_weights_name}"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Incorrect path: {path} does not exist.")
    
    return path

def get_dino(size = "small",target_data = "single_cell"):
    # load the original DINOv2 model with the correct architecture and parameters.
    model=torch.hub.load('facebookresearch/dinov2', DINO_NETWORKS_INFOS[size]['model_name'],pretrained= not target_data == "single_cell")

    # make correct state dict for loading
    if target_data == "single_cell":
        weights_path = get_path_to_weights(target_data,DINO_NETWORKS_INFOS[size][f"{target_data}_weights_filename"])
        pretrained = torch.load(weights_path, map_location=torch.device('cpu'))
        new_state_dict = {}
        for key, value in pretrained['teacher'].items():
            if 'dino_head' in key or "ibot_head" in key:
                pass
            else:
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        pretrained = new_state_dict
    #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
        pos_embed = nn.Parameter(torch.zeros(1, 257,DINO_NETWORKS_INFOS[size]["out_dim"]))
        model.pos_embed = pos_embed

        model.load_state_dict(pretrained, strict=True)
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(param.numel() for param in model.parameters())

    # Set model to evaluation mode and move to the best available device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model.forward_features ,total_params

def get_dino_w_resize_and_model_nb_of_params(size = "small",target_data = "single_cell"):
    """Returns a function that resizes input images and passes them through the DINO Bloom model."""
    model, model_nb_params = get_dino(size,target_data)
    
    def resize_and_encode(input_tensor):
        input_tensor = resize_224(input_tensor)
        with torch.no_grad():
            output = model(input_tensor)
        return output["x_norm_clstoken"]

    return resize_and_encode, model_nb_params


if __name__ == '__main__':
    model = get_dino_w_resize_and_model_nb_of_params(size="small",target_data='single_cell')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = torch.rand((2,3,226,227),device=device)
    print(model(dummy_input).size())