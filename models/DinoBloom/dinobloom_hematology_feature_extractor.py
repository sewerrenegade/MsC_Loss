import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import transforms
#from datasets.SCEMILA.base_SCEMILA import SCEMILAimage_base

DINOBLOOM_DEFAULT_MEAN = (0.485, 0.456, 0.406)
DINOBLOOM_DEFAULT_STD = (0.229, 0.224, 0.225)
DINOBLOOM_NETWORKS_INFOS = {"small":{"out_dim":384,"weights_filename":"DinoBloom-S.pth","model_name":"dinov2_vits14"},
                 "big":{"out_dim":768,"weights_filename":"DinoBloom-B.pth","model_name":"dinov2_vitb14"},
                 "large":{"out_dim":1024,"weights_filename":"DinoBloom-L.pth","model_name":"dinov2_vitl14"},}
DINOBLOOM_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=DINOBLOOM_DEFAULT_MEAN, std=DINOBLOOM_DEFAULT_STD),
])
DEFAULT_PATCH_NUM=16
DINOBLOOM_DEFAULT_IMAGE_DIM=224
eval_model="dinov2_vits14"

def get_path_to_weights(network_weights_name):
    return f"models/DinoBloom/weights/{network_weights_name}"

def resize_224(tensor):
    """
    Resizes the input tensor to 224x224.
    Handles both single images [C, H, W] and batches [B, C, H, W].
    """
    # If the input tensor is a batch of images
    if tensor.ndimension() == 4:
        # Apply resize for each image in the batch
        return torch.stack([F.resize(img, [224, 224]) for img in tensor])
    
    # If the input tensor is a single image
    elif tensor.ndimension() == 3:
        return F.resize(tensor, [224, 224])
    
    # Raise error if the tensor is not [C, H, W] or [B, C, H, W]
    else:
        raise ValueError("Input tensor must have 3 or 4 dimensions (C, H, W) or (B, C, H, W).")

def get_dino_bloom_w_resize(size = "small"):
    dinobloom = get_dino_bloom(size)
    def resize_and_encode(input):
        input = resize_224(input)
        # print(f"dino input shape: {input.shape}")
        dino_out = dinobloom(input)
        # print(f"dino input shape: {dino_out['x_norm_clstoken'].shape}")
        return dino_out["x_norm_clstoken"]
    return resize_and_encode

def get_dino_bloom(size="small"):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GETTING FRESH DINO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    # Load the original DINOv2 model with the correct architecture and parameters.
    network_infos = DINOBLOOM_NETWORKS_INFOS[size]
    model = torch.hub.load('facebookresearch/dinov2', network_infos["model_name"])
    
    # Load finetuned weights
    pretrained = torch.load(get_path_to_weights(network_infos["weights_filename"]), 
                            map_location=torch.device('cpu'), weights_only=True)
    
    # Prepare the correct state dict for loading
    new_state_dict = {}
    for key, value in pretrained['teacher'].items():
        if 'dino_head' in key or "ibot_head" in key:
            continue
        new_key = key.replace('backbone.', '')
        new_state_dict[new_key] = value

    # Add positional embedding for 224x224 images (16x16 patches)
    pos_embed = nn.Parameter(torch.zeros(1, 257, network_infos["out_dim"]))
    model.pos_embed = pos_embed

    # Load state dict
    model.load_state_dict(new_state_dict, strict=True)
    for param in model.parameters():
        param.requires_grad = False

    # Set to eval mode
    model.eval()

    model.cuda()
    
    return model.forward_features

# def get_dino_bloom(size = "small"):
#     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!GETTING FRESH DINO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     # load the original DINOv2 model with the correct architecture and parameters.
#     network_infos= DINOBLOOM_NETWORKS_INFOS[size]
#     model=torch.hub.load('facebookresearch/dinov2', network_infos["model_name"])
#     # load finetuned weights
#     pretrained = torch.load(get_path_to_weights(network_infos["weights_filename"]), map_location=torch.device('cpu'),weights_only= True)
#     # make correct state dict for loading
#     new_state_dict = {}
#     for key, value in pretrained['teacher'].items():
#         if 'dino_head' in key or "ibot_head" in key:
#             pass
#         else:
#             new_key = key.replace('backbone.', '')
#             new_state_dict[new_key] = value

#     #corresponds to 224x224 image. patch size=14x14 => 16*16 patches
#     pos_embed = nn.Parameter(torch.zeros(1, 257, network_infos["out_dim"]))
#     model.pos_embed = pos_embed
#     model.load_state_dict(new_state_dict, strict=True)
#     model.cuda()
#     return model.forward_features