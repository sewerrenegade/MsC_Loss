
import torch.nn as nn
import torch

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.name = "mse"
    
    def forward(self, space_1, space_2):
        loss = self.loss_fn(space_1,space_2)
        return loss,{}