import torch

class UncertWeighting(torch.nn.Module):
    '''https://arxiv.org/abs/1705.07115'''
    def __init__(self, n_losses):
        super(UncertWeighting, self).__init__()
        self.n_losses = n_losses
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_losses))  # Initialized on CPU

    def forward(self, losses):
        # Move log_vars to correct device if necessary        
        stds = torch.exp(self.log_vars).to(losses.device)
        coeffs = 1 / (2 * stds.pow(2))
        multi_task_losses = losses * coeffs + self.log_vars
        total_loss = multi_task_losses.sum()
        return total_loss, multi_task_losses

    def get_weights(self):
        stds = torch.exp(self.log_vars)
        coeffs = 1 / (2 * stds.pow(2))
        return {f"loss_{i+1}_weight": coeffs[i].item() for i in range(self.n_losses)}
