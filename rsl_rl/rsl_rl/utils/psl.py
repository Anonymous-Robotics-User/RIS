import torch

class PSL():
    def __init__(self):
        obs_indices = []
        obs_multiplier = []
        action_indices = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8] # Joint Rotation
        action_multiplier = [-1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]

        self.obs_indices = torch.tensor(obs_indices, device=self.device, requires_grad=False) 
        self.obs_multiplier = torch.tensor(obs_multiplier, device=self.device, requires_grad=False)
        self.action_indices = torch.tensor(action_indices, device=self.device, requires_grad=False)
        self.action_multiplier = torch.tensor(action_multiplier, device=self.device, requires_grad=False)