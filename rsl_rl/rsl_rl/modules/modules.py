from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import unpad_trajectories

class Weight_EMA_Update(object):
    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = (self.decay)*state_dict[key] + (1-self.decay)*new_state_dict[key]
        self.model.load_state_dict(state_dict)