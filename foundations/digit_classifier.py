import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Linear(28*28, 512)
        self.l2 = nn.ReLU()
        self.l3 = nn.Dropout(0.2)
        self.l4 = nn.Linear(512, 10)
        self.l5 = nn.Sigmoid()
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        y = self.l1(torch.reshape(images, (images.shape[0], -1)))
        y = self.l2(y)
        y = self.l3(y)
        y = self.l4(y)
        y = self.l5(y)

        return torch.round(y, decimals=4)
        # Return the model's prediction to 4 decimal places
