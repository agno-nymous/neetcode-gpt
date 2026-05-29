import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.l1 = nn.Embedding(vocabulary_size, 16)
        self.l2 = nn.Linear(16, 1)
        self.l3 = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        y = self.l1(x) #B, T, embed_dim
        y = torch.mean(y, axis=1,)# keepdim= True)
        y = self.l2(y)
        return self.l3(y)


