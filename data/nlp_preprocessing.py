import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive + negative
        sent_list = [sent.split() for sent in combined]
        token_list = set()
        for sent in sent_list:
            for w in sent:
                token_list.add(w)

        token2id = {v:k+1 for k, v in enumerate(sorted(list(token_list)))}

        final = []
        for sent in sent_list:
            final.append(
                torch.Tensor([token2id.get(x) for x in sent])
            )
        # final = final

        return torch.nn.utils.rnn.pad_sequence(final, batch_first=True).tolist()
