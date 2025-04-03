import torch
import numpy as np


def to_tensor(state):
    # state -> tensor
    port = int(state[1])
    protocol = int(state[2])
    other = np.insert(state[3:], 0, state[0])

    tensor_port = torch.tensor([port], dtype=torch.long)
    tensor_protocol = torch.tensor([protocol], dtype=torch.long)
    tensor_other = torch.tensor(other, dtype=torch.float32).unsqueeze(0)

    return [tensor_port, tensor_protocol, tensor_other]
