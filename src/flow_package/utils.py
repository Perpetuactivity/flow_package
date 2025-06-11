import torch
import numpy as np


def to_tensor(state, device="cpu"):
    # state -> tensor
    port = np.int64(state[0])
    protocol = np.int64(state[1])
    # other = np.insert(state[2:], 0, state[0])
    other = np.array(state[2:], dtype=np.float32)

    tensor_port = torch.tensor([port], dtype=torch.long, device=device)
    tensor_protocol = torch.tensor([protocol], dtype=torch.long, device=device)
    tensor_other = torch.tensor(other, dtype=torch.float32, device=device).unsqueeze(0)

    return [tensor_port, tensor_protocol, tensor_other]
