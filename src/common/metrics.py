import torch
import numpy as np

def supcon_acc(logits, labels):
    """
    Top-1 retrieval accuracy for SupCon batches.
    Works with torch or numpy labels.
    """
    # Ensure torch tensors
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).to(logits.device)
    elif not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=logits.device)

    preds = logits.argmax(dim=1)
    correct = (labels[preds] == labels)

    return correct.float().mean().item()

