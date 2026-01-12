import torch
import torch.nn.functional as F


def binary_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.view(-1)
    labels = labels.view(-1).float()
    return F.binary_cross_entropy_with_logits(logits, labels)


def binary_accuracy(logits, labels):
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, device=logits.device)

    logits = logits.view(-1)
    labels = labels.view(-1)

    preds = (logits > 0).long()
    return (preds == labels).float().mean().item()

