import torch


def supcon_acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Top-1 retrieval accuracy for supervised contrastive batches.

    Args:
        logits: Tensor of shape (N, N)
        labels: Tensor of shape (N,)

    Returns:
        Accuracy as a float
    """
    # For each anchor, select the most similar image
    preds = logits.argmax(dim=1)

    # Check if retrieved item has the same label
    correct = labels[preds] == labels

    return correct.float().mean().item()

