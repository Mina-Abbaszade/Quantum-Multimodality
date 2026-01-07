import torch


def supcon_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Supervised Contrastive Loss (SupCon).

    Args:
        logits: Tensor of shape (N, N),
                where logits[i, j] = cos(s_i, v_j) / temperature
        labels: LongTensor of shape (N,),
                class labels repeated k times
        temperature: Temperature parameter (tau)

    Returns:
        Scalar loss tensor
    """
    N = logits.size(0)

    # Mask for positives (exclude self-comparisons)
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask.fill_diagonal_(False)

    # Scale logits (if already scaled, temperature should match)
    L = logits / max(temperature, 1e-6)

    # Exponentiate
    expL = torch.exp(L)

    # Numerator: sum over positives
    numerator = (expL * mask.float()).sum(dim=1)

    # Denominator: sum over all
    denominator = expL.sum(dim=1).clamp(min=1e-8)

    # Average log-prob over positives
    pos_count = mask.sum(dim=1).clamp(min=1).float()
    loss = - (torch.log((numerator + 1e-8) / denominator) / pos_count).mean()

    return loss

