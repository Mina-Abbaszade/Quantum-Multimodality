import torch
import torch.nn.functional as F
from lambeq import PytorchQuantumModel


class InfoNCEModel(PytorchQuantumModel):
    """
    Quantum InfoNCE model for sentence–image similarity.

    This model:
    - computes quantum circuit outputs for sentence and image diagrams
    - flattens and L2-normalizes embeddings
    - returns temperature-scaled cosine similarity logits (N x N)

    The model is task-agnostic and reusable across:
    - single-object training
    - relational alignment
    - relational classification (as a backbone)
    """

    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def forward(self, diagram_pairs):
        """
        Args:
            diagram_pairs: list of (sentence_diagram, image_diagram)

        Returns:
            logits: Tensor of shape (N, N)
                    logits[i, j] = cos(s_i, v_j) / temperature
        """
        # Unpack sentence and image circuits
        sent_diagrams, img_diagrams = zip(*diagram_pairs)

        # Compute quantum circuit outputs
        out_s = self.get_diagram_output(sent_diagrams)
        out_i = self.get_diagram_output(img_diagrams)

        # Flatten to (N, D)
        N = out_s.size(0)
        out_s = out_s.view(N, -1).to(torch.float32)
        out_i = out_i.view(N, -1).to(torch.float32)

        # L2 normalization (safe)
        out_s = F.normalize(out_s, dim=1, eps=1e-8)
        out_i = F.normalize(out_i, dim=1, eps=1e-8)

        # Cosine similarity matrix
        logits = torch.matmul(out_s, out_i.T)

        # Numerical stability
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        temperature = max(self.temperature, 1e-6)
        logits = logits / temperature
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)

        return logits

