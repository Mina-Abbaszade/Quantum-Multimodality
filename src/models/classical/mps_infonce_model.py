import torch
import torch.nn.functional as F
from lambeq import PytorchModel


class InfoNCEModel(PytorchModel):
    """
    Trainer-compatible InfoNCE model for MPS baseline.

    - Sentence side: lambeq MPS circuits (trainable)
    - Image side: fixed CLIP feature vectors (classical)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.temperature = 0.07

    @classmethod
    def from_diagrams(cls, diagrams, **kwargs):
        """
        Allow temperature argument in from_diagrams().
        """
        temperature = kwargs.pop("temperature", 0.07)
        model = super().from_diagrams(diagrams, **kwargs)
        model.temperature = temperature
        return model

    def forward(self, batch):
        """
        batch: list of (sentence_circuit, image_feature) pairs
        """
        sent_circuits, image_feats = zip(*batch)

        # Sentence embeddings (from MPS circuits)
        out_s = self.get_diagram_output(list(sent_circuits))
        out_s = out_s.reshape(out_s.size(0), -1)
        out_s = F.normalize(out_s, dim=1)

        # Image embeddings (classical CLIP vectors)
        import numpy as np

        image_feats = [
            torch.from_numpy(f) if isinstance(f, np.ndarray) else f
            for f in image_feats
        ]

        out_i = torch.stack(image_feats).to(
            out_s.device, dtype=torch.float32
        )
        out_i = F.normalize(out_i, dim=1)

        # Cosine similarity
        logits = out_s @ out_i.T
        return logits / self.temperature

