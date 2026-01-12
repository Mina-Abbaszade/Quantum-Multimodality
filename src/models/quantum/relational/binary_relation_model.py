import torch
import torch.nn as nn
import torch.nn.functional as F
from lambeq import PytorchQuantumModel


class BinaryRelationModel(PytorchQuantumModel):
    """
    Binary classifier for relational tasks (left vs right).

    Input: (sentence_circuit, image_circuit)
    Output: logit ∈ ℝ
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = None

    def _init_classifier(self, input_dim: int):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
        )

    def forward(self, diagram_pairs):
        sent, img = zip(*diagram_pairs)

        out_s = self.get_diagram_output(sent)
        out_i = self.get_diagram_output(img)

        B = out_s.size(0)
        out_s = out_s.view(B, -1).float()
        out_i = out_i.view(B, -1).float()

        out_s = F.normalize(out_s, dim=1)
        out_i = F.normalize(out_i, dim=1)

        z = torch.cat(
            [
                out_s,
                out_i,
                out_s * out_i,
                torch.abs(out_s - out_i),
            ],
            dim=1,
        )

        if self.classifier is None:
            self._init_classifier(z.size(1))
            self.classifier = self.classifier.to(z.device)

        logits = self.classifier(z)
        return logits.squeeze(-1)

