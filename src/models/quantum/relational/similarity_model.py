# src/models/quantum/relational/similarity_model.py

import torch
import torch.nn.functional as F
from lambeq import PytorchQuantumModel


class SimilarityModel(PytorchQuantumModel):
    """
    Quantum similarity model for relational image–caption matching.

    Input:  list of (sentence_circuit, image_circuit)
    Output: cosine similarity in [-1, 1]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, diagram_pairs):
        sent, img = zip(*diagram_pairs)

        out_s = self.get_diagram_output(sent)
        out_i = self.get_diagram_output(img)

        B = out_s.size(0)
        out_s = out_s.view(B, -1)
        out_i = out_i.view(B, -1)

        out_s = F.normalize(out_s, dim=1)
        out_i = F.normalize(out_i, dim=1)

        similarity = F.cosine_similarity(out_s, out_i, dim=1)
        return similarity.unsqueeze(1)

