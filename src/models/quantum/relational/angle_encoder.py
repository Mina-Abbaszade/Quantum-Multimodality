"""
Relational ANGLE encoding (quantum)

- Sentence circuits:
    Sim4Ansatz over grammatical diagrams (lambeq parser output)

- Image circuits:
    IQPAnsatz with angle parameters (e.g. PCA-reduced CLIP features)

This file ONLY builds circuits.
NO datasets, NO training loops, NO losses.
"""

from typing import List
import numpy as np

from lambeq import BobcatParser, AtomicType
from lambeq.ansatz.circuit import Sim4Ansatz, IQPAnsatz
from lambeq.backend.grammar import Diagram
from lambeq.backend.quantum import Ty, Box

# ======================================================
# Sentence circuits (RELATIONAL)
# ======================================================

# src/models/quantum/relational/angle_encoder.py

from lambeq import (
    AtomicType,
    IQPAnsatz,
    Sim4Ansatz,
    BobcatParser,
)
from lambeq.backend.grammar import Ty, Box
from lambeq.rewrite import RemoveCupsRewriter

# ============================================================
# Sentence (Relational Grammar → Quantum Circuits)
# ============================================================

def build_relational_sentence_ansatz():
    """
    Quantum ansatz for relational sentence circuits.

    This matches your previous relational experiments:
    - nouns, sentence, prepositional phrase
    - relational structure already encoded in diagrams
    """

    shape1_type = Ty('shape1')
    shape2_type = Ty('shape2')

    ansatz = Sim4Ansatz(
        {
            AtomicType.NOUN: 1,
            AtomicType.SENTENCE: 9,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape1_type: 5,
            shape2_type: 5,
        },
        n_layers=3,
    )

    return ansatz


def build_relational_sentence_circuits(sentence_diagrams, sentence_ansatz):
    """
    Apply the relational sentence ansatz to *pre-built* sentence diagrams.

    IMPORTANT:
    - sentence_diagrams are produced by draw_sentence_diag(...)
    - NO Bobcat parsing happens here
    """

    return [sentence_ansatz(diag) for diag in sentence_diagrams]


# ============================================================
# Image (Angle / IQP Encoding → Quantum Circuits)
# ============================================================

def build_relational_image_ansatz():
    """
    IQP-based image ansatz for relational tasks.

    - PCA-reduced CLIP features (dim=8)
    - encoded as angles
    - mapped onto 9 qubits (padding / expressivity)
    """

    shape1_type = Ty('shape1')
    shape2_type = Ty('shape2')
    clip_shape  = Ty('shape3')
    image_type  = Ty('image')

    ansatz_image = IQPAnsatz(
        {
            AtomicType.NOUN: 9,
            AtomicType.SENTENCE: 9,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape1_type: 5,
            shape2_type: 5,
            clip_shape: 9,   # ⬅ 9 qubits for PCA(8)
            image_type: 1,
        },
        n_layers=1,
    )

    # Image placeholder (no grammar, just a state injection)
    clip_embedding = Box(
        name="CLIP_SHAPE",
        dom=Ty(),
        cod=clip_shape,
    )

    diagram = clip_embedding
    image_circ_skeleton = ansatz_image(diagram)

    return image_circ_skeleton


def build_relational_image_circuits(image_vectors, image_circ_skeleton):
    """
    Instantiate image circuits by binding PCA features to IQP parameters.

    Args:
        image_vectors: np.ndarray of shape (N, 8)
        image_circ_skeleton: circuit returned by build_relational_image_ansatz
    """

    # Map symbol names → symbols
    symbol_dict = {str(s): s for s in image_circ_skeleton.free_symbols}

    # IMPORTANT: fixed ordering (matches PCA dimension = 8)
    ordered_symbol_names = [
        'CLIP_SHAPE__shape3_0',
        'CLIP_SHAPE__shape3_1',
        'CLIP_SHAPE__shape3_2',
        'CLIP_SHAPE__shape3_3',
        'CLIP_SHAPE__shape3_4',
        'CLIP_SHAPE__shape3_5',
        'CLIP_SHAPE__shape3_6',
        'CLIP_SHAPE__shape3_7',
    ]

    ordered_symbols = [symbol_dict[name] for name in ordered_symbol_names]

    # Instantiate circuits
    circuits = [
        image_circ_skeleton.lambdify(*ordered_symbols)(*vec)
        for vec in image_vectors
    ]

    return circuits

