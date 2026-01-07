from typing import List, Tuple

from lambeq.backend.grammar import Ty, Box
from lambeq import BobcatParser, AtomicType
from lambeq.ansatz.circuit import Sim4Ansatz
from lambeq.ansatz import IQPAnsatz


# -------------------------
# Type definitions
# -------------------------

shape_type = Ty('shape')
clip_type = Ty('clip_shape')
image_type = Ty('clip_shape')


# -------------------------
# Diagram templates
# -------------------------

shape_box = Box(name='SHAPE', dom=Ty(), cod=shape_type)
clip_embedding_box = Box(name='CLIP_SHAPE', dom=Ty(), cod=image_type)

# Image diagram template (same for all samples)
image_diagram = clip_embedding_box


# -------------------------
# Ansatz builders
# -------------------------

def build_sentence_ansatz(
    n_layers: int = 10,
    n_qubits: int = 9
):
    """
    Builds an IQP sentence ansatz for angle encoding.
    """
    return IQPAnsatz(
        {
            AtomicType.NOUN: n_qubits,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape_type: 5,
            clip_type: n_qubits,
            image_type: 1,
        },
        n_layers=n_layers
    )


def build_image_ansatz(
    n_layers: int = 1,
    n_qubits: int = 9
):
    """
    Builds a Sim4 image ansatz for CLIP-based angle encoding.
    """
    return Sim4Ansatz(
        {
            AtomicType.NOUN: n_qubits,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape_type: 5,
            clip_type: n_qubits,
            image_type: 3,
        },
        n_layers=n_layers
    )


# -------------------------
# Circuit construction
# -------------------------

def build_sentence_circuits(
    captions: List[str],
    ansatz
):
    """
    Parses captions and converts them into quantum sentence circuits.
    """
    parser = BobcatParser(verbose='suppress')
    diagrams = [parser.sentence2diagram(c) for c in captions]
    circuits = [ansatz(d) for d in diagrams]
    return circuits


def build_image_circuits(
    image_encodings,
    ansatz,
    ordered_symbols
):
    """
    Converts CLIP angle-encoded vectors into quantum image circuits.

    image_encodings: numpy array of shape (N, D)
    ordered_symbols: list of lambeq symbols matching circuit parameters
    """
    image_circ_template = ansatz(image_diagram)
    circuits = [
        image_circ_template.lambdify(*ordered_symbols)(
            *encoding
        )
        for encoding in image_encodings
    ]
    return circuits

