# src/models/quantum/amplitude_encoder.py

import numpy as np
import pennylane as qml
from lambeq import BobcatParser
from lambeq.ansatz.circuit import IQPAnsatz
from lambeq.backend.quantum import Ty, Box, qubit, to_circuital

# --------------------------------------------------
# Sentence side (quantum text)
# --------------------------------------------------

from lambeq import AtomicType
from lambeq.ansatz.circuit import Sim4Ansatz
from lambeq.backend.quantum import Ty

def build_sentence_ansatz(n_layers: int = 3):
    """
    Sentence ansatz identical to angle/OHE setups.
    """
    shape_type = Ty("shape")
    clip_type  = Ty("clip_shape")
    image_type = Ty("image")

    return Sim4Ansatz(
        {
            AtomicType.NOUN: 9,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape_type: 5,
            clip_type: 9,
            image_type: 1,
        },
        n_layers=n_layers
    )

def build_sentence_circuits(captions, ansatz):
    """
    Convert captions into lambeq quantum circuits.
    """
    parser = BobcatParser(verbose="suppress")
    diagrams = [parser.sentence2diagram(c) for c in captions]
    circuits = [ansatz(d) for d in diagrams]
    return circuits


# --------------------------------------------------
# Image side (amplitude encoding)
# --------------------------------------------------

def _amplitude_qnode(n_qubits: int):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(features):
        qml.AmplitudeEmbedding(
            features=features,
            wires=range(n_qubits),
            normalize=True
        )
        return qml.state()

    return circuit


def amplitude_encode_images(image_features, n_qubits: int = 9):
    """
    Encode CLIP features into quantum statevectors via amplitude encoding.
    """
    qnode = _amplitude_qnode(n_qubits)

    encoded = []
    for feat in image_features:
        state = qnode(feat)
        encoded.append(state)

    return encoded


def build_amplitude_image_circuits(encoded_states, n_qubits: int = 9):
    """
    Wrap amplitude-encoded statevectors into lambeq diagrams.
    """
    clip_type = Ty().tensor(*[qubit] * n_qubits)
    circuits = []

    for state in encoded_states:
        box = Box("CLIP", dom=Ty(), cod=clip_type)
        box.data = state
        circuits.append(box.to_diagram())

    return circuits

