from lambeq import AtomicType
from lambeq.ansatz import MPSAnsatz
from lambeq.backend.grammar import Ty, Box
from lambeq.backend.tensor import Dim
from lambeq import BobcatParser


# --------------------------------------------------
# MPS ansatz builder
# --------------------------------------------------

def build_mps_ansatz(
    dim: int = 512,
    bond_dim: int = 10,
):
    """
    Build a classical MPS ansatz for sentence–image matching.
    """
    shape_type = Ty("shape")
    clip_type = Ty("clip_shape")
    image_type = Ty("image")

    return MPSAnsatz(
        {
            AtomicType.NOUN: Dim(dim),
            AtomicType.SENTENCE: Dim(dim),
            AtomicType.PREPOSITIONAL_PHRASE: Dim(dim),
            clip_type: Dim(dim),
            image_type: Dim(dim),
        },
        bond_dim=bond_dim,
    )


# --------------------------------------------------
# Sentence circuit construction
# --------------------------------------------------

def build_mps_sentence_circuits(
    captions,
    ansatz,
):
    """
    Build MPS sentence circuits from captions.
    """
    parser = BobcatParser(verbose="suppress")

    diagrams = [
        parser.sentence2diagram(caption)
        for caption in captions
    ]

    circuits = [
        ansatz(diagram)
        for diagram in diagrams
    ]

    return circuits


# --------------------------------------------------
# Image diagram template
# --------------------------------------------------

def build_mps_image_diagram():
    """
    Return a symbolic image placeholder diagram for MPS.
    """
    clip_type = Ty("clip_shape")
    return Box("CLIP_SHAPE", dom=Ty(), cod=clip_type)

