import numpy as np
import pandas as pd
from typing import Iterable, Tuple

from lambeq.backend.grammar import Ty, Box
from lambeq.ansatz import IQPAnsatz
from lambeq import AtomicType


# --------------------------------------------------
# Constants
# --------------------------------------------------

ALLOWED_SHAPES = {"cylinder", "sphere", "cube", "cone"}
DEFAULT_ORDER = ("cylinder", "sphere", "cube", "cone")


# --------------------------------------------------
# Data utilities (symbolic, not CLIP-based)
# --------------------------------------------------

def to_dataframe(triples: Iterable[Tuple[str, str, str]]) -> pd.DataFrame:
    """
    Convert dataset_pairs split into a clean DataFrame with shape labels.
    """
    rows = []
    for raw_cap, shape_cap, full_path in triples:
        shape = shape_cap.strip().lower()
        if shape == "":
            continue
        shape = shape.split()[-1]
        rows.append(
            {
                "raw_caption": raw_cap,
                "shape": shape,
                "image_file_path": full_path,
            }
        )

    df = pd.DataFrame(rows)

    bad = ~df["shape"].isin(ALLOWED_SHAPES)
    if bad.any():
        df = df.loc[~bad].reset_index(drop=True)

    return df.reset_index(drop=True)


def encode_image_split_onehot(
    df: pd.DataFrame,
    label_col: str = "shape",
    order=DEFAULT_ORDER,
    *,
    pad_to: int | None = None,
    repeat_to: int | None = None,
    dtype=np.float32,
    strict: bool = True,
) -> np.ndarray:
    """
    Convert shape labels into one-hot vectors.
    """
    order = tuple(order)
    idx = {name.lower(): i for i, name in enumerate(order)}
    D = len(order)

    feats = []
    for lab in df[label_col].astype(str).str.lower():
        if lab not in idx:
            if strict:
                raise ValueError(f"Unknown label '{lab}'. Expected one of {order}.")
            vec = np.zeros(D, dtype=dtype)
        else:
            vec = np.zeros(D, dtype=dtype)
            vec[idx[lab]] = 1.0

        if repeat_to is not None:
            if repeat_to % D != 0:
                raise ValueError(
                    f"repeat_to={repeat_to} not multiple of base dim {D}."
                )
            vec = np.tile(vec, repeat_to // D)

        if pad_to is not None:
            if pad_to < vec.size:
                raise ValueError(f"pad_to={pad_to} < current size {vec.size}.")
            if pad_to > vec.size:
                vec = np.concatenate(
                    [vec, np.zeros(pad_to - vec.size, dtype=dtype)]
                )

        feats.append(vec)

    return np.vstack(feats)


# --------------------------------------------------
# OHE quantum ansatz builders
# --------------------------------------------------

def build_ohe_sentence_ansatz(n_layers: int = 5):
    """
    Sentence ansatz for symbolic quantum (OHE) encoding.
    """
    shape_type = Ty("shape")
    clip_type = Ty("clip_shape")
    image_type = Ty("image")

    return IQPAnsatz(
        {
            AtomicType.NOUN: 5,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape_type: 5,
            clip_type: 5,
            image_type: 5,
        },
        n_layers=n_layers,
    )


def build_ohe_image_ansatz(n_layers: int = 1):
    """
    Image ansatz for symbolic quantum (OHE) encoding.
    """
    shape_type = Ty("shape")
    clip_type = Ty("clip_shape")
    image_type = Ty("image")

    return IQPAnsatz(
        {
            AtomicType.NOUN: 5,
            AtomicType.SENTENCE: 1,
            AtomicType.PREPOSITIONAL_PHRASE: 1,
            shape_type: 5,
            clip_type: 5,
            image_type: 5,
        },
        n_layers=n_layers,
    )

# --------------------------------------------------
# OHE image circuit construction
# --------------------------------------------------

def build_ohe_image_circuits(
    image_diagram,
    image_ansatz,
    image_features: np.ndarray,
):
    """
    Build quantum image circuits from one-hot encoded features
    using lambdify.

    Args:
        image_diagram: lambeq Box used as image placeholder
        image_ansatz: IQPAnsatz for OHE images
        image_features: np.ndarray of shape (N, D)

    Returns:
        List of lambeq quantum diagrams
    """
    image_circ_template = image_ansatz(image_diagram)

    # Extract and order free symbols exactly as lambeq expects
    symbol_dict = {str(s): s for s in image_circ_template.free_symbols}
    ordered_symbols = [
        symbol_dict[name] for name in sorted(symbol_dict.keys())
    ]

    circuits = [
        image_circ_template.lambdify(*ordered_symbols)(*feat)
        for feat in image_features
    ]

    return circuits

# --------------------------------------------------
# OHE sentence circuit construction
# --------------------------------------------------

from lambeq import BobcatParser


def build_ohe_sentence_circuits(
    captions,
    sentence_ansatz,
):
    """
    Build quantum sentence circuits for symbolic quantum (OHE) encoding.

    Args:
        captions: list of cleaned shape captions (e.g. ["cube", "sphere"])
        sentence_ansatz: IQPAnsatz returned by build_ohe_sentence_ansatz

    Returns:
        List of lambeq quantum sentence circuits
    """
    parser = BobcatParser(verbose="suppress")

    diagrams = [
        parser.sentence2diagram(caption)
        for caption in captions
    ]

    circuits = [
        sentence_ansatz(diagram)
        for diagram in diagrams
    ]

    return circuits

