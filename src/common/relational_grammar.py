# ============================================================
# Relational Grammar: Right–Left & Swap-Noun Diagrams
# ============================================================

from lambeq import AtomicType
from lambeq.backend.grammar import Word, Cup, Id, Box, Swap
from lambeq import RemoveCupsRewriter


# ------------------------------------------------------------
# Right-vs-Left sentence diagram
# ------------------------------------------------------------

def RvL_sentence_diagram(N, S, remove_cups, noun1, preposition, noun2):

    if preposition == "right":
        prep = "isRightOf"
    elif preposition == "left":
        prep = "isLeftOf"
    else:
        raise ValueError(f"Unknown preposition: {preposition}")

    _noun1 = Word(noun1, N)
    _noun2 = Word(noun2, N)

    _prep = Word(prep, N.r @ S @ N.l)

    cup1 = Cup(N, N.r)
    cup2 = Cup(N.l, N)

    sentence = (_noun1 @ _prep @ _noun2) >> (cup1 @ Id(S) @ cup2)
    return sentence


# ------------------------------------------------------------
# Swap-noun sentence diagram
# ------------------------------------------------------------

def SwapNoun_sentence_diagram(N, S, remove_cups, noun1, preposition, noun2):

    class IsLeftOf(Box):
        def __init__(self):
            super().__init__("isLeftOf", N @ N, S)

    class IsRightOf(Box):
        def __init__(self):
            swap = Swap(N, N)
            left = IsLeftOf()
            super().__init__("isRightOf", N @ N, S)
            self.diagram = swap >> left

    _noun1 = Word(noun1, N)
    _noun2 = Word(noun2, N)

    if preposition == "left":
        prep = IsLeftOf()
        diagram = (_noun1 @ _noun2) >> prep
    elif preposition == "right":
        prep = IsRightOf()
        diagram = (_noun1 @ _noun2) >> prep.diagram
    else:
        raise ValueError(f"Unknown preposition: {preposition}")

    return remove_cups(diagram)


# ------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------

def draw_sentence_diag(df, dataset_selector=1):
    """
    dataset_selector:
        1 → Right-vs-Left
        2 → Swap-Noun
    """

    remove_cups = RemoveCupsRewriter()
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    diagrams = []

    for _, row in df.iterrows():
        noun1 = row["noun1"]
        prep  = row["preposition"]
        noun2 = row["noun2"]

        if dataset_selector == 1:
            d = RvL_sentence_diagram(N, S, remove_cups, noun1, prep, noun2)
        elif dataset_selector == 2:
            d = SwapNoun_sentence_diagram(N, S, remove_cups, noun1, prep, noun2)
        else:
            raise ValueError("dataset_selector must be 1 or 2")

        diagrams.append(d)

    return diagrams

