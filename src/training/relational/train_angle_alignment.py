# ============================================================
# Relational Image–Caption Matching (Angle Encoding)
# Task: Image–Caption Alignment (with positives + negatives)
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lambeq import Dataset, PytorchTrainer
from lambeq import RemoveCupsRewriter

from src.common.relational_grammar import draw_sentence_diag
from src.models.quantum.relational.angle_encoder import (
    build_relational_sentence_ansatz,
    build_relational_sentence_circuits,
    build_relational_image_ansatz,
    build_relational_image_circuits,
)
from src.models.quantum.relational.similarity_model import SimilarityModel


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_ROOT = "data/relational"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "ood_val.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "ood_test.csv")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def flip_relation(sentence: str) -> str:
    """Create a hard negative by flipping left/right."""
    if " left " in sentence:
        return sentence.replace(" left ", " right ")
    elif " right " in sentence:
        return sentence.replace(" right ", " left ")
    else:
        raise ValueError(f"Cannot flip relation in sentence: {sentence}")


def build_matching_dataset(df, sen_circuits, img_circuits):
    """
    Build image–caption matching dataset with positives + negatives.
    """
    pairs = []
    labels = []

    sentence_to_index = {
        s: i for i, s in enumerate(df["sentence"].tolist())
    }

    for i, row in df.iterrows():
        # ------------------
        # Positive pair
        # ------------------
        pairs.append((sen_circuits[i], img_circuits[i]))
        labels.append(1)

        # ------------------
        # Negative pair (relation flipped)
        # ------------------
        flipped = flip_relation(row["sentence"])
        if flipped not in sentence_to_index:
            continue

        j = sentence_to_index[flipped]
        pairs.append((sen_circuits[j], img_circuits[i]))
        labels.append(0)

    return pairs, torch.tensor(labels, dtype=torch.float32)


# ------------------------------------------------------------
# Load CSV datasets
# ------------------------------------------------------------

df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

# Image features (PCA-reduced, stored as lists)
train_image_feats = np.vstack(df_train["image_vector"].apply(eval))
val_image_feats   = np.vstack(df_val["image_vector"].apply(eval))
test_image_feats  = np.vstack(df_test["image_vector"].apply(eval))


# ------------------------------------------------------------
# Sentence circuits (relational grammar)
# ------------------------------------------------------------

remove_cups = RemoveCupsRewriter()

sentence_ansatz = build_relational_sentence_ansatz()

# Build grammatical diagrams (RvL dataset)
train_sentence_diagrams = draw_sentence_diag(df_train, dataset_selector=1)
val_sentence_diagrams   = draw_sentence_diag(df_val,   dataset_selector=1)
test_sentence_diagrams  = draw_sentence_diag(df_test,  dataset_selector=1)

# Apply quantum ansatz
train_sen = build_relational_sentence_circuits(
    train_sentence_diagrams,
    sentence_ansatz
)
val_sen = build_relational_sentence_circuits(
    val_sentence_diagrams,
    sentence_ansatz
)
test_sen = build_relational_sentence_circuits(
    test_sentence_diagrams,
    sentence_ansatz
)


# ------------------------------------------------------------
# Image circuits (angle / IQP encoding)
# ------------------------------------------------------------

image_ansatz = build_relational_image_ansatz()

train_img = build_relational_image_circuits(train_image_feats, image_ansatz)
val_img   = build_relational_image_circuits(val_image_feats,   image_ansatz)
test_img  = build_relational_image_circuits(test_image_feats,  image_ansatz)


# ------------------------------------------------------------
# Build matching datasets (positives + negatives)
# ------------------------------------------------------------

train_pairs, train_labels = build_matching_dataset(
    df_train, train_sen, train_img
)
val_pairs, val_labels = build_matching_dataset(
    df_val, val_sen, val_img
)
test_pairs, test_labels = build_matching_dataset(
    df_test, test_sen, test_img
)

train_dataset = Dataset(train_pairs, train_labels, batch_size=32)
val_dataset   = Dataset(val_pairs,   val_labels,   batch_size=32)
test_dataset  = Dataset(test_pairs,  test_labels,  batch_size=32)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

all_circuits = (
    train_sen + train_img +
    val_sen   + val_img +
    test_sen  + test_img
)

model = SimilarityModel.from_diagrams(all_circuits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ------------------------------------------------------------
# Loss & metrics
# ------------------------------------------------------------

def cosine_contrastive_loss(sim, labels):
    """
    Positive pairs -> similarity → +1
    Negative pairs -> similarity → -1
    """
    sim = sim.view(-1)
    labels = labels.view(-1)

    pos_loss = labels * (1.0 - sim).pow(2)
    neg_loss = (1.0 - labels) * (sim + 1.0).pow(2)

    return (pos_loss + neg_loss).mean()


def alignment_accuracy(sim, labels):
    sim = sim.view(-1)

    if not torch.is_tensor(labels):
        labels = torch.tensor(labels, dtype=torch.float32, device=sim.device)
    else:
        labels = labels.float().to(sim.device)

    labels = labels.view(-1)
    preds = (sim > 0.0).float()
    return (preds == labels).float().mean().item()

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

trainer = PytorchTrainer(
    model=model,
    loss_function=cosine_contrastive_loss,
    optimizer=torch.optim.Adam,
    learning_rate=1e-2,
    epochs=10,
    evaluate_functions={"acc": alignment_accuracy},
    evaluate_on_train=True,
    verbose="text",
    seed=SEED,
)

trainer.fit(train_dataset, val_dataset)


# ------------------------------------------------------------
# Final test evaluation
# ------------------------------------------------------------

model.eval()
accs = []

for batch, labels in test_dataset:
    with torch.no_grad():
        sim = model(batch)
    accs.append(alignment_accuracy(sim, labels))

print(f"Final test accuracy: {np.mean(accs):.4f}")

