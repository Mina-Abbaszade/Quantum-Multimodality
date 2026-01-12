# ============================================================
# Relational Binary Relation Classification (Angle Encoding)
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch

from lambeq import Dataset, PytorchTrainer

from src.common.relational_grammar import draw_sentence_diag
from src.models.quantum.relational.angle_encoder import (
    build_relational_sentence_ansatz,
    build_relational_sentence_circuits,
    build_relational_image_ansatz,
    build_relational_image_circuits,
)
from src.models.quantum.relational.binary_relation_model import BinaryRelationModel
from src.common.binary_metrics import binary_loss, binary_accuracy


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 63
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

DATA_ROOT = "data/relational"

TRAIN_CSV = os.path.join(DATA_ROOT, "train_binary.csv")
VAL_CSV   = os.path.join(DATA_ROOT, "ood_val_binary.csv")
TEST_CSV  = os.path.join(DATA_ROOT, "ood_test_binary.csv")


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

df_train = pd.read_csv(TRAIN_CSV)
df_val   = pd.read_csv(VAL_CSV)
df_test  = pd.read_csv(TEST_CSV)

train_labels = torch.tensor(df_train["label"].values)
val_labels   = torch.tensor(df_val["label"].values)
test_labels  = torch.tensor(df_test["label"].values)

train_image_feats = np.vstack(df_train["image_vector_pca"].apply(eval))
val_image_feats   = np.vstack(df_val["image_vector_pca"].apply(eval))
test_image_feats  = np.vstack(df_test["image_vector_pca"].apply(eval))


# ------------------------------------------------------------
# Sentence circuits
# ------------------------------------------------------------

sentence_ansatz = build_relational_sentence_ansatz()

train_sentence_diagrams = draw_sentence_diag(df_train, dataset_selector=1)
val_sentence_diagrams   = draw_sentence_diag(df_val, dataset_selector=1)
test_sentence_diagrams  = draw_sentence_diag(df_test, dataset_selector=1)

train_sen = build_relational_sentence_circuits(train_sentence_diagrams, sentence_ansatz)
val_sen   = build_relational_sentence_circuits(val_sentence_diagrams, sentence_ansatz)
test_sen  = build_relational_sentence_circuits(test_sentence_diagrams, sentence_ansatz)


# ------------------------------------------------------------
# Image circuits (angle encoding)
# ------------------------------------------------------------

print(type(df_train["image_vector_pca"].iloc[0]))
print(len(df_train["image_vector_pca"].iloc[0]))
print(df_train["image_vector_pca"].iloc[0])

image_ansatz = build_relational_image_ansatz()

train_img = build_relational_image_circuits(train_image_feats, image_ansatz)
val_img   = build_relational_image_circuits(val_image_feats, image_ansatz)
test_img  = build_relational_image_circuits(test_image_feats, image_ansatz)


# ------------------------------------------------------------
# Pair datasets
# ------------------------------------------------------------

train_pairs = list(zip(train_sen, train_img))
val_pairs   = list(zip(val_sen, val_img))
test_pairs  = list(zip(test_sen, test_img))

train_dataset = Dataset(train_pairs, train_labels, batch_size=16)
val_dataset   = Dataset(val_pairs, val_labels, batch_size=16)
test_dataset  = Dataset(test_pairs, test_labels, batch_size=16)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

all_circuits = (
    train_sen + train_img +
    val_sen   + val_img +
    test_sen  + test_img
)

model = BinaryRelationModel.from_diagrams(all_circuits)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

trainer = PytorchTrainer(
    model=model,
    loss_function=binary_loss,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    epochs=10,
    evaluate_functions={"acc": binary_accuracy},
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
        logits = model(batch)
    accs.append(binary_accuracy(logits, labels))

print(f"Final test accuracy: {np.mean(accs):.4f}")

