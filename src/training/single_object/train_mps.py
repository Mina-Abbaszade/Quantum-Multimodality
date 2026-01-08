import random
import numpy as np
import torch

from lambeq import Dataset, PytorchTrainer

from src.common.data import build_dataset_pairs
from src.common.losses import supcon_loss
from src.common.metrics import supcon_acc

from src.models.classical.mps_encoder import (
    build_mps_ansatz,
    build_mps_sentence_circuits,
    build_mps_image_diagram,
)

from src.models.classical.clip_image_encoder import (
    load_clip,
    encode_image_split,
)

from src.models.classical.mps_infonce_model import InfoNCEModel


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------
# Paths (EDIT THIS)
# --------------------------------------------------

IMAGE_ROOT = "/path/to/single_object_dataset"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset_pairs = build_dataset_pairs(IMAGE_ROOT)

records_train = dataset_pairs["train"]
records_val   = dataset_pairs["OOD_val"]
records_test  = dataset_pairs["OOD_test"]


# --------------------------------------------------
# Sentence circuits (MPS)
# --------------------------------------------------

train_shapes = [shape for _, shape, _ in records_train]
val_shapes   = [shape for _, shape, _ in records_val]
test_shapes  = [shape for _, shape, _ in records_test]

ansatz = build_mps_ansatz(dim=512, bond_dim=10)

train_sen = build_mps_sentence_circuits(train_shapes, ansatz)
val_sen   = build_mps_sentence_circuits(val_shapes,   ansatz)
test_sen  = build_mps_sentence_circuits(test_shapes,  ansatz)


# --------------------------------------------------
# Image features (CLIP)
# --------------------------------------------------

import pandas as pd

df_train = pd.DataFrame(records_train, columns=["raw", "shape", "image_file_path"])
df_val   = pd.DataFrame(records_val,   columns=["raw", "shape", "image_file_path"])
df_test  = pd.DataFrame(records_test,  columns=["raw", "shape", "image_file_path"])

clip_model, preprocess, device = load_clip()

train_feats = encode_image_split(df_train, clip_model, preprocess, device)
val_feats   = encode_image_split(df_val,   clip_model, preprocess, device)
test_feats  = encode_image_split(df_test,  clip_model, preprocess, device)


# --------------------------------------------------
# SupCon batching (identical to quantum)
# --------------------------------------------------

from collections import defaultdict

def build_supcon_dataset(records, sen, img, k=8, batch_size=32):
    shape_to_idxs = defaultdict(list)
    for i, (_, shape, _) in enumerate(records):
        shape_to_idxs[shape].append(i)

    shapes = sorted(shape_to_idxs.keys())

    all_pairs, all_labels = [], []
    for s_idx, s in enumerate(shapes):
        for i in shape_to_idxs[s][:k]:
            all_pairs.append((sen[i], img[i]))
            all_labels.append(s_idx)

    return Dataset(
        all_pairs,
        torch.tensor(all_labels),
        batch_size=batch_size,
    )


train_dataset = build_supcon_dataset(records_train, train_sen, train_feats)
val_dataset   = build_supcon_dataset(records_val,   val_sen,   val_feats)
test_dataset  = build_supcon_dataset(records_test,  test_sen,  test_feats)


# --------------------------------------------------
# Train
# --------------------------------------------------

model = InfoNCEModel.from_diagrams(
    train_dataset.diagrams,
    temperature=0.07,
)

trainer = PytorchTrainer(
    model=model,
    loss_function=supcon_loss,
    optimizer=torch.optim.Adam,
    learning_rate=1e-2,
    epochs=10,
    evaluate_functions={"acc": supcon_acc},
    evaluate_on_train=True,
    seed=SEED,
)

trainer.fit(train_dataset, val_dataset)


# --------------------------------------------------
# Test
# --------------------------------------------------

model.eval()
accs = []

with torch.no_grad():
    for diagrams, labels in test_dataset:
        logits = model(diagrams)
        accs.append(supcon_acc(logits, labels))

print(f"Final OOD test accuracy (MPS): {np.mean(accs):.3f}")

