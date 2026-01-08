# ==================================================
# MPS single-object training (text = MPS, image = CLIP)
# ==================================================

import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# --------------------------------------------------
# lambeq imports
# --------------------------------------------------
from lambeq import (
    Dataset,
    PytorchTrainer,
    BobcatParser,
    AtomicType
)
from lambeq.ansatz import MPSAnsatz
from lambeq.backend.grammar import Ty, Box
from lambeq.backend.tensor import Dim

# --------------------------------------------------
# Project imports
# --------------------------------------------------
from src.common.data import build_dataset_pairs
from src.models.classical.mps_infonce_model import InfoNCEModel
from src.models.classical.clip_image_encoder import load_clip, encode_image_split
from src.common.losses import supcon_loss, supcon_acc


# ==================================================
# Dataset
# ==================================================

IMAGE_ROOT = "data/single_object"
dataset_pairs = build_dataset_pairs(IMAGE_ROOT)

records_train = dataset_pairs["train"]
records_val   = dataset_pairs["OOD_val"]
records_test  = dataset_pairs["OOD_test"]


# ==================================================
# Sentence circuits (MPS)
# ==================================================

shape_type = Ty("shape")
clip_type  = Ty("clip_shape")
image_type = Ty("image")

shape_box = Box("SHAPE", Ty(), shape_type)
clip_box  = Box("CLIP_SHAPE", Ty(), clip_type)

parser = BobcatParser(verbose="suppress")

dim = 512
ansatz = MPSAnsatz(
    {
        AtomicType.NOUN: Dim(dim),
        AtomicType.SENTENCE: Dim(dim),
        AtomicType.PREPOSITIONAL_PHRASE: Dim(dim),
        clip_type: Dim(dim),
        image_type: Dim(dim),
    },
    bond_dim=10,
)

def build_sentence_circuits(records):
    shapes = [shape for _, shape, _ in records]
    diagrams = [parser.sentence2diagram(s) for s in shapes]
    return [ansatz(d) for d in diagrams]

train_sen = build_sentence_circuits(records_train)
val_sen   = build_sentence_circuits(records_val)
test_sen  = build_sentence_circuits(records_test)


# ==================================================
# Image features (CLIP)
# ==================================================

df_train = pd.DataFrame(records_train, columns=["raw", "shape", "image_file_path"])
df_val   = pd.DataFrame(records_val,   columns=["raw", "shape", "image_file_path"])
df_test  = pd.DataFrame(records_test,  columns=["raw", "shape", "image_file_path"])

clip_model, preprocess, device = load_clip()

train_feats = encode_image_split(df_train, clip_model, preprocess, device)
val_feats   = encode_image_split(df_val,   clip_model, preprocess, device)
test_feats  = encode_image_split(df_test,  clip_model, preprocess, device)


# ==================================================
# SupCon dataset construction
# ==================================================

def build_supcon_dataset(records, sen, img, k=8, batch_size=32):
    shape_to_idxs = defaultdict(list)
    for i, (_, shape, _) in enumerate(records):
        shape_to_idxs[shape].append(i)

    shapes = sorted(shape_to_idxs.keys())

    all_pairs, all_labels = [], []
    for label, s in enumerate(shapes):
        for i in shape_to_idxs[s][:k]:
            all_pairs.append((sen[i], img[i]))
            all_labels.append(label)

    return Dataset(
        all_pairs,
        torch.tensor(all_labels),
        batch_size=batch_size,
    )

train_dataset = build_supcon_dataset(records_train, train_sen, train_feats)
val_dataset   = build_supcon_dataset(records_val,   val_sen,   val_feats)
test_dataset  = build_supcon_dataset(records_test,  test_sen,  test_feats)


# ==================================================
# Model
# ==================================================

all_circuits = train_sen + val_sen + test_sen

model = InfoNCEModel.from_diagrams(
    all_circuits,
    temperature=0.07,
)


# ==================================================
# Training
# ==================================================

trainer = PytorchTrainer(
    model=model,
    loss_function=supcon_loss,
    optimizer=torch.optim.Adam,
    learning_rate=1e-3,
    epochs=10,
    evaluate_functions={"acc": supcon_acc},
    evaluate_on_train=True,
    verbose="text",
)

trainer.fit(train_dataset, val_dataset)


# ==================================================
# Test evaluation
# ==================================================

model.eval()
with torch.no_grad():
    accs = []
    for batch, labels in test_dataset:
        logits = model(batch)
        accs.append(supcon_acc(logits, labels))

print(f"Final test accuracy: {np.mean(accs):.3f}")

