import os
import random
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.decomposition import PCA
from lambeq import Dataset, PytorchTrainer
from lambeq import IQPAnsatz, AtomicType
from lambeq.backend.grammar import Ty, Box

from src.common.data import build_dataset_pairs
from src.common.losses import supcon_loss
from src.common.metrics import supcon_acc
from src.models.similarity_model import InfoNCEModel
from src.models.quantum.angle_encoder import (
    build_sentence_ansatz,
    build_image_ansatz,
    build_sentence_circuits,
    build_image_circuits,
)

# -------------------------
# Reproducibility
# -------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------
# Paths (EDIT THIS)
# -------------------------

IMAGE_ROOT = "data/single_object"

# -------------------------
# Load dataset
# -------------------------

dataset_pairs = build_dataset_pairs(IMAGE_ROOT)

train_records = dataset_pairs["train"]
val_records   = dataset_pairs["OOD_val"]
test_records  = dataset_pairs["OOD_test"]

train_captions = [s for _, s, _ in train_records]
val_captions   = [s for _, s, _ in val_records]
test_captions  = [s for _, s, _ in test_records]

# -------------------------
# Build sentence circuits
# -------------------------

sentence_ansatz = build_sentence_ansatz()
train_sen = build_sentence_circuits(train_captions, sentence_ansatz)
val_sen   = build_sentence_circuits(val_captions, sentence_ansatz)
test_sen  = build_sentence_circuits(test_captions, sentence_ansatz)

# -------------------------
# Encode images with CLIP + PCA
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

def encode_images(records):
    feats = []
    for _, _, path in records:
        image = Image.open(path).convert("RGB")

        #  correct CLIP preprocessing (ONCE)
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(image_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        feats.append(feat.cpu().numpy().flatten())

    return np.vstack(feats)

train_feats = encode_images(train_records)
val_feats   = encode_images(val_records)
test_feats  = encode_images(test_records)

pca = PCA(n_components=8)
train_feats = pca.fit_transform(train_feats)
val_feats   = pca.transform(val_feats)
test_feats  = pca.transform(test_feats)

# -------------------------
# Build image circuits
# -------------------------

# --------------------------------------------------
# Image diagram template (CLIP placeholder)
# --------------------------------------------------

clip_shape = Ty('clip_shape')
image_diagram = Box(
    name='CLIP_SHAPE',
    dom=Ty(),
    cod=clip_shape
)

# --------------------------------------------------
# Image ansatz (angle encoding, single-object)
# --------------------------------------------------

clip_shape = Ty('clip_shape')
image_type = Ty('image')

image_ansatz = IQPAnsatz(
    {
        AtomicType.NOUN: 9,
        AtomicType.SENTENCE: 1,
        AtomicType.PREPOSITIONAL_PHRASE: 1,
        clip_shape: 9,   # 9 angles for CLIP features
        image_type: 1,
    },
    n_layers=1
)

ordered_symbols = list(image_ansatz(image_diagram).free_symbols)

train_img = build_image_circuits(train_feats, image_ansatz, ordered_symbols)
val_img   = build_image_circuits(val_feats, image_ansatz, ordered_symbols)
test_img  = build_image_circuits(test_feats, image_ansatz, ordered_symbols)

# -------------------------
# Build SupCon datasets
# -------------------------

def build_supcon_dataset(records, sen_circuits, img_circuits, batch_size=32):
    shapes = sorted({s for _, s, _ in records})
    shape_to_idxs = {s: [] for s in shapes}
    for i, (_, s, _) in enumerate(records):
        shape_to_idxs[s].append(i)

    pairs, labels = [], []
    for s_idx, s in enumerate(shapes):
        idxs = shape_to_idxs[s][:8]
        for i in idxs:
            pairs.append((sen_circuits[i], img_circuits[i]))
            labels.append(s_idx)

    return Dataset(pairs, torch.tensor(labels), batch_size=batch_size)

train_dataset = build_supcon_dataset(train_records, train_sen, train_img)
val_dataset   = build_supcon_dataset(val_records,   val_sen,   val_img)
test_dataset  = build_supcon_dataset(test_records,  test_sen,  test_img)

# -------------------------
# Train
# -------------------------

all_circuits = (
    train_sen + train_img +
    val_sen   + val_img +
    test_sen  + test_img
)
model = InfoNCEModel.from_diagrams(all_circuits)

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

# -------------------------
# Test
# -------------------------

model.eval()
accs = []
with torch.no_grad():
    for diagrams, labels in test_dataset:
        logits = model(diagrams)
        accs.append(supcon_acc(logits, labels))

print(f"Final OOD test accuracy: {np.mean(accs):.3f}")

# ============================================================
# Save pretrained shape parameters (for multi-stage learning)
# ============================================================

import os

SAVE_DIR = "pretrained"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, "single_object_angle_params.txt")

final_weights = model.weights.detach().cpu().numpy()
symbols = model.symbols

# Only save shape-related parameters
SHAPE_TOKENS = {"cone", "cube", "sphere", "cylinder"}

with open(SAVE_PATH, "w") as f:
    for sym, val in zip(symbols, final_weights):
        name = sym.name
        if any(tok in name for tok in SHAPE_TOKENS):
            f.write(f"{name}: {val:.6f}\n")

print(f"[OK] Saved pretrained shape parameters → {SAVE_PATH}")

