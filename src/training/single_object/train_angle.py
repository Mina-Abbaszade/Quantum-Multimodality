import os
import random
import numpy as np
import torch
import clip
from sklearn.decomposition import PCA
from lambeq import Dataset, PytorchTrainer

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

IMAGE_ROOT = "/path/to/single_object_dataset"

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
        img = preprocess(
            clip_model.visual.preprocess(
                clip_model.visual.preprocess(Image.open(path).convert("RGB"))
            )
        )
        with torch.no_grad():
            f = clip_model.encode_image(img.unsqueeze(0).to(device))
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy().flatten())
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

image_ansatz = build_image_ansatz()
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

model = InfoNCEModel.from_diagrams(
    train_dataset.diagrams,
    temperature=0.07
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

