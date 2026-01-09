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
from src.models.quantum.amplitude_encoder import (
    build_sentence_ansatz,
    build_sentence_circuits,
    amplitude_encode_images,
    build_amplitude_image_circuits,
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

train_caps = [s for _, s, _ in train_records]
val_caps   = [s for _, s, _ in val_records]
test_caps  = [s for _, s, _ in test_records]

# -------------------------
# Sentence circuits
# -------------------------

sentence_ansatz = build_sentence_ansatz()
train_sen = build_sentence_circuits(train_caps, sentence_ansatz)
val_sen   = build_sentence_circuits(val_caps, sentence_ansatz)
test_sen  = build_sentence_circuits(test_caps, sentence_ansatz)

# -------------------------
# Image features (CLIP + PCA)
# -------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

from PIL import Image
import numpy as np
import torch

def encode_images(records):
    feats = []
    for _, _, path in records:
        image = Image.open(path).convert("RGB")

        # Correct CLIP preprocessing (ONCE)
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = clip_model.encode_image(image_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        feats.append(feat.cpu().numpy().flatten())

    return np.vstack(feats)

train_feats = encode_images(train_records)
val_feats   = encode_images(val_records)
test_feats  = encode_images(test_records)

assert train_feats.shape[1] == 512
assert val_feats.shape[1] == 512
assert test_feats.shape[1] == 512

# -------------------------
# Amplitude encoding
# -------------------------

train_states = amplitude_encode_images(train_feats)
val_states   = amplitude_encode_images(val_feats)
test_states  = amplitude_encode_images(test_feats)

train_img = build_amplitude_image_circuits(train_states)
val_img   = build_amplitude_image_circuits(val_states)
test_img  = build_amplitude_image_circuits(test_states)

# -------------------------
# SupCon dataset builder
# -------------------------

def build_supcon_dataset(records, sen, img, batch_size=32):
    shapes = sorted({s for _, s, _ in records})
    shape_to_idxs = {s: [] for s in shapes}
    for i, (_, s, _) in enumerate(records):
        shape_to_idxs[s].append(i)

    pairs, labels = [], []
    for s_idx, s in enumerate(shapes):
        for i in shape_to_idxs[s][:8]:
            pairs.append((sen[i], img[i]))
            labels.append(s_idx)

    return Dataset(pairs, torch.tensor(labels), batch_size=batch_size)

train_dataset = build_supcon_dataset(train_records, train_sen, train_img)
val_dataset   = build_supcon_dataset(val_records, val_sen, val_img)
test_dataset  = build_supcon_dataset(test_records, test_sen, test_img)

# -------------------------
# Train
# -------------------------

# Register all quantum circuits (sentence + image)
all_circuits = (
    train_sen + train_img +
    val_sen   + val_img +
    test_sen  + test_img
)

model = InfoNCEModel.from_diagrams(
    all_circuits,
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

print(f"Final OOD test accuracy (amplitude): {np.mean(accs):.3f}")

