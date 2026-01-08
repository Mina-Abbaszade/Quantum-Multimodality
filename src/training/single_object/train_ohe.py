import random
import numpy as np
import torch

from lambeq import Dataset, PytorchTrainer

from src.common.data import build_dataset_pairs
from src.common.losses import supcon_loss
from src.common.metrics import supcon_acc
from src.models.similarity_model import InfoNCEModel

from src.models.quantum.ohe_encoder import (
    to_dataframe,
    encode_image_split_onehot,
    build_ohe_sentence_ansatz,
    build_ohe_image_ansatz,
    build_ohe_sentence_circuits,
    build_ohe_image_circuits,
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

df_train = to_dataframe(dataset_pairs["train"])
df_val   = to_dataframe(dataset_pairs["OOD_val"])
df_test  = to_dataframe(dataset_pairs["OOD_test"])

# -------------------------
# Sentence circuits (OHE)
# -------------------------

sentence_ansatz = build_ohe_sentence_ansatz()

train_sen = build_ohe_sentence_circuits(
    df_train["shape"].tolist(),
    sentence_ansatz
)

val_sen = build_ohe_sentence_circuits(
    df_val["shape"].tolist(),
    sentence_ansatz
)

test_sen = build_ohe_sentence_circuits(
    df_test["shape"].tolist(),
    sentence_ansatz
)

# -------------------------
# Image circuits (OHE)
# -------------------------

# One-hot image features (4-dim symbolic)
train_feats = encode_image_split_onehot(df_train, pad_to=4)
val_feats   = encode_image_split_onehot(df_val,   pad_to=4)
test_feats  = encode_image_split_onehot(df_test,  pad_to=4)

# Image placeholder diagram
from lambeq.backend.grammar import Ty, Box
shape_type = Ty("shape")
clip_type  = Ty("clip_shape")
image_type = Ty("image")

image_diagram = Box("CLIP_SHAPE", dom=Ty(), cod=clip_type)

image_ansatz = build_ohe_image_ansatz()

train_img = build_ohe_image_circuits(image_diagram, image_ansatz, train_feats)
val_img   = build_ohe_image_circuits(image_diagram, image_ansatz, val_feats)
test_img  = build_ohe_image_circuits(image_diagram, image_ansatz, test_feats)

# -------------------------
# SupCon dataset builder
# -------------------------

def build_supcon_dataset(records, sen, img, batch_size=32):
    shapes = sorted(set(records["shape"]))
    shape_to_idxs = {s: [] for s in shapes}

    for i, s in enumerate(records["shape"]):
        shape_to_idxs[s].append(i)

    pairs, labels = [], []
    for s_idx, s in enumerate(shapes):
        for i in shape_to_idxs[s][:8]:
            pairs.append((sen[i], img[i]))
            labels.append(s_idx)

    return Dataset(pairs, torch.tensor(labels), batch_size=batch_size)


train_dataset = build_supcon_dataset(df_train, train_sen, train_img)
val_dataset   = build_supcon_dataset(df_val,   val_sen,   val_img)
test_dataset  = build_supcon_dataset(df_test,  test_sen,  test_img)

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

print(f"Final OOD test accuracy (OHE): {np.mean(accs):.3f}")

