import torch
import numpy as np
from PIL import Image
import clip


# --------------------------------------------------
# CLIP image encoder (classical)
# --------------------------------------------------

def load_clip(device=None):
    """
    Load CLIP ViT-B/32 model and preprocessing.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, device


def encode_image_split(
    df,
    model,
    preprocess,
    device,
):
    """
    Encode images into normalized CLIP feature vectors (512-dim).

    Returns:
        torch.Tensor of shape (N, 512)
    """
    feats = []

    for _, row in df.iterrows():
        img = Image.open(row["image_file_path"]).convert("RGB")
        img_input = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feat = model.encode_image(img_input)
            feat = feat / feat.norm(dim=-1, keepdim=True)

        feats.append(feat.cpu().numpy().flatten())

    return torch.tensor(np.vstack(feats), dtype=torch.float32)

