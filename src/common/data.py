import os
from typing import Dict, List, Tuple

# Color words to ignore in captions
COLORS_TO_IGNORE = {
    'blue', 'purple', 'cyan', 'brown',
    'red', 'yellow', 'gray', 'green'
}


def build_caption_image_pairs(split_root: str) -> List[Tuple[str, str]]:
    """
    Scans subdirectories of split_root.
    Each subfolder is named by the raw caption (e.g. 'blue sphere').

    Returns:
        List of (raw_caption, image_relative_path)
    """
    pairs = []
    for folder in os.listdir(split_root):
        folder_path = os.path.join(split_root, folder)
        if not os.path.isdir(folder_path):
            continue

        raw_caption = folder.replace('_', ' ')
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.join(folder, fname)
                pairs.append((raw_caption, rel_path))

    return pairs


def clean_caption(raw_caption: str) -> str:
    """
    Removes color words from a raw caption.
    """
    tokens = raw_caption.lower().split()
    shape_tokens = [t for t in tokens if t not in COLORS_TO_IGNORE]
    return ' '.join(shape_tokens)


def build_dataset_pairs(image_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Builds dataset pairs for single-object training.

    Expected directory structure:
        image_root/
            train/
            OOD_val/
            OOD_test/

    Returns:
        Dict mapping split name to list of:
        (raw_caption, cleaned_shape_caption, full_image_path)
    """
    splits = ['train', 'OOD_val', 'OOD_test']
    dataset_pairs: Dict[str, List[Tuple[str, str, str]]] = {}

    for split in splits:
        split_dir = os.path.join(image_root, split)
        raw_pairs = build_caption_image_pairs(split_dir)

        cleaned = []
        for raw_cap, rel_path in raw_pairs:
            shape_cap = clean_caption(raw_cap)
            full_path = os.path.join(split_dir, rel_path)
            cleaned.append((raw_cap, shape_cap, full_path))

        dataset_pairs[split] = cleaned

    return dataset_pairs

