#!/usr/bin/env python3
"""
compute_angle_pairs.py  --  regenerate predicted-vs-GT correction angles
                            WITHOUT retraining.

This loads your already-saved checkpoints and runs *inference only* (forward
passes, no optimisation, no 2000-epoch loop) to reproduce, for every limb
hemisphere, the pair:

        (ground-truth correction angle, predicted correction angle)

It writes two CSVs that compute_stats.py then turns into median / ICC /
Bland-Altman:

    angle_pairs_cv.csv    -- pooled over the 5 cross-validation folds
                             (each fold's checkpoint evaluated on that fold's
                              validation split, reconstructed with the same
                              SEED so it matches training exactly)
    angle_pairs_test.csv  -- the fixed 80/10/10 hold-out test split, scored
                             with the final global checkpoint

The dataset class, letterbox transform, hemisphere assignment and Miniaci /
Fujisawa geometry are reproduced verbatim from the training notebook so the
numbers are identical to what training produced -- only the throw-away of the
per-case errors is fixed (the notebook kept only mean/std/max).

Ground truth here is whatever is stored in the COCO annotation file. If that
file already contains the mean-of-three-observers landmarks, then the angle
errors below are measured against the mean observer, exactly as in
Przystalski et al. 2023.

Requirements: the CKD repo (for the model + extract_coordinates), the saved
checkpoints, and the annotation JSON. A GPU is used if available but inference
also runs on CPU.

Edit the CONFIG block, then:  python compute_angle_pairs.py
"""

import json
import math
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ============================================================ CONFIG -- EDIT ME
CKD_PATH = "../notebooks/CKD"  # path to the CKD repo (has models.py, utils.py)
DATA_DIR = "/tf/data/hto/xrays"
COCO_JSON_PATH = os.path.join(DATA_DIR, "hto_annotations.json")
if not os.path.exists(COCO_JSON_PATH):
    COCO_JSON_PATH = "hto_annotations.json"

CV_CHECKPOINTS = [f"/tf/notebooks/kfolds_models/best_model_fold{i}.pt" for i in range(1, 6)]  # fold 1..5
GLOBAL_CHECKPOINT = "/tf/notebooks/kfolds_models/best_model_global.pt"     # final fixed-split model

# These MUST match the values used during training, or the splits/geometry drift.
SEED = 42
TARGET_SIZE = 768
HEATMAP_SCALE = 0.5
SIGMA = 6.0
BATCH_SIZE = 4
MODEL_VARIANT = "small_p16"
N_SPLITS = 5

OUT_CV = "/tf/notebooks/angle_pairs_cv.csv"
OUT_TEST = "/tf/notebooks/angle_pairs_test.csv"
# =============================================================================

sys.path.append(os.path.abspath(CKD_PATH))
from models import (                                          # noqa: E402
    Conformer_tiny_patch16_keypoint_half_heatmap,
    Conformer_small_patch16_keypoint_half_heatmap,
    Conformer_small_patch32_keypoint_half_heatmap,
    Conformer_base_patch16_keypoint_half_heatmap,
)
from utils import extract_coordinates                          # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model_map = {
    "tiny": Conformer_tiny_patch16_keypoint_half_heatmap,
    "small_p16": Conformer_small_patch16_keypoint_half_heatmap,
    "small_p32": Conformer_small_patch32_keypoint_half_heatmap,
    "base": Conformer_base_patch16_keypoint_half_heatmap,
}


# ----------------------------------------------------------- preprocessing (verbatim)
def preprocess_global_image(img, target_size=512):
    """Letterbox-resize *img* to a square canvas of *target_size* pixels."""
    orig_w, orig_h = img.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    final_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    final_img.paste(resized, (pad_left, pad_top))
    return final_img, scale, (pad_left, pad_top)


def _rep_x(ann):
    """Representative x for hemisphere assignment (mean of visible kp x's)."""
    kps = ann.get("keypoints", [])
    xs = [kps[i] for i in range(0, len(kps), 3) if kps[i] > 0]
    if xs:
        return sum(xs) / len(xs)
    bbox = ann.get("bbox", [0, 0, 0, 0])
    return bbox[0] + bbox[2] / 2


# ----------------------------------------------------------- dataset (eval path, verbatim)
class GlobalRadiographKeypointDataset(Dataset):
    """COCO 12-keypoint dataset. Inference only -- no augmentation is applied
    because we never use split='train' here. Splitting is reproduced exactly
    so that `indices` (from KFold) select the same images as during training."""

    def __init__(self, coco_json_path, split="val", split_ratios=(0.8, 0.1, 0.1),
                 target_size=512, heatmap_scale=0.25, sigma=2.0, seed=42, indices=None):
        super().__init__()
        self.target_size = target_size
        self.heatmap_scale = heatmap_scale
        self.sigma = sigma
        self.num_keypoints = 12
        self.split = split

        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        images_info = {img["id"]: img for img in coco_data.get("images", [])}
        anns_by_img = {}
        for ann in coco_data.get("annotations", []):
            anns_by_img.setdefault(ann.get("image_id"), []).append(ann)

        valid_samples = []
        for img_id, anns in anns_by_img.items():
            if img_id not in images_info:
                continue
            img_info = images_info[img_id]
            img_w = img_info.get("width", 2860)

            by_cat = {}
            for ann in anns:
                by_cat.setdefault(ann.get("category_id"), []).append(ann)

            kps_flat = [-1.0, -1.0, 0] * 12
            has_kp = False
            for cat_id, cat_anns in by_cat.items():
                sorted_anns = sorted(cat_anns, key=_rep_x)
                if len(sorted_anns) == 2:
                    assignments = [(sorted_anns[0], 0), (sorted_anns[1], 6)]
                elif len(sorted_anns) == 1:
                    x = _rep_x(sorted_anns[0])
                    base = 0 if x < img_w / 2.0 else 6
                    assignments = [(sorted_anns[0], base)]
                else:
                    assignments = [(sorted_anns[0], 0), (sorted_anns[-1], 6)]

                for ann, base in assignments:
                    kps = ann.get("keypoints", [])
                    if cat_id == 1 and len(kps) >= 3:
                        kps_flat[base * 3:(base + 1) * 3] = [kps[0], kps[1], 2 if kps[0] > 0 else 0]
                        if kps[0] > 0:
                            has_kp = True
                    elif cat_id == 2 and len(kps) >= 9:
                        for k in range(3):
                            s = base + 1 + k
                            kps_flat[s * 3:(s + 1) * 3] = [kps[k * 3], kps[k * 3 + 1], kps[k * 3 + 2]]
                            if kps[k * 3 + 2] > 0:
                                has_kp = True
                    elif cat_id == 3 and len(kps) >= 6:
                        for k in range(2):
                            s = base + 4 + k
                            kps_flat[s * 3:(s + 1) * 3] = [kps[k * 3], kps[k * 3 + 1], kps[k * 3 + 2]]
                            if kps[k * 3 + 2] > 0:
                                has_kp = True

            if has_kp:
                filename = img_info.get("file_name")
                img_dir = os.path.dirname(coco_json_path) or "."
                if not os.path.exists(os.path.join(img_dir, filename)):
                    alt = os.path.join("/tf/data/hto/xrays", os.path.basename(filename))
                    if os.path.exists(alt):
                        img_dir = "/tf/data/hto/xrays"
                valid_samples.append({
                    "img_path": os.path.join(img_dir, filename),
                    "orig_size": (img_w, img_info.get("height", 8000)),
                    "keypoints": kps_flat,
                })

        valid_samples.sort(key=lambda x: x["img_path"])
        random.seed(seed)
        random.shuffle(valid_samples)

        if indices is not None:
            self.samples = [valid_samples[i] for i in indices]
        else:
            n = len(valid_samples)
            train_end = int(n * split_ratios[0])
            val_end = train_end + int(n * split_ratios[1])
            self.samples = {
                "train": valid_samples[:train_end],
                "val": valid_samples[train_end:val_end],
                "test": valid_samples[val_end:],
            }.get(split, valid_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            img = Image.open(sample["img_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", sample["orig_size"], color=(128, 128, 128))

        processed_img, scale, padding = preprocess_global_image(img, self.target_size)

        final_kps = []
        for i in range(self.num_keypoints):
            kp_x = sample["keypoints"][i * 3]
            kp_y = sample["keypoints"][i * 3 + 1]
            kp_v = sample["keypoints"][i * 3 + 2]
            if kp_v > 0 and kp_x >= 0 and kp_y >= 0:
                final_kps.append([kp_x * scale + padding[0], kp_y * scale + padding[1]])
            else:
                final_kps.append([-1.0, -1.0])

        img_tensor = torch.from_numpy(np.array(processed_img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) \
            / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        return {
            "image": img_tensor,
            "keypoint": torch.tensor(final_kps, dtype=torch.float32),
            "img_path": sample["img_path"],
            "orig_size": torch.tensor(sample["orig_size"]),
        }


# ----------------------------------------------------------- geometry (verbatim)
def map_global_to_orig(kp_final, orig_size, target_size=512):
    orig_w, orig_h = orig_size
    scale = min(target_size / orig_w, target_size / orig_h)
    pad_left = (target_size - int(orig_w * scale)) // 2
    pad_top = (target_size - int(orig_h * scale)) // 2
    return np.array([(kp_final[0] - pad_left) / scale, (kp_final[1] - pad_top) / scale])


def calculate_intersection(p1, p2, target_y):
    if p2[0] == p1[0]:
        return p1[0]
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    if m == 0:
        return p1[0] if abs(target_y - p1[1]) < 1e-9 else float("nan")
    return (target_y - p1[1]) / m + p1[0]


def evaluate_side_geometry(points):
    """Miniaci correction angle alpha for one leg hemisphere."""
    ankle_c = (points["ankle_inner"] + points["ankle_outer"]) / 2.0
    fujisawa = points["knee_inner"] + 0.625 * (points["knee_outer"] - points["knee_inner"])
    tx = calculate_intersection(points["femur_head"], fujisawa, ankle_c[1])
    target_at_ankle = np.array([tx, ankle_c[1]])
    v_orig = ankle_c - points["ost_point"]
    v_target = target_at_ankle - points["ost_point"]
    raw = abs(math.atan2(v_orig[1], v_orig[0]) - math.atan2(v_target[1], v_target[0]))
    alpha = min(raw, 2 * math.pi - raw) * 180.0 / math.pi
    return alpha


_SIDE_KEYS = ["femur_head", "knee_inner", "ost_point",
              "knee_outer", "ankle_inner", "ankle_outer"]


# ----------------------------------------------------------- inference -> pairs
@torch.no_grad()
def collect_pairs(model, loader, dev):
    """Return list of (gt_angle, pred_angle, img_path, side) for valid hemispheres."""
    model.eval()
    pairs = []
    for batch in loader:
        pred_hms = torch.sigmoid(model(batch["image"].to(dev)))
        coords = extract_coordinates(pred_hms.cpu(), scale_factor=1.0 / HEATMAP_SCALE).numpy()
        gts = batch["keypoint"].numpy()
        sizes = batch["orig_size"].numpy()
        paths = batch["img_path"]
        for b in range(len(coords)):
            for base, side in ((0, "left"), (6, "right")):
                pg_, pp_ = {}, {}
                for k_off, name in enumerate(_SIDE_KEYS):
                    s = base + k_off
                    if gts[b][s][0] >= 0:
                        pg_[name] = map_global_to_orig(gts[b][s], sizes[b], TARGET_SIZE)
                        pp_[name] = map_global_to_orig(coords[b][s], sizes[b], TARGET_SIZE)
                if all(k in pg_ for k in _SIDE_KEYS) and all(k in pp_ for k in _SIDE_KEYS):
                    ga = evaluate_side_geometry(pg_)
                    pa = evaluate_side_geometry(pp_)
                    pairs.append((ga, pa, os.path.basename(paths[b]), side))
    return pairs


def load_model(checkpoint_path):
    model = _model_map[MODEL_VARIANT](num_keypoints=12).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def make_loader(indices=None, split="val"):
    ds = GlobalRadiographKeypointDataset(
        COCO_JSON_PATH, split=split, target_size=TARGET_SIZE,
        heatmap_scale=HEATMAP_SCALE, sigma=SIGMA, seed=SEED, indices=indices)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False), len(ds)


def main():
    import pandas as pd
    from sklearn.model_selection import KFold

    if not os.path.exists(COCO_JSON_PATH):
        sys.exit(f"Annotation file not found: {COCO_JSON_PATH}")

    print(f"Device: {device}")

    # -------- cross-validation pairs (matches training folds via identical SEED) ----
    n_total = GlobalRadiographKeypointDataset(
        COCO_JSON_PATH, split="all", target_size=TARGET_SIZE,
        heatmap_scale=HEATMAP_SCALE, sigma=SIGMA, seed=SEED).__len__()
    print(f"Total samples: {n_total}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_rows = []
    for fold, (_, val_idx) in enumerate(kf.split(np.arange(n_total))):
        ckpt = CV_CHECKPOINTS[fold]
        if not os.path.exists(ckpt):
            print(f"  [fold {fold + 1}] checkpoint missing: {ckpt} -- skipped")
            continue
        loader, n_val = make_loader(indices=val_idx, split="val")
        model = load_model(ckpt)
        pairs = collect_pairs(model, loader, device)
        for ga, pa, fname, side in pairs:
            cv_rows.append({"fold": fold + 1, "image": fname, "side": side,
                            "gt_angle": ga, "pred_angle": pa})
        print(f"  [fold {fold + 1}] {n_val} images -> {len(pairs)} hemispheres "
              f"({ckpt})")

    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(OUT_CV, index=False)
        print(f"Wrote {OUT_CV}  ({len(cv_rows)} hemispheres)")

    # -------- fixed-split test pairs (global checkpoint) ----------------------------
    if os.path.exists(GLOBAL_CHECKPOINT):
        loader, n_test = make_loader(indices=None, split="test")
        model = load_model(GLOBAL_CHECKPOINT)
        pairs = collect_pairs(model, loader, device)
        test_rows = [{"image": f, "side": s, "gt_angle": ga, "pred_angle": pa}
                     for ga, pa, f, s in pairs]
        pd.DataFrame(test_rows).to_csv(OUT_TEST, index=False)
        print(f"Wrote {OUT_TEST}  ({len(test_rows)} hemispheres from {n_test} test images)")
    else:
        print(f"Global checkpoint missing: {GLOBAL_CHECKPOINT} -- test CSV skipped")

    print("\nDone. Now run:  python compute_stats.py "
          f"{OUT_CV} {OUT_TEST}")


if __name__ == "__main__":
    main()
