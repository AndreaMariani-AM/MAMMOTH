"""
Tutorial: Mammoth model instantiation, weight shapes, single-expert heatmap, and top-k patches.

This script demonstrates in order:
  1. How to instantiate the Mammoth model and load weights from a MIL checkpoint.
  2. How dispatch weights are shaped (batch, patches, experts, heads, slots).
  3. How to generate a heatmap for a single expert given an expert index.
  4. How to extract the top 10 patches per expert and save them into a single folder.

Paths: Uses config from examples/config/paths.py (TCGA_EXAMPLE_H5, TCGA_EXAMPLE_SVS, etc.).
      Override with --ckpt, --h5, --wsi, --out-dir.

Run from repo root: python examples/tutorial_mammoth_visualization.py [options]
"""

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_examples_dir = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import h5py
except ImportError:
    h5py = None
try:
    import openslide
except ImportError:
    openslide = None

from src.mammoth import Mammoth

# Centralized paths (config/paths.py under examples)
try:
    from config.paths import TCGA_EXAMPLE_H5, TCGA_EXAMPLE_SVS
except Exception:
    TCGA_EXAMPLE_H5 = ""
    TCGA_EXAMPLE_SVS = ""

H5_FEAT_KEY = "feats"
COORDS_KEY = "coords"
PATCH_SIZE_AT_EXTRACTION = 256
FEATURE_EXTRACTION_MAG = 20
LEVEL0_MAG_FALLBACK = 40
MAX_THUMBNAIL_SIDE = 4000
NUM_EXPERTS = 30
NUM_SLOTS = 10
HEATMAP_ALPHA = 0.3


# -----------------------------------------------------------------------------
# 1. Instantiate Mammoth and load weights
# -----------------------------------------------------------------------------


def load_mammoth_state_dict(ckpt_path, device="cpu"):
    """
    Load checkpoint and return state_dict containing only the Mammoth submodule.
    Full MIL checkpoints often store the model under 'model' with prefix 'mlp.router.mammoth.'
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    model_sd = ckpt.get("model", ckpt)
    prefix = "mlp.router.mammoth."
    stripped = {
        k[len(prefix) :]: v for k, v in model_sd.items() if k.startswith(prefix)
    }
    if not stripped:
        raise ValueError(
            f"No keys with prefix {prefix!r} in checkpoint. Keys: {list(model_sd.keys())[:10]}..."
        )
    return stripped


def build_mammoth(ckpt_path, device="cpu"):
    """
    Build Mammoth with architecture matching the lung_tp53_abmil training setup,
    load weights from checkpoint, and return the model in eval mode.
    """
    mammoth = Mammoth(
        input_dim=1024,
        dim=512,
        num_experts=NUM_EXPERTS,
        num_slots=NUM_SLOTS,
        num_heads=16,
        lora_rank=16,
        auto_rank=False,
        slot_dim=256,
        share_lora_weights=True,
        keep_slots=True,
        dropout=0.0,
    )
    sd = load_mammoth_state_dict(ckpt_path, device)
    mammoth.load_state_dict(sd, strict=True)
    mammoth.to(device)
    mammoth.eval()
    return mammoth


def print_weight_shapes(state_dict):
    """Print shape of each tensor in the Mammoth state dict (for tutorial clarity)."""
    print("Mammoth state dict weight shapes:")
    for k, v in sorted(state_dict.items()):
        print(f"  {k}: {tuple(v.shape)}")


# -----------------------------------------------------------------------------
# 2. Load features, run forward, get dispatch weights and their shape
# -----------------------------------------------------------------------------


def load_h5_feats_and_coords(h5_path, feat_key=None):
    """Load patch features (N, D) and coords (N, 2) from H5. Coords in level-0 pixels."""
    if h5py is None:
        raise ImportError("h5py is required")
    feat_key = feat_key or H5_FEAT_KEY
    with h5py.File(h5_path, "r") as f:
        if feat_key not in f and feat_key == "feats":
            feat_key = "features"
        if feat_key not in f:
            raise KeyError(
                f"Feature key {feat_key!r} or 'features' not in {list(f.keys())}"
            )
        feats = np.array(f[feat_key], dtype=np.float32)
        if COORDS_KEY not in f:
            raise KeyError(f"Coords key {COORDS_KEY!r} not in {list(f.keys())}")
        coords = np.array(f[COORDS_KEY], dtype=np.float64)
    if feats.ndim == 3 and feats.shape[0] == 1:
        feats = feats.squeeze(0)
    if coords.ndim == 3 and coords.shape[0] == 1:
        coords = coords.squeeze(0)
    if coords.shape[1] >= 2:
        coords = coords[:, :2]
    return feats, coords


def compute_dispatch_weights(mammoth, feats, device="cpu", normalize=True):
    """
    Run Mammoth forward with return_weights=True.
    Returns:
      dispatch_weights: (1, N, E, H, S) then normalized over (E,H,S) per patch.
      scores: (N, E) per-patch per-expert scores (mean over heads and slots).
    """
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    with torch.no_grad():
        _, dispatch_weights = mammoth(x, return_weights=True)
    # dispatch_weights: (batch, seq, num_experts, num_heads, num_slots)
    if normalize:
        dispatch_weights = dispatch_weights / dispatch_weights.sum(
            dim=(2, 3, 4), keepdim=True
        )
    w = dispatch_weights[0]  # (N, E, H, S)
    scores = w.mean(dim=(2, 3))  # (N, E)
    return dispatch_weights.cpu().numpy(), scores.cpu().numpy()


# -----------------------------------------------------------------------------
# 3. Single-expert heatmap
# -----------------------------------------------------------------------------


def percentile_scores(scores_e):
    """Map per-patch scores for one expert to [0,1] percentile."""
    n = scores_e.size
    ranks = np.argsort(np.argsort(scores_e, axis=0), axis=0).astype(np.float64)
    return (ranks / (n - 1)) if n > 1 else np.zeros_like(scores_e, dtype=np.float64)


def get_wsi_thumbnail(svs_path, max_side=MAX_THUMBNAIL_SIDE):
    """Return RGB thumbnail (H,W,3), scale factor, level0 mag and dimensions."""
    if openslide is None:
        raise ImportError("openslide is required to read WSI")
    slide = openslide.OpenSlide(str(svs_path))
    level0_w, level0_h = slide.dimensions
    for key in ("openslide.objective-power", "aperio.AppMag"):
        try:
            mag = float(slide.properties.get(key, 0) or 0)
            if mag > 0:
                level0_mag = mag
                break
        except (TypeError, ValueError):
            continue
    else:
        level0_mag = float(LEVEL0_MAG_FALLBACK)
    scale = min(1.0, max_side / max(level0_w, level0_h))
    new_w, new_h = int(round(level0_w * scale)), int(round(level0_h * scale))
    thumb = slide.get_thumbnail((new_w, new_h))
    thumb = np.array(thumb.convert("RGB"))
    slide.close()
    actual_h, actual_w = thumb.shape[:2]
    scale = actual_w / level0_w
    return thumb, scale, level0_mag, level0_w, level0_h


def patch_size_at_level0(level0_mag):
    """Patch side length in level-0 pixels (features at 20x, 256 px)."""
    if level0_mag is None or level0_mag <= 0 or level0_mag <= FEATURE_EXTRACTION_MAG:
        level0_mag = LEVEL0_MAG_FALLBACK
    return PATCH_SIZE_AT_EXTRACTION * (level0_mag / FEATURE_EXTRACTION_MAG)


def build_overlay_rgba(
    coords_level0, percentile_e, scale, thumb_w, thumb_h, patch_size_level0
):
    """Build RGBA overlay (thumb_h, thumb_w, 4) for one expert using turbo colormap."""
    try:
        cmap = matplotlib.colormaps["turbo"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("turbo")
    patch_thumb = max(1, int(round(patch_size_level0 * scale)))
    px = np.round(coords_level0[:, 0] * scale).astype(np.int32)
    py = np.round(coords_level0[:, 1] * scale).astype(np.int32)
    pct = np.clip(percentile_e.astype(np.float64), 0.0, 1.0)
    M = px.size
    if M == 0:
        return np.zeros((thumb_h, thumb_w, 4), dtype=np.float32)
    r_grid = np.arange(patch_thumb, dtype=np.int32)
    rr = (py[:, None, None] + r_grid[None, :, None]) + np.zeros(
        (1, 1, patch_thumb), dtype=np.int32
    )
    cc = (px[:, None, None] + r_grid[None, None, :]) + np.zeros(
        (1, patch_thumb, 1), dtype=np.int32
    )
    valid_dst = (rr >= 0) & (rr < thumb_h) & (cc >= 0) & (cc < thumb_w)
    valid_flat = valid_dst.ravel()
    row_flat = rr.ravel()[valid_flat]
    col_flat = cc.ravel()[valid_flat]
    pct_flat = np.repeat(pct, patch_thumb * patch_thumb)[valid_flat]
    n_pixels = thumb_h * thumb_w
    sum_pct = np.zeros(n_pixels, dtype=np.float64)
    count_pct = np.zeros(n_pixels, dtype=np.float64)
    idx_flat = row_flat * thumb_w + col_flat
    np.add.at(sum_pct, idx_flat, pct_flat)
    np.add.at(count_pct, idx_flat, 1.0)
    count_pct = np.maximum(count_pct, 1.0)
    avg_pct_2d = (sum_pct / count_pct).reshape(thumb_h, thumb_w)
    valid = (count_pct > 0).reshape(thumb_h, thumb_w)
    overlay = np.zeros((thumb_h, thumb_w, 4), dtype=np.float32)
    overlay[valid] = np.array(cmap(avg_pct_2d[valid]), dtype=np.float32)
    overlay[valid, 3] = HEATMAP_ALPHA
    return overlay


def blend_overlay(thumb, overlay):
    """Composite RGBA overlay onto RGB thumbnail. Returns (H,W,3) float [0,1]."""
    alpha = np.asarray(overlay[:, :, 3], dtype=np.float32)[:, :, np.newaxis]
    rgb_overlay = overlay[:, :, :3]
    thumb_f = np.asarray(thumb, dtype=np.float32) / 255.0
    return np.clip(alpha * rgb_overlay + (1.0 - alpha) * thumb_f, 0.0, 1.0)


def save_single_expert_heatmap(
    thumb, coords_level0, scores_e, scale, patch_size_level0, out_path, expert_id
):
    """
    Generate and save a heatmap for a single expert.
    scores_e: (N,) per-patch scores for that expert (e.g. from heatmap_scores[:, expert_id]).
    """
    thumb_h, thumb_w = thumb.shape[:2]
    pct_e = percentile_scores(scores_e)
    overlay = build_overlay_rgba(
        coords_level0, pct_e, scale, thumb_w, thumb_h, patch_size_level0
    )
    blended = blend_overlay(thumb, overlay)
    fig, ax = plt.subplots(1, 1, figsize=(thumb_w / 150, thumb_h / 150))
    ax.imshow(blended)
    ax.set_axis_off()
    ax.text(
        thumb_w * 0.02,
        thumb_h * 0.98,
        str(expert_id),
        transform=ax.transData,
        ha="left",
        va="top",
        fontsize=10,
        color="white",
        weight="bold",
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# 4. Top-k patches per expert into a single folder
# -----------------------------------------------------------------------------


def topk_patch_indices(scores_e, k=10):
    """Return patch indices with highest score for this expert (descending)."""
    n = scores_e.size
    k = min(k, n)
    return np.argsort(scores_e)[::-1][:k]


def crop_patch_at_index(thumb, coords_level0, scale, patch_size_level0, patch_idx):
    """Crop thumbnail at one patch location. Returns (patch_thumb, patch_thumb, 3)."""
    thumb_h, thumb_w = thumb.shape[:2]
    patch_thumb = max(1, int(round(patch_size_level0 * scale)))
    crop = np.zeros((patch_thumb, patch_thumb, 3), dtype=thumb.dtype)
    x0, y0 = coords_level0[patch_idx, 0], coords_level0[patch_idx, 1]
    px, py = int(round(x0 * scale)), int(round(y0 * scale))
    src_y0, src_x0 = max(0, py), max(0, px)
    src_y1, src_x1 = min(thumb_h, py + patch_thumb), min(thumb_w, px + patch_thumb)
    crop_y0, crop_x0 = src_y0 - py, src_x0 - px
    dy, dx = src_y1 - src_y0, src_x1 - src_x0
    if dy > 0 and dx > 0:
        crop[crop_y0 : crop_y0 + dy, crop_x0 : crop_x0 + dx] = thumb[
            src_y0:src_y1, src_x0:src_x1
        ]
    return crop


def save_topk_patches_per_expert_to_folder(
    thumb, coords_level0, scale, patch_size_level0, heatmap_scores, k, out_dir
):
    """
    Get top k patches per expert and save each as a separate image in out_dir.
    heatmap_scores: (N, E). Saves expert_{e:02d}_rank_{r:02d}.png (e=expert index, r=rank 0..k-1).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in range(heatmap_scores.shape[1]):
        indices = topk_patch_indices(heatmap_scores[:, e], k)
        for rank, patch_idx in enumerate(indices):
            crop = crop_patch_at_index(
                thumb, coords_level0, scale, patch_size_level0, patch_idx
            )
            name = f"expert_{e:02d}_rank_{rank:02d}.png"
            plt.imsave(out_dir / name, crop)
    print(
        f"Saved {heatmap_scores.shape[1] * min(k, heatmap_scores.shape[0])} patches to {out_dir}"
    )


# -----------------------------------------------------------------------------
# Main: run all steps
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Tutorial: Mammoth viz — model, weights, heatmap, top-k patches"
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Mammoth checkpoint (full MIL ckpt with mlp.router.mammoth.*)",
    )
    ap.add_argument("--h5", type=str, default=TCGA_EXAMPLE_H5, help="H5 features path")
    ap.add_argument(
        "--wsi",
        type=str,
        default=TCGA_EXAMPLE_SVS,
        help="WSI path (for thumbnail/heatmap/patches)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (default: tutorial_out under repo)",
    )
    ap.add_argument(
        "--expert",
        type=int,
        default=0,
        metavar="E",
        help="Expert index for single-expert heatmap (0..29)",
    )
    ap.add_argument(
        "--topk", type=int, default=10, help="Number of top patches per expert to save"
    )
    ap.add_argument("--device", type=str, default="cpu", help="Device for model")
    ap.add_argument(
        "--no-heatmap", action="store_true", help="Skip single-expert heatmap"
    )
    ap.add_argument(
        "--no-topk", action="store_true", help="Skip saving top-k patches to folder"
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir or _repo_root / "tutorial_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.ckpt:
        raise SystemExit(
            "Provide --ckpt path to a MIL checkpoint containing Mammoth weights (e.g. mlp.router.mammoth.*)."
        )
    if not args.h5 or not Path(args.h5).exists():
        raise SystemExit(f"H5 path not set or missing: {args.h5}")
    if (not args.no_heatmap or not args.no_topk) and (
        not args.wsi or not Path(args.wsi).exists()
    ):
        raise SystemExit(
            f"WSI path required for heatmap/topk and must exist: {args.wsi}"
        )

    device = args.device

    # --- 1. Instantiate Mammoth and show weight shapes ---
    print("Step 1: Building Mammoth and loading weights...")
    mammoth = build_mammoth(args.ckpt, device)
    sd = load_mammoth_state_dict(args.ckpt, device)
    print_weight_shapes(sd)

    # --- 2. Load features, run forward, show dispatch weight shape ---
    print("\nStep 2: Loading H5 features and computing dispatch weights...")
    feats, coords = load_h5_feats_and_coords(args.h5)
    print(f"  feats shape: {feats.shape}, coords shape: {coords.shape}")
    dispatch_weights, scores = compute_dispatch_weights(mammoth, feats, device)
    print(f"  dispatch_weights shape (batch, N, E, H, S): {dispatch_weights.shape}")
    print(f"  scores (N, E) shape: {scores.shape}")

    # Percentile scores for visualization (same scale across experts)
    heatmap_scores = np.zeros_like(scores, dtype=np.float64)
    for e in range(scores.shape[1]):
        heatmap_scores[:, e] = percentile_scores(scores[:, e])

    # Load WSI once if we need heatmap and/or top-k patches
    thumb = scale = patch_size_level0 = None
    if (
        (not args.no_heatmap or not args.no_topk)
        and args.wsi
        and Path(args.wsi).exists()
    ):
        thumb, scale, level0_mag, level0_w, level0_h = get_wsi_thumbnail(args.wsi)
        if np.nanmax(coords) <= 1.1 and np.nanmin(coords) >= -0.01:
            coords = coords * np.array([level0_w, level0_h], dtype=np.float64)
        patch_size_level0 = patch_size_at_level0(level0_mag)

    # --- 3. Single-expert heatmap ---
    if not args.no_heatmap and thumb is not None:
        print(f"\nStep 3: Generating heatmap for expert index {args.expert}...")
        heatmap_path = out_dir / f"heatmap_expert_{args.expert:02d}.png"
        save_single_expert_heatmap(
            thumb,
            coords,
            scores[:, args.expert],
            scale,
            patch_size_level0,
            heatmap_path,
            args.expert,
        )
        print(f"  Saved {heatmap_path}")

    # --- 4. Top-k patches per expert into one folder ---
    if not args.no_topk and thumb is not None:
        print(
            f"\nStep 4: Saving top {args.topk} patches per expert to {out_dir / 'topk_patches'}..."
        )
        save_topk_patches_per_expert_to_folder(
            thumb,
            coords,
            scale,
            patch_size_level0,
            heatmap_scores,
            args.topk,
            out_dir / "topk_patches",
        )
    elif not args.no_topk:
        print("\nStep 4: Skipped (no WSI path).")

    print("\nDone.")


if __name__ == "__main__":
    main()
