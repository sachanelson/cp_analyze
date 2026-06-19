import os
import numpy as np
import pandas as pd
import tifffile as tf

#
# cp_measure.py - Per-label intensity measurement on 3-D image + mask pairs
#
# measure_labels(img_zyx, mask_zyx, metrics)
#   Measure per-label statistics for a single channel volume.
#   Returns a DataFrame with one row per label.
#
# measure_file(img_path, mask_path, channel_idx, metrics, output_folder,
#              img_base, mask_base)
#   Load files, run measurement, write results to output_folder/measure_<tag>.txt
#   Returns (output_path, DataFrame).
#

VALID_METRICS = ("mean", "max", "min", "std", "volume")


def measure_labels(img_zyx, mask_zyx, metrics=VALID_METRICS):
    """
    Measure per-label statistics of img_zyx within each label of mask_zyx.

    Parameters
    ----------
    img_zyx  : np.ndarray  (Z,Y,X) float or int image
    mask_zyx : np.ndarray  (Z,Y,X) int32 label mask (0 = background)
    metrics  : sequence of str  any subset of VALID_METRICS

    Returns
    -------
    pd.DataFrame  columns: label_id + requested metrics
    """
    for m in metrics:
        if m not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{m}'; valid: {VALID_METRICS}")

    label_ids = np.unique(mask_zyx)
    label_ids = label_ids[label_ids != 0]

    img_f = img_zyx.astype(np.float32)
    rows  = []
    for lab in label_ids:
        region = img_f[mask_zyx == lab]
        row = {"label_id": int(lab)}
        if "volume" in metrics:
            row["volume"] = int(region.size)
        if "mean" in metrics:
            row["mean"]   = float(region.mean()) if region.size else float("nan")
        if "max" in metrics:
            row["max"]    = float(region.max())  if region.size else float("nan")
        if "min" in metrics:
            row["min"]    = float(region.min())  if region.size else float("nan")
        if "std" in metrics:
            row["std"]    = float(region.std())  if region.size else float("nan")
        rows.append(row)

    cols = ["label_id"] + [m for m in ("volume","mean","max","min","std") if m in metrics]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def measure_file(img_path, mask_path, channel_idx,
                 metrics=VALID_METRICS, output_folder=".",
                 img_base=None, mask_base=None):
    """
    Load an image file and a mask file, measure per-label statistics,
    and write results to output_folder/measure_<img_base>_ch<n>_<mask_base>.txt

    Parameters
    ----------
    img_path     : str   Path to image TIFF (Z,C,Y,X) or (Z,Y,X)
    mask_path    : str   Path to mask TIFF  (Z,Y,X) int32
    channel_idx  : int   Channel index to measure (ignored for 3-D images)
    metrics      : sequence[str]
    output_folder: str   Root output folder
    img_base     : str   Override base name for image (default: stem of img_path)
    mask_base    : str   Override base name for mask  (default: stem of mask_path)

    Returns
    -------
    out_path : str          Full path to written .txt file
    df       : pd.DataFrame  Measurement table
    """
    if img_base  is None: img_base  = os.path.splitext(os.path.basename(img_path))[0]
    if mask_base is None: mask_base = os.path.splitext(os.path.basename(mask_path))[0]

    # Check for rescaled image first
    rescaled_dir = os.path.join(os.path.dirname(os.path.dirname(mask_path)), "Rescaled")
    rescaled_path = os.path.join(rescaled_dir, f"{img_base}_rescaled.tif")
    if os.path.exists(rescaled_path):
        img = tf.imread(rescaled_path)
        print(f"[cp_measure] Using rescaled image: {os.path.basename(rescaled_path)}")
    else:
        img = tf.imread(img_path)
    
    mask = tf.imread(mask_path).astype(np.int32)

    if img.ndim == 4:
        nc = img.shape[1]
        if channel_idx >= nc:
            raise ValueError(f"channel_idx={channel_idx} out of range for image with {nc} channels")
        img_zyx = img[:, channel_idx]
    elif img.ndim == 3:
        img_zyx = img
    else:
        raise ValueError(f"Unexpected image shape {img.shape}")

    if mask.ndim == 2:
        mask = mask[np.newaxis]

    if img_zyx.shape != mask.shape:
        raise ValueError(
            f"Image shape {img_zyx.shape} does not match mask shape {mask.shape}. "
            f"Ensure you're using matching rescaled/original files."
        )

    df = measure_labels(img_zyx, mask, metrics)
    df.insert(0, "mask_file",  mask_base)
    df.insert(0, "channel",    channel_idx)
    df.insert(0, "image_file", img_base)

    os.makedirs(output_folder, exist_ok=True)
    tag      = f"{img_base}_ch{channel_idx}_{mask_base}"
    out_path = os.path.join(output_folder, f"measure_{tag}.txt")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[cp_measure] {len(df)} labels → {out_path}")
    return out_path, df


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    out = "/Volumes/SG2/cellpose_work/testOutput"
    img_p  = os.path.join(out, "input_cellnuc2ch_z25-34.tif")
    mask_p = os.path.join(out, "test1_full3ch_masks.tif")
    if not (os.path.exists(img_p) and os.path.exists(mask_p)):
        print("Test files not found"); sys.exit(1)
    path, df = measure_file(img_p, mask_p, channel_idx=1,
                            metrics=("mean","max","min","std","volume"),
                            output_folder=out)
    print(df.head())
