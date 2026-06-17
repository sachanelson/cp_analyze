import os
import numpy as np
import tifffile as tf
from scipy.ndimage import gaussian_filter, median_filter
from cellpose import models
from cellpose.io import logger_setup

#
# cp_segment.py - Cellpose segmentation module callable from cp_manager
#
# segment(img, channel_spec, base_name, output_folder,
#         filter_type, filter_size, z_range)
#
# Parameters
# ----------
# img           : np.ndarray  Multichannel 3-D array, shape (Z, C, Y, X)
#                             Assumed channel layout:
#                               3-ch: ch0=C (cytoplasm), ch1=N (nucleus), ch2=S1 (signal)
#                               4-ch: ch0=C, ch1=N, ch2=S1, ch3=S2 (second signal)
# channel_spec  : str         "C"  = cytoplasm only       → sends ch0 alone
#                             "CN" = cytoplasm + nucleus   → sends ch0+ch1
#                             "S1" = signal channel 1      → sends ch2 alone
#                             "S2" = signal channel 2      → sends ch3 alone (4-ch only)
# base_name     : str         Base filename for output (no extension)
# output_folder : str         Root output folder; masks written to output_folder/Masks/
# filter_type   : str         "None", "Median", or "Gaussian"
# filter_size   : tuple(x,y,z) Kernel/sigma size (x, y, z order).
# z_range       : tuple(int,int)  (z_start, z_end) inclusive, 0-based.
#
# Returns
# -------
# mask_path : str   Full path to the saved mask TIFF
# n_masks   : int   Number of unique masks (excluding background)
#

STITCH_THR = 0.5
MIN_SIZE   = 200

VALID_SPECS = ("C", "CN", "S1", "S2")

# Human-readable description of each spec
_SPEC_DESC = {
    "C":  "cytoplasm only (ch0)",
    "CN": "cytoplasm (ch0) + nucleus (ch1)",
    "S1": "signal ch1 (ch2 of input)",
    "S2": "signal ch2 (ch3 of input)",
}

_model = None   # lazy-loaded singleton


def _get_model():
    global _model
    if _model is None:
        logger_setup()
        _model = models.CellposeModel(gpu=True)
    return _model


def _apply_filter(vol_zyx, filter_type, filter_size):
    """Apply spatial filter to a (Z,Y,X) uint16 volume. Returns uint16."""
    fz, fy, fx = filter_size[2], filter_size[1], filter_size[0]   # (z,y,x)
    arr = vol_zyx.astype(np.float32)
    if filter_type == "Median":
        arr = median_filter(arr, size=(fz, fy, fx))
    elif filter_type == "Gaussian":
        arr = gaussian_filter(arr, sigma=(fz, fy, fx), truncate=3.5)
    return np.clip(arr, 0, 65535).astype(np.uint16)


def segment(img, channel_spec, base_name, output_folder,
            filter_type="None", filter_size=(2, 2, 2),
            z_range=None):
    """
    Run Cellpose segmentation on a slice of a multichannel 3-D image.

    Channel layout assumed (Z, C, Y, X):
      3-ch: ch0=C, ch1=N, ch2=S1
      4-ch: ch0=C, ch1=N, ch2=S1, ch3=S2
    """
    channel_spec = channel_spec.upper()
    if channel_spec == "S":   # alias
        channel_spec = "S1"
    if channel_spec not in VALID_SPECS:
        raise ValueError(f"channel_spec must be one of {VALID_SPECS}; got '{channel_spec}'")
    if filter_type not in ("None", "Median", "Gaussian"):
        raise ValueError(f"filter_type must be 'None', 'Median', or 'Gaussian'; got '{filter_type}'")

    if img.ndim != 4:
        raise ValueError(f"img must be 4-D (Z, C, Y, X); got shape {img.shape}")

    n_ch = img.shape[1]

    # Guard: S2 requires at least 4 channels
    if channel_spec == "S2" and n_ch < 4:
        raise ValueError(
            f"channel_spec='S2' requires a 4-channel image (ch3=S2), "
            f"but image has only {n_ch} channel(s).")

    # ----------------------------------------------------------------
    # 1. Extract Z range
    # ----------------------------------------------------------------
    nz_total = img.shape[0]
    if z_range is None:
        z_start, z_end = 0, nz_total - 1
    else:
        z_start, z_end = int(z_range[0]), int(z_range[1])
    z_slice = slice(z_start, z_end + 1)

    # ----------------------------------------------------------------
    # 2. Extract required channels
    #    Layout: ch0=C, ch1=N, ch2=S1, ch3=S2
    # ----------------------------------------------------------------
    if channel_spec == "CN":
        cell_vol = img[z_slice, 0]   # C
        nuc_vol  = img[z_slice, 1]   # N
        ch_desc  = "ch0=C + ch1=N"
    elif channel_spec == "C":
        cell_vol = img[z_slice, 0]   # C only
        nuc_vol  = None
        ch_desc  = "ch0=C"
    elif channel_spec == "S1":
        cell_vol = img[z_slice, 2]   # S1
        nuc_vol  = None
        ch_desc  = "ch2=S1"
    elif channel_spec == "S2":
        cell_vol = img[z_slice, 3]   # S2
        nuc_vol  = None
        ch_desc  = "ch3=S2"

    # ----------------------------------------------------------------
    # 3. Apply filter to signal channel only
    # ----------------------------------------------------------------
    if filter_type != "None":
        cell_vol = _apply_filter(cell_vol, filter_type, filter_size)

    # ----------------------------------------------------------------
    # 4. Build input array for Cellpose
    #    For CN: send 2-ch stack so cellpose can use nuclear guidance.
    #    For all others: send the single extracted channel only.
    # ----------------------------------------------------------------
    if nuc_vol is not None:
        cp_input     = np.stack([cell_vol, nuc_vol], axis=1)  # (Z, 2, Y, X)
        channel_axis = 1
    else:
        cp_input     = cell_vol   # (Z, Y, X) — single channel
        channel_axis = None

    # ----------------------------------------------------------------
    # 5. Diagnostics
    # ----------------------------------------------------------------
    print(f"[cp_segment] Segmenting '{base_name}'")
    print(f"  spec      : {channel_spec}  ({_SPEC_DESC[channel_spec]})")
    print(f"  user spec : {ch_desc}")
    print(f"  filter    : {filter_type}")
    print(f"  z range   : {z_start}–{z_end}")
    print(f"  img shape : {img.shape}  (n_ch={n_ch})")
    print(f"  cp input  : shape={cp_input.shape}, dtype={cp_input.dtype}, "
          f"channel_axis={channel_axis}")

    # ----------------------------------------------------------------
    # 6. Run Cellpose
    # ----------------------------------------------------------------
    model = _get_model()
    kwargs = dict(
        z_axis=0,
        do_3D=False,
        stitch_threshold=STITCH_THR,
        min_size=MIN_SIZE,
    )
    if channel_axis is not None:
        kwargs["channel_axis"] = channel_axis

    masks, _, _ = model.eval(cp_input, **kwargs)
    masks = masks.astype(np.int32)
    n_masks = int(len(np.unique(masks)) - 1)
    print(f"[cp_segment] Done — {n_masks} masks found")

    # ----------------------------------------------------------------
    # 6. Save mask file to output_folder/Masks/
    # ----------------------------------------------------------------
    masks_dir = os.path.join(output_folder, "Masks")
    os.makedirs(masks_dir, exist_ok=True)

    tag = f"{channel_spec}_z{z_start}-{z_end}"
    if filter_type != "None":
        tag += f"_{filter_type.lower()}"
    mask_name = f"{base_name}_{tag}_masks.tif"
    mask_path = os.path.join(masks_dir, mask_name)

    tf.imwrite(mask_path, masks, metadata={"axes": "ZYX"})
    print(f"[cp_segment] Saved: {mask_path}")

    return mask_path, n_masks


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    test_file = "/Volumes/SG2/cellpose_work/B1FL.EMCR.YFPH.Exp9.TT4_YFP_PS6_Nuen_2_CellNuc.tif"
    out_dir   = "/Volumes/SG2/cellpose_work/testOutput"

    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    img = tf.imread(test_file)   # (62, 2, 1432, 2355)
    print(f"Loaded: {img.shape}, {img.dtype}")

    path, n = segment(
        img,
        channel_spec  = "CN",
        base_name     = "selftest",
        output_folder = out_dir,
        filter_type   = "Median",
        filter_size   = (2, 2, 2),
        z_range       = (25, 34),
    )
    print(f"Result: {path}  ({n} masks)")
