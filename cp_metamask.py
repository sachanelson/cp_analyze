import os
import warnings
import numpy as np
import tifffile as tf
from skimage.draw import polygon as sk_polygon

#
# cp_metamask.py - Set operations on Cellpose mask files
#
# meta_mask(mask_files, output_folder, operation,
#           threshold=0.5, polygon=None, polygon_mode="In")
#
# Parameters
# ----------
# mask_files    : list[str]   1 or 2 paths to int32 mask TIFFs (Z,Y,X).
#                             1-file ops: "Superset" (no-op passthrough).
#                             2-file ops: all operations.
# output_folder : str         Root output folder; results written to
#                             output_folder/Masks/Metamask/
# operation     : str         "Intersect" | "Parse" | "Subsets" | "Superset"
# threshold     : float|int   Overlap criterion.
#                             0 < t < 1  → fractional overlap of smaller mask
#                             t >= 1     → absolute voxel count
# polygon       : array-like  Optional. Shape (N,2) of (y,x) vertices as
#                             exported from Napari (viewer.layers[...].data[0]).
#                             Applied in the Y-X plane; same polygon used for
#                             all Z planes.
# polygon_mode  : str         "In" | "Out" | "Both"  (ignored if no polygon)
#
# Returns
# -------
# written_files : list[str]   Full paths of written mask TIFFs
# n_written     : int         Number of files written
#
# Output naming scheme
# --------------------
# <base_A>[_<base_B>]_<operation>_<group>_<polygon_tag>.tif
# where group ∈ {intersect, superset, A_only, B_only, A_and_B,
#                A_not_B, B_not_A, A_in_B}  and
#       polygon_tag ∈ {inpoly, outpoly, ""} (empty = no polygon)
#
# Operation × polygon_mode → number of output files
# ---------------------------------------------------
# Intersect  / Superset  :  In|Out → 1,   Both → 2
# Parse                  :  In|Out → 2,   Both → 4
# Subsets                :  In|Out → 3,   Both → 6
#

VALID_OPS   = ("Intersect", "Parse", "Subsets", "Superset")
VALID_PMODES = ("In", "Out", "Both")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mask(path):
    """Load a mask TIFF and return a (Z,Y,X) int32 array."""
    arr = tf.imread(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis]   # promote to (1,Y,X)
    if arr.dtype != np.int32:
        arr = arr.astype(np.int32)
    return arr


def _poly_mask_2d(polygon_yx, shape_yx):
    """
    Rasterise a (N,2) polygon (y,x) onto a 2-D boolean array of shape_yx.
    Uses skimage.draw.polygon.
    """
    rr, cc = sk_polygon(polygon_yx[:, 0], polygon_yx[:, 1], shape=shape_yx)
    m = np.zeros(shape_yx, dtype=bool)
    m[rr, cc] = True
    return m


def _poly_mask_3d(polygon_yx, shape_zyx):
    """Broadcast the 2-D polygon mask to (Z,Y,X)."""
    pm2 = _poly_mask_2d(polygon_yx, shape_zyx[1:])
    return np.broadcast_to(pm2[np.newaxis], shape_zyx).copy()


def _overlap_ids(mask_a, mask_b, threshold):
    """
    Return the set of label IDs in mask_a that overlap mask_b
    above `threshold`.

    threshold < 1  → fraction of the A-label's voxels that fall in any B-label
    threshold >= 1 → absolute voxel count overlap
    """
    ids_a   = np.unique(mask_a)
    ids_a   = ids_a[ids_a != 0]
    b_nonzero = mask_b != 0
    in_both = set()
    for lab in ids_a:
        region = mask_a == lab
        overlap = int(np.count_nonzero(region & b_nonzero))
        if threshold >= 1:
            if overlap >= threshold:
                in_both.add(lab)
        else:
            frac = overlap / max(1, int(np.count_nonzero(region)))
            if frac >= threshold:
                in_both.add(lab)
    return in_both


def _keep_labels(mask, label_ids):
    """Return a copy of mask keeping only the given label IDs (zero elsewhere)."""
    out = np.zeros_like(mask)
    for lab in label_ids:
        out[mask == lab] = lab
    return out


def _save_mask(arr, path):
    """Save int32 mask array as plain TIFF (no ImageJ flag)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tf.imwrite(path, arr.astype(np.int32), metadata={"axes": "ZYX"})


def _base(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]


def _out_path(out_dir, name):
    return os.path.join(out_dir, name + ".tif")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def meta_mask(mask_files, output_folder,
              operation, threshold=0.5,
              polygon=None, polygon_mode="In"):
    """
    Perform set operations on one or two mask TIFFs.

    See module docstring for full parameter / return documentation.
    """
    # ---- Validate inputs ----
    if operation not in VALID_OPS:
        raise ValueError(f"operation must be one of {VALID_OPS}; got '{operation}'")
    if polygon_mode not in VALID_PMODES:
        raise ValueError(f"polygon_mode must be one of {VALID_PMODES}; got '{polygon_mode}'")
    if not (1 <= len(mask_files) <= 2):
        raise ValueError("mask_files must contain 1 or 2 paths")
    if len(mask_files) == 1 and operation in ("Parse", "Subsets", "Intersect"):
        raise ValueError(f"operation '{operation}' requires 2 mask files")

    if polygon is None and polygon_mode != "In":
        warnings.warn(
            "polygon_mode is ignored because no polygon was provided; "
            "treating as polygon_mode='In' (whole volume).",
            UserWarning
        )
        polygon_mode = "In"

    # ---- Load masks ----
    mask_a = _load_mask(mask_files[0])
    mask_b = _load_mask(mask_files[1]) if len(mask_files) == 2 else None

    if mask_b is not None and mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes do not match: {mask_a.shape} vs {mask_b.shape}"
        )

    shape_zyx = mask_a.shape

    # ---- Build polygon masks ----
    if polygon is not None:
        poly_arr = np.asarray(polygon)
        if poly_arr.ndim == 3:
            # Napari exports shapes as (1, N, 3) with coords (z,y,x)
            poly_arr = poly_arr[0][:, 1:]   # drop z → (N,2) y,x
        elif poly_arr.ndim == 2 and poly_arr.shape[1] == 3:
            poly_arr = poly_arr[:, 1:]       # (N,3) → (N,2)
        pm_in  = _poly_mask_3d(poly_arr, shape_zyx)
        pm_out = ~pm_in
    else:
        pm_in  = np.ones(shape_zyx, dtype=bool)
        pm_out = np.zeros(shape_zyx, dtype=bool)   # empty — not used

    # Which spatial subsets to process
    if polygon is None or polygon_mode == "In":
        regions = [("", pm_in)]
    elif polygon_mode == "Out":
        regions = [("outpoly", pm_out)]
    else:  # Both
        regions = [("inpoly", pm_in), ("outpoly", pm_out)]

    # ---- Output directory ----
    out_dir = os.path.join(output_folder, "Masks", "Metamask")
    os.makedirs(out_dir, exist_ok=True)

    base_a = _base(mask_files[0])
    base_b = _base(mask_files[1]) if mask_b is not None else None
    prefix = f"{base_a}_{base_b}" if base_b else base_a

    written = []

    # ---- Compute set groups per region ----
    for poly_tag, region_mask in regions:
        suffix = f"_{poly_tag}" if poly_tag else ""

        # Restrict masks to this spatial region
        a_reg = mask_a.copy(); a_reg[~region_mask] = 0
        b_reg = mask_b.copy() if mask_b is not None else None
        if b_reg is not None:
            b_reg[~region_mask] = 0

        if operation == "Superset":
            # Union of all labelled voxels; re-label to avoid collision
            out = a_reg.copy()
            if b_reg is not None:
                # Offset B labels so they don't collide with A
                max_a = int(a_reg.max())
                b_off = b_reg.copy()
                b_off[b_off != 0] += max_a
                # Where A is 0, fill with offset-B
                out[out == 0] = b_off[out == 0]
            name = f"{prefix}_Superset{suffix}"
            p = _out_path(out_dir, name)
            _save_mask(out, p); written.append(p)

        elif operation == "Intersect":
            # Labels in A that overlap B above threshold
            in_both = _overlap_ids(a_reg, b_reg, threshold)
            out = _keep_labels(a_reg, in_both)
            name = f"{prefix}_Intersect{suffix}"
            p = _out_path(out_dir, name)
            _save_mask(out, p); written.append(p)

        elif operation == "Parse":
            # Partition A labels: those overlapping B, those not
            in_both  = _overlap_ids(a_reg, b_reg, threshold)
            ids_a    = set(np.unique(a_reg)) - {0}
            not_in_b = ids_a - in_both

            in_b_mask  = _keep_labels(a_reg, in_both)
            out_b_mask = _keep_labels(a_reg, not_in_b)

            for grp, arr in (("A_in_B", in_b_mask), ("A_not_B", out_b_mask)):
                name = f"{prefix}_Parse_{grp}{suffix}"
                p = _out_path(out_dir, name)
                _save_mask(arr, p); written.append(p)

        elif operation == "Subsets":
            # Three groups: A-only, B-only, A∩B
            in_both_a = _overlap_ids(a_reg, b_reg, threshold)
            in_both_b = _overlap_ids(b_reg, a_reg, threshold)
            ids_a = set(np.unique(a_reg)) - {0}
            ids_b = set(np.unique(b_reg)) - {0}

            a_only   = ids_a - in_both_a
            b_only   = ids_b - in_both_b
            # A∩B: keep A labels that are "in both" (from A's perspective)
            a_and_b  = in_both_a

            a_only_mask  = _keep_labels(a_reg, a_only)
            b_only_mask  = _keep_labels(b_reg, b_only)
            a_and_b_mask = _keep_labels(a_reg, a_and_b)

            for grp, arr in (
                ("A_only",  a_only_mask),
                ("B_only",  b_only_mask),
                ("A_and_B", a_and_b_mask),
            ):
                name = f"{prefix}_Subsets_{grp}{suffix}"
                p = _out_path(out_dir, name)
                _save_mask(arr, p); written.append(p)

    log_row = {
        "MaskA":        os.path.basename(mask_files[0]),
        "MaskB":        os.path.basename(mask_files[1]) if len(mask_files) > 1 else "",
        "Operation":    operation,
        "Threshold":    threshold,
        "PolygonMode":  polygon_mode if polygon is not None else "N/A",
        "HasPolygon":   polygon is not None,
        "N_OutputFiles": len(written),
        "OutputFiles":  ";".join(os.path.basename(p) for p in written),
        "Timestamp":    __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return written, len(written), log_row


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, sys

    out = "/Volumes/SG2/cellpose_work/testOutput"
    f1  = os.path.join(out, "Masks", "test1_full3ch_masks.tif")
    f2  = os.path.join(out, "Masks", "test1_cellnuc2ch_masks.tif")

    # Fall back to testOutput root if Masks subdir not present
    if not os.path.exists(f1):
        f1 = os.path.join(out, "test1_full3ch_masks.tif")
    if not os.path.exists(f2):
        f2 = os.path.join(out, "test1_cellnuc2ch_masks.tif")

    if not (os.path.exists(f1) and os.path.exists(f2)):
        print(f"Test mask files not found under {out}")
        sys.exit(1)

    for op in ("Intersect", "Parse", "Subsets", "Superset"):
        files_in = [f1, f2] if op != "Superset" else [f1, f2]
        paths, n, log_row = meta_mask(
            files_in, out,
            operation=op, threshold=0.3,
            polygon=None, polygon_mode="In"
        )
        print(f"{op}: {n} file(s)")
        for p in paths:
            print(f"  {p}  ({np.unique(tf.imread(p)).size - 1} masks)")
