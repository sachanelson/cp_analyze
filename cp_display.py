import os
import glob
import re
import numpy as np
import tifffile as tf
import napari
from magicgui.widgets import Label, Container, ComboBox, FloatSpinBox, PushButton, CheckBox
from magicgui import magicgui

#
# cp_display.py - Napari viewer callable from cp_manager
#
# open_viewer(records, output_folder,
#             show_masks=True, show_metamasks=True)
#
# Parameters
# ----------
# records        : list[FileRecord]  Included FileRecord objects (with .path, .filename)
# output_folder  : str               Root output folder; masks in Masks/, metamasks in Masks/Metamask/
# show_masks     : bool              Load files from output_folder/Masks/
# show_metamasks : bool              Load files from output_folder/Masks/Metamask/
#
# Each call opens a fresh, independent Napari window (non-blocking).
# Loads raw image channels from the first included record as reference images,
# then adds all mask/metamask files as label layers.
# A filter widget (min/max voxels, min mean signal) and click-to-inspect panel
# are docked on the right, mirroring cp_view.py.
#

def open_viewer(records, output_folder,
                show_masks=True, show_metamasks=True,
                preloaded_arrays=None):
    """
    Open a new Napari window showing raw images from `records` and
    mask/metamask label layers from the output folder.
    Returns immediately (non-blocking).
    """
    masks_dir    = os.path.join(output_folder, "Masks")
    metamask_dir = os.path.join(masks_dir, "Metamask")

    # ---- Collect mask files ----
    mask_files = []
    if show_masks and os.path.isdir(masks_dir):
        mask_files += sorted(glob.glob(os.path.join(masks_dir, "*.tif")))
    if show_metamasks and os.path.isdir(metamask_dir):
        mask_files += sorted(glob.glob(os.path.join(metamask_dir, "*.tif")))

    import time
    t0_total = time.perf_counter()

    # ---- Load raw images from all included records ----
    # Use preloaded_arrays cache if provided (avoids re-reading large TIFFs)
    raw_images = []
    for rec in records:
        try:
            t0 = time.perf_counter()
            img = preloaded_arrays.get(rec.path) if preloaded_arrays is not None else None
            if img is not None:
                print(f"[cp_display] Cache HIT  {rec.filename} ({time.perf_counter()-t0:.2f}s)")
            else:
                img = tf.imread(rec.path)
                if preloaded_arrays is not None:
                    preloaded_arrays[rec.path] = img
                print(f"[cp_display] Cache MISS {rec.filename} — read in {time.perf_counter()-t0:.2f}s")
            raw_images.append((rec.filename, img))
        except Exception as e:
            print(f"[cp_display] Could not load {rec.filename}: {e}")

    # ---- Initialize mask list ----
    loaded_masks = []

    if not raw_images and not mask_files:
        print("[cp_display] Nothing to display.")
        return

    # ---- Open Napari viewer ----
    viewer = napari.Viewer(title=f"cp_display — {os.path.basename(output_folder)}")

    colormaps = ["green", "cyan", "magenta", "yellow", "blue", "red"]
    for i, (name, img) in enumerate(raw_images):
        # If multi-channel (Z,C,Y,X), add each channel separately
        if img.ndim == 4:
            for ch in range(img.shape[1]):
                cmap = colormaps[ch % len(colormaps)]
                viewer.add_image(img[:, ch], name=f"{name}_ch{ch}",
                                 colormap=cmap, visible=(ch == 0))
        else:
            cmap = colormaps[i % len(colormaps)]
            viewer.add_image(img, name=name, colormap=cmap, visible=(i == 0))

    # Track layer metadata: name -> {'path': full_path, 'is_metamask': bool, 'filtered': bool}
    layer_metadata = {}
    
    t0_masks = time.perf_counter()
    for p in mask_files:
        try:
            t0 = time.perf_counter()
            arr = preloaded_arrays.get(p) if preloaded_arrays is not None else None
            if arr is not None:
                print(f"[cp_display] Cache HIT  mask {os.path.basename(p)} ({time.perf_counter()-t0:.2f}s)")
            else:
                arr = tf.imread(p)
                if preloaded_arrays is not None:
                    preloaded_arrays[p] = arr
                print(f"[cp_display] Cache MISS mask {os.path.basename(p)} — read in {time.perf_counter()-t0:.2f}s")
            name = os.path.basename(p)
            is_metamask = metamask_dir in p
            loaded_masks.append((name, arr.astype(np.int32)))
            layer_metadata[name] = {'path': p, 'is_metamask': is_metamask, 'filtered': False}
        except Exception as e:
            print(f"[cp_display] Could not load mask {p}: {e}")
    print(f"[cp_display] All masks loaded in {time.perf_counter()-t0_masks:.2f}s")

    t0_napari = time.perf_counter()
    for name, arr in loaded_masks:
        viewer.add_labels(arr, name=name, visible=True)
    print(f"[cp_display] add_labels took {time.perf_counter()-t0_napari:.2f}s")

    # ---- Add a dedicated Shapes layer for polygon-based mask deletion ----
    poly_layer = viewer.add_shapes(name="Polygon Selection", edge_color="yellow",
                                   face_color="transparent", edge_width=2)
    poly_layer.mode = 'add_polygon'

    # ---- Build lookup from mask layer name to source raw image data ----
    # Store all raw images with their channels for mean signal calculation
    # Key: mask layer name, Value: dict with 'img' (Z,C,Y,X or Z,Y,X), 'path', 'name'
    mask_to_raw = {}
    for mask_name, _ in loaded_masks:
        # Parse mask filename to find source: e.g., "B1FL_TT1_S1_z0-60_median_masks.tif"
        # Extract base name by removing mask-related suffixes
        base_match = re.match(r'(.+?)_(?:CN|S1|S2)_(?:z\d+-\d+)(?:_[^_]+)?_masks\.tif$', mask_name)
        if base_match:
            base = base_match.group(1)
        else:
            # Fallback: remove common suffixes
            base = mask_name.replace('_masks.tif', '').replace('_CN_', '_').replace('_S1_', '_').replace('_S2_', '_')
            # Remove z-range and filter suffixes
            base = re.sub(r'_z\d+-\d+', '', base)
            base = re.sub(r'_(?:median|gaussian|none)$', '', base, flags=re.IGNORECASE)
        
        # Find matching raw image
        raw_data = None
        for raw_name, raw_arr in raw_images:
            # Match if base is contained in raw filename (or vice versa)
            if base in raw_name or raw_name.replace('.tif', '') in base:
                raw_data = {'img': raw_arr, 'path': rec.path if 'rec' in locals() else '', 'name': raw_name}
                break
        
        # If no match found, use first raw image as fallback
        if raw_data is None and raw_images:
            raw_name, raw_arr = raw_images[0]
            raw_data = {'img': raw_arr, 'path': rec.path if 'rec' in locals() else '', 'name': raw_name}
        
        mask_to_raw[mask_name] = raw_data

    # ---- Filter widget (mirrors cp_view) ----
    if loaded_masks:
        first_mask_name, first_mask_arr = loaded_masks[0]
        n_initial = int(len(np.unique(first_mask_arr)) - 1)
        lbl_count = Label(value=f"Masks shown: {n_initial} / {n_initial}")

        # Track which layers have been filtered
        filtered_layer_names = set()
        
        @magicgui(
            call_button="Apply filters to visible",
            min_size={"label": "Min voxels",    "value": 200,    "min": 0, "max": 10_000_000},
            max_size={"label": "Max voxels",    "value": 500_000,"min": 0, "max": 100_000_000},
            min_mean={"label": "Min mean signal","value": 0,     "min": 0, "max": 65535},
        )
        def filter_widget(min_size: int = 200,
                          max_size: int = 500_000,
                          min_mean: int = 0):
            nonlocal filtered_layer_names
            # Apply filters to all visible mask (Labels) layers only
            filtered_layers = []
            total_before = 0
            total_after = 0
            
            for layer in viewer.layers:
                if not isinstance(layer, napari.layers.Labels):
                    continue
                if not layer.visible:
                    continue
                    
                masks = layer.data
                labels, counts = np.unique(masks, return_counts=True)
                n_before = len(labels) - 1  # exclude background
                total_before += n_before
                
                # Look up correct raw image and channel for this layer
                mean_ch = ch_selector.value
                signal_img = None
                raw_data = mask_to_raw.get(layer.name)
                if raw_data is not None and raw_data['img'] is not None:
                    raw_arr = raw_data['img']
                    if raw_arr.ndim == 4 and mean_ch < raw_arr.shape[1]:
                        candidate = raw_arr[:, mean_ch]
                    elif raw_arr.ndim == 3:
                        candidate = raw_arr
                    else:
                        candidate = None
                    if candidate is not None and candidate.shape == masks.shape:
                        signal_img = candidate
                
                # --- Vectorized filter: build a LUT then apply in one indexing step ---
                max_lbl = int(labels.max()) if len(labels) > 0 else 0
                lut = np.zeros(max_lbl + 1, dtype=masks.dtype)
                
                # Size filter: mark labels that pass into the LUT
                size_ok = (counts >= min_size) & (counts <= max_size)
                keep_labels = labels[size_ok & (labels != 0)]
                lut[keep_labels] = keep_labels
                
                # Mean signal filter: compute all label means in one pass with bincount
                if signal_img is not None and min_mean > 0 and len(keep_labels) > 0:
                    flat_mask = masks.ravel().astype(np.int64)
                    flat_sig  = signal_img.ravel().astype(np.float64)
                    lut_size  = max_lbl + 1
                    label_sums   = np.bincount(flat_mask, weights=flat_sig, minlength=lut_size)
                    label_counts = np.bincount(flat_mask,                   minlength=lut_size)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        label_means = np.where(label_counts > 0,
                                               label_sums / label_counts, 0.0)
                    # Zero out labels in LUT that are below mean threshold
                    lut[label_means[:lut_size] < min_mean] = 0
                
                # Apply LUT: single vectorized remap - O(N_voxels)
                out = lut[masks]
                n_after = int(len(np.unique(out)) - 1)
                total_after += n_after
                layer.data = out
                
                # Mark this layer as filtered
                if layer.name in layer_metadata:
                    layer_metadata[layer.name]['filtered'] = True
                    filtered_layer_names.add(layer.name)
                    
                filtered_layers.append(f"{layer.name}: {n_before}→{n_after}")
                
            lbl_count.value = f"Masks shown: {total_after}"
            print(f"[filter] Applied to visible layers: {filtered_layers}")
            print(f"[filter] min_size={min_size}, max_size={max_size}, min_mean={min_mean}")
            print(f"[filter] Total: {total_before} → {total_after} labels")
            print(f"[filter] Filtered layer names tracked: {filtered_layer_names}")

        # ---- Write filtered masks widgets (added to same panel) ----
        from datetime import datetime
        import shutil
        
        btn_write = PushButton(text="Write filtered masks")
        chk_overwrite = CheckBox(text="Backup current", value=True)
        lbl_write_status = Label(value="")
        
        # Combine filter widget and write widgets into one panel
        combined_panel = Container(
            widgets=[filter_widget, btn_write, chk_overwrite, lbl_write_status],
            labels=False
        )
        viewer.window.add_dock_widget(combined_panel, area="right", name="Mask Filters")
        
        def on_write_clicked():
            nonlocal filtered_layer_names
            
            if not filtered_layer_names:
                lbl_write_status.value = "No filters applied yet"
                print("[write] No filters have been applied to any layers")
                return
            
            do_backup = chk_overwrite.value
            date_str = datetime.now().strftime("%Y-%m-%d")
            
            # Separate filtered masks by type (regular vs metamask)
            regular_masks = []
            metamask_masks = []
            
            for name in filtered_layer_names:
                meta = layer_metadata.get(name)
                if meta and meta['filtered']:
                    if meta['is_metamask']:
                        metamask_masks.append((name, meta['path']))
                    else:
                        regular_masks.append((name, meta['path']))
            
            if not regular_masks and not metamask_masks:
                lbl_write_status.value = "No filtered masks to write"
                return
            
            written = []
            backed_up = []
            errors = []
            
            def write_mask(name, path, backup_dir):
                layer = next((l for l in viewer.layers if l.name == name), None)
                if layer is None:
                    errors.append(f"Layer {name} not found")
                    return
                if do_backup and os.path.exists(path):
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_path = os.path.join(backup_dir, os.path.basename(path))
                    shutil.copy2(path, backup_path)
                    backed_up.append(os.path.basename(path))
                    print(f"[write] Backed up {os.path.basename(path)} → {backup_dir}")
                arr = layer.data.astype(np.int32)
                n_labels = int(len(np.unique(arr)) - 1)
                tf.imwrite(path, arr)
                written.append(name)
                print(f"[write] Wrote {path} ({n_labels} labels)")
                # Update cache so next viewer open gets the modified data
                if preloaded_arrays is not None and path in preloaded_arrays:
                    preloaded_arrays[path] = arr
                    print(f"[write] Cache updated for {os.path.basename(path)}")
            
            # Process regular masks
            if regular_masks:
                backup_dir = os.path.join(masks_dir, f"old_{date_str}")
                for name, path in regular_masks:
                    try:
                        write_mask(name, path, backup_dir)
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        print(f"[write] Error writing {name}: {e}")
            
            # Process metamasks
            if metamask_masks:
                backup_dir = os.path.join(metamask_dir, f"old_{date_str}")
                for name, path in metamask_masks:
                    try:
                        write_mask(name, path, backup_dir)
                    except Exception as e:
                        errors.append(f"{name}: {e}")
                        print(f"[write] Error writing metamask {name}: {e}")
            
            # Update status
            if written:
                status = f"Wrote {len(written)} files"
                if backed_up:
                    status += f", backed up {len(backed_up)}"
                lbl_write_status.value = status
                # Clear the filtered set since we've written them
                filtered_layer_names.clear()
            elif errors:
                lbl_write_status.value = f"Errors: {len(errors)}"
            
            print(f"[write] Summary: {len(written)} written, {len(backed_up)} backed up, {len(errors)} errors")
        
        btn_write.clicked.connect(on_write_clicked)

        # ---- Click-to-inspect panel ----
        lbl_id     = Label(value="Label ID:  —")
        lbl_voxels = Label(value="Voxels:    —")
        lbl_mean   = Label(value="Mean signal: —")
        lbl_layer  = Label(value="Layer(s):  —")
        lbl_help   = Label(value="Click/drag to inspect | Select Labels layer, press '3' for polygon tool")
        lbl_error  = Label(value="", visible=False)  # Error message label
        
        # Channel selector for mean signal calculation
        channel_choices = [("Ch 0 (C/cytoplasm)", 0)]
        if len(raw_images) > 0:
            _, sample_img = raw_images[0]
            if sample_img.ndim == 4:
                n_ch = sample_img.shape[1]
                channel_choices = []
                ch_names = ["C (cytoplasm)", "N (nucleus)", "S1 (signal 1)", "S2 (signal 2)"]
                for i in range(min(n_ch, 4)):
                    name = ch_names[i] if i < len(ch_names) else f"Ch {i}"
                    channel_choices.append((f"Ch {i}: {name}", i))
        
        ch_selector = ComboBox(choices=channel_choices, value=0, label="Mean signal channel")
        
        # Toggle button to activate/deactivate inspector
        btn_toggle = PushButton(text="Inspect: OFF")
        inspector_state = {'active': False}
        
        # Track last inspected label and layer for delete/restore operations
        last_inspected = {'layer': None, 'label': None, 'layer_name': None}

        # Track drag state for region selection
        drag_state = {'start_pos': None, 'is_dragging': False, 'start_coords': None}
        
        def on_toggle_inspector():
            inspector_state['active'] = not inspector_state['active']
            if inspector_state['active']:
                btn_toggle.text = "Inspect: ON"
                lbl_help.value = "Click/drag to inspect  |  Press button to turn off"
                # Set all Labels layers to pick mode; make the visible one active
                visible_labels = []
                for layer in viewer.layers:
                    if isinstance(layer, napari.layers.Labels):
                        layer.mode = 'pick'
                        if layer.visible:
                            visible_labels.append(layer)
                if len(visible_labels) == 1:
                    viewer.layers.selection.active = visible_labels[0]
                    lbl_error.visible = False
                elif len(visible_labels) > 1:
                    viewer.layers.selection.active = visible_labels[0]
                    lbl_error.value = f"WARNING: {len(visible_labels)} mask layers visible — hide all but one"
                    lbl_error.visible = True
                else:
                    lbl_error.value = "WARNING: No mask layers are visible"
                    lbl_error.visible = True
            else:
                btn_toggle.text = "Inspect: OFF"
                lbl_help.value = "Press 'Inspect: OFF' to enable inspector"
                drag_state['start_pos'] = None
                drag_state['is_dragging'] = False
                # Restore pan_zoom mode on all Labels layers
                for layer in viewer.layers:
                    if isinstance(layer, napari.layers.Labels):
                        layer.mode = 'pan_zoom'
        
        btn_toggle.clicked.connect(on_toggle_inspector)
        
        # Delete/Restore buttons for label editing
        btn_delete_label = PushButton(text="Delete label")
        btn_restore_labels = PushButton(text="Restore labels")
        lbl_delete_status = Label(value="")
        
        info_panel = Container(
            widgets=[btn_toggle, lbl_count, lbl_id, lbl_voxels, lbl_mean, lbl_layer, ch_selector, 
                     lbl_help, lbl_error, btn_delete_label, btn_restore_labels, lbl_delete_status], 
            labels=False
        )
        viewer.window.add_dock_widget(info_panel, area="right", name="Label Inspector")
        
        def get_visible_mask_layers(v):
            """Get all visible Labels layers."""
            return [layer for layer in v.layers 
                    if isinstance(layer, napari.layers.Labels) and layer.visible]
        
        def inspect_single_click(v, event):
            """Handle single click to inspect one mask."""
            visible_mask_layers = get_visible_mask_layers(v)
            
            if not visible_mask_layers:
                lbl_error.value = "ERROR: No mask layers visible!"
                lbl_error.visible = True
                lbl_id.value = "Label ID:  —"
                lbl_voxels.value = "Voxels: —"
                lbl_mean.value = "Mean signal: —"
                lbl_layer.value = "Layer(s):  —"
                return
            else:
                lbl_error.visible = False
            
            mean_ch = ch_selector.value
            hits = []
            
            for layer in visible_mask_layers:
                c = tuple(int(round(c)) for c in layer.world_to_data(event.position))
                if any(co < 0 or co >= s for co, s in zip(c, layer.data.shape)):
                    continue
                val = int(layer.data[c])
                if val != 0:
                    hits.append((layer, c, val))
            
            if not hits:
                lbl_id.value = "Label ID:  (background)"
                lbl_voxels.value = "Voxels: —"
                lbl_mean.value = "Mean signal: —"
                lbl_layer.value = f"Layer(s):  {len(visible_mask_layers)} visible, no mask at click"
                return
            
            target_layer, coords, lbl = hits[0]
            
            if len(hits) == 1:
                layer_summary = target_layer.name
            else:
                layer_names = [h[0].name for h in hits]
                layer_summary = f"{target_layer.name} (top of {len(hits)}: {', '.join(layer_names)})"
            
            raw_data = mask_to_raw.get(target_layer.name)
            if raw_data is None or raw_data['img'] is None:
                raise RuntimeError(f"No raw image found for mask '{target_layer.name}'")
            
            raw_arr = raw_data['img']
            if raw_arr.ndim == 4 and mean_ch < raw_arr.shape[1]:
                signal_img = raw_arr[:, mean_ch]
            elif raw_arr.ndim == 3:
                signal_img = raw_arr
            else:
                raise RuntimeError(f"Cannot extract channel {mean_ch} from raw image with shape {raw_arr.shape}")
            
            if signal_img.shape != target_layer.data.shape:
                raise RuntimeError(
                    f"Shape mismatch: raw image channel {mean_ch} has shape {signal_img.shape}, "
                    f"but mask '{target_layer.name}' has shape {target_layer.data.shape}"
                )
            
            mask = target_layer.data == lbl
            voxels = int(mask.sum())
            mean_s = float(signal_img[mask].mean())
            
            lbl_id.value = f"Label ID:  {lbl}"
            lbl_voxels.value = f"Voxels:    {voxels:,}"
            lbl_mean.value = f"Mean signal: {mean_s:.1f} (Ch {mean_ch})"
            lbl_layer.value = f"Layer(s):  {layer_summary}"
            
            last_inspected['layer'] = target_layer
            last_inspected['label'] = lbl
            last_inspected['layer_name'] = target_layer.name
            
            print(f"[inspect] layer={target_layer.name}, ch={mean_ch}, label={lbl}, voxels={voxels:,}, mean={mean_s:.1f}")
            if len(hits) > 1:
                print(f"[inspect] Note: {len(hits)} visible layers have masks at this position")
        
        def inspect_region(v, start_pos, end_pos):
            """Handle drag region selection to find all masks in rectangle."""
            print(f"[region] inspect_region called: start={start_pos}, end={end_pos}")
            visible_mask_layers = get_visible_mask_layers(v)
            
            if not visible_mask_layers:
                lbl_error.value = "ERROR: No mask layers visible!"
                lbl_error.visible = True
                return
            else:
                lbl_error.visible = False
            
            mean_ch = ch_selector.value
            
            # Get Z from current view (use current slice)
            current_z = v.dims.current_step[0] if len(v.dims.current_step) > 0 else 0
            
            all_region_labels = []  # List of (layer_name, label_id, voxels, mean_signal)
            
            for layer in visible_mask_layers:
                # Convert world positions to data coordinates
                start_data = tuple(int(round(c)) for c in layer.world_to_data(start_pos))
                end_data = tuple(int(round(c)) for c in layer.world_to_data(end_pos))
                
                # Build slice for the region (Z, Y, X)
                z_slice = slice(current_z, current_z + 1)  # Single Z slice
                y_slice = slice(min(start_data[1], end_data[1]), max(start_data[1], end_data[1]) + 1)
                x_slice = slice(min(start_data[2], end_data[2]), max(start_data[2], end_data[2]) + 1)
                
                # Check bounds
                if y_slice.start < 0:
                    y_slice = slice(0, y_slice.stop)
                if x_slice.start < 0:
                    x_slice = slice(0, x_slice.stop)
                if y_slice.stop > layer.data.shape[1]:
                    y_slice = slice(y_slice.start, layer.data.shape[1])
                if x_slice.stop > layer.data.shape[2]:
                    x_slice = slice(x_slice.start, layer.data.shape[2])
                
                if y_slice.start >= y_slice.stop or x_slice.start >= x_slice.stop:
                    continue
                
                # Extract region
                region = layer.data[z_slice, y_slice, x_slice]
                unique_labels = np.unique(region)
                unique_labels = unique_labels[unique_labels != 0]  # Exclude background
                
                if len(unique_labels) == 0:
                    continue
                
                # Get raw data for mean signal calculation
                raw_data = mask_to_raw.get(layer.name)
                signal_img = None
                if raw_data is not None and raw_data['img'] is not None:
                    raw_arr = raw_data['img']
                    if raw_arr.ndim == 4 and mean_ch < raw_arr.shape[1]:
                        signal_img = raw_arr[:, mean_ch]
                    elif raw_arr.ndim == 3:
                        signal_img = raw_arr
                
                for lbl in unique_labels:
                    mask = layer.data == lbl
                    voxels = int(mask.sum())
                    mean_s = float("nan")
                    if signal_img is not None and signal_img.shape == layer.data.shape:
                        mean_s = float(signal_img[mask].mean())
                    all_region_labels.append((layer.name, int(lbl), voxels, mean_s))
            
            if not all_region_labels:
                lbl_id.value = "Region: no masks found"
                lbl_voxels.value = "—"
                lbl_mean.value = "—"
                lbl_layer.value = f"Layer(s):  {len(visible_mask_layers)} visible"
                last_inspected['layer'] = None
                last_inspected['label'] = None
                last_inspected['layer_name'] = None
                print(f"[region] No masks found in dragged region at Z={current_z}")
                return
            
            # Find largest mask for single-label operations
            largest = max(all_region_labels, key=lambda x: x[2])
            target_layer_name, target_lbl, target_voxels, target_mean = largest
            
            # Find the layer object
            target_layer = None
            for layer in visible_mask_layers:
                if layer.name == target_layer_name:
                    target_layer = layer
                    break
            
            # Build summary
            total_masks = len(all_region_labels)
            layer_counts = {}
            for ln, lbl, vox, mean in all_region_labels:
                layer_counts[ln] = layer_counts.get(ln, 0) + 1
            
            layer_summary = f"{target_layer_name} ({layer_counts[target_layer_name]} masks)"
            if len(layer_counts) > 1:
                other_layers = [f"{ln}:{cnt}" for ln, cnt in layer_counts.items() if ln != target_layer_name]
                layer_summary += f" + {', '.join(other_layers)}"
            
            lbl_id.value = f"Region: {total_masks} masks, largest={target_lbl}"
            lbl_voxels.value = f"Voxels:    {target_voxels:,} (largest)"
            lbl_mean.value = f"Mean signal: {target_mean:.1f} (Ch {mean_ch})"
            lbl_layer.value = f"Layer(s):  {layer_summary}"
            
            # Store largest for delete/restore
            if target_layer:
                last_inspected['layer'] = target_layer
                last_inspected['label'] = target_lbl
                last_inspected['layer_name'] = target_layer_name
            
            print(f"[region] Found {total_masks} masks in region at Z={current_z}")
            print(f"[region] Largest: {target_layer_name} label={target_lbl}, voxels={target_voxels:,}")
            for ln, lbl, vox, mean in sorted(all_region_labels, key=lambda x: -x[2])[:5]:
                print(f"[region]   {ln}:{lbl} = {vox:,} voxels, mean={mean:.1f}")
        
        def on_mouse_drag(layer, event):
            if not inspector_state['active']:
                return
            if event.button != 1:
                return

            start_screen = np.array(event.pos)
            start_world  = event.position
            is_dragging  = False

            try:
                yield

                while event.type == 'mouse_move':
                    dist = np.linalg.norm(np.array(event.pos) - start_screen)
                    if dist > 5:
                        is_dragging = True
                    yield

                end_world = event.position
                dist = np.linalg.norm(np.array(event.pos) - start_screen)
                print(f"[drag] screen dist={dist:.1f}px, is_dragging={is_dragging}, start={start_world}, end={end_world}")

                if not is_dragging:
                    inspect_single_click(viewer, event)
                else:
                    inspect_region(viewer, start_world, end_world)

            except Exception as e:
                print(f"[inspect] error: {e}")
                lbl_error.value = f"Error: {str(e)[:50]}"
                lbl_error.visible = True

        registered = 0
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.mouse_drag_callbacks.append(on_mouse_drag)
                registered += 1
                print(f"[inspect] Registered drag callback on layer: {layer.name}")
        print(f"[inspect] Drag callback registered on {registered} Labels layers")

        def _on_layer_inserted(event):
            if isinstance(event.value, napari.layers.Labels):
                event.value.mouse_drag_callbacks.append(on_mouse_drag)
        viewer.layers.events.inserted.connect(_on_layer_inserted)

        # ---- Delete label functionality ----
        def on_delete_label():
            if last_inspected['layer'] is None or last_inspected['label'] is None:
                lbl_delete_status.value = "No label selected - click a mask first"
                return
            
            layer = last_inspected['layer']
            lbl = last_inspected['label']
            
            try:
                # Count voxels before deletion for reporting
                mask = layer.data == lbl
                voxels_before = int(mask.sum())
                
                if voxels_before == 0:
                    lbl_delete_status.value = f"Label {lbl} already deleted or not found"
                    return
                
                # Delete the label (set to 0)
                layer.data[mask] = 0
                
                # Mark layer as modified (for tracking purposes)
                if layer.name in layer_metadata:
                    layer_metadata[layer.name]['filtered'] = True
                    filtered_layer_names.add(layer.name)
                
                lbl_delete_status.value = f"Deleted label {lbl} ({voxels_before:,} voxels) from {layer.name}"
                print(f"[delete] Removed label {lbl} ({voxels_before:,} voxels) from layer '{layer.name}'")
                
                # Clear the last inspected since it's now deleted
                last_inspected['label'] = None
                
            except Exception as e:
                lbl_delete_status.value = f"Error deleting: {str(e)[:40]}"
                print(f"[delete] Error: {e}")
        
        # ---- Restore labels functionality ----
        def on_restore_labels():
            if last_inspected['layer_name'] is None:
                lbl_delete_status.value = "No layer selected - click a mask first"
                return
            
            layer_name = last_inspected['layer_name']
            
            try:
                # Find the layer in the viewer
                layer = None
                for l in viewer.layers:
                    if l.name == layer_name:
                        layer = l
                        break
                
                if layer is None:
                    lbl_delete_status.value = f"Layer '{layer_name}' not found"
                    return
                
                # Get original file path
                meta = layer_metadata.get(layer_name)
                if meta is None:
                    lbl_delete_status.value = f"No metadata for layer '{layer_name}'"
                    return
                
                original_path = meta['path']
                if not os.path.exists(original_path):
                    lbl_delete_status.value = f"Original file not found: {original_path}"
                    return
                
                # Reload the original data
                original_data = tf.imread(original_path).astype(np.int32)
                
                # Count labels before/after for reporting
                n_before = len(np.unique(layer.data)) - 1
                layer.data = original_data
                n_after = len(np.unique(layer.data)) - 1
                
                # Clear filtered flag since we've restored to original
                meta['filtered'] = False
                if layer_name in filtered_layer_names:
                    filtered_layer_names.discard(layer_name)
                
                lbl_delete_status.value = f"Restored {layer_name}: {n_before}→{n_after} labels"
                print(f"[restore] Reloaded '{layer_name}' from {original_path}")
                print(f"[restore] Labels: {n_before} → {n_after}")
                
            except Exception as e:
                lbl_delete_status.value = f"Error restoring: {str(e)[:40]}"
                print(f"[restore] Error: {e}")
        
        btn_delete_label.clicked.connect(on_delete_label)
        btn_restore_labels.clicked.connect(on_restore_labels)

        # ---- Polygon delete panel ----
        from skimage.draw import polygon as sk_polygon

        poly_z_scope = ComboBox(
            choices=[("Current Z slice only", "current"), ("All Z slices", "all")],
            value="current",
            label="Z scope"
        )
        poly_threshold = FloatSpinBox(value=0.5, min=0.01, max=1.0, step=0.05, label="Overlap threshold")
        btn_poly_delete = PushButton(text="Delete masks in polygon")
        btn_poly_clear = PushButton(text="Clear polygons")
        lbl_poly_status = Label(value="")

        poly_panel = Container(
            widgets=[poly_z_scope, poly_threshold, btn_poly_delete, btn_poly_clear, lbl_poly_status],
            labels=True
        )
        viewer.window.add_dock_widget(poly_panel, area="right", name="Polygon Delete")

        def on_poly_delete():
            shapes = poly_layer.data
            if len(shapes) == 0:
                lbl_poly_status.value = "No polygon drawn on 'Polygon Selection' layer"
                return

            visible_mask_layers = [l for l in viewer.layers
                                   if isinstance(l, napari.layers.Labels) and l.visible]
            if not visible_mask_layers:
                lbl_poly_status.value = "No visible mask layers"
                return

            threshold = poly_threshold.value
            z_scope = poly_z_scope.value
            current_z = int(viewer.dims.current_step[0])
            total_deleted = 0

            for mask_layer in visible_mask_layers:
                masks = mask_layer.data  # (Z, Y, X)
                H, W = masks.shape[1], masks.shape[2]
                out = masks.copy()

                # Rasterize each polygon shape into a 2D mask
                combined_poly_mask = np.zeros((H, W), dtype=bool)
                for shape in shapes:
                    # shape is (N,3) in world coords (z,y,x) — take y,x columns
                    world_pts = np.array(shape)
                    data_pts = np.array([mask_layer.world_to_data(pt) for pt in world_pts])
                    rr, cc = sk_polygon(data_pts[:, 1], data_pts[:, 2], shape=(H, W))
                    combined_poly_mask[rr, cc] = True

                # Determine which Z slices to examine
                if z_scope == "current":
                    z_slices = slice(current_z, current_z + 1)
                else:
                    z_slices = slice(None)

                scoped = masks[z_slices].astype(np.int64)   # (Zs, Y, X)
                max_lbl = int(scoped.max()) if scoped.size > 0 else 0
                if max_lbl == 0:
                    mask_layer.data = out
                    continue

                n_bins = max_lbl + 1

                # Count total voxels per label in scope
                flat_scoped = scoped.ravel()
                total_counts = np.bincount(flat_scoped, minlength=n_bins)

                # Count in-polygon voxels per label in scope
                # combined_poly_mask is (H,W); broadcast across Z axis
                in_poly_scoped = scoped[:, combined_poly_mask].ravel()
                in_poly_counts = np.bincount(in_poly_scoped, minlength=n_bins)

                # Compute overlap fraction; avoid division by zero
                with np.errstate(invalid='ignore', divide='ignore'):
                    overlap = np.where(total_counts > 0,
                                       in_poly_counts / total_counts, 0.0)

                # Labels to delete: overlap >= threshold (exclude background 0)
                delete_mask = (overlap >= threshold)
                delete_mask[0] = False
                delete_labels = np.where(delete_mask)[0].astype(np.int64)
                n_deleted = len(delete_labels)

                if n_deleted > 0:
                    # Build LUT: keep label unless it's in delete set
                    lut = np.arange(max_lbl + 1, dtype=masks.dtype)
                    lut[delete_labels] = 0
                    out = lut[masks.clip(0, max_lbl)]

                mask_layer.data = out
                total_deleted += n_deleted

                if mask_layer.name in layer_metadata:
                    layer_metadata[mask_layer.name]['filtered'] = True
                    filtered_layer_names.add(mask_layer.name)

                print(f"[poly_delete] {mask_layer.name}: deleted {n_deleted} labels "
                      f"(threshold={threshold:.2f}, z_scope={z_scope})")

            lbl_poly_status.value = (f"Deleted {total_deleted} labels across "
                                     f"{len(visible_mask_layers)} layer(s)")

        def on_poly_clear():
            poly_layer.data = []
            lbl_poly_status.value = "Polygons cleared"

        btn_poly_delete.clicked.connect(on_poly_delete)
        btn_poly_clear.clicked.connect(on_poly_clear)

    print(f"[cp_display] Opened viewer with {len(raw_images)} image(s), "
          f"{len(loaded_masks)} mask layer(s)")
