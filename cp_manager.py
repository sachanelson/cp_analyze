import os
import sys
import re
import glob
import subprocess
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QTableView, QHeaderView, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QSplitter, QProgressDialog, QCheckBox,
    QListWidget, QDialogButtonBox, QDialog,
)

#
# cp_manager.py - PyQt6 interface for coordinating TIF image analysis
#

COMPONENT_ONE_OPTIONS = ["C", "N", "S"]

# ---------------------------------------------------------------------------
# Array LRU cache (evicts least-recently-used entries when over memory cap)
# ---------------------------------------------------------------------------

class _ArrayLRUCache:
    """LRU cache for numpy arrays, capped by total memory usage (in GB).
    
    Evicts the least-recently-used entry whenever adding a new array would
    push total cached memory over max_gb. Supports dict-style get/set and
    the `in` operator so it's a drop-in replacement for a plain dict cache.
    """
    def __init__(self, max_gb=4.0):
        self._max_bytes = int(max_gb * 1024 ** 3)
        self._store = OrderedDict()  # path -> array, ordered by recency
        self._sizes = {}             # path -> nbytes

    def _total(self):
        return sum(self._sizes.values())

    def get(self, key, default=None):
        if key not in self._store:
            return default
        self._store.move_to_end(key)  # mark as recently used
        return self._store[key]

    def __contains__(self, key):
        return key in self._store

    def __getitem__(self, key):
        self._store.move_to_end(key)
        return self._store[key]

    def __setitem__(self, key, arr):
        nbytes = arr.nbytes if hasattr(arr, 'nbytes') else 0
        # If this single array exceeds the cap, don't cache it at all
        if nbytes > self._max_bytes:
            print(f"[cache] Skipping cache for {os.path.basename(key)} "
                  f"({nbytes/1024**3:.2f} GB > cap {self._max_bytes/1024**3:.1f} GB)")
            return
        # Evict LRU entries until there is room
        while self._store and (self._total() + nbytes > self._max_bytes):
            evicted_key, _ = self._store.popitem(last=False)
            freed = self._sizes.pop(evicted_key, 0)
            print(f"[cache] Evicted {os.path.basename(evicted_key)} "
                  f"({freed/1024**2:.0f} MB); cache now "
                  f"{self._total()/1024**2:.0f} MB")
        if key in self._store:
            self._sizes.pop(key, None)
        self._store[key] = arr
        self._store.move_to_end(key)
        self._sizes[key] = nbytes
        print(f"[cache] Cached {os.path.basename(key)} "
              f"({nbytes/1024**2:.0f} MB); total {self._total()/1024**2:.0f} MB "
              f"/ {self._max_bytes/1024**2:.0f} MB")

    def set_max_gb(self, gb):
        self._max_bytes = int(gb * 1024 ** 3)
        print(f"[cache] Cap updated to {gb} GB ({self._max_bytes/1024**2:.0f} MB)")

    def clear(self):
        self._store.clear()
        self._sizes.clear()

    def __len__(self):
        return len(self._store)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class FileRecord:
    """Holds metadata and user-editable fields for one TIF file."""
    def __init__(self, path, shape_x, shape_y, shape_z, shape_c):
        self.path       = path
        self.filename   = os.path.basename(path)
        self.size_mb    = os.path.getsize(path) / (1024 ** 2)
        self.date       = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        self.shape_x    = shape_x
        self.shape_y    = shape_y
        self.shape_z    = shape_z
        self.shape_c    = shape_c
        self.zstrt      = 0
        self.znd        = max(0, shape_z - 1)
        self.magnification = "20X"
        self.microscope = ""
        self.included   = True   # checkbox: include in segmentation/analysis
        self.flip_xy    = False  # per-file X↔Y flip
        # Channel descriptions: list of strings like "N_DAPI", "C_pS6647"
        self.ch_descs   = ["" for _ in range(shape_c)]


class TifTableModel(QAbstractTableModel):
    """Table model backed by a list of FileRecord objects."""

    FIXED_COLS = ["Include", "FlipXY", "Filename", "Size (MB)", "Date", "ShapeX", "ShapeY", "ShapeZ", "ShapeC",
                  "zstrt", "znd", "Magnification", "Microscope"]
    EDITABLE_FIXED = {"zstrt", "znd", "Magnification", "Microscope"}

    def __init__(self, records=None, n_channels=3):
        super().__init__()
        self.records    = records or []
        self.n_channels = n_channels

    def _col_headers(self):
        ch_headers = [f"Ch{i}" for i in range(self.n_channels)]
        return self.FIXED_COLS + ch_headers

    def rowCount(self, parent=QModelIndex()):
        return len(self.records)

    def columnCount(self, parent=QModelIndex()):
        return len(self._col_headers())

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            headers = self._col_headers()
            if section < len(headers):
                return headers[section]
        return QVariant()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()
        r = self.records[index.row()]
        col = index.column()
        headers = self._col_headers()
        h = headers[col]

        if h == "Include":
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if r.included else Qt.CheckState.Unchecked
            return QVariant()

        if h == "FlipXY":
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if r.flip_xy else Qt.CheckState.Unchecked
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if   h == "Filename":      return r.filename
            elif h == "Size (MB)":     return f"{r.size_mb:.1f}"
            elif h == "Date":          return r.date
            elif h == "ShapeX":        return str(r.shape_x)
            elif h == "ShapeY":        return str(r.shape_y)
            elif h == "ShapeZ":        return str(r.shape_z)
            elif h == "ShapeC":        return str(r.shape_c)
            elif h == "zstrt":         return str(r.zstrt)
            elif h == "znd":           return str(r.znd)
            elif h == "Magnification": return r.magnification
            elif h == "Microscope":    return r.microscope
            elif h.startswith("Ch"):
                ch_idx = int(h[2:])
                if ch_idx < len(r.ch_descs):
                    return r.ch_descs[ch_idx]
            return QVariant()

        if role == Qt.ItemDataRole.BackgroundRole:
            if h in self.EDITABLE_FIXED or h.startswith("Ch"):
                return QColor(255, 255, 220)   # light yellow = editable
        return QVariant()

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid():
            return False
        r = self.records[index.row()]
        headers = self._col_headers()
        h = headers[index.column()]

        if h == "Include" and role == Qt.ItemDataRole.CheckStateRole:
            r.included = (value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked)
            self.dataChanged.emit(index, index)
            return True

        if h == "FlipXY" and role == Qt.ItemDataRole.CheckStateRole:
            r.flip_xy = (value == Qt.CheckState.Checked.value or value == Qt.CheckState.Checked)
            self.dataChanged.emit(index, index)
            return True

        if role != Qt.ItemDataRole.EditRole:
            return False

        if h == "zstrt":
            try:
                r.zstrt = int(value)
            except ValueError:
                return False
        elif h == "znd":
            try:
                r.znd = int(value)
            except ValueError:
                return False
        elif h == "Magnification":
            r.magnification = value
        elif h == "Microscope":
            r.microscope = value
        elif h.startswith("Ch"):
            ch_idx = int(h[2:])
            if ch_idx < len(r.ch_descs):
                r.ch_descs[ch_idx] = value
        else:
            return False

        self.dataChanged.emit(index, index)
        return True

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        headers = self._col_headers()
        h = headers[index.column()]
        if h in ("Include", "FlipXY"):
            return base | Qt.ItemFlag.ItemIsUserCheckable
        if h in self.EDITABLE_FIXED or h.startswith("Ch"):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def set_records(self, records, n_channels):
        self.beginResetModel()
        self.records    = records
        self.n_channels = n_channels
        self.endResetModel()


# ---------------------------------------------------------------------------
# Channel description editor widget (comp1 combo + comp2 lineedit + Apply)
# ---------------------------------------------------------------------------

class MeasurePairsModel(QAbstractTableModel):
    """Table model for image/mask pairs with include toggle and channel selection."""
    
    HEADERS = ["Include", "Image File", "Mask File", "Channel"]
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pairs = []  # list of dicts with keys: included, image_name, image_path, mask_name, mask_path, channel
    
    def set_pairs(self, pairs):
        self.beginResetModel()
        self._pairs = pairs
        self.endResetModel()
    
    def get_included_pairs(self):
        """Return list of pairs where included=True."""
        return [p for p in self._pairs if p.get('included', False)]
    
    def set_all_included(self, included):
        """Set all pairs to included or excluded."""
        for p in self._pairs:
            p['included'] = included
        self.dataChanged.emit(self.index(0, 0), self.index(len(self._pairs) - 1, 0))
    
    def rowCount(self, parent=QModelIndex()):
        return len(self._pairs)
    
    def columnCount(self, parent=QModelIndex()):
        return len(self.HEADERS)
    
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        return QVariant()
    
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._pairs):
            return QVariant()
        
        pair = self._pairs[index.row()]
        col = index.column()
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if col == 0:
                return ""
            elif col == 1:
                return pair.get('image_name', '')
            elif col == 2:
                return pair.get('mask_name', '')
            elif col == 3:
                return str(pair.get('channel', 0))
        
        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            return Qt.CheckState.Checked if pair.get('included', False) else Qt.CheckState.Unchecked
        
        return QVariant()
    
    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.row() >= len(self._pairs):
            return False
        
        pair = self._pairs[index.row()]
        col = index.column()
        
        if role == Qt.ItemDataRole.CheckStateRole and col == 0:
            pair['included'] = (value == Qt.CheckState.Checked.value)
            self.dataChanged.emit(index, index)
            return True
        
        if role == Qt.ItemDataRole.EditRole:
            if col == 3:  # Channel column
                try:
                    pair['channel'] = int(value)
                    self.dataChanged.emit(index, index)
                    return True
                except ValueError:
                    return False
        
        return False
    
    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        
        if index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        elif index.column() == 3:
            flags |= Qt.ItemFlag.ItemIsEditable
        
        return flags


class ChannelDescEditor(QWidget):
    def __init__(self, table_view, model, main_win, parent=None):
        super().__init__(parent)
        self.table_view = table_view
        self.model      = model
        self.main_win   = main_win

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Channel descriptor:"))
        self.comp1 = QComboBox()
        self.comp1.addItems(COMPONENT_ONE_OPTIONS)
        self.comp1.setFixedWidth(55)
        layout.addWidget(self.comp1)

        layout.addWidget(QLabel("_"))
        self.comp2 = QLineEdit()
        self.comp2.setPlaceholderText("e.g. DAPI")
        self.comp2.setFixedWidth(120)
        layout.addWidget(self.comp2)

        btn_apply = QPushButton("Apply to selected cell")
        btn_apply.clicked.connect(self.apply_to_selected)
        layout.addWidget(btn_apply)

        btn_match = QPushButton("Match File 1")
        btn_match.setToolTip("Copy channel descriptions and Microscope from row 1 to all other rows")
        btn_match.clicked.connect(self.match_file1)
        layout.addWidget(btn_match)

        layout.addStretch()

    def apply_to_selected(self):
        indexes = self.table_view.selectedIndexes()
        if not indexes:
            return
        desc = f"{self.comp1.currentText()}_{self.comp2.text().strip()}"
        for idx in indexes:
            headers = self.model._col_headers()
            if headers[idx.column()].startswith("Ch"):
                self.model.setData(idx, desc)

    def match_file1(self):
        records = self.model.records
        if len(records) < 2:
            return
        src = records[0]
        for rec in records[1:]:
            rec.microscope = src.microscope
            rec.ch_descs   = list(src.ch_descs)
        # Notify view of full data change
        top_left     = self.model.index(1, 0)
        bottom_right = self.model.index(len(records) - 1, self.model.columnCount() - 1)
        self.model.dataChanged.emit(top_left, bottom_right)


# ---------------------------------------------------------------------------
# Segmentation Table window (non-modal, persistent)
# ---------------------------------------------------------------------------

class _SegTableWindow(QWidget):
    """
    Non-modal window showing the segmentation results table.
    Each row has an 'Include' checkbox so the user can mark which mask
    files to use as inputs for Metamask operations.
    Stays open when focus returns to CpManager.
    """

    INCLUDE_COL = 0   # checkbox column index

    def __init__(self, df, filepath="", parent=None):
        super().__init__(parent,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowMinMaxButtonsHint |
                         Qt.WindowType.WindowCloseButtonHint)
        self.setWindowTitle("Segmentation Table")
        self.resize(1000, 350)

        self._included = []   # list of bool, one per row
        self._df       = pd.DataFrame()
        self._filepath = filepath

        layout = QVBoxLayout(self)
        self.lbl_path = QLabel("")
        layout.addWidget(self.lbl_path)

        self.table_view = QTableView()
        self.table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setAlternatingRowColors(True)
        layout.addWidget(self.table_view)

        self.refresh(df, filepath)

    # ---- inner model ----
    class _Model(QAbstractTableModel):
        def __init__(self, df, included):
            super().__init__()
            self._df       = df
            self._included = included   # mutable list

        def rowCount(self, p=QModelIndex()):    return len(self._df)
        def columnCount(self, p=QModelIndex()): return len(self._df.columns) + 1

        def _col_name(self, col):
            if col == 0: return "Include"
            return str(self._df.columns[col - 1])

        def headerData(self, s, o, role=Qt.ItemDataRole.DisplayRole):
            if role != Qt.ItemDataRole.DisplayRole: return QVariant()
            if o == Qt.Orientation.Horizontal: return self._col_name(s)
            return str(s + 1)

        def flags(self, index):
            base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if index.column() == 0:
                return base | Qt.ItemFlag.ItemIsUserCheckable
            return base

        def data(self, index, role=Qt.ItemDataRole.DisplayRole):
            if not index.isValid(): return QVariant()
            row, col = index.row(), index.column()
            if col == 0:
                if role == Qt.ItemDataRole.CheckStateRole:
                    return Qt.CheckState.Checked if self._included[row] else Qt.CheckState.Unchecked
                return QVariant()
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._df.iloc[row, col - 1])
            return QVariant()

        def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
            if index.column() == 0 and role == Qt.ItemDataRole.CheckStateRole:
                self._included[index.row()] = (
                    value == Qt.CheckState.Checked.value or
                    value == Qt.CheckState.Checked)
                self.dataChanged.emit(index, index)
                return True
            return False

    # ---- public API ----
    def refresh(self, df, filepath=""):
        self._df       = df if not df.empty else pd.DataFrame()
        self._filepath = filepath
        self._included = [True] * len(self._df)
        self.lbl_path.setText(f"File: {filepath}" if filepath else "(no file)")
        model = _SegTableWindow._Model(self._df, self._included)
        self.table_view.setModel(model)

    def checked_mask_files(self):
        """Return list of MaskFile paths for rows with Include=True."""
        if self._df.empty or "MaskFile" not in self._df.columns:
            return []
        out_dir = ""
        if self._filepath:
            out_dir = os.path.dirname(self._filepath)
        masks_dir = os.path.join(out_dir, "Masks") if out_dir else ""
        result = []
        for i, inc in enumerate(self._included):
            if inc:
                fname = str(self._df.iloc[i]["MaskFile"])
                full  = os.path.join(masks_dir, fname) if masks_dir else fname
                result.append(full)
        return result


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class CpManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cellpose Analysis Manager")
        self.resize(1400, 800)

        self.records         = []
        self.n_channels      = 3
        self.display_script  = ""
        self.output_folder   = ""
        self.seg_results     = []   # list of dicts for segment.txt
        self.metamask_results= []   # list of dicts for metamasks.txt
        self.measure_df      = pd.DataFrame()   # accumulated measurements
        self.seg_table_win   = None  # persistent non-modal seg table window
        self._img_cache      = _ArrayLRUCache(max_gb=20.0)  # avoids re-reading on display

        tabs = QTabWidget()
        self.tabs = tabs
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_main_tab(),    "Data Manager")
        tabs.addTab(self._build_measure_tab(), "Measure")
        tabs.addTab(self._build_plot_tab(),    "Plot")

    # ------------------------------------------------------------------
    def _build_main_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ---- Top bar: folder selector + channel count + Flip XY ----
        top = QHBoxLayout()

        self.btn_folder = QPushButton("Select Input Folder")
        self.btn_folder.clicked.connect(self.select_folder)
        top.addWidget(self.btn_folder)

        self.lbl_folder = QLineEdit()
        self.lbl_folder.setReadOnly(True)
        self.lbl_folder.setPlaceholderText("No folder selected")
        self.lbl_folder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        top.addWidget(self.lbl_folder)

        top.addWidget(QLabel("Channels:"))
        self.spin_channels = QSpinBox()
        self.spin_channels.setRange(1, 16)
        self.spin_channels.setValue(3)
        self.spin_channels.setFixedWidth(55)
        top.addWidget(self.spin_channels)

        btn_apply_flip = QPushButton("Apply Flip XY to included")
        btn_apply_flip.setToolTip("Set FlipXY=checked on all currently included rows")
        btn_apply_flip.clicked.connect(self._apply_flip_to_included)
        top.addWidget(btn_apply_flip)

        top.addWidget(QLabel("Cache (GB):"))
        self.spin_cache_gb = QSpinBox()
        self.spin_cache_gb.setRange(1, 96)
        self.spin_cache_gb.setValue(20)
        self.spin_cache_gb.setFixedWidth(55)
        self.spin_cache_gb.setToolTip("Max RAM used by the image cache (LRU eviction when exceeded)")
        self.spin_cache_gb.valueChanged.connect(self._img_cache.set_max_gb)
        top.addWidget(self.spin_cache_gb)

        layout.addLayout(top)

        # ---- Output folder row ----
        out_row = QHBoxLayout()
        self.btn_out_choose = QPushButton("Set Output Folder")
        self.btn_out_choose.clicked.connect(self.choose_output_folder)
        out_row.addWidget(self.btn_out_choose)

        self.lbl_outfolder = QLineEdit()
        self.lbl_outfolder.setReadOnly(True)
        self.lbl_outfolder.setPlaceholderText("Defaults to <input_folder>/<date>/ after folder selection")
        self.lbl_outfolder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        out_row.addWidget(self.lbl_outfolder)

        self.btn_create_out = QPushButton("Create Folder")
        self.btn_create_out.setToolTip("Create the output folder on disk if it does not exist")
        self.btn_create_out.clicked.connect(self.create_output_folder)
        out_row.addWidget(self.btn_create_out)

        layout.addLayout(out_row)

        # ---- Channel descriptor editor ----
        self.model      = TifTableModel(n_channels=3)
        self.table_view = QTableView()
        self.table_view.setModel(self.model)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table_view.setAlternatingRowColors(True)

        self.ch_editor = ChannelDescEditor(self.table_view, self.model, self)
        layout.addWidget(self.ch_editor)

        # ---- Table (constrained height) ----
        self.table_view.setMaximumHeight(220)
        layout.addWidget(self.table_view)

        # ---- Write Data List + table viewer buttons ----
        write_row = QHBoxLayout()
        btn_write = QPushButton("Write / Update Data List")
        btn_write.setToolTip("Save table to dataFiles.txt in the output folder")
        btn_write.clicked.connect(self.write_data_list)
        write_row.addWidget(btn_write)

        btn_seg_table = QPushButton("Segmentation Table")
        btn_seg_table.setToolTip("Show segment.txt in a popup")
        btn_seg_table.clicked.connect(self.show_seg_table)
        write_row.addWidget(btn_seg_table)

        btn_mm_table = QPushButton("Metamask Table")
        btn_mm_table.setToolTip("Show metamasks.txt in a popup")
        btn_mm_table.clicked.connect(self.show_metamask_table)
        write_row.addWidget(btn_mm_table)

        write_row.addStretch()
        layout.addLayout(write_row)

        # ---- Segmentation parameters section ----
        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep1)
        layout.addWidget(QLabel("<b>Segmentation (cp_segment)</b>"))

        seg_grid = QGridLayout()
        seg_grid.setColumnStretch(1, 1)

        # Channel spec
        seg_grid.addWidget(QLabel("Channel spec:"), 0, 0)
        self.combo_ch_spec = QComboBox()
        self.combo_ch_spec.addItems(["C", "CN", "S1", "S2"])
        self.combo_ch_spec.setFixedWidth(70)
        seg_grid.addWidget(self.combo_ch_spec, 0, 1, Qt.AlignmentFlag.AlignLeft)

        # Filter type
        seg_grid.addWidget(QLabel("Filter:"), 1, 0)
        self.combo_filter = QComboBox()
        self.combo_filter.addItems(["None", "Median", "Gaussian"])
        self.combo_filter.setFixedWidth(100)
        self.combo_filter.currentTextChanged.connect(self._on_filter_changed)
        seg_grid.addWidget(self.combo_filter, 1, 1, Qt.AlignmentFlag.AlignLeft)

        # Filter size (x, y, z)
        filter_size_row = QHBoxLayout()
        filter_size_row.addWidget(QLabel("Filter size (x,y,z):"))
        self.spin_fx = QSpinBox(); self.spin_fx.setRange(1,20); self.spin_fx.setValue(2); self.spin_fx.setFixedWidth(50)
        self.spin_fy = QSpinBox(); self.spin_fy.setRange(1,20); self.spin_fy.setValue(2); self.spin_fy.setFixedWidth(50)
        self.spin_fz = QSpinBox(); self.spin_fz.setRange(1,20); self.spin_fz.setValue(2); self.spin_fz.setFixedWidth(50)
        filter_size_row.addWidget(self.spin_fx)
        filter_size_row.addWidget(QLabel("x")); filter_size_row.addWidget(self.spin_fy)
        filter_size_row.addWidget(QLabel("x")); filter_size_row.addWidget(self.spin_fz)
        filter_size_row.addStretch()
        self.filter_size_widget = QWidget()
        self.filter_size_widget.setLayout(filter_size_row)
        self.filter_size_widget.setEnabled(False)
        seg_grid.addWidget(self.filter_size_widget, 2, 0, 1, 4)

        # Downscale option
        self.chk_downscale = QCheckBox("Downscale 2× (XY only)")
        self.chk_downscale.setToolTip("Downscale input images 2-fold in X and Y before segmentation")
        seg_grid.addWidget(self.chk_downscale, 3, 0, 1, 2, Qt.AlignmentFlag.AlignLeft)

        # Run segmentation button
        self.btn_run_seg = QPushButton("Run Segmentation")
        self.btn_run_seg.clicked.connect(self.run_segmentation)
        seg_grid.addWidget(self.btn_run_seg, 4, 0, 1, 4)

        layout.addLayout(seg_grid)

        # ---- Metamask controls ----
        sep_mm = QFrame(); sep_mm.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep_mm)
        layout.addWidget(QLabel("<b>Metamask (cp_metamask)</b>"))

        mm_grid = QGridLayout()
        mm_grid.setColumnStretch(1, 1)

        mm_grid.addWidget(QLabel("Mask A:"), 0, 0)
        self.combo_mm_a = QComboBox()
        self.combo_mm_a.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mm_grid.addWidget(self.combo_mm_a, 0, 1)

        mm_grid.addWidget(QLabel("Mask B:"), 1, 0)
        self.combo_mm_b = QComboBox()
        self.combo_mm_b.addItem("(none)", userData="")
        self.combo_mm_b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mm_grid.addWidget(self.combo_mm_b, 1, 1)

        btn_mm_refresh = QPushButton("Refresh from seg table")
        btn_mm_refresh.setToolTip("Populate Mask A/B from checked rows in Segmentation Table")
        btn_mm_refresh.clicked.connect(self._refresh_mm_combos)
        mm_grid.addWidget(btn_mm_refresh, 2, 0, 1, 2)

        mm_grid.addWidget(QLabel("Operation:"), 3, 0)
        self.combo_mm_op = QComboBox()
        self.combo_mm_op.addItems(["Intersect", "Parse", "Subsets", "Superset"])
        self.combo_mm_op.setFixedWidth(110)
        mm_grid.addWidget(self.combo_mm_op, 3, 1, Qt.AlignmentFlag.AlignLeft)

        mm_grid.addWidget(QLabel("Threshold:"), 4, 0)
        self.spin_mm_thr = QDoubleSpinBox()
        self.spin_mm_thr.setRange(0.0, 100000.0)
        self.spin_mm_thr.setDecimals(3)
        self.spin_mm_thr.setValue(0.5)
        self.spin_mm_thr.setFixedWidth(90)
        self.spin_mm_thr.setToolTip("0–1: fractional overlap; ≥1: voxel count")
        mm_grid.addWidget(self.spin_mm_thr, 4, 1, Qt.AlignmentFlag.AlignLeft)

        mm_grid.addWidget(QLabel("Polygon mode:"), 5, 0)
        self.combo_mm_pmode = QComboBox()
        self.combo_mm_pmode.addItems(["In", "Out", "Both"])
        self.combo_mm_pmode.setFixedWidth(80)
        mm_grid.addWidget(self.combo_mm_pmode, 5, 1, Qt.AlignmentFlag.AlignLeft)

        self.btn_run_mm = QPushButton("Run Metamask")
        self.btn_run_mm.clicked.connect(self.run_metamask)
        mm_grid.addWidget(self.btn_run_mm, 6, 0, 1, 2)

        layout.addLayout(mm_grid)

        # ---- Display section ----
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("<b>Display:</b>"))

        self.chk_show_masks     = QCheckBox("Masks")
        self.chk_show_masks.setChecked(True)
        self.chk_show_metamasks = QCheckBox("Metamasks")
        self.chk_show_metamasks.setChecked(True)
        disp_row.addWidget(self.chk_show_masks)
        disp_row.addWidget(self.chk_show_metamasks)
        disp_row.addStretch()

        self.btn_display = QPushButton("Display")
        self.btn_display.clicked.connect(self.open_display)
        disp_row.addWidget(self.btn_display)
        layout.addLayout(disp_row)

        return widget

    # ------------------------------------------------------------------
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder", "")
        if not folder:
            return
        self.lbl_folder.setText(folder)
        self.n_channels = self.spin_channels.value()
        self._load_folder(folder)

        out_dir = self._resolve_output_folder(folder)
        self.output_folder = out_dir
        self.lbl_outfolder.setText(out_dir)
        self._autoload_tables(out_dir)

    # ------------------------------------------------------------------
    def _resolve_output_folder(self, input_folder):
        """
        Detect date-named subfolders (YYYY-MM-DD) inside input_folder.
        - 0 found → return today's date folder (not yet created)
        - 1 found → use it automatically
        - >1 found → ask user: new folder or pick an existing one
        """
        date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        date_dirs = sorted([
            d for d in os.listdir(input_folder)
            if date_pat.match(d) and os.path.isdir(os.path.join(input_folder, d))
        ])

        today = datetime.now().strftime("%Y-%m-%d")
        today_path = os.path.join(input_folder, today)

        if len(date_dirs) == 0:
            return today_path

        if len(date_dirs) == 1:
            chosen = os.path.join(input_folder, date_dirs[0])
            return chosen

        # Multiple date folders — ask user
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Output Folder")
        vbox = QVBoxLayout(dlg)
        vbox.addWidget(QLabel(
            f"Multiple date folders found in:\n{input_folder}\n\n"
            "Select an existing folder, or choose 'New' to create today's."
        ))
        lst = QListWidget()
        lst.addItem(f"[New]  {today_path}")
        for d in date_dirs:
            lst.addItem(os.path.join(input_folder, d))
        lst.setCurrentRow(0)
        vbox.addWidget(lst)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted and lst.currentItem():
            text = lst.currentItem().text()
            # Strip the "[New]  " prefix if present
            path = text.replace("[New]  ", "", 1)
            return path

        # Fallback: today
        return today_path

    def _load_datafiles_txt(self, out_dir):
        """Patch in-memory records from dataFiles.txt if it exists in out_dir.
        Overwrites per-file editable fields (zstrt, znd, magnification, microscope,
        ch_descs, included, flip_xy) for any filename that matches a loaded record."""
        path = os.path.join(out_dir, "dataFiles.txt")
        if not os.path.exists(path):
            return
        try:
            df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        except Exception as e:
            print(f"[autoload] Could not load dataFiles.txt: {e}")
            return

        # Validate ShapeC consistency and update Channels spinbox
        if "ShapeC" in df.columns:
            c_vals = df["ShapeC"].dropna().unique()
            c_ints = []
            for v in c_vals:
                try: c_ints.append(int(v))
                except (ValueError, TypeError): pass
            if len(set(c_ints)) > 1:
                QMessageBox.critical(
                    self, "Channel mismatch",
                    f"dataFiles.txt contains mixed ShapeC values: {sorted(set(c_ints))}\n"
                    "All files must have the same number of channels."
                )
                return
            if len(c_ints) == 1:
                n = c_ints[0]
                self.n_channels = n
                self.spin_channels.blockSignals(True)
                self.spin_channels.setValue(n)
                self.spin_channels.blockSignals(False)

        # Build lookup: filename → row dict
        by_name = {row["Filename"]: row for row in df.to_dict("records")
                   if "Filename" in row}

        patched = 0
        for rec in self.model.records:
            row = by_name.get(rec.filename)
            if row is None:
                continue
            # editable scalar fields
            for field, attr, cast in [
                ("zstrt",         "zstrt",         int),
                ("znd",           "znd",           int),
                ("Magnification", "magnification", str),
                ("Microscope",    "microscope",    str),
            ]:
                if field in row and row[field] != "":
                    try: setattr(rec, attr, cast(row[field]))
                    except ValueError: pass
            # checkboxes
            for field, attr in [("Include", "included"), ("FlipXY", "flip_xy")]:
                if field in row and row[field] != "":
                    val = row[field].strip().lower()
                    setattr(rec, attr, val in ("true", "1", "yes"))
            # channel descriptions
            for col in row:
                if col.startswith("Ch"):
                    try:
                        idx = int(col[2:])
                        if idx < len(rec.ch_descs):
                            rec.ch_descs[idx] = row[col]
                    except ValueError:
                        pass
            patched += 1

        if patched:
            self.model.beginResetModel()
            self.model.endResetModel()
            print(f"[autoload] Patched {patched} record(s) from {path}")

    def _autoload_tables(self, out_dir):
        """Load dataFiles.txt, segment.txt and metamasks.txt from out_dir if they exist."""
        self._load_datafiles_txt(out_dir)
        seg_path = os.path.join(out_dir, "segment.txt")
        if os.path.exists(seg_path):
            try:
                df = pd.read_csv(seg_path, sep="\t", keep_default_na=False)
                self.seg_results = df.to_dict("records")
                print(f"[autoload] Loaded {len(self.seg_results)} rows from {seg_path}")
            except Exception as e:
                print(f"[autoload] Could not load segment.txt: {e}")
        mm_path = os.path.join(out_dir, "metamasks.txt")
        if os.path.exists(mm_path):
            try:
                df = pd.read_csv(mm_path, sep="\t", keep_default_na=False)
                self.metamask_results = df.to_dict("records")
                print(f"[autoload] Loaded {len(self.metamask_results)} rows from {mm_path}")
            except Exception as e:
                print(f"[autoload] Could not load metamasks.txt: {e}")
        # Always open / refresh the seg table window
        self.show_seg_table()

    def choose_output_folder(self):
        start = self.output_folder or self.lbl_folder.text() or ""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", start,
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks
        )
        if folder:
            self.output_folder = folder
            self.lbl_outfolder.setText(folder)
            self._autoload_tables(folder)

    def create_output_folder(self):
        path = self.lbl_outfolder.text().strip()
        if not path:
            QMessageBox.warning(self, "No path", "No output folder path set.")
            return
        if os.path.exists(path):
            QMessageBox.information(self, "Exists", f"Folder already exists:\n{path}")
            return
        try:
            os.makedirs(path)
            QMessageBox.information(self, "Created", f"Created:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create folder:\n{e}")

    def write_data_list(self):
        out_dir = self.output_folder
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please set an output folder first.")
            return
        if not os.path.exists(out_dir):
            reply = QMessageBox.question(
                self, "Folder missing",
                f"Output folder does not exist:\n{out_dir}\n\nCreate it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                os.makedirs(out_dir)
            else:
                return

        records = self.model.records
        if not records:
            QMessageBox.information(self, "No data", "No files loaded yet.")
            return

        headers = self.model._col_headers()
        rows = []
        for r in records:
            row = {}
            for h in headers:
                if   h == "Include":       row[h] = r.included
                elif h == "FlipXY":        row[h] = r.flip_xy
                elif h == "Filename":      row[h] = r.filename
                elif h == "Size (MB)":     row[h] = f"{r.size_mb:.1f}"
                elif h == "Date":          row[h] = r.date
                elif h == "ShapeX":        row[h] = r.shape_x
                elif h == "ShapeY":        row[h] = r.shape_y
                elif h == "ShapeZ":        row[h] = r.shape_z
                elif h == "ShapeC":        row[h] = r.shape_c
                elif h == "zstrt":         row[h] = r.zstrt
                elif h == "znd":           row[h] = r.znd
                elif h == "Magnification": row[h] = r.magnification
                elif h == "Microscope":    row[h] = r.microscope
                elif h.startswith("Ch"):
                    ch_idx = int(h[2:])
                    row[h] = r.ch_descs[ch_idx] if ch_idx < len(r.ch_descs) else ""
            rows.append(row)

        df = pd.DataFrame(rows, columns=headers)
        out_path = os.path.join(out_dir, "dataFiles.txt")
        df.to_csv(out_path, sep="\t", index=False)
        QMessageBox.information(self, "Saved", f"Data list written to:\n{out_path}")

    def _load_folder(self, folder):
        tif_paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".tif", ".tiff"))
        )
        if not tif_paths:
            QMessageBox.information(self, "No TIFs", "No .tif/.tiff files found in the selected folder.")
            return

        records = []
        errors  = []
        n_ch    = self.n_channels

        for path in tif_paths:
            try:
                img = self._img_cache.get(path)
                if img is None:
                    img = tf.imread(path)
                    self._img_cache[path] = img
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
                continue

            shape = img.shape
            ndim  = len(shape)

            # Handle 1-channel case: expect 3 dims (Z, Y, X)
            if n_ch == 1:
                if ndim != 3:
                    errors.append(
                        f"{os.path.basename(path)}: expected 3D for 1 channel, got shape {shape}"
                    )
                    continue
                dims_sorted = sorted(enumerate(shape), key=lambda x: -x[1])
                x_ax = dims_sorted[0][0]
                y_ax = dims_sorted[1][0]
                z_ax = dims_sorted[2][0]
                rec = FileRecord(path,
                                 shape_x=shape[x_ax],
                                 shape_y=shape[y_ax],
                                 shape_z=shape[z_ax],
                                 shape_c=1)
                records.append(rec)
                continue

            # Multi-channel: find channel axis
            c_axes = [i for i, s in enumerate(shape) if s == n_ch]
            if len(c_axes) == 0:
                errors.append(
                    f"{os.path.basename(path)}: no axis with dim={n_ch} (channels). "
                    f"Shape={shape}. Check channel count."
                )
                continue
            if len(c_axes) > 1:
                chosen = self._resolve_channel_axis(os.path.basename(path), shape, c_axes, n_ch)
                if chosen is None:
                    continue
                c_ax = chosen
            else:
                c_ax = c_axes[0]

            spatial = [(i, s) for i, s in enumerate(shape) if i != c_ax]
            spatial_sorted = sorted(spatial, key=lambda x: -x[1])
            x_ax = spatial_sorted[0][0]
            y_ax = spatial_sorted[1][0]
            z_ax = spatial_sorted[2][0] if len(spatial_sorted) > 2 else None

            if z_ax is None:
                errors.append(
                    f"{os.path.basename(path)}: only 2 spatial dims found in shape {shape}."
                )
                continue

            rec = FileRecord(path,
                             shape_x=shape[x_ax],
                             shape_y=shape[y_ax],
                             shape_z=shape[z_ax],
                             shape_c=n_ch)
            records.append(rec)

        if errors:
            QMessageBox.warning(self, "Load errors", "\n".join(errors))

        self.records = records
        self.model.set_records(records, n_ch)
        self.table_view.resizeColumnsToContents()

    def _resolve_channel_axis(self, filename, shape, candidates, n_ch):
        """Ask the user which axis is the channel axis when multiple match."""
        from PyQt6.QtWidgets import QInputDialog
        items = [f"Axis {i} (dim={shape[i]})" for i in candidates]
        item, ok = QInputDialog.getItem(
            self,
            "Ambiguous channel axis",
            f"{filename}\nMultiple axes have dim={n_ch}: {candidates}\nSelect the channel axis:",
            items, 0, False
        )
        if not ok:
            return None
        return candidates[items.index(item)]

    def _apply_flip_to_included(self):
        """Toggle FlipXY=True on all currently included rows."""
        for i, rec in enumerate(self.model.records):
            if rec.included:
                rec.flip_xy = True
        top_left     = self.model.index(0, 0)
        bottom_right = self.model.index(self.model.rowCount() - 1,
                                        self.model.columnCount() - 1)
        self.model.dataChanged.emit(top_left, bottom_right)

    def show_seg_table(self):
        """Open / refresh the non-modal Segmentation Table window."""
        out_dir  = self.output_folder
        seg_path = os.path.join(out_dir, "segment.txt") if out_dir else ""
        if self.seg_results:
            df = pd.DataFrame(self.seg_results)
        elif seg_path and os.path.exists(seg_path):
            try:
                df = pd.read_csv(seg_path, sep="\t", keep_default_na=False)
                self.seg_results = df.to_dict("records")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load segment.txt:\n{e}"); return
        else:
            df = pd.DataFrame()   # empty — window still opens

        if self.seg_table_win is None or not self.seg_table_win.isVisible():
            self.seg_table_win = _SegTableWindow(df, seg_path, parent=self)
            self.seg_table_win.show()
        else:
            self.seg_table_win.refresh(df, seg_path)
            self.seg_table_win.raise_()
        
        # Also refresh measure pairs when showing segmentation table
        self._refresh_measure_pairs()

    def show_metamask_table(self):
        """Show metamasks.txt in a resizable popup table."""
        out_dir = self.output_folder
        mm_path = os.path.join(out_dir, "metamasks.txt") if out_dir else ""
        if self.metamask_results:
            df = pd.DataFrame(self.metamask_results)
        elif mm_path and os.path.exists(mm_path):
            try:
                df = pd.read_csv(mm_path, sep="\t", keep_default_na=False)
                self.metamask_results = df.to_dict("records")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load metamasks.txt:\n{e}"); return
        else:
            QMessageBox.information(self, "No data",
                "No metamask results yet.\nRun meta_mask first."); return
        self._show_df_simple(df, "Metamask Table", mm_path)

    @staticmethod
    def _show_df_simple(df, title, filepath=""):
        """Simple modal read-only DataFrame popup (for metamasks etc)."""
        class _M(QAbstractTableModel):
            def __init__(self, d): super().__init__(); self._d = d
            def rowCount(self, p=QModelIndex()):    return len(self._d)
            def columnCount(self, p=QModelIndex()): return len(self._d.columns)
            def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
                if role == Qt.ItemDataRole.DisplayRole:
                    return str(self._d.iloc[idx.row(), idx.column()])
                return QVariant()
            def headerData(self, s, o, role=Qt.ItemDataRole.DisplayRole):
                if role != Qt.ItemDataRole.DisplayRole: return QVariant()
                return str(self._d.columns[s]) if o == Qt.Orientation.Horizontal else str(s+1)
        dlg = QDialog()
        dlg.setWindowTitle(title); dlg.resize(900, 400)
        vbox = QVBoxLayout(dlg)
        if filepath: vbox.addWidget(QLabel(f"File: {filepath}"))
        tv = QTableView(); tv.setModel(_M(df))
        tv.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tv.horizontalHeader().setStretchLastSection(True)
        tv.setAlternatingRowColors(True)
        vbox.addWidget(tv)
        dlg.exec()

    # ------------------------------------------------------------------
    def _on_filter_changed(self, text):
        self.filter_size_widget.setEnabled(text != "None")

    def _refresh_mm_combos(self):
        """Populate Mask A / Mask B combos from checked rows in the seg table."""
        out_dir   = self.output_folder
        masks_dir = os.path.join(out_dir, "Masks") if out_dir else ""

        # Collect candidates: checked rows from seg table, then all Masks/ files
        checked = []
        if self.seg_table_win and self.seg_table_win.isVisible():
            checked = self.seg_table_win.checked_mask_files()

        # Fall back to all files in Masks/ if nothing is checked
        if not checked and masks_dir and os.path.isdir(masks_dir):
            checked = sorted(glob.glob(os.path.join(masks_dir, "*.tif")))
            metamask_dir = os.path.join(masks_dir, "Metamask")
            if os.path.isdir(metamask_dir):
                checked += sorted(glob.glob(os.path.join(metamask_dir, "*.tif")))

        self.combo_mm_a.clear()
        self.combo_mm_b.clear()
        self.combo_mm_b.addItem("(none)", userData="")
        for p in checked:
            label = os.path.basename(p)
            self.combo_mm_a.addItem(label, userData=p)
            self.combo_mm_b.addItem(label, userData=p)

    def run_metamask(self):
        from cp_metamask import meta_mask
        out_dir = self.output_folder
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please set an output folder first.")
            return
        mask_a = self.combo_mm_a.currentData()
        mask_b = self.combo_mm_b.currentData()  # "" if (none)
        if not mask_a:
            QMessageBox.warning(self, "No mask", "Please select Mask A.\nUse 'Refresh from seg table' first.")
            return
        mask_files = [mask_a] if not mask_b else [mask_a, mask_b]
        operation  = self.combo_mm_op.currentText()
        threshold  = self.spin_mm_thr.value()
        pmode      = self.combo_mm_pmode.currentText()
        try:
            written, n, log_row = meta_mask(
                mask_files    = mask_files,
                output_folder = out_dir,
                operation     = operation,
                threshold     = threshold,
                polygon       = None,
                polygon_mode  = pmode,
            )
            self.metamask_results.append(log_row)
            mm_path = os.path.join(out_dir, "metamasks.txt")
            pd.DataFrame(self.metamask_results).to_csv(mm_path, sep="\t", index=False)
            QMessageBox.information(self, "Done",
                f"{n} metamask file(s) written.\n" +
                "\n".join(os.path.basename(p) for p in written) +
                f"\n\nTable saved to:\n{mm_path}")
        except Exception as e:
            QMessageBox.critical(self, "Metamask error", str(e))

    def open_display(self):
        from cp_display import open_viewer
        records = [r for r in self.model.records if r.included]
        if not records:
            QMessageBox.warning(self, "No files", "No files are checked for inclusion.")
            return
        out_dir = self.output_folder
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please set an output folder first.")
            return
        open_viewer(
            records           = records,
            output_folder     = out_dir,
            show_masks        = self.chk_show_masks.isChecked(),
            show_metamasks    = self.chk_show_metamasks.isChecked(),
            preloaded_arrays  = self._img_cache,
        )

    def run_segmentation(self):
        """Load each included file, reorder axes, call cp_segment, save segment.txt."""
        from cp_segment import segment as cp_segment

        records = [r for r in self.model.records if r.included]
        if not records:
            QMessageBox.warning(self, "No files", "No files are checked for inclusion.")
            return
        out_dir = self.output_folder
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please set an output folder first.")
            return
        os.makedirs(out_dir, exist_ok=True)

        data_list_path = os.path.join(out_dir, "dataFiles.txt")
        if not os.path.exists(data_list_path):
            self.write_data_list()

        ch_spec     = self.combo_ch_spec.currentText()
        filter_type = self.combo_filter.currentText()
        filter_size = (self.spin_fx.value(), self.spin_fy.value(), self.spin_fz.value())

        # Guard: S2 is only valid for 4-channel images
        if ch_spec == "S2" and self.n_channels < 4:
            QMessageBox.critical(
                self, "Invalid channel spec",
                f"Channel spec 'S2' requires a 4-channel image, "
                f"but the loaded files have {self.n_channels} channel(s).\n\n"
                "Segmentation will not be performed."
            )
            return

        progress = QProgressDialog("Running segmentation…", "Cancel", 0, len(records), self)
        progress.setWindowTitle("Segmentation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        new_rows = []
        errors   = []
        for i, rec in enumerate(records):
            progress.setValue(i)
            progress.setLabelText(f"[{i+1}/{len(records)}] {rec.filename}")
            QApplication.processEvents()
            if progress.wasCanceled():
                break
            try:
                img = tf.imread(rec.path)
                img = self._reorder_to_zcyx(img, rec, rec.flip_xy)
                base = os.path.splitext(rec.filename)[0]
                mask_path, n_masks = cp_segment(
                    img,
                    channel_spec  = ch_spec,
                    base_name     = base,
                    output_folder = out_dir,
                    filter_type   = filter_type,
                    filter_size   = filter_size,
                    z_range       = (rec.zstrt, rec.znd),
                    downscale     = self.chk_downscale.isChecked(),
                )
                new_rows.append({
                    "Filename":     rec.filename,
                    "ChannelSpec":  ch_spec,
                    "FilterType":   filter_type,
                    "FilterSize":   str(filter_size),
                    "zstrt":        rec.zstrt,
                    "znd":          rec.znd,
                    "MaskFile":     os.path.basename(mask_path),
                    "N_Masks":      n_masks,
                    "Timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            except Exception as e:
                errors.append(f"{rec.filename}: {e}")

        progress.setValue(len(records))

        self.seg_results.extend(new_rows)
        seg_path = os.path.join(out_dir, "segment.txt")
        pd.DataFrame(self.seg_results).to_csv(seg_path, sep="\t", index=False)

        msg = f"Segmentation complete.\n{len(new_rows)} file(s) processed.\nResults: {seg_path}"
        if errors:
            msg += "\n\nErrors:\n" + "\n".join(errors)
        QMessageBox.information(self, "Done", msg)

    @staticmethod
    def _reorder_to_zcyx(img, rec, flip_xy=False):
        """Reorder img to (Z, C, Y, X) using the shape metadata in rec.
        If flip_xy is True, swap the Y and X axes after reordering."""
        shape = img.shape
        if img.ndim == 3:
            if flip_xy:
                return np.transpose(img, (0, 2, 1))   # (Z,X,Y) → (Z,Y,X)
            return img
        target = {"Z": rec.shape_z, "C": rec.shape_c,
                  "Y": rec.shape_y, "X": rec.shape_x}
        remaining = list(range(img.ndim))
        assigned  = {}
        for label in ("C", "Z", "Y", "X"):
            sz = target[label]
            matches = [ax for ax in remaining if shape[ax] == sz]
            if not matches:
                raise ValueError(
                    f"Cannot find axis for {label}={sz} in shape {shape}")
            assigned[label] = matches[0]
            remaining.remove(matches[0])
        if flip_xy:
            assigned["Y"], assigned["X"] = assigned["X"], assigned["Y"]
        order = [assigned["Z"], assigned["C"], assigned["Y"], assigned["X"]]
        return np.transpose(img, order)


    # ------------------------------------------------------------------
    # Measure tab
    # ------------------------------------------------------------------
    def _build_measure_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # ---- Image/Mask pairs table ----
        layout.addWidget(QLabel("<b>Image → Mask pairs</b>"))
        
        # Create table for pairs
        self.tbl_measure = QTableView()
        self.tbl_measure.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.tbl_measure.horizontalHeader().setStretchLastSection(True)
        self.tbl_measure.setAlternatingRowColors(True)
        layout.addWidget(self.tbl_measure)

        # Initialize model
        self.measure_pairs_model = MeasurePairsModel(self)
        self.tbl_measure.setModel(self.measure_pairs_model)

        # ---- Buttons ----
        btn_row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh pairs")
        btn_refresh.clicked.connect(self._refresh_measure_pairs)
        btn_row.addWidget(btn_refresh)
        btn_sel_all = QPushButton("Select all")
        btn_sel_all.clicked.connect(lambda: self._set_all_pairs(True))
        btn_row.addWidget(btn_sel_all)
        btn_sel_none = QPushButton("Select none")
        btn_sel_none.clicked.connect(lambda: self._set_all_pairs(False))
        btn_row.addWidget(btn_sel_none)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ---- Metrics ----
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)
        layout.addWidget(QLabel("<b>Metrics</b>"))
        metric_row = QHBoxLayout()
        self.chk_metrics = {}
        for m in ("mean", "max", "min", "std", "volume"):
            cb = QCheckBox(m)
            cb.setChecked(True)
            self.chk_metrics[m] = cb
            metric_row.addWidget(cb)
        metric_row.addStretch()
        layout.addLayout(metric_row)

        # ---- Run + status ----
        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)
        run_row = QHBoxLayout()
        btn_run_msr = QPushButton("Run Measurements")
        btn_run_msr.clicked.connect(self.run_measurements)
        run_row.addWidget(btn_run_msr)
        run_row.addStretch()
        layout.addLayout(run_row)

        self.lbl_msr_status = QLabel("")
        layout.addWidget(self.lbl_msr_status)
        layout.addStretch()
        return widget

    def _refresh_measure_pairs(self):
        """Populate the measure pairs table from data manager and segmentation results."""
        pairs = []
        
        # Get image files from data manager (included records)
        img_files = [(r.filename, r.path) for r in self.model.records if r.included]
        
        # Get mask files from seg_results and metamask_results
        mask_files = set()
        out_dir = self.output_folder
        masks_dir = os.path.join(out_dir, "Masks") if out_dir else ""
        metamask_dir = os.path.join(masks_dir, "Metamask") if masks_dir else ""
        
        for d in (masks_dir, metamask_dir):
            if d and os.path.isdir(d):
                for f in sorted(glob.glob(os.path.join(d, "*.tif"))):
                    mask_files.add((os.path.basename(f), f))
        
        # Create pairs (match by filename similarity if possible, otherwise all combinations)
        if img_files and mask_files:
            for img_name, img_path in img_files:
                # Find best matching mask (same base name)
                base = img_name.replace('.tif', '').replace('.tiff', '')
                best_mask = None
                for mask_name, mask_path in mask_files:
                    mask_base = mask_name.replace('_masks.tif', '').replace('_metamask.tif', '')
                    if base in mask_base or mask_base in base:
                        best_mask = (mask_name, mask_path)
                        break
                # If no match found, use first mask
                if best_mask is None:
                    best_mask = list(mask_files)[0]
                
                pairs.append({
                    'included': True,
                    'image_name': img_name,
                    'image_path': img_path,
                    'mask_name': best_mask[0],
                    'mask_path': best_mask[1],
                    'channel': 0,
                })
        
        self.measure_pairs_model.set_pairs(pairs)
        self.lbl_msr_status.setText(f"Loaded {len(pairs)} image/mask pair(s)")

    def _set_all_pairs(self, included):
        """Set all pairs to included or excluded."""
        self.measure_pairs_model.set_all_included(included)

    def run_measurements(self):
        from cp_measure import measure_file
        out_dir = self.output_folder
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please set an output folder first.")
            return
        
        # Get included pairs
        pairs = self.measure_pairs_model.get_included_pairs()
        if not pairs:
            QMessageBox.warning(self, "No pairs selected", 
                "No image/mask pairs are selected for measurement.\n"
                "Click 'Refresh pairs' to load pairs, then check the 'Include' column.")
            return
        
        metrics = [m for m, cb in self.chk_metrics.items() if cb.isChecked()]
        if not metrics:
            QMessageBox.warning(self, "No metrics", "Select at least one metric.")
            return
        
        msr_dir = os.path.join(out_dir, "Measurements")
        os.makedirs(msr_dir, exist_ok=True)
        
        total_files = 0
        total_labels = 0
        errors = []
        
        for pair in pairs:
            img_path = pair['image_path']
            mask_path = pair['mask_path']
            ch = pair['channel']
            
            try:
                out_path, df = measure_file(
                    img_path, mask_path,
                    channel_idx   = ch,
                    metrics       = metrics,
                    output_folder = msr_dir,
                )
                self.measure_df = pd.concat([self.measure_df, df], ignore_index=True)
                total_files += 1
                total_labels += len(df)
                print(f"[measure] {pair['image_name']} + {pair['mask_name']}: {len(df)} labels")
            except Exception as e:
                err_msg = f"{pair['image_name']}: {str(e)}"
                errors.append(err_msg)
                print(f"[measure] ERROR: {err_msg}")
        
        # Update status
        status_msg = f"Measured {total_labels} labels in {total_files} file(s)"
        if errors:
            status_msg += f", {len(errors)} error(s)"
        self.lbl_msr_status.setText(status_msg)
        
        # Refresh plot combos and show summary
        self._populate_plot_combos()
        if errors:
            QMessageBox.warning(self, "Completed with errors", 
                f"{status_msg}\n\nErrors:\n" + "\n".join(errors[:5]))
        else:
            QMessageBox.information(self, "Done", f"{status_msg}\nResults in: {msr_dir}")

    # ------------------------------------------------------------------
    # Plot tab
    # ------------------------------------------------------------------
    def _build_plot_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Plot type:"))
        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems(["Histogram", "Scatter", "Violin"])
        self.combo_plot_type.setFixedWidth(100)
        self.combo_plot_type.currentTextChanged.connect(self._on_plot_type_changed)
        ctrl.addWidget(self.combo_plot_type)

        ctrl.addWidget(QLabel("X axis:"))
        self.combo_plot_x = QComboBox()
        self.combo_plot_x.setFixedWidth(120)
        ctrl.addWidget(self.combo_plot_x)

        self.lbl_plot_y = QLabel("Y axis:")
        ctrl.addWidget(self.lbl_plot_y)
        self.combo_plot_y = QComboBox()
        self.combo_plot_y.setFixedWidth(120)
        ctrl.addWidget(self.combo_plot_y)

        ctrl.addWidget(QLabel("Group by:"))
        self.combo_plot_grp = QComboBox()
        self.combo_plot_grp.setFixedWidth(120)
        self.combo_plot_grp.addItem("(none)", userData=None)
        ctrl.addWidget(self.combo_plot_grp)

        btn_plot = QPushButton("Plot")
        btn_plot.clicked.connect(self.draw_plot)
        ctrl.addWidget(btn_plot)

        btn_load = QPushButton("Load measure.txt…")
        btn_load.clicked.connect(self.load_measure_file)
        ctrl.addWidget(btn_load)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig    = Figure(figsize=(8, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        return widget

    def _on_plot_type_changed(self, ptype):
        is_scatter = (ptype == "Scatter")
        self.lbl_plot_y.setVisible(is_scatter)
        self.combo_plot_y.setVisible(is_scatter)

    def _populate_plot_combos(self):
        df = self.measure_df
        if df.empty:
            return
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = list(df.columns)
        for combo in (self.combo_plot_x, self.combo_plot_y):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(num_cols)
            idx = combo.findText(cur)
            if idx >= 0: combo.setCurrentIndex(idx)
        cur_grp = self.combo_plot_grp.currentText()
        self.combo_plot_grp.clear()
        self.combo_plot_grp.addItem("(none)", userData=None)
        for c in cat_cols:
            self.combo_plot_grp.addItem(c, userData=c)
        idx = self.combo_plot_grp.findText(cur_grp)
        if idx >= 0: self.combo_plot_grp.setCurrentIndex(idx)

    def load_measure_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load measurement file", self.output_folder or "",
            "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep="\t")
            self.measure_df = pd.concat([self.measure_df, df], ignore_index=True)
            self._populate_plot_combos()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def draw_plot(self):
        df = self.measure_df
        if df.empty:
            QMessageBox.information(self, "No data",
                "Run measurements or load a measure.txt file first.")
            return
        ptype = self.combo_plot_type.currentText()
        xcol  = self.combo_plot_x.currentText()
        ycol  = self.combo_plot_y.currentText()
        grp   = self.combo_plot_grp.currentData()

        if xcol not in df.columns:
            QMessageBox.warning(self, "Bad column", f"Column '{xcol}' not in data."); return
        if ptype == "Scatter" and ycol not in df.columns:
            QMessageBox.warning(self, "Bad column", f"Column '{ycol}' not in data."); return

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        groups = df[grp].unique() if grp and grp in df.columns else [None]
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if ptype == "Histogram":
            for i, g in enumerate(groups):
                data = df[df[grp] == g][xcol].dropna() if g is not None else df[xcol].dropna()
                label = str(g) if g is not None else xcol
                ax.hist(data, bins=40, alpha=0.6, color=colors[i % len(colors)], label=label)
            ax.set_xlabel(xcol); ax.set_ylabel("Count")
            if grp: ax.legend(title=grp)

        elif ptype == "Scatter":
            for i, g in enumerate(groups):
                sub = df[df[grp] == g] if g is not None else df
                label = str(g) if g is not None else None
                ax.scatter(sub[xcol].dropna(), sub[ycol].dropna(),
                           s=8, alpha=0.5, color=colors[i % len(colors)], label=label)
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            if grp: ax.legend(title=grp)

        elif ptype == "Violin":
            if grp and grp in df.columns:
                group_data = [df[df[grp] == g][xcol].dropna().values for g in groups]
                vp = ax.violinplot(group_data, showmedians=True)
                ax.set_xticks(range(1, len(groups) + 1))
                ax.set_xticklabels([str(g) for g in groups], rotation=30, ha="right")
                ax.set_xlabel(grp)
            else:
                ax.violinplot(df[xcol].dropna().values, showmedians=True)
                ax.set_xticks([1]); ax.set_xticklabels([xcol])
            ax.set_ylabel(xcol)

        ax.set_title(f"{ptype}: {xcol}" + (f" vs {ycol}" if ptype == "Scatter" else ""))
        self.canvas.draw()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = CpManager()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
