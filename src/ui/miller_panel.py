""" miller_panel.py
Defines MillerPanel: A widget for specifying miller indices
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton, QSpinBox
from PyQt6.QtCore import pyqtSignal

class MillerPanel(QWidget):
    # Define custom signals to tell the main window when things change
    plane_visibility_toggled = pyqtSignal(bool)
    plane_indices_changed = pyqtSignal()
    dir_visibility_toggled = pyqtSignal(bool)
    dir_indices_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lattice_type = 'fcc' # Default
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setColumnStretch(0, 0)  # Button column (fixed width)
        layout.setColumnStretch(1, 1)  # Indices column (expandable)
        
        # ===== ROW 0: PLANE =====
        self.show_plane_cb = QPushButton("Show Plane")
        self.show_plane_cb.setCheckable(True)
        self.show_plane_cb.setMinimumWidth(100)
        self.show_plane_cb.toggled.connect(self.plane_visibility_toggled.emit)
        layout.addWidget(self.show_plane_cb, 0, 0)
        
        # FCC plane indices
        self.plane_fcc_widget = QWidget()
        plane_fcc_layout = QGridLayout(self.plane_fcc_widget)
        plane_fcc_layout.setContentsMargins(0, 0, 0, 0)
        plane_fcc_layout.setSpacing(3)
        
        self.plane_h_spin = self._create_spinbox(1, self.plane_indices_changed.emit)
        self.plane_k_spin = self._create_spinbox(0, self.plane_indices_changed.emit)
        self.plane_l_spin = self._create_spinbox(0, self.plane_indices_changed.emit)
        
        plane_fcc_layout.addWidget(self.plane_h_spin, 0, 0)
        plane_fcc_layout.addWidget(self.plane_k_spin, 0, 1)
        plane_fcc_layout.addWidget(self.plane_l_spin, 0, 2)
        layout.addWidget(self.plane_fcc_widget, 0, 1)
        
        # HCP plane indices
        self.plane_hcp_widget = QWidget()
        plane_hcp_layout = QGridLayout(self.plane_hcp_widget)
        plane_hcp_layout.setContentsMargins(0, 0, 0, 0)
        plane_hcp_layout.setSpacing(3)
        
        self.plane_h_b_spin = self._create_spinbox(1, self._update_plane_hcp_i)
        self.plane_k_b_spin = self._create_spinbox(0, self._update_plane_hcp_i)
        self.plane_i_b_spin = self._create_spinbox(-1, None)
        self.plane_i_b_spin.setEnabled(False) # Auto-calculated
        self.plane_l_b_spin = self._create_spinbox(0, self.plane_indices_changed.emit)
        
        plane_hcp_layout.addWidget(self.plane_h_b_spin, 0, 0)
        plane_hcp_layout.addWidget(self.plane_k_b_spin, 0, 1)
        plane_hcp_layout.addWidget(self.plane_i_b_spin, 0, 2)
        plane_hcp_layout.addWidget(self.plane_l_b_spin, 0, 3)
        layout.addWidget(self.plane_hcp_widget, 0, 1)
        self.plane_hcp_widget.setVisible(False)
        
        # ===== ROW 1: DIRECTION =====
        self.show_dir_cb = QPushButton("Show Direction")
        self.show_dir_cb.setCheckable(True)
        self.show_dir_cb.setMinimumWidth(100)
        self.show_dir_cb.toggled.connect(self.dir_visibility_toggled.emit)
        layout.addWidget(self.show_dir_cb, 1, 0)
        
        # FCC direction indices
        self.dir_fcc_widget = QWidget()
        dir_fcc_layout = QGridLayout(self.dir_fcc_widget)
        dir_fcc_layout.setContentsMargins(0, 0, 0, 0)
        dir_fcc_layout.setSpacing(3)
        
        self.dir_h_spin = self._create_spinbox(1, self.dir_indices_changed.emit)
        self.dir_k_spin = self._create_spinbox(0, self.dir_indices_changed.emit)
        self.dir_l_spin = self._create_spinbox(0, self.dir_indices_changed.emit)
        
        dir_fcc_layout.addWidget(self.dir_h_spin, 0, 0)
        dir_fcc_layout.addWidget(self.dir_k_spin, 0, 1)
        dir_fcc_layout.addWidget(self.dir_l_spin, 0, 2)
        layout.addWidget(self.dir_fcc_widget, 1, 1)
        
        # HCP direction indices
        self.dir_hcp_widget = QWidget()
        dir_hcp_layout = QGridLayout(self.dir_hcp_widget)
        dir_hcp_layout.setContentsMargins(0, 0, 0, 0)
        dir_hcp_layout.setSpacing(3)
        
        self.dir_h_b_spin = self._create_spinbox(1, self._update_dir_hcp_i)
        self.dir_k_b_spin = self._create_spinbox(0, self._update_dir_hcp_i)
        self.dir_i_b_spin = self._create_spinbox(-1, None)
        self.dir_i_b_spin.setEnabled(False) # Auto-calculated
        self.dir_l_b_spin = self._create_spinbox(0, self.dir_indices_changed.emit)
        
        dir_hcp_layout.addWidget(self.dir_h_b_spin, 0, 0)
        dir_hcp_layout.addWidget(self.dir_k_b_spin, 0, 1)
        dir_hcp_layout.addWidget(self.dir_i_b_spin, 0, 2)
        dir_hcp_layout.addWidget(self.dir_l_b_spin, 0, 3)
        layout.addWidget(self.dir_hcp_widget, 1, 1)
        self.dir_hcp_widget.setVisible(False)
        
        # Initialize the i-index
        self._update_dir_hcp_i()

    def _create_spinbox(self, default_val, connect_func):
        """Helper to reduce spinbox boilerplate."""
        spin = QSpinBox()
        spin.setRange(-10, 10)
        spin.setValue(default_val)
        if connect_func:
            spin.valueChanged.connect(connect_func)
        return spin

    def _update_plane_hcp_i(self):
        """Auto-sync HCP plane i-index as -(h+k) and trigger update."""
        self.plane_i_b_spin.blockSignals(True)
        self.plane_i_b_spin.setValue(-(self.plane_h_b_spin.value() + self.plane_k_b_spin.value()))
        self.plane_i_b_spin.blockSignals(False)
        self.plane_indices_changed.emit()

    def _update_dir_hcp_i(self):
        """Auto-sync HCP direction i-index as -(h+k) and trigger update."""
        self.dir_i_b_spin.blockSignals(True)
        self.dir_i_b_spin.setValue(-(self.dir_h_b_spin.value() + self.dir_k_b_spin.value()))
        self.dir_i_b_spin.blockSignals(False)
        self.dir_indices_changed.emit()

    def set_lattice_type(self, lattice_type):
        """Swap between showing FCC (3 indices) and HCP (4 indices) inputs."""
        self.lattice_type = lattice_type
        is_hcp = lattice_type == 'hcp'
        
        self.plane_fcc_widget.setVisible(not is_hcp)
        self.plane_hcp_widget.setVisible(is_hcp)
        self.dir_fcc_widget.setVisible(not is_hcp)
        self.dir_hcp_widget.setVisible(is_hcp)
        
        # Reset visibility buttons if we switch to random
        if lattice_type == 'random':
            self.show_plane_cb.setChecked(False)
            self.show_dir_cb.setChecked(False)

    def get_plane_indices(self):
        """Return indices based on current lattice type."""
        if self.lattice_type == 'fcc':
            return np.array([self.plane_h_spin.value(), self.plane_k_spin.value(), self.plane_l_spin.value()])
        else:
            return np.array([self.plane_h_b_spin.value(), self.plane_k_b_spin.value(), 
                             self.plane_i_b_spin.value(), self.plane_l_b_spin.value()])

    def get_dir_indices(self):
        """Return indices based on current lattice type."""
        if self.lattice_type == 'fcc':
            return np.array([self.dir_h_spin.value(), self.dir_k_spin.value(), self.dir_l_spin.value()])
        else:
            return np.array([self.dir_h_b_spin.value(), self.dir_k_b_spin.value(), 
                             self.dir_i_b_spin.value(), self.dir_l_b_spin.value()])
    
    def set_plane_indices(self, indices):
        """Set the plane spinboxes from an array of indices."""
        if self.lattice_type == 'fcc' and len(indices) >= 3:
            # Block signals so we don't trigger multiple UI updates
            self.plane_h_spin.blockSignals(True)
            self.plane_k_spin.blockSignals(True)
            
            self.plane_h_spin.setValue(int(indices[0]))
            self.plane_k_spin.setValue(int(indices[1]))
            
            self.plane_h_spin.blockSignals(False)
            self.plane_k_spin.blockSignals(False)
            
            # Setting the final spinbox will automatically fire plane_indices_changed
            self.plane_l_spin.setValue(int(indices[2]))
            
        elif self.lattice_type == 'hcp' and len(indices) >= 3:
            # HCP arrays are saved with 4 indices: [h, k, i, l]
            l_val = int(indices[3]) if len(indices) >= 4 else int(indices[2])
            
            self.plane_h_b_spin.blockSignals(True)
            self.plane_k_b_spin.blockSignals(True)
            self.plane_l_b_spin.blockSignals(True)
            
            self.plane_h_b_spin.setValue(int(indices[0]))
            self.plane_k_b_spin.setValue(int(indices[1]))
            self.plane_l_b_spin.setValue(l_val)
            
            self.plane_h_b_spin.blockSignals(False)
            self.plane_k_b_spin.blockSignals(False)
            self.plane_l_b_spin.blockSignals(False)
            
            # This helper auto-calculates 'i' and emits the plane_indices_changed signal
            self._update_plane_hcp_i()

    def set_dir_indices(self, indices):
        """Set the direction spinboxes from an array of indices."""
        if self.lattice_type == 'fcc' and len(indices) >= 3:
            self.dir_h_spin.blockSignals(True)
            self.dir_k_spin.blockSignals(True)
            
            self.dir_h_spin.setValue(int(indices[0]))
            self.dir_k_spin.setValue(int(indices[1]))
            
            self.dir_h_spin.blockSignals(False)
            self.dir_k_spin.blockSignals(False)
            
            # Setting the final spinbox will automatically fire dir_indices_changed
            self.dir_l_spin.setValue(int(indices[2]))
            
        elif self.lattice_type == 'hcp' and len(indices) >= 3:
            # HCP arrays are saved with 4 indices: [h, k, i, l]
            l_val = int(indices[3]) if len(indices) >= 4 else int(indices[2])
            
            self.dir_h_b_spin.blockSignals(True)
            self.dir_k_b_spin.blockSignals(True)
            self.dir_l_b_spin.blockSignals(True)
            
            self.dir_h_b_spin.setValue(int(indices[0]))
            self.dir_k_b_spin.setValue(int(indices[1]))
            self.dir_l_b_spin.setValue(l_val)
            
            self.dir_h_b_spin.blockSignals(False)
            self.dir_k_b_spin.blockSignals(False)
            self.dir_l_b_spin.blockSignals(False)
            
            # This helper auto-calculates 'i' and emits the dir_indices_changed signal
            self._update_dir_hcp_i()