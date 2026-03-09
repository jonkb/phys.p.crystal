""" lattice_panel.py
Defines LatticePanel: A widget for defining the placement of atoms in a lattice
Three modes: Random, FCC, HCP
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton)
from PyQt6.QtCore import pyqtSignal

class LatticePanel(QWidget):
    # Signals to tell the main window when it needs to update the 3D graph
    method_changed = pyqtSignal(str)
    randomize_requested = pyqtSignal()
    params_changed = pyqtSignal() 
    # Display names & internal names
    method_map = {"Random": "random", "FCC": "fcc", "HCP": "hcp"}
    inv_method_map = {v: k for k, v in method_map.items()}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_method = 'random'
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- Method Dropdown ---
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Lattice:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "FCC", "HCP"])
        self.method_combo.currentTextChanged.connect(self._on_combo_changed)
        method_layout.addWidget(self.method_combo)
        main_layout.addLayout(method_layout)

        # --- Random Controls ---
        self.random_widget = QWidget()
        random_layout = QGridLayout(self.random_widget)
        random_layout.setContentsMargins(0, 0, 0, 0)
        random_layout.setSpacing(5)
        
        random_layout.addWidget(QLabel("Number of Points:"), 0, 0)
        self.N_spin = QSpinBox()
        self.N_spin.setRange(1, 100000)
        self.N_spin.setValue(100)
        self.N_spin.setSingleStep(10)
        random_layout.addWidget(self.N_spin, 0, 1)

        self.shuffle_btn = QPushButton("Randomize Positions")
        self.shuffle_btn.setFixedHeight(40)
        self.shuffle_btn.clicked.connect(self.randomize_requested.emit)
        random_layout.addWidget(self.shuffle_btn, 1, 0, 1, 2)
        
        main_layout.addWidget(self.random_widget)

        # --- Crystal Controls ---
        self.crystal_widget = QWidget()
        crystal_layout = QGridLayout(self.crystal_widget)
        crystal_layout.setContentsMargins(0, 0, 0, 0)
        crystal_layout.setSpacing(5)

        crystal_layout.addWidget(QLabel("Lattice Parameters:"), 0, 0, 1, 2)
        
        self.a_spin = self._create_double_spinbox(1.0)
        self.b_spin = self._create_double_spinbox(1.0)
        self.c_spin = self._create_double_spinbox(1.0)
        crystal_layout.addWidget(QLabel("a:"), 1, 0)
        crystal_layout.addWidget(self.a_spin, 1, 1)
        crystal_layout.addWidget(QLabel("b:"), 2, 0)
        crystal_layout.addWidget(self.b_spin, 2, 1)
        crystal_layout.addWidget(QLabel("c:"), 3, 0)
        crystal_layout.addWidget(self.c_spin, 3, 1)

        crystal_layout.addWidget(QLabel("Euler Angles (°):"), 4, 0, 1, 2)
        self.alpha_spin = self._create_double_spinbox(0.0, -180, 180, 1.0)
        self.beta_spin = self._create_double_spinbox(0.0, -180, 180, 1.0)
        self.gamma_spin = self._create_double_spinbox(0.0, -180, 180, 1.0)
        crystal_layout.addWidget(QLabel("α:"), 5, 0)
        crystal_layout.addWidget(self.alpha_spin, 5, 1)
        crystal_layout.addWidget(QLabel("β:"), 6, 0)
        crystal_layout.addWidget(self.beta_spin, 6, 1)
        crystal_layout.addWidget(QLabel("γ:"), 7, 0)
        crystal_layout.addWidget(self.gamma_spin, 7, 1)

        main_layout.addWidget(self.crystal_widget)
        self.crystal_widget.setVisible(False)

    def _create_double_spinbox(self, default, min_val=0.1, max_val=10.0, step=0.1):
        """Helper to reduce boilerplate for the 6 double spinboxes."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self.params_changed.emit)
        return spin

    def _on_combo_changed(self, text):
        """Handle internal visibility and tell the parent the method changed."""
        self.current_method = self.method_map.get(text, "random")
        
        show_crystal = self.current_method in ['fcc', 'hcp']
        self.crystal_widget.setVisible(show_crystal)
        self.random_widget.setVisible(not show_crystal)
        
        self.method_changed.emit(self.current_method)

    # --- Data Accessors for the Main Window ---
    def get_method(self): 
        return self.current_method
    def get_point_count(self): 
        return self.N_spin.value()
    def get_lattice_params(self): 
        return [self.a_spin.value(), self.b_spin.value(), self.c_spin.value()]
    def get_euler_angles(self): 
        return [self.alpha_spin.value(), self.beta_spin.value(), self.gamma_spin.value()]

    # --- Setters ---
    def set_method(self, method_name):
        """Sets the placement method dropdown."""
        method_displayname = self.inv_method_map[method_name]
        index = self.method_combo.findText(method_displayname)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        else:
            print("Lattice placement method not found. Falling back to random.")
            self.method_combo.setCurrentIndex(0)

    def set_point_count(self, n):
        """Sets the number of points for random placement."""
        # Assuming you have a spinbox for N named 'N_spin'
        self.N_spin.setValue(n)

    def set_lattice_params(self, prms):
        """Sets the lattice parameters [a, b, c]."""
        # Assuming you have spinboxes for a, b, c
        if len(prms) >= 3:
            self.a_spin.setValue(float(prms[0]))
            self.b_spin.setValue(float(prms[1]))
            self.c_spin.setValue(float(prms[2]))

    def set_euler_angles(self, angles):
        """Sets the Euler rotation angles [phi, theta, psi]."""
        # Assuming you have spinboxes for euler angles
        if len(angles) >= 3:
            self.alpha_spin.setValue(float(angles[0]))
            self.beta_spin.setValue(float(angles[1]))
            self.gamma_spin.setValue(float(angles[2]))
