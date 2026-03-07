""" interatomic_panel.py
Defines InteratomicPanel: A widget for defining the interatomic potential
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, 
                             QLabel, QDoubleSpinBox, QStackedWidget, 
                             QGridLayout)

from numpy import sqrt

class InteratomicPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Dropdown for selecting the potential type
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Lennard-Jones", "Morse"])
        model_layout.addWidget(self.type_combo)
        layout.addLayout(model_layout)

        # --- Create Pages ---
        self.lj_page = self.create_lj_page()
        self.morse_page = self.create_morse_page()

        # Add to main layout
        layout.addWidget(self.lj_page)
        layout.addWidget(self.morse_page)        
        
        # 3. Connect dropdown to stacked widget
        self.type_combo.currentIndexChanged.connect(self.toggle_pages)
        # Default to page 0
        self.toggle_pages(0)

    def create_lj_page(self):
        """Creates the page containing Lennard-Jones parameters."""
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        
        # Epsilon (Depth)
        lbl_eps = QLabel("ε (Depth):")
        self.epsilon_spin = self.create_spinbox(decimals=4)
        self.epsilon_spin.setValue(0.1) # Default
        
        # Sigma (r0)
        lbl_sig = QLabel("σ (r0):")
        self.sigma_spin = self.create_spinbox(decimals=4)
        self.sigma_spin.setValue(0.89 / sqrt(2)) # Default
        
        grid.addWidget(lbl_eps, 0, 0)
        grid.addWidget(self.epsilon_spin, 0, 1)
        grid.addWidget(lbl_sig, 1, 0)
        grid.addWidget(self.sigma_spin, 1, 1)
        
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        return page

    def create_morse_page(self):
        """Creates the page containing Morse parameters."""
        page = QWidget()
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        
        # De (Depth)
        lbl_de = QLabel("De (Depth):")
        self.de_spin = self.create_spinbox(decimals=4)
        self.de_spin.setValue(0.1) # Default
        
        # a (Slope/Width)
        lbl_a = QLabel("a (Slope):")
        self.a_spin = self.create_spinbox(decimals=4)
        self.a_spin.setValue(5.0)
        
        # req (Equilibrium distance)
        lbl_req = QLabel("req:")
        self.req_spin = self.create_spinbox(decimals=4)
        self.req_spin.setValue(1.0/sqrt(2))
        
        grid.addWidget(lbl_de, 0, 0)
        grid.addWidget(self.de_spin, 0, 1)
        grid.addWidget(lbl_a, 1, 0)
        grid.addWidget(self.a_spin, 1, 1)
        grid.addWidget(lbl_req, 2, 0)
        grid.addWidget(self.req_spin, 2, 1)
        
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        return page
    
    def toggle_pages(self, index):
        """Show the selected page and completely hide the other."""
        if index == 0:  # Lennard-Jones
            self.lj_page.setVisible(True)
            self.morse_page.setVisible(False)
        elif index == 1:  # Morse
            self.lj_page.setVisible(False)
            self.morse_page.setVisible(True)

    def create_spinbox(self, decimals=3, max_val=1e5):
        """Helper to create uniform double spinboxes."""
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(0.0, max_val) # Assuming standard positive inputs for depth/distance
        spin.setSingleStep(0.1)
        return spin

    # --- Data Extraction & Loading ---
    
    def get_potential_data(self):
        """Returns a dictionary of the currently selected potential and its params."""
        pot_type = self.type_combo.currentText()
        
        if pot_type == "Lennard-Jones":
            return {
                "type": "Lennard-Jones",
                "epsilon_depth": self.epsilon_spin.value(),
                "sigma_r0": self.sigma_spin.value()
            }
        elif pot_type == "Morse":
            return {
                "type": "Morse",
                "De_depth": self.de_spin.value(),
                "a_slope": self.a_spin.value(),
                "req": self.req_spin.value()
            }

    def set_potential_data(self, data):
        """Populates the UI from a loaded dictionary."""
        pot_type = data.get("type", "Lennard-Jones")
        self.type_combo.setCurrentText(pot_type)
        
        if pot_type == "Lennard-Jones":
            self.epsilon_spin.setValue(data.get("epsilon_depth", 0.0))
            self.sigma_spin.setValue(data.get("sigma_r0", 0.0))
        elif pot_type == "Morse":
            self.de_spin.setValue(data.get("De_depth", 0.0))
            self.a_spin.setValue(data.get("a_slope", 0.0))
            self.req_spin.setValue(data.get("req", 0.0))