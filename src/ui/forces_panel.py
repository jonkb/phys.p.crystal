""" forces_panel.py
Defines ForcesPanel: A widget for defining forces
"""

from PyQt6.QtWidgets import (QVBoxLayout, QGridLayout, QLabel, QWidget,
    QDoubleSpinBox, QMessageBox
)
from .select_regions_panel import SelectRegionsPanel

class ForcesPanel(SelectRegionsPanel):
    def __init__(self, initial_domain=None, parent=None):
        super().__init__(item_name="Force", initial_domain=initial_domain, parent=parent)

    def setup_custom_ui(self, layout):
        """Inject the 3 vector spinboxes into the base layout."""

        vec_layout = QVBoxLayout()
        vec_layout.addWidget(QLabel("<b>Vector (x, y, z):</b>"))

        vec_wig = QWidget()
        vec_grid = QGridLayout(vec_wig)
        vec_grid.setContentsMargins(0, 0, 0, 0)
        """
        # Specify that the first column shrinks and the next two share evenly
        vec_grid.setColumnStretch(0, 0)
        vec_grid.setColumnStretch(1, 1)
        vec_grid.setColumnStretch(2, 1)
        """
        
        self.spins = {}
        for i, axis in enumerate(['x', 'y', 'z']):
            spin = QDoubleSpinBox()
            spin.setRange(-1e6, 1e6) # Allow large forces
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            self.spins[axis] = spin
            lbl = QLabel(f"{axis.upper()}:")
            vec_grid.addWidget(lbl, i, 0)
            vec_grid.addWidget(spin, i, 1)
            
        vec_layout.addWidget(vec_wig)
        layout.addLayout(vec_layout)

    def clear_custom_ui(self):
        for spin in self.spins.values():
            spin.setValue(0.0)

    def load_custom_data(self, data):
        for i, axis in enumerate(['x', 'y', 'z']):
            self.spins[axis].setValue(data[i])

    def get_custom_data(self):
        # Force vector
        return [spin.value() for spin in self.spins.values()]

    def validate_custom_data(self, data):
        # Optional: ensure it's not a zero vector
        if not any(data):
            QMessageBox.warning(self, "Invalid Input", 
                "Force vector cannot be zero in all directions.")
            return False
        return True