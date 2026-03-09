""" body_forces_panel.py
Defines BodyForcesPanel: A widget for editing body forces

Currently just atom_mass is contained here... under the justification that the
    inertial d'Alembert forces are often considered body forces
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QDoubleSpinBox)

class BodyForcesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Setup UI
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Atom Mass
        mass_layout = QHBoxLayout()
        mass_layout.addWidget(QLabel("Atomic mass:"))
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(1e-6, 1e6)
        self.mass_spin.setValue(1.0)
        mass_layout.addWidget(self.mass_spin)
        main_layout.addLayout(mass_layout)

        # Gravity (TODO)
    
    def get_mass(self):
        """Return atom_mass"""
        return self.mass_spin.value()