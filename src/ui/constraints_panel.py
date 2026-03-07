""" constraints_panel.py
Defines ConstraintsPanel: A widget for defining constraints
"""

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QCheckBox, QMessageBox
from .select_regions_panel import SelectRegionsPanel

class ConstraintsPanel(SelectRegionsPanel):
    def __init__(self, initial_domain=None, parent=None):
        super().__init__(item_name="Constraint", initial_domain=initial_domain, 
            parent=parent)

    def setup_custom_ui(self, layout):
        dof_layout = QHBoxLayout()
        dof_layout.addWidget(QLabel("<b>Constrain DOF:</b>"))
        
        self.cbs = {}
        for axis in ['x', 'y', 'z']:
            cb = QCheckBox(axis.upper())
            self.cbs[axis] = cb
            dof_layout.addWidget(cb)
            
        dof_layout.addStretch()
        layout.addLayout(dof_layout)

    def clear_custom_ui(self):
        for cb in self.cbs.values():
            cb.setChecked(False)

    def load_custom_data(self, data):
        for i, axis in enumerate(['x', 'y', 'z']):
            self.cbs[axis].setChecked(data[i])

    def get_custom_data(self):
        # DOF constrained boolean array [x,y,z]
        return [cb.isChecked() for cb in self.cbs.values()]

    def validate_custom_data(self, data):
        if not any(data):
            QMessageBox.warning(self, "Invalid Input", "Select at least one DOF to constrain.")
            return False
        return True