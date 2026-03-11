""" forces_panel.py
Defines ForcesPanel: A widget for defining forces
"""

from PyQt6.QtWidgets import (QVBoxLayout, QGridLayout, QLabel, QWidget,
    QMessageBox, QLineEdit
)
from .select_regions_panel import SelectRegionsPanel
import sympy as sp

class ForcesPanel(SelectRegionsPanel):
    fi_lbls = ['f_x', 'f_y', 'f_z']

    def __init__(self, initial_domain=None, parent=None):
        super().__init__(item_name="Force", initial_domain=initial_domain, parent=parent)

    def setup_custom_ui(self, layout):
        """Inject the force component inputs into the base layout."""

        vec_layout = QVBoxLayout()
        vec_layout.addWidget(QLabel("<b>Force Components:</b> (SymPy f_i(t))"))

        vec_wig = QWidget()
        vec_grid = QGridLayout(vec_wig)
        vec_grid.setContentsMargins(0, 0, 0, 0)
        
        # Generate inputs for each force component
        self.inputs = {}
        for i, fi_lbl in enumerate(self.fi_lbls):
            line_edit = QLineEdit("0.0")
            line_edit.setPlaceholderText(f"e.g. 10*sin(t)")
            self.inputs[fi_lbl] = line_edit
            lbl = QLabel(f"{fi_lbl}:")
            vec_grid.addWidget(lbl, i, 0)
            vec_grid.addWidget(line_edit, i, 1)
            
        vec_layout.addWidget(vec_wig)
        layout.addLayout(vec_layout)

    def clear_custom_ui(self):
        for line_edit in self.inputs.values():
            line_edit.setText("0.0")

    def load_custom_data(self, data):
        for i, fi_lbl in enumerate(self.fi_lbls):
            self.inputs[fi_lbl].setText(data[i])

    def get_custom_data(self):
        # Force vector
        return [self.inputs[lbl].text().strip() for lbl in self.fi_lbls]

    def validate_custom_data(self, data):
        # Validate that SymPy can parse all three expressions
        exprs = []
        for i, expr_str in enumerate(data):
            try:
                exprs.append(sp.sympify(expr_str))
            except Exception as e:
                QMessageBox.warning(self, "Invalid Math", 
                                    f"Could not parse the {self.fi_lbls[i]} expression:\n{e}")
                return False
        # Check if they're all zero
        is_zero = [fi.simplify().is_zero for fi in exprs]
        if all(is_zero):
            QMessageBox.warning(self, "Invalid Input",
                "Force vector cannot be zero in all directions.")
            return False
        return True