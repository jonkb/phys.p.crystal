""" simulation_panel.py
Defines SimulationPanel: A widget for setting simulation options
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                             QDoubleSpinBox, QSpinBox, QGridLayout)
#from PyQt6.QtCore import Qt
from .collabsible_box import CollapsibleBox
from .scientific_spinbox import ScientificSpinBox

class SimulationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # --- 1. Time Vector Sub-Accordion ---
        time_accordion = CollapsibleBox("Time Vector")
        time_widget = QWidget()
        time_grid = QGridLayout(time_widget)
        time_grid.setContentsMargins(0, 0, 0, 0)
        
        # t1 (End Time)
        lbl_t1 = QLabel("End Time (t1):")
        #lbl_t1.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.t1_spin = QDoubleSpinBox()
        self.t1_spin.setRange(0.001, 1e6)
        self.t1_spin.setDecimals(3)
        self.t1_spin.setSingleStep(0.1)
        self.t1_spin.setValue(2.0)
        
        # Nt (Number of steps)
        lbl_nt = QLabel("Steps (Nt):")
        #lbl_nt.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.nt_spin = QSpinBox()
        self.nt_spin.setRange(1, 1000000)
        self.nt_spin.setSingleStep(10)
        self.nt_spin.setValue(50)
        
        time_grid.addWidget(lbl_t1, 0, 0)
        time_grid.addWidget(self.t1_spin, 0, 1)
        time_grid.addWidget(lbl_nt, 1, 0)
        time_grid.addWidget(self.nt_spin, 1, 1)
        time_grid.setColumnStretch(0, 0)
        time_grid.setColumnStretch(1, 1)
        
        time_accordion.addWidget(time_widget)
        layout.addWidget(time_accordion)


        # --- 2. Solver Options Sub-Accordion ---
        solver_accordion = CollapsibleBox("Solver Options")
        solver_widget = QWidget()
        solver_grid = QGridLayout(solver_widget)
        solver_grid.setContentsMargins(0, 0, 0, 0)
        
        # Tolerance
        lbl_tol = QLabel("Tolerance (tol):")
        #lbl_tol.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.tol_spin = ScientificSpinBox()
        self.tol_spin.setValue(1e-5)
        
        # Max Steps
        lbl_max_steps = QLabel("Max Steps:")
        #lbl_max_steps.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.max_steps_spin = ScientificSpinBox()
        self.max_steps_spin.setValue(1e5) 
        
        solver_grid.addWidget(lbl_tol, 0, 0)
        solver_grid.addWidget(self.tol_spin, 0, 1)
        solver_grid.addWidget(lbl_max_steps, 1, 0)
        solver_grid.addWidget(self.max_steps_spin, 1, 1)
        solver_grid.setColumnStretch(0, 0)
        solver_grid.setColumnStretch(1, 1)
        
        solver_accordion.addWidget(solver_widget)
        layout.addWidget(solver_accordion)
        
        # Push everything to the top
        #layout.addStretch()

    def get_simulation_data(self):
        """Packs the UI inputs into a clean dictionary for the exporter."""

        # Helper function to strip floating point artifacts
        def clean_float(val):
            return float(f"{val:.6g}")

        return {
            "time": {
                "t1": clean_float(self.t1_spin.value()),
                "Nt": self.nt_spin.value()
            },
            "options": {
                "tol": clean_float(self.tol_spin.value()),
                "max_steps": int(self.max_steps_spin.value())
            }
        }
