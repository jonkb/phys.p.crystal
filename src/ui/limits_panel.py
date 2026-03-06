""" limits_panel.py
Defines LimitsPanel: A widget for selecting domain limits
"""

from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QDoubleSpinBox
from PyQt6.QtCore import pyqtSignal

class LimitsPanel(QWidget):
    # Define a custom signal that emits a list of tuples: [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    limits_changed = pyqtSignal(list)

    def __init__(self, initial_limits=None, parent=None):
        super().__init__(parent)
        
        # Fallback limits if none are provided
        if initial_limits is None:
            initial_limits = [(-10, 10), (-10, 10), (-10, 10)]
            
        self.limits = initial_limits
        self.setup_ui()

    def setup_ui(self):
        # We use 'self' as the parent layout because this IS the widget
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        # Specify that the first column shrinks and the next two share evenly
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        
        # Headers
        layout.addWidget(QLabel("Axis"), 0, 0)
        layout.addWidget(QLabel("Min"), 0, 1)
        layout.addWidget(QLabel("Max"), 0, 2)

        self.spins = []
        axes = ["X:", "Y:", "Z:"]
        
        # Generate the spinboxes programmatically to save space
        for i, axis_name in enumerate(axes):
            layout.addWidget(QLabel(axis_name), i+1, 0)
            
            # Min SpinBox
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-100, 100)
            min_spin.setValue(self.limits[i][0])
            min_spin.valueChanged.connect(self.emit_limits)
            layout.addWidget(min_spin, i+1, 1)
            
            # Max SpinBox
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-100, 100)
            max_spin.setValue(self.limits[i][1])
            max_spin.valueChanged.connect(self.emit_limits)
            layout.addWidget(max_spin, i+1, 2)
            
            # Save references so we can read their values later
            self.spins.append((min_spin, max_spin))

    def emit_limits(self):
        """Read all current spinbox values and emit them to anyone listening."""
        new_limits = [
            (self.spins[0][0].value(), self.spins[0][1].value()), # X
            (self.spins[1][0].value(), self.spins[1][1].value()), # Y
            (self.spins[2][0].value(), self.spins[2][1].value())  # Z
        ]
        # Broadcast the signal
        self.limits_changed.emit(new_limits)

    def set_limits(self, new_limits):
        """Update the spinboxes with new limits programmatically.
        new_limits (list): A list of tuples/lists [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        """
        if new_limits is None or len(new_limits) != 3:
            return

        # Update internal state
        self.limits = new_limits

        # Iterate through the X, Y, and Z spinbox pairs
        for i in range(3):
            min_val, max_val = new_limits[i]
            min_spin, max_spin = self.spins[i]
            
            # Block signals to prevent 6 separate 'limits_changed' emissions
            min_spin.blockSignals(True)
            max_spin.blockSignals(True)
            
            min_spin.setValue(float(min_val))
            max_spin.setValue(float(max_val))
            
            # Unblock signals
            min_spin.blockSignals(False)
            max_spin.blockSignals(False)
