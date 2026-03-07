"""scientific_spinbox.py
A subclass of QDoubleSpinBox that uses scientific notation
"""

from PyQt6.QtWidgets import QDoubleSpinBox
from PyQt6.QtGui import QValidator

class ScientificSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(-1e100)
        self.setMaximum(1e100)
        self.setDecimals(1000) # Required to prevent arbitrary rounding

    def validate(self, text, pos):
        """Allows intermediate typing states like '1e-' before they become valid floats."""
        text = text.replace(",", ".")
        if text in ["", "-", "+"] or text.lower().endswith("e") or text.lower().endswith("e-") or text.lower().endswith("e+"):
            return QValidator.State.Intermediate, text, pos
        try:
            float(text)
            return QValidator.State.Acceptable, text, pos
        except ValueError:
            return QValidator.State.Invalid, text, pos

    def valueFromText(self, text):
        """Converts the typed scientific string back into a Python float."""
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return self.value()

    def textFromValue(self, value):
        """Formats the float into a scientific string (e.g., 1e-05)."""
        # :g uses scientific notation automatically for very large/small numbers
        # If you strictly want scientific everywhere, use "{:.2e}".format(value)
        return f"{value:.8g}" 

    def stepBy(self, steps):
        """Overrides the up/down arrows to multiply/divide by 10."""
        value = self.value()
        if value == 0:
            self.setValue(1e-5 if steps > 0 else -1e-5)
        else:
            self.setValue(value * (10.0 ** steps))
