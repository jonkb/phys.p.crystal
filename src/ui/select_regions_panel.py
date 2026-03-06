""" select_regions_panel.py
Defines a parent class for panels that select regions of points
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QLineEdit, 
                             QPushButton, QMessageBox)
from PyQt6.QtCore import pyqtSignal
from .limits_panel import LimitsPanel

class SelectRegionsPanel(QWidget):
    """Base class for panels that apply settings to a specific 3D subregion."""
    
    # Generic signals
    data_changed = pyqtSignal(dict)
    active_region_changed = pyqtSignal(list)

    def __init__(self, item_name="Item", initial_domain=None, parent=None):
        super().__init__(parent)
        self.item_name = item_name
        self.items_dict = {}
        if initial_domain is None:
            # Default limits for new constraints based on the graph size
            self.default_limits = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
        else:
            self.default_limits = initial_domain
        self.current_limits = self.default_limits.copy()
        
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # 1. Selection Dropdown
        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("Editing:"))
        self.combo = QComboBox()
        self.combo.addItem(f"+ New {self.item_name}")
        self.combo.currentTextChanged.connect(self.on_selection_changed)
        select_layout.addWidget(self.combo)
        main_layout.addLayout(select_layout)

        # 2. Name Input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(f"e.g., Top {self.item_name}")
        name_layout.addWidget(self.name_input)
        main_layout.addLayout(name_layout)

        # 3. Subdomain Limits
        main_layout.addWidget(QLabel("<b>Affected Region:</b>"))
        self.limits_widget = LimitsPanel(initial_limits=self.default_limits)
        self.limits_widget.limits_changed.connect(self.update_current_limits)
        main_layout.addWidget(self.limits_widget)

        # 4. CUSTOM UI HOOK (Subclasses will inject their UI here)
        self.setup_custom_ui(main_layout)

        # 5. Action Buttons
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton(f"Save New {self.item_name}")
        self.save_btn.clicked.connect(self.save_item)
        
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.setStyleSheet("color: red;")
        self.delete_btn.setEnabled(False)
        self.delete_btn.clicked.connect(self.delete_item)
        
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.delete_btn)
        main_layout.addLayout(btn_layout)

    def update_current_limits(self, new_limits):
        self.current_limits = new_limits
        self.active_region_changed.emit(self.current_limits)

    def on_selection_changed(self, selection):
        if selection.startswith("+ New") or not selection:
            self.name_input.setText("")
            self.name_input.setEnabled(True)
            self.clear_custom_ui()
            self.delete_btn.setEnabled(False)
            self.save_btn.setText(f"Save New {self.item_name}")
        elif selection in self.items_dict:
            data = self.items_dict[selection]
            self.name_input.setText(selection)
            self.name_input.setEnabled(False)
            
            self.limits_widget.set_limits(data['limits'])
            self.current_limits = data['limits']
            
            self.load_custom_data(data['payload'])
            self.delete_btn.setEnabled(True)
            self.save_btn.setText(f"Update {self.item_name}")
            
        self.active_region_changed.emit(self.current_limits)

    def save_item(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Input", "Please provide a name.")
            return

        payload = self.get_custom_data()
        if not self.validate_custom_data(payload):
            return # Subclass handles the warning message

        is_new = name not in self.items_dict
        self.items_dict[name] = {"limits": self.current_limits, "payload": payload}

        if is_new:
            self.combo.blockSignals(True)
            self.combo.addItem(name)
            self.combo.setCurrentText(name)
            self.combo.blockSignals(False)
            self.on_selection_changed(name)

        self.data_changed.emit(self.items_dict)

    def delete_item(self):
        name = self.combo.currentText()
        if name in self.items_dict:
            del self.items_dict[name]
            self.combo.removeItem(self.combo.findText(name))
            self.combo.setCurrentIndex(0)
            self.data_changed.emit(self.items_dict)

    def get_items(self):
        return self.items_dict

    # --- Methods to be overridden by subclasses ---
    def setup_custom_ui(self, layout): pass
    def clear_custom_ui(self): pass
    def load_custom_data(self, data): pass
    def get_custom_data(self): return {}
    def validate_custom_data(self, data): return True
