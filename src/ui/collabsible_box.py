""" collapsible_box.py
Defines CollapsibleBox: a custom widget for accordion menus

The methods addWidget & addLayout expose the inner QVBoxLayout, so the 
CollapsibleBox can be treated as a Widget or as a Layout
"""

from PyQt6.QtWidgets import QWidget, QToolButton, QSizePolicy, QVBoxLayout
from PyQt6.QtCore import Qt

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        
        # The toggle button (Header)
        self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
        self.toggle_button.setStyleSheet("""
            QToolButton { 
                border: none; 
                font-weight: bold; 
                text-align: left;
                padding: 5px;
                background-color: #101010;
                border-radius: 3px;
            }
            QToolButton:hover { background-color: #404040; }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle_button.pressed.connect(self.on_pressed)

        # The content area
        self.content_area = QWidget()
        self.content_area.setVisible(False) # Start collapsed
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(10, 5, 0, 0) # Indent the content slightly

        # Main layout for the custom widget
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

    def on_pressed(self):
        # Toggle visibility and arrow direction
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow if not checked else Qt.ArrowType.RightArrow)
        self.content_area.setVisible(not checked)

    def addWidget(self, widget):
        """Allows directly adding widgets to the collapsible area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Allows directly adding layouts to the collapsible area."""
        self.content_layout.addLayout(layout)
