""" Start the user interface
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.design_lattice import DesignLattice

def app_design_lattice():
    app = QApplication(sys.argv)
    viewer = DesignLattice()
    viewer.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    # Fix for Linux OpenGL environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    # Open design lattice GUI
    app_design_lattice()
