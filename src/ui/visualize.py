""" Classes for visualizing spheres in 3D using PyQt6
"""

import os
import sys
import numpy as np
import h5py
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QSlider, QFrame
)
from PyQt6.QtDataVisualization import (
    Q3DScatter, QScatterDataProxy, QScatter3DSeries, 
    QScatterDataItem, QAbstract3DSeries
)
from PyQt6.QtGui import QVector3D, QColor
from PyQt6.QtCore import Qt, QTimer

# For diagnostics
from PyQt6.QtGui import QOpenGLContext
from PyQt6.QtOpenGL import QOpenGLFunctions_2_0


def hline(parent):
    # Add a horizontal dividing line to parent
    divider = QFrame()
    divider.setFrameShape(QFrame.Shape.HLine)
    divider.setFrameShadow(QFrame.Shadow.Sunken)
    parent.addWidget(divider)

class SphereGraph(Q3DScatter):
    def __init__(self):
        super().__init__()

        # Variables
        self.grid_spacing = 2

        # Camera setup
        self.setOrthoProjection(True)
        self.activeTheme().setType(self.activeTheme().Theme.ThemeEbony)

        # Set up axes
        self.setup_axes()
        self.axes_limits(np.array([(-8, 8), (-5, 5), (-3, 3)]))

        # Setup the Series
        self.series = QScatter3DSeries()
        self.series.setMesh(QAbstract3DSeries.Mesh.MeshSphere)
        self.series.setItemSize(0.12) 
        self.series.setBaseColor(QColor(0, 180, 255))
        self.addSeries(self.series)

        # Setup line series for direction visualization
        self.line_proxy = QScatterDataProxy()
        self.line_series = QScatter3DSeries(self.line_proxy)
        self.line_series.setMesh(QAbstract3DSeries.Mesh.MeshSphere) #MeshPoint)
        self.line_series.setItemSize(0.05)
        self.line_series.setBaseColor(QColor(0, 255, 0))  # Green for directions
        self.addSeries(self.line_series)

    def setup_axes(self):
        # Add axes labels
        self.axisX().setTitle("X")
        self.axisY().setTitle("Y")
        self.axisZ().setTitle("Z")
        self.axisX().setTitleVisible(True)
        self.axisY().setTitleVisible(True)
        self.axisZ().setTitleVisible(True)
        # Make coordinate system right-handed 
        self.axisZ().setReversed(True)

    def axes_limits(self, limits):
        self.limits = limits
        xlim, ylim, zlim = limits
        # Lock axes limits
        self.axisX().setRange(*xlim)
        self.axisY().setRange(*ylim)
        self.axisZ().setRange(*zlim)
        # Force equal aspect ratio
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        z_span = zlim[1] - zlim[0]
        ratio_hy = max(x_span, z_span) / y_span
        ratio_xz =  x_span / z_span
        self.setAspectRatio(ratio_hy)
        self.setHorizontalAspectRatio(ratio_xz)
        # Set grid line spacing
        self.axisX().setSegmentCount(int(x_span / self.grid_spacing))
        self.axisY().setSegmentCount(int(y_span / self.grid_spacing))
        self.axisZ().setSegmentCount(int(z_span / self.grid_spacing))


class SolPlotter(QMainWindow):
    """Display a time‑dependent solution of sphere positions.

    Parameters
    ----------
    solution : np.ndarray, shape (Nt, Nx, 3)
        Coordinates of Nx spheres for Nt time steps.
    """
    def __init__(self, solution):
        super().__init__()
        assert isinstance(solution, np.ndarray)
        assert solution.ndim == 3 and solution.shape[2] == 3
        self.solution = solution
        self.Nt = solution.shape[0]

        self.setWindowTitle("Solution Plotter")
        self.resize(1920, 1080)

        # create 3D graph
        self.graph = SphereGraph()
        # adjust axes to cover all data at once
        all_pts = solution.reshape(-1, 3)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        self.graph.axes_limits(np.array([
            (mins[0], maxs[0]),
            (mins[1], maxs[1]),
            (mins[2], maxs[2])
        ]))

        container = QWidget.createWindowContainer(self.graph)

        # Slider for frame count at the bottom
        slider_layout = QHBoxLayout()
        self.setup_slider_panel(slider_layout)

        # assemble main layout
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.addWidget(container, 1)
        main_layout.addLayout(slider_layout)
        self.setCentralWidget(central)

        # show initial timestep
        self.show_timestep(0)

    def setup_slider_panel(self, parent_layout):
        # Setup the Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)

        # Setup the Timer for animation
        self.timer = QTimer()
        fps = 10 #30
        self.timer.setInterval(int(1000/fps))
        self.timer.timeout.connect(self.advance_frame)

        # slider and label
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.Nt-1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider_label = QLabel("0")

        parent_layout.addWidget(self.play_button)
        parent_layout.addWidget(QLabel("Frame:"))
        parent_layout.addWidget(self.slider, 1)
        parent_layout.addWidget(self.slider_label)

    def on_slider_changed(self):
        self.slider.blockSignals(True)
        val = self.slider.value()
        self.slider_label.setText(str(val))
        self.show_timestep(val)
        self.slider.blockSignals(False)

    def show_timestep(self, idx):
        idx = int(idx)
        if idx < 0:
            idx = 0
        if idx >= self.Nt:
            idx = self.Nt - 1
        coords = self.solution[idx]
        #proxy = QScatterDataProxy()
        items = [
            QScatterDataItem(QVector3D(float(x), float(y), float(z)))
            for x, y, z in coords
        ]
        #proxy.addItems(items)
        #self.graph.series.setDataProxy(proxy)
        self.graph.series.dataProxy().resetArray(items)
    
    def toggle_playback(self):
        if self.play_button.isChecked():
            self.play_button.setText("Pause")
            self.timer.start()
        else:
            self.play_button.setText("Play")
            self.timer.stop()
    
    def advance_frame(self):
        next_frame = self.slider.value() + 1
        
        if next_frame >= self.Nt:
            next_frame = 0 # Loop back to start
        
        self.slider.setValue(next_frame)


def print_gpu_info():
    """Diagnostic check for GPU"""
    ctx = QOpenGLContext.currentContext()
    if ctx:
        # Use the versioned functions class
        funcs = QOpenGLFunctions_2_0()
        funcs.initializeOpenGLFunctions() # This connects it to the current context
        
        # OpenGL Enums
        GL_VENDOR = 0x1F00
        GL_RENDERER = 0x1F01
        
        # Get the bytes and decode to string
        vendor = funcs.glGetString(GL_VENDOR)
        renderer = funcs.glGetString(GL_RENDERER)
        
        print(f"\n--- Graphics Hardware Report ---")
        print(f"Vendor:   {vendor}")
        print(f"Renderer: {renderer}")
        print(f"--------------------------------\n")
    else:
        print("No active OpenGL context. Make sure to call this after window.show()!")

def plot_sol_file(path):
    """Plot the solution saved in the given file"""

    msg_invalid = "The provided file was not a valid Crystal Physics result file"
    with h5py.File(path, 'r') as f:
        assert f.attrs["file_type"] == "simulation_result", msg_invalid
        # Load result
        xs = np.array(f["result"]["coords"])
        app_plot_sol(xs)

def app_plot_sol(data):
    """Plot the given solution"""
    app = QApplication(sys.argv)
    viewer = SolPlotter(data)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Fix for Linux OpenGL environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    #sol_path = os.path.join(src_dir, "../data/sol_20260303-123015.csv")
    sol_path = os.path.join(src_dir, "../data/sol_20260303-195122.csv")
    plot_sol_file(sol_path)

    """
    data = np.random.rand(20, 100, 3)
    app_plot_sol(data)
    """