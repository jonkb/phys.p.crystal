""" Classes for visualizing spheres in 3D using PyQt6
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QDoubleSpinBox, QSpinBox, QFrame, QComboBox
)
from PyQt6.QtDataVisualization import (
    Q3DScatter, QScatterDataProxy, QScatter3DSeries, 
    QScatterDataItem, QAbstract3DSeries
)
from PyQt6.QtGui import QVector3D, QColor
from PyQt6.QtCore import Qt

# For diagnostics
from PyQt6.QtGui import QOpenGLContext
from PyQt6.QtOpenGL import QOpenGLFunctions_2_0

# Custom imports
from crystal import generate_FCC


class SphereGraph(Q3DScatter):
    def __init__(self):
        super().__init__()

        # Variables
        self.grid_spacing = 2

        # Camera setup
        self.setOrthoProjection(True)
        self.activeTheme().setType(self.activeTheme().Theme.ThemeEbony)

        # Set up axes
        self.limits = np.array([(-8, 8), (-5, 5), (-3, 3)])
        self.setup_axes()
        self.axes_limits(self.limits)

        # Setup the Series
        self.series = QScatter3DSeries()
        self.series.setMesh(QAbstract3DSeries.Mesh.MeshSphere)
        self.series.setItemSize(0.12) 
        self.series.setBaseColor(QColor(0, 180, 255))
        self.addSeries(self.series)

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


class DesignLattice(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Sphere Visualization")
        self.resize(1920, 1080)

        # 1. Initialize the 3D Graph
        self.graph = SphereGraph()

        # Placement method
        self.placement_method = 'random'
        
        # Lattice parameters
        self.lattice_a = 1.0
        self.lattice_b = 1.0
        self.lattice_c = 1.0
        
        # Euler angles (in degrees)
        self.euler_alpha = 0.0
        self.euler_beta = 0.0
        self.euler_gamma = 0.0
        
        # Wrap the graph in a QWidget container for the layout
        self.graph_container = QWidget.createWindowContainer(self.graph)

        # 2. Setup the UI Layout (Horizontal: Left Panel + Right Graph)
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add Axes Limits Controls
        self.setup_limits_controls(left_layout)

        self.hline(left_layout)
        
        # Add Placement Method Selection
        self.setup_placement_controls(left_layout)
        
        self.hline(left_layout)
        
        # Add stretch to push controls to the top
        left_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 0)  # Left panel with minimal width
        main_layout.addWidget(self.graph_container, 1)  # Graph takes remaining space
        
        self.setCentralWidget(central_widget)
        
        # Initial render
        self.generate_spheres()

    def hline(self, parent):
        # Add a horizontal dividing line to parent
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        parent.addWidget(divider)

    def setup_limits_controls(self, parent_layout):
        """Create input fields for axes limits in a grid layout."""
        # Create a grid layout for aligned controls
        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)
        
        # Headers
        grid_layout.addWidget(QLabel("Axis"), 0, 0)
        grid_layout.addWidget(QLabel("Min"), 0, 1)
        grid_layout.addWidget(QLabel("Max"), 0, 2)
        
        # X axis
        grid_layout.addWidget(QLabel("X:"), 1, 0)
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-100, 100)
        self.x_min_spin.setValue(self.graph.limits[0][0])
        self.x_min_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.x_min_spin, 1, 1)
        
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-100, 100)
        self.x_max_spin.setValue(self.graph.limits[0][1])
        self.x_max_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.x_max_spin, 1, 2)
        
        # Y axis
        grid_layout.addWidget(QLabel("Y:"), 2, 0)
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-100, 100)
        self.y_min_spin.setValue(self.graph.limits[1][0])
        self.y_min_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.y_min_spin, 2, 1)
        
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-100, 100)
        self.y_max_spin.setValue(self.graph.limits[1][1])
        self.y_max_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.y_max_spin, 2, 2)
        
        # Z axis
        grid_layout.addWidget(QLabel("Z:"), 3, 0)
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-100, 100)
        self.z_min_spin.setValue(self.graph.limits[2][0])
        self.z_min_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.z_min_spin, 3, 1)
        
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-100, 100)
        self.z_max_spin.setValue(self.graph.limits[2][1])
        self.z_max_spin.valueChanged.connect(self.apply_limits)
        grid_layout.addWidget(self.z_max_spin, 3, 2)
        
        parent_layout.addLayout(grid_layout)

    def setup_placement_controls(self, parent_layout):
        """Create controls for placement method selection and lattice parameters."""
        # Placement method dropdown
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Placement:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Random", "FCC", "HCP"])
        self.method_combo.currentTextChanged.connect(self.on_placement_method_changed)
        method_layout.addWidget(self.method_combo)
        parent_layout.addLayout(method_layout)
        
        # Container for random sphere placement controls
        self.random_controls_widget = QWidget()
        self.random_controls_layout = QVBoxLayout(self.random_controls_widget)
        self.random_controls_layout.setContentsMargins(0, 0, 0, 0)

        # Number of random spheres
        random_grid = QGridLayout()
        random_grid.setSpacing(5)
        random_grid.addWidget(QLabel("Number of Points:"), 0, 0)
        self.N_spin = QSpinBox()
        self.N_spin.setRange(1, 100000)
        self.N_spin.setValue(100)
        self.N_spin.setSingleStep(10)
        random_grid.addWidget(self.N_spin, 0, 1)
        
        # Randomize Button
        self.shuffle_button = QPushButton("Randomize Positions")
        self.shuffle_button.setFixedHeight(40)
        self.shuffle_button.clicked.connect(self.generate_spheres)
        random_grid.addWidget(self.shuffle_button, 1, 0, 1, 2)
        
        # Add random placement controls
        self.random_controls_layout.addLayout(random_grid)
        parent_layout.addWidget(self.random_controls_widget)
        
        # Container for crystallographic controls (initially hidden)
        self.crystal_controls_widget = QWidget()
        self.crystal_controls_layout = QVBoxLayout(self.crystal_controls_widget)
        self.crystal_controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Lattice parameters (a, b, c)
        lattice_grid = QGridLayout()
        lattice_grid.setSpacing(5)
        lattice_grid.addWidget(QLabel("Lattice Parameters:"), 0, 0, 1, 2)
        
        lattice_grid.addWidget(QLabel("a:"), 1, 0)
        self.a_spin = QDoubleSpinBox()
        self.a_spin.setRange(0.1, 10.0)
        self.a_spin.setValue(1.0)
        self.a_spin.setSingleStep(0.1)
        self.a_spin.valueChanged.connect(self.on_lattice_params_changed)
        lattice_grid.addWidget(self.a_spin, 1, 1)
        
        lattice_grid.addWidget(QLabel("b:"), 2, 0)
        self.b_spin = QDoubleSpinBox()
        self.b_spin.setRange(0.1, 10.0)
        self.b_spin.setValue(1.0)
        self.b_spin.setSingleStep(0.1)
        self.b_spin.valueChanged.connect(self.on_lattice_params_changed)
        lattice_grid.addWidget(self.b_spin, 2, 1)
        
        lattice_grid.addWidget(QLabel("c:"), 3, 0)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.1, 10.0)
        self.c_spin.setValue(1.0)
        self.c_spin.setSingleStep(0.1)
        self.c_spin.valueChanged.connect(self.on_lattice_params_changed)
        lattice_grid.addWidget(self.c_spin, 3, 1)
        
        # Euler angles
        lattice_grid.addWidget(QLabel("Euler Angles (°):"), 4, 0, 1, 2)
        
        lattice_grid.addWidget(QLabel("α:"), 5, 0)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(-180, 180)
        self.alpha_spin.setValue(0.0)
        self.alpha_spin.setSingleStep(1.0)
        self.alpha_spin.valueChanged.connect(self.on_euler_angles_changed)
        lattice_grid.addWidget(self.alpha_spin, 5, 1)
        
        lattice_grid.addWidget(QLabel("β:"), 6, 0)
        self.beta_spin = QDoubleSpinBox()
        self.beta_spin.setRange(-180, 180)
        self.beta_spin.setValue(0.0)
        self.beta_spin.setSingleStep(1.0)
        self.beta_spin.valueChanged.connect(self.on_euler_angles_changed)
        lattice_grid.addWidget(self.beta_spin, 6, 1)
        
        lattice_grid.addWidget(QLabel("γ:"), 7, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(-180, 180)
        self.gamma_spin.setValue(0.0)
        self.gamma_spin.setSingleStep(1.0)
        self.gamma_spin.valueChanged.connect(self.on_euler_angles_changed)
        lattice_grid.addWidget(self.gamma_spin, 7, 1)
        
        self.crystal_controls_layout.addLayout(lattice_grid)
        parent_layout.addWidget(self.crystal_controls_widget)
        
        # Initially hide crystal controls
        self.crystal_controls_widget.setVisible(False)

    def apply_limits(self):
        """Automatically update axes limits from spinbox values."""
        self.graph.limits = [
            (self.x_min_spin.value(), self.x_max_spin.value()),
            (self.y_min_spin.value(), self.y_max_spin.value()),
            (self.z_min_spin.value(), self.z_max_spin.value())
        ]
        self.graph.axes_limits(self.graph.limits)

    def update_data(self, data_np):
        """Standard method to accept a (N, 3) NumPy array."""
        if data_np.ndim != 2 or data_np.shape[1] != 3:
            print("Invalid shape. Needs (N, 3)")
            return

        new_proxy = QScatterDataProxy()
        items = []
        for row in data_np:
            items.append(QScatterDataItem(QVector3D(float(row[0]), float(row[1]), float(row[2]))))
        
        new_proxy.addItems(items)
        self.graph.series.setDataProxy(new_proxy)
    
    def generate_spheres(self):
        """Generate sphere coordinates based on placement method."""
        if self.placement_method == 'random':
            self.generate_random_placement()
        elif self.placement_method == 'fcc':
            self.generate_fcc_placement()
        elif self.placement_method == 'hcp':
            self.generate_hcp_placement()
    
    def generate_random_placement(self):
        """Generate random sphere placement within the domain."""
        point_count = int(self.N_spin.value())
        xnew = np.random.uniform(*self.graph.limits[0], point_count)
        ynew = np.random.uniform(*self.graph.limits[1], point_count)
        znew = np.random.uniform(*self.graph.limits[2], point_count)
        new_coords = np.array([xnew, ynew, znew]).T
        self.update_data(new_coords)
    
    def generate_fcc_placement(self):
        """Generate FCC (Face-Centered Cubic) lattice."""

        domain = self.graph.limits
        lattice_prms = np.array([self.lattice_a, self.lattice_b, self.lattice_c])
        euler_angles = np.radians(np.array([self.euler_alpha, self.euler_beta, 
                self.euler_gamma]))
        coords = generate_FCC(domain, lattice_prms, euler_angles)
        
        self.update_data(coords)
    
    def generate_hcp_placement(self):
        """Generate HCP (Hexagonal Close Packed) lattice."""
        a = self.lattice_a
        c = self.lattice_c
        
        # HCP lattice parameters (ideal ratio c/a = sqrt(8/3))
        # HCP basis vectors
        basis_vectors = np.array([
            [a, 0, 0],
            [-a/2, a*np.sqrt(3)/2, 0],
            [0, 0, c]
        ])
        
        # HCP atom positions in conventional cell
        hcp_positions = np.array([
            [0.0, 0.0, 0.0],
            [1/3, 1/3, 0.5]
        ])
        
        # Generate lattice
        coords = []
        nx, ny, nz = 4, 4, 3
        for i in range(-nx, nx):
            for j in range(-ny, ny):
                for k in range(-nz, nz):
                    cell_origin = np.array([i, j, k]) @ basis_vectors
                    for frac_pos in hcp_positions:
                        atom_pos = cell_origin + frac_pos @ basis_vectors
                        coords.append(atom_pos)
        
        coords = np.array(coords)
        
        # Apply rotation
        coords = self.apply_euler_rotation(coords)
        
        # Filter points within domain
        coords = self.filter_in_domain(coords)
        
        if len(coords) > 0:
            self.update_data(coords)
    
    def apply_euler_rotation(self, coords):
        """Apply ZYZ Euler angle rotation to coordinates."""
        alpha = np.radians(self.euler_alpha)
        beta = np.radians(self.euler_beta)
        gamma = np.radians(self.euler_gamma)
        
        # ZYZ Euler angle rotation matrices
        Rz_alpha = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        
        Ry_beta = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        
        Rz_gamma = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: R = Rz(gamma) * Ry(beta) * Rz(alpha)
        R = Rz_gamma @ Ry_beta @ Rz_alpha
        
        return coords @ R.T
    
    def filter_in_domain(self, coords):
        """Filter coordinates to keep only those within the domain."""
        xlim, ylim, zlim = self.graph.limits
        mask = (
            (coords[:, 0] >= xlim[0]) & (coords[:, 0] <= xlim[1]) &
            (coords[:, 1] >= ylim[0]) & (coords[:, 1] <= ylim[1]) &
            (coords[:, 2] >= zlim[0]) & (coords[:, 2] <= zlim[1])
        )
        return coords[mask]
    
    def on_placement_method_changed(self, method_text):
        """Handle placement method change."""
        method_map = {"Random": "random", "FCC": "fcc", "HCP": "hcp"}
        self.placement_method = method_map.get(method_text, "random")

        # Show/hide crystal controls
        show_crystal = self.placement_method in ['fcc', 'hcp']
        self.crystal_controls_widget.setVisible(show_crystal)
        # Show/hide random controls
        self.random_controls_widget.setVisible(not show_crystal)
        
        # Regenerate spheres
        self.generate_spheres()

    def on_lattice_params_changed(self):
        """Handle lattice parameter changes."""
        self.lattice_a = self.a_spin.value()
        self.lattice_b = self.b_spin.value()
        self.lattice_c = self.c_spin.value()
        self.generate_spheres()
    
    def on_euler_angles_changed(self):
        """Handle Euler angle changes."""
        self.euler_alpha = self.alpha_spin.value()
        self.euler_beta = self.beta_spin.value()
        self.euler_gamma = self.gamma_spin.value()
        self.generate_spheres()


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

if __name__ == "__main__":
    # Fix for Linux OpenGL environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    app = QApplication(sys.argv)
    viewer = DesignLattice()
    window_geometry = viewer.geometry()
    viewer.show()

    print_gpu_info()

    sys.exit(app.exec())