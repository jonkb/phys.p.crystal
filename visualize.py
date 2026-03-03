""" Classes for visualizing spheres in 3D using PyQt6
"""

import sys
import os
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, 
    QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QDoubleSpinBox, QSpinBox, QFrame, QComboBox
)
from PyQt6.QtDataVisualization import (
    Q3DScatter, QScatterDataProxy, QScatter3DSeries, 
    QScatterDataItem, QAbstract3DSeries, QCustom3DItem
)
from PyQt6.QtGui import QVector3D, QColor, QQuaternion
from PyQt6.QtCore import Qt

# For diagnostics
from PyQt6.QtGui import QOpenGLContext
from PyQt6.QtOpenGL import QOpenGLFunctions_2_0

# Custom imports
import crystal

script_dir = os.path.dirname(os.path.abspath(__file__))
unit_plane_path = os.path.join(script_dir, "unit_plane.obj")

plane_texture_path = os.path.join(script_dir, "plane_tex.png")

def gen_plane_texture():
    """Generate texture image for a plane (semi-transparent red)"""
    if not os.path.exists(plane_texture_path):
        from PyQt6.QtGui import QImage, QColor
        tex = QImage(4, 4, QImage.Format.Format_RGBA8888)
        tex.fill(QColor(255, 0, 0, 120))
        tex.save(plane_texture_path)

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
        
        #self.hline(left_layout)
        
        # Add stretch to push controls to the top
        left_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 0)  # Left panel with minimal width
        main_layout.addWidget(self.graph_container, 1)  # Graph takes remaining space
        
        self.setCentralWidget(central_widget)
        
        # Initial scene
        self.generate_spheres()
        self.setup_plane()

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
        
        # Add Miller index controls for planes/directions
        self.setup_miller_controls(self.crystal_controls_layout)
        
        parent_layout.addWidget(self.crystal_controls_widget)
        
        # Initially hide crystal controls
        self.crystal_controls_widget.setVisible(False)

    def setup_miller_controls(self, parent_layout):
        """Add Miller index inputs and visibility toggles for planes and directions."""
        # wrapper
        self.miller_widget = QWidget()
        miller_layout = QVBoxLayout(self.miller_widget)
        miller_layout.setContentsMargins(0, 0, 0, 0)

        # Start with a horizontal divider
        self.hline(miller_layout)
        
        # Create main table layout
        table_layout = QGridLayout()
        table_layout.setSpacing(8)
        table_layout.setColumnStretch(0, 0)  # Button column (fixed width)
        table_layout.setColumnStretch(1, 1)  # Indices column (expandable)
        
        # ===== ROW 0: PLANE =====
        self.show_plane_cb = QPushButton("Show Plane")
        self.show_plane_cb.setCheckable(True)
        self.show_plane_cb.setMinimumWidth(100)
        self.show_plane_cb.toggled.connect(self.update_plane_visibility)
        table_layout.addWidget(self.show_plane_cb, 0, 0)
        
        # FCC plane indices
        self.plane_fcc_widget = QWidget()
        plane_fcc_layout = QGridLayout(self.plane_fcc_widget)
        plane_fcc_layout.setContentsMargins(0, 0, 0, 0)
        plane_fcc_layout.setSpacing(3)
        self.plane_h_spin = QSpinBox()
        self.plane_h_spin.setRange(-10, 10)
        self.plane_h_spin.setValue(1)
        self.plane_h_spin.valueChanged.connect(self.update_plane_orientation)
        plane_fcc_layout.addWidget(self.plane_h_spin, 0, 0)
        self.plane_k_spin = QSpinBox()
        self.plane_k_spin.setRange(-10, 10)
        self.plane_k_spin.setValue(0)
        self.plane_k_spin.valueChanged.connect(self.update_plane_orientation)
        plane_fcc_layout.addWidget(self.plane_k_spin, 0, 1)
        self.plane_l_spin = QSpinBox()
        self.plane_l_spin.setRange(-10, 10)
        self.plane_l_spin.setValue(0)
        self.plane_l_spin.valueChanged.connect(self.update_plane_orientation)
        plane_fcc_layout.addWidget(self.plane_l_spin, 0, 2)
        table_layout.addWidget(self.plane_fcc_widget, 0, 1)
        
        # HCP plane indices
        self.plane_hcp_widget = QWidget()
        plane_hcp_layout = QGridLayout(self.plane_hcp_widget)
        plane_hcp_layout.setContentsMargins(0, 0, 0, 0)
        plane_hcp_layout.setSpacing(3)
        self.plane_h_b_spin = QSpinBox()
        self.plane_h_b_spin.setRange(-10, 10)
        self.plane_h_b_spin.setValue(1)
        self.plane_h_b_spin.valueChanged.connect(self._update_plane_hcp_i)
        plane_hcp_layout.addWidget(self.plane_h_b_spin, 0, 0)
        self.plane_k_b_spin = QSpinBox()
        self.plane_k_b_spin.setRange(-10, 10)
        self.plane_k_b_spin.setValue(0)
        self.plane_k_b_spin.valueChanged.connect(self._update_plane_hcp_i)
        plane_hcp_layout.addWidget(self.plane_k_b_spin, 0, 1)
        self.plane_i_b_spin = QSpinBox()
        self.plane_i_b_spin.setRange(-10, 10)
        self.plane_i_b_spin.setValue(-1)
        self.plane_i_b_spin.setEnabled(False)
        plane_hcp_layout.addWidget(self.plane_i_b_spin, 0, 2)
        self.plane_l_b_spin = QSpinBox()
        self.plane_l_b_spin.setRange(-10, 10)
        self.plane_l_b_spin.setValue(0)
        self.plane_l_b_spin.valueChanged.connect(self.update_plane_orientation)
        plane_hcp_layout.addWidget(self.plane_l_b_spin, 0, 3)
        table_layout.addWidget(self.plane_hcp_widget, 0, 1)
        self.plane_hcp_widget.setVisible(False)
        
        # ===== ROW 1: DIRECTION =====
        # (plane visibility handled above)
        self.show_dir_cb = QPushButton("Show Direction")
        self.show_dir_cb.setCheckable(True)
        self.show_dir_cb.setMinimumWidth(100)
        table_layout.addWidget(self.show_dir_cb, 1, 0)
        
        # FCC direction indices
        self.dir_fcc_widget = QWidget()
        dir_fcc_layout = QGridLayout(self.dir_fcc_widget)
        dir_fcc_layout.setContentsMargins(0, 0, 0, 0)
        dir_fcc_layout.setSpacing(3)
        self.dir_h_spin = QSpinBox()
        self.dir_h_spin.setRange(-10, 10)
        self.dir_h_spin.setValue(1)
        dir_fcc_layout.addWidget(self.dir_h_spin, 0, 0)
        self.dir_k_spin = QSpinBox()
        self.dir_k_spin.setRange(-10, 10)
        self.dir_k_spin.setValue(0)
        dir_fcc_layout.addWidget(self.dir_k_spin, 0, 1)
        self.dir_l_spin = QSpinBox()
        self.dir_l_spin.setRange(-10, 10)
        self.dir_l_spin.setValue(0)
        dir_fcc_layout.addWidget(self.dir_l_spin, 0, 2)
        table_layout.addWidget(self.dir_fcc_widget, 1, 1)
        
        # HCP direction indices
        self.dir_hcp_widget = QWidget()
        dir_hcp_layout = QGridLayout(self.dir_hcp_widget)
        dir_hcp_layout.setContentsMargins(0, 0, 0, 0)
        dir_hcp_layout.setSpacing(3)
        self.dir_h_b_spin = QSpinBox()
        self.dir_h_b_spin.setRange(-10, 10)
        self.dir_h_b_spin.setValue(1)
        dir_hcp_layout.addWidget(self.dir_h_b_spin, 0, 0)
        self.dir_k_b_spin = QSpinBox()
        self.dir_k_b_spin.setRange(-10, 10)
        self.dir_k_b_spin.setValue(0)
        dir_hcp_layout.addWidget(self.dir_k_b_spin, 0, 1)
        self.dir_i_b_spin = QSpinBox()
        self.dir_i_b_spin.setRange(-10, 10)
        self.dir_i_b_spin.setValue(-1)
        self.dir_i_b_spin.setEnabled(False)
        dir_hcp_layout.addWidget(self.dir_i_b_spin, 0, 2)
        self.dir_l_b_spin = QSpinBox()
        self.dir_l_b_spin.setRange(-10, 10)
        self.dir_l_b_spin.setValue(0)
        dir_hcp_layout.addWidget(self.dir_l_b_spin, 0, 3)
        table_layout.addWidget(self.dir_hcp_widget, 1, 1)
        self.dir_hcp_widget.setVisible(False)
        
        miller_layout.addLayout(table_layout)
        self.dir_i_b_spin.blockSignals(True)
        self.dir_i_b_spin.setValue(-(self.dir_h_b_spin.value() + self.dir_k_b_spin.value()))
        self.dir_i_b_spin.blockSignals(False)
        
        parent_layout.addWidget(self.miller_widget)

    def setup_plane(self):
        """Create plane on graph to be used later"""
        # Plane custom item (create now but hide)
        self.plane_item = QCustom3DItem()
        self.plane_item.setMeshFile(unit_plane_path)
        self.plane_item.setTextureFile(plane_texture_path)
        self.plane_item.setScalingAbsolute(False)
        self.scale_plane()
        # Add but hide by default
        self.graph.addCustomItem(self.plane_item)
        self.plane_item.setVisible(False)

    def scale_plane(self):
        """Automatically scale & place the plane object"""
        # Find longest length in domain
        domain = np.array(self.graph.limits)
        lengths = domain[:,1] - domain[:,0]
        L = np.sqrt(np.sum(lengths**2))
        # Note: the default size of the plane is weirdly 5x5
        self.plane_item.setScaling(QVector3D(2*L, 0.01, 2*L))
        center = QVector3D( *np.mean(domain, axis=1) )
        self.plane_item.setPosition(center)

    def quat_from_two_vectors(self, v0, v1):
        """Return a QQuaternion that rotates unit vector v0 to unit vector v1.
        Uses the shortest rotation; handles edge cases where vectors are
        parallel or opposite.
        """
        v0 = np.array(v0, dtype=float)
        v1 = np.array(v1, dtype=float)
        n0 = v0 / np.linalg.norm(v0)
        n1 = v1 / np.linalg.norm(v1)
        dot = float(np.dot(n0, n1))
        if dot > 0.999999:
            return QQuaternion()  # identity
        if dot < -0.999999:
            # 180 degree rotation: pick an orthogonal axis
            axis = np.array([1.0, 0.0, 0.0])
            if abs(n0[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis / np.linalg.norm(axis)
            return QQuaternion(0.0, float(axis[0]), float(axis[1]), float(axis[2]))
        axis = np.cross(n0, n1)
        w = 1.0 + dot
        q = np.array([w, axis[0], axis[1], axis[2]], dtype=float)
        q = q / np.linalg.norm(q)
        return QQuaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def update_plane_orientation(self):
        """Compute plane normal from Miller indices and rotate the plane item.
        Uses `crystal.miller_plane` to obtain the normal in physical coords,
        then sets the plane item's rotation as a QQuaternion.
        """
        if not hasattr(self, 'plane_item') or self.plane_item is None:
            return
        # Choose indices depending on lattice
        if self.placement_method == 'fcc':
            miller = [self.plane_h_spin.value(), self.plane_k_spin.value(), self.plane_l_spin.value()]
            lattice_type = 'FCC'
        else:
            miller = [self.plane_h_b_spin.value(), self.plane_k_b_spin.value(), self.plane_l_b_spin.value()]
            lattice_type = 'HCP'

        lattice_prms = np.array([self.lattice_a, self.lattice_b, self.lattice_c])
        euler_angles = np.radians(np.array([self.euler_alpha, self.euler_beta, self.euler_gamma]))
        normal = crystal.miller_plane(lattice_type, miller, lattice_prms, euler_angles)

        default_normal = np.array([0.0, 1.0, 0.0])
        q = self.quat_from_two_vectors(default_normal, normal)
        self.plane_item.setRotation(q)

    def _update_plane_hcp_i(self):
        """Auto-sync HCP plane i-index as -(h+k) and update orientation."""
        if not hasattr(self, 'plane_i_b_spin'):
            return
        self.plane_i_b_spin.blockSignals(True)
        self.plane_i_b_spin.setValue(-(self.plane_h_b_spin.value() + self.plane_k_b_spin.value()))
        self.plane_i_b_spin.blockSignals(False)
        self.update_plane_orientation()

    def _update_dir_hcp_i(self):
        """Auto-sync HCP direction i-index as -(h+k)."""
        if not hasattr(self, 'dir_i_b_spin'):
            return
        self.dir_i_b_spin.blockSignals(True)
        self.dir_i_b_spin.setValue(-(self.dir_h_b_spin.value() + self.dir_k_b_spin.value()))
        self.dir_i_b_spin.blockSignals(False)

    def apply_limits(self):
        """Automatically update axes limits from spinbox values."""
        self.graph.limits = [
            (self.x_min_spin.value(), self.x_max_spin.value()),
            (self.y_min_spin.value(), self.y_max_spin.value()),
            (self.z_min_spin.value(), self.z_max_spin.value())
        ]
        self.graph.axes_limits(self.graph.limits)
        # Update the scaling of the plane
        self.scale_plane()
        # Regenerate the lattice to fill the domain
        self.generate_spheres()

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
            self.generate_crystal("FCC")
        elif self.placement_method == 'hcp':
            self.generate_crystal("HCP")
    
    def generate_random_placement(self):
        """Generate random sphere placement within the domain."""
        point_count = int(self.N_spin.value())
        xnew = np.random.uniform(*self.graph.limits[0], point_count)
        ynew = np.random.uniform(*self.graph.limits[1], point_count)
        znew = np.random.uniform(*self.graph.limits[2], point_count)
        new_coords = np.array([xnew, ynew, znew]).T
        self.update_data(new_coords)
    
    def generate_crystal(self, lattice_type):
        """ Generate a crystal lattice
        lattice_type (str): "HCP" or "FCC"
        """
        domain = self.graph.limits
        lattice_prms = np.array([self.lattice_a, self.lattice_b, self.lattice_c])
        euler_angles = np.radians(np.array([self.euler_alpha, self.euler_beta, 
                self.euler_gamma]))
        
        coords = crystal.generate_lattice(lattice_type, domain, lattice_prms, euler_angles)
        
        self.update_data(coords)

    def on_placement_method_changed(self, method_text):
        """Handle placement method change."""
        method_map = {"Random": "random", "FCC": "fcc", "HCP": "hcp"}
        self.placement_method = method_map.get(method_text, "random")

        # Show/hide crystal controls
        show_crystal = self.placement_method in ['fcc', 'hcp']
        self.crystal_controls_widget.setVisible(show_crystal)
        # Show/hide random controls
        self.random_controls_widget.setVisible(not show_crystal)
        
        # Toggle the appropriate plane/direction index widgets
        if hasattr(self, 'plane_fcc_widget') and hasattr(self, 'plane_hcp_widget'):
            is_hcp = self.placement_method == 'hcp'
            self.plane_fcc_widget.setVisible(not is_hcp)
            self.plane_hcp_widget.setVisible(is_hcp)
            self.dir_fcc_widget.setVisible(not is_hcp)
            self.dir_hcp_widget.setVisible(is_hcp)

        # Regenerate spheres
        self.generate_spheres()

    def update_plane_visibility(self, visible):
        """Show or hide the custom plane object in the graph."""
        if self.plane_item is None:
            return
        self.plane_item.setVisible(visible)
    
    def on_lattice_params_changed(self):
        """Handle lattice parameter changes."""
        self.lattice_a = self.a_spin.value()
        self.lattice_b = self.b_spin.value()
        self.lattice_c = self.c_spin.value()
        self.generate_spheres()
        self.update_plane_orientation()
    
    def on_euler_angles_changed(self):
        """Handle Euler angle changes."""
        self.euler_alpha = self.alpha_spin.value()
        self.euler_beta = self.beta_spin.value()
        self.euler_gamma = self.gamma_spin.value()
        self.generate_spheres()
        self.update_plane_orientation()


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