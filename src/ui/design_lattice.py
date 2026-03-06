""" Design a crystal lattice & simulation
"""

import os
import numpy as np
import h5py

# Core PyQt6 UI components
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog
)

# 3D Visualization components
from PyQt6.QtDataVisualization import (
    QScatterDataProxy, QScatterDataItem, QCustom3DItem
)
from PyQt6.QtGui import QVector3D, QQuaternion

# Custom modules
from config import res_dir, data_dir
from .visualize import SphereGraph 
import crystal
from util import isonow
from .collabsible_box import CollapsibleBox
from .limits_panel import LimitsPanel
from .miller_panel import MillerPanel
from .lattice_panel import LatticePanel
from .constraints_panel import ConstraintsPanel
from .forces_panel import ForcesPanel
from .body_forces_panel import BodyForcesPanel

# Paths
unit_plane_path = os.path.join(res_dir, "unit_plane.obj")
texture_red_path = os.path.join(res_dir, "tex_red.png")


class DesignLattice(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crystal Physics")
        self.resize(1920, 1080)

        # 1. Initialize the 3D Graph
        self.graph = SphereGraph()

        # Wrap the graph in a QWidget container for the layout
        self.graph_container = QWidget.createWindowContainer(self.graph)

        # 2. Setup the UI Layout (Horizontal: Left Panel + Right Graph)
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for controls
        left_panel = QWidget()
        self.setup_left_panel(left_panel)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 0)  # Left panel with minimal width
        main_layout.addWidget(self.graph_container, 1)  # Graph takes remaining space
        
        self.setCentralWidget(central_widget)
        
        # Initial scene
        self.generate_spheres()
        self.setup_plane()
        self.line_visible = False
    
    def setup_left_panel(self, left_panel):
        """Set up the left panel layout"""
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Create the Lattice accordion menu
        placement_accordion = CollapsibleBox("Atom Placement")
        
        # Add Domain Limits Controls
        limits_accordion = CollapsibleBox("Domain Limits")
        self.limits_panel = LimitsPanel(initial_limits=self.graph.limits)
        self.limits_panel.limits_changed.connect(self.apply_limits)
        limits_accordion.addWidget(self.limits_panel)
        placement_accordion.addWidget(limits_accordion)

        # Add Placement Method Selection
        lattice_accordion = CollapsibleBox("Lattice Design")
        self.lattice_panel = LatticePanel()
        self.lattice_panel.method_changed.connect(self.on_placement_method_changed)
        self.lattice_panel.randomize_requested.connect(self.generate_spheres)
        self.lattice_panel.params_changed.connect(self.refresh_lattice_visuals)
        lattice_accordion.addWidget(self.lattice_panel)
        placement_accordion.addWidget(lattice_accordion)

        # Add lattice accordion to left layout
        left_layout.addWidget(placement_accordion)
        
        # Add miller indices visualization section
        self.miller_accordion = CollapsibleBox("Planes && Directions")
        self.miller_panel = MillerPanel()
        self.miller_panel.plane_visibility_toggled.connect(self.update_plane_visibility)
        self.miller_panel.plane_indices_changed.connect(self.update_plane_orientation)
        self.miller_panel.dir_visibility_toggled.connect(self.update_line_visibility)
        self.miller_panel.dir_indices_changed.connect(self.draw_line)
        self.miller_accordion.addWidget(self.miller_panel)
        self.miller_accordion.setVisible(False) # Hidden by default
        left_layout.addWidget(self.miller_accordion)

        # Edit forces
        forces_accordion = CollapsibleBox("Forces")
        # Edit body forces
        body_forces_accordion = CollapsibleBox("Body Forces")
        self.body_forces_panel = BodyForcesPanel()
        body_forces_accordion.addWidget(self.body_forces_panel)
        forces_accordion.addWidget(body_forces_accordion)
        # Edit interatomic forces
        interatomic_accordion = CollapsibleBox("Interatomic Potential")
        interatomic_accordion.addWidget(QLabel("TODO"))
        forces_accordion.addWidget(interatomic_accordion)
        # Add applied forces
        appl_forces_accordion = CollapsibleBox("Applied Forces")
        self.appl_forces_panel = ForcesPanel(initial_domain=self.graph.limits)
        appl_forces_accordion.addWidget(self.appl_forces_panel)
        forces_accordion.addWidget(appl_forces_accordion)
        left_layout.addWidget(forces_accordion)

        # Add Constraints
        constraints_accordion = CollapsibleBox("Constraints")
        self.constraints_panel = ConstraintsPanel(initial_domain=self.graph.limits)
        # TODO: In the future, highlight atoms that are constrained
        #self.constraints_panel.active_region_changed.connect(self.highlight_region)
        constraints_accordion.addWidget(self.constraints_panel)
        left_layout.addWidget(constraints_accordion)

        # Add Simulation Settings
        sim_accordion = CollapsibleBox("Simulation")
        sim_accordion.addWidget(QLabel("TODO"))
        left_layout.addWidget(sim_accordion)
        
        # Add stretch to push controls to the top
        left_layout.addStretch()
        
        # Add save button at the bottom of left panel
        self.save_button = QPushButton("Save Input File")
        self.save_button.setFixedHeight(40)
        self.save_button.clicked.connect(self.save_inp)
        left_layout.addWidget(self.save_button)

    def setup_plane(self):
        """Create plane on graph to be used later"""
        # Plane custom item (create now but hide)
        self.plane_item = QCustom3DItem()
        self.plane_item.setMeshFile(unit_plane_path)
        self.plane_item.setTextureFile(texture_red_path)
        self.plane_item.setScalingAbsolute(False)
        self.scale_plane()
        # Add but hide by default
        self.graph.addCustomItem(self.plane_item)
        self.plane_item.setVisible(False)

    def draw_line(self):
        """Populate direction line series with points"""
        if self.line_visible:
            # Find longest length in domain
            domain = np.array(self.graph.limits)
            lengths = domain[:,1] - domain[:,0]
            L = np.sqrt(np.sum(lengths**2))
            lattice_type = self.lattice_panel.get_method().upper()
            miller = self.miller_panel.get_dir_indices()
            if lattice_type == "HCP":
                miller = crystal.bravais_miller(miller, False)
            lattice_prms = np.array(self.lattice_panel.get_lattice_params())
            euler_angles = np.radians(self.lattice_panel.get_euler_angles())

            # Unit vector along direction
            vec = crystal.miller_vec(lattice_type, miller, lattice_prms, euler_angles)

            # Create points along the line
            num_points = int(L*20)
            t = np.linspace(-L/2, L/2, num_points)
            line_points = t.reshape(-1, 1) * vec.reshape(1, -1)
            
            # Add points to the line series
            pt2item = lambda pt: QScatterDataItem(QVector3D(float(pt[0]), 
                    float(pt[1]), float(pt[2])))
            items = [pt2item(x) for x in line_points]
            self.graph.line_proxy.resetArray(items)
        else:
            self.graph.line_proxy.resetArray([])

    def update_line_visibility(self, visible):
        """Toggle visibility of the direction line series."""
        # setVisible not working properly
        #self.graph.line_series.setVisible(visible)
        self.line_visible = visible
        self.draw_line()

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
        indices = self.miller_panel.get_plane_indices()
        if self.lattice_panel.get_method() == 'fcc':
            miller = indices
            lattice_type = 'FCC'
        else:
            miller = indices[[0,1,3]]
            lattice_type = 'HCP'

        lattice_prms = self.lattice_panel.get_lattice_params()
        euler_angles = np.radians(self.lattice_panel.get_euler_angles())
        normal = crystal.miller_plane(lattice_type, miller, lattice_prms, euler_angles)

        default_normal = np.array([0.0, 1.0, 0.0])
        q = self.quat_from_two_vectors(default_normal, normal)
        self.plane_item.setRotation(q)

    def apply_limits(self, new_limits):
        """Update axes limits from spinbox values."""
        self.graph.axes_limits(new_limits)
        # Update the scaling of the plane
        self.scale_plane()
        # Regenerate the lattice to fill the domain
        self.generate_spheres()

    def update_data(self, data_np):
        """Standard method to accept a (N, 3) NumPy array.

        The latest coordinates are stored in ``self.current_coords`` so that they
        can be exported when the save button is pressed.
        """
        if data_np.ndim != 2 or data_np.shape[1] != 3:
            print("Invalid shape. Needs (N, 3)")
            return

        # remember for export
        self.current_coords = data_np.copy()

        new_proxy = QScatterDataProxy()
        items = []
        for row in data_np:
            items.append(QScatterDataItem(QVector3D(float(row[0]), float(row[1]), float(row[2]))))
        
        new_proxy.addItems(items)
        self.graph.series.setDataProxy(new_proxy)
    
    def generate_spheres(self):
        """Generate sphere coordinates based on placement method."""
        placement_method = self.lattice_panel.get_method()
        if placement_method == 'random':
            self.generate_random_placement()
        elif placement_method == 'fcc':
            self.generate_crystal("FCC")
        elif placement_method == 'hcp':
            self.generate_crystal("HCP")
    
    def generate_random_placement(self):
        """Generate random sphere placement within the domain."""
        point_count = int(self.lattice_panel.get_point_count())
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
        lattice_prms = self.lattice_panel.get_lattice_params()
        euler_angles = np.radians(self.lattice_panel.get_euler_angles())
        
        coords = crystal.generate_lattice(lattice_type, domain, lattice_prms, euler_angles)
        
        self.update_data(coords)

    def on_placement_method_changed(self, method_name):
        """Handle placement method change."""

        # Show/hide appropriate controls
        show_crystal = method_name in ['fcc', 'hcp']
        self.miller_accordion.setVisible(show_crystal)
        # Signal for the miller panel to update based on method
        self.miller_panel.set_lattice_type(method_name)

        self.refresh_lattice_visuals()

    def update_plane_visibility(self, visible):
        """Show or hide the custom plane object in the graph."""
        if self.plane_item is None:
            return
        self.plane_item.setVisible(visible)

    def refresh_lattice_visuals(self):
        """Triggered whenever any spinbox changes in the placement panel."""
        self.generate_spheres()
        self.update_plane_orientation()
        self.draw_line()

    def save_inp(self):
        """Open save dialog and write simulation input file

        Defaults the dialog to the ``data`` directory adjacent to the
        repository root. 
        Extension: *.inp.h5

        HDF Hierarchy
        -------------
        /root
            /lattice
                /setup
                coordinates
            /visualization
            /forces
                /body
                /interatomic
                /forces
                /constraints
            /simulation
                /time
                /options
        """
        # Get current coordinates
        coords = getattr(self, 'current_coords', None)
        if coords is None or coords.size == 0:
            print("WARNING: No points found")
        
        # Default save location & filename 
        default_name = f"sim_{isonow()}.inp.h5"
        default_path = os.path.join(data_dir, default_name)
        # Prompt for filename
        fname, _ = QFileDialog.getSaveFileName(self, "Save positions",
                default_path, "HDF5 Files (*.h5 *.hdf5)")
        if not fname:
            return
        if not fname.lower().endswith('.h5'):
            fname += '.h5'

        # Write to file
        try:
            with h5py.File(fname, 'w') as f:
                # --- Root Level Metadata ---
                f.attrs['program_name'] = "phys.p.crystal: Crystal Physics"
                f.attrs['file_type'] = "simulation_input"
                f.attrs['timestamp'] = isonow()
                f.attrs['units'] = "NONE" # TODO

                # -- Group 1: Lattice Design --
                grp_lat = f.create_group('lattice')

                # Lattice setup -- options from DesignLattice
                grp_lat_setup = grp_lat.create_group('setup')
                grp_lat_setup.create_dataset('domain_limits', data=np.array(self.graph.limits))
                placement_method = self.lattice_panel.get_method()
                grp_lat_setup.attrs['type'] = placement_method
                if placement_method == 'random':
                    grp_lat_setup.attrs['N'] = self.lattice_panel.get_point_count()
                else:
                    grp_lat_setup.attrs['prms'] = self.lattice_panel.get_lattice_params()
                    grp_lat_setup.attrs['euler_angles'] = self.lattice_panel.get_euler_angles()
                
                # Initial coordinates
                grp_lat.create_dataset('coordinates', data=coords, compression="gzip")

                # -- Group 2: Visualization Options --
                grp_viz = f.create_group('visualization')
                grp_viz.attrs['plane_visible'] = self.miller_panel.show_plane_cb.isChecked()
                grp_viz.attrs['plane_indices'] = self.miller_panel.get_plane_indices()
                grp_viz.attrs['direction_visible'] = self.miller_panel.show_dir_cb.isChecked()
                grp_viz.attrs['direction_indices'] = self.miller_panel.get_dir_indices()
                # TODO: Consider adding camera position

                # -- Group 3: Forces --
                grp_force = f.create_group('forces')

                # Inertial and body forces
                grp_fbody = grp_force.create_group('body')
                grp_fbody.attrs['atom_mass'] = 1.0
                # TODO: gravity

                # Interatomic force potential
                #   TODO: Replace hardcoded with left panel options
                #   TODO: Add Morse potential function
                grp_fIA = grp_force.create_group('interatomic')
                grp_fIA.attrs['potential_type'] = "Lennard-Jones"
                grp_fIA.attrs['epsilon_depth'] = 0.1
                grp_fIA.attrs['sigma_r0'] = 0.89 / np.sqrt(2) # Good for FCC, a=1

                # Applied Forces
                grp_fA = grp_force.create_group('applied')
                #   TODO (list of forces)

                # -- Group 4: Constraints --
                grp_fCn = f.create_group('constraints')
                #   TODO

                # -- Group 5: Simulation options --
                grp_sim = f.create_group('simulation')

                # Time vector
                grp_simT = grp_sim.create_group('time')
                grp_simT.attrs['t1'] = 2.0
                grp_simT.attrs['Nt'] = 50

                # Solver options
                grp_simOpt = grp_sim.create_group('options')
                grp_simOpt.attrs['tol'] = 1e-5
                grp_simOpt.attrs['max_steps'] = int(1e5)

            # Success
            print("Simulation input file saved to:")
            print(fname)
        except Exception as e:
            print(f"Failed to save file: {e}")
