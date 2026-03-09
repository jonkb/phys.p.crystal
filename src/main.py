""" Start the user interface
"""

import sys
import argparse
import h5py
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ui.design_lattice import DesignLattice
from ui.visualize import plot_sol_file

def run_argparse():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Crystal Physics"
    )
    
    # File to load
    parser.add_argument(
        "inp_file",
        nargs='?', # Optional
        type=str, 
        default=None,
        help=("Path to a Crystal Physics HDF5 file (e.g., sim001.res.h5)."
            " If no file is provided, the design crystal module is started "
            " with default settings.")
    )

    # Whether to load into the crystal designer or into the visualization module
    parser.add_argument(
        "-v", "--visualize",
        action='store_true',
        help=("Whether to load this file into the visualization module instead"
        " of the design crystal module")
    )

    # Parse what the user typed in the terminal
    return parser.parse_args()

def app_design_lattice(inp_file=None):
    app = QApplication(sys.argv)
    viewer = DesignLattice()
    # Load in the file contents, if passed
    if inp_file is not None:
        viewer.load_inp(inp_file)
    # Show & run until close
    viewer.show()
    sys.exit(app.exec())

def validate_h5(h5_filepath):
    """Confirm that the given file was created by the Crystal Physics program
    Also return whether it's an input file or a results file
    Returns (success_bool, msg)
    """

    msg_invalid = "The provided file was not a valid Crystal Physics file"
    if h5_filepath[-3:] != ".h5":
        return False, msg_invalid

    with h5py.File(h5_filepath, 'r') as f:
        if not "program_name" in f.attrs:
            return False, msg_invalid
        if not f.attrs["program_name"] == "phys.p.crystal: Crystal Physics":
            return False, msg_invalid
        if not "file_type" in f.attrs:
            return False, msg_invalid
        # Check which type of file it is
        file_type = f.attrs["file_type"] 
        if file_type in ["simulation_input", "simulation_result"]:
            return True, file_type
        else:
            return False, msg_invalid

if __name__ == "__main__":
    # Fix for Linux OpenGL environments
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # Parse CLI arguments
    args = run_argparse()

    if args.inp_file is not None:
        # If a file was specified, confirm that it's valid
        success, file_type = validate_h5(args.inp_file)
        if args.visualize:
            if file_type != "simulation_result":
                print("Only a results file can be loaded with -v")
                quit()
            # Visualize simulation results
            plot_sol_file(args.inp_file)
        else:
            # Load design_lattice with settings from inp_file
            app_design_lattice(args.inp_file)
    else:
        if args.visualize:
            print("A results file must be specified to open the visualization with -v")
            quit()
        # Open design lattice GUI with default settings
        app_design_lattice()
