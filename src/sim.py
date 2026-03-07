""" sim.py
Run a simulation from an input file (*.inp.h5)
"""

import os
import argparse
import h5py
import numpy as np
import jax.numpy as jnp

#from config import data_dir
from autoDyn import AutoEL, simulate
from ui.visualize import app_plot_sol
from util import tic, toc, isonow
from lattice_energy import LJLagrangian

msg_load_err = "Invalid input file"

def in_region(limits, x0):
    """Return a boolean mask for whether each point in x0 is in limits"""
    in_x = (x0[:, 0] >= limits[0, 0]) & (x0[:, 0] <= limits[0, 1])
    in_y = (x0[:, 1] >= limits[1, 0]) & (x0[:, 1] <= limits[1, 1])
    in_z = (x0[:, 2] >= limits[2, 0]) & (x0[:, 2] <= limits[2, 1])
    return in_x & in_y & in_z

def parse_applied_forces(force_dict, x0):
    """ Convert a dictionary of (constant) applied forces to an (N,3) array
    representing the resultant of all forces acting on the particles
    """
    # Initialize an array of zeros with the same shape as x0
    F_resultant = np.zeros_like(x0, dtype=float)
    
    # Iterate over each applied force region
    for region_name, group in force_dict.items():
        # Get the bounding box limits [3x2 array]
        limits = np.array(group['limits'])
        
        # Find which atoms fall within the bounding box
        mask = in_region(limits, x0)
        
        # Add the applied force to the atoms within the region
        F_resultant[mask, :] += group.attrs['vector']
        
    return F_resultant

def parse_constraints(con_dict, x0):
    """ Convert a dictionary of constraints to an (N,3) boolean array
    representing the constrained degrees of freedom for each particle.
    """
    # Initialize an array of False (unconstrained) with the same shape as x0
    con = np.zeros_like(x0, dtype=bool)
    
    # Iterate over each constraint region
    for region_name, group in con_dict.items():
        # Get the bounding box limits
        limits = np.array(group['limits'])

        # Find which atoms fall within the bounding box
        mask = in_region(limits, x0)
        
        # Apply the constraints using logical OR (so overlapping regions don't overwrite each other)
        con[mask, :] |= group.attrs['dof']
        
    return con

def load_inp(input_filepath):
    """ Read the contents of the provided hdf5 file
    Return the essential information for the simulation
    """
    inp_data = {}
    with h5py.File(input_filepath, 'r') as f:
        assert f.attrs['file_type'] == "simulation_input", msg_load_err

        # Pull out atomic coordinates
        x0 = np.array(f['lattice']['coordinates'])
        inp_data['x0'] = x0

        # Pull out body forces
        grp_fbody = f['forces']['body']
        inp_data['atom_mass'] = grp_fbody.attrs['atom_mass']
        # TODO: gravity

        # Pull out interatomic potential
        inp_data['potential'] = dict(f['forces']['interatomic'].attrs)

        # Applied forces
        force_dict = dict(f['forces']['applied'])
        inp_data['applied_forces'] = parse_applied_forces(force_dict, x0)

        # Constraints
        con_dict = dict(f['constraints'])
        inp_data['constraints'] = parse_constraints(con_dict, x0)

        # Simulation options
        grp_sim = f['simulation']
        inp_data['t1'] = grp_sim['time'].attrs['t1']
        inp_data['Nt'] = grp_sim['time'].attrs['Nt']
        # Casting to standard Python types solved a weird diffrax / jax error
        inp_data['tol'] = float(grp_sim['options'].attrs['tol'])
        inp_data['max_steps'] = int(grp_sim['options'].attrs['max_steps'])
    return inp_data

def save_res(inp_file, out_file, ys, xs, t_wallclock):
    """Save simulation results
        HDF Hierarchy
        -------------
        /root
            /input -- Copy of input HDF (see design_lattice.save_inp)
            /result
    """
    # Load the input file so it can be copied over to the output
    with h5py.File(inp_file, 'r') as f_in, h5py.File(out_file, 'w') as f_out:
        try:
            # --- Root Level Metadata ---
            f_out.attrs['program_name'] = "phys.p.crystal: Crystal Physics"
            f_out.attrs['file_type'] = "simulation_result"
            f_out.attrs['timestamp'] = isonow()

            # -- Group 1: Inputs --
            #   This line copies the input HDF directly to out_file
            f_out.copy(f_in, 'input')

            # -- Group 2: Results --
            grp_res = f_out.create_group('result')
            grp_res.attrs['wallclock_time'] = t_wallclock
            grp_res.create_dataset('states', data=ys, compression="gzip", chunks=True)
            grp_res.create_dataset('coords', data=xs, compression="gzip", chunks=True)

        except Exception as e:
            print(f"Failed to save file: {e}")
        else:
            # Success
            print("Simulation results file saved to:", out_file)


def run_simulation(inp_data):
    """Main simulation logic."""

    # Start timing
    times = tic()

    # Unpack inp_data
    x0 = jnp.array(inp_data['x0'])
    atom_mass = inp_data['atom_mass']
    t1 = inp_data['t1']
    Nt = inp_data['Nt']
    tol = inp_data['tol']
    max_steps = inp_data['max_steps']

    # Array specifying which DOF are constrained
    con = jnp.array(inp_data['constraints'])
    free_idx = jnp.where(~con)

    # Lagrangian
    #   TODO: Other potential functions
    if inp_data['potential']['type'] == "Lennard-Jones":
        epsilon_depth = inp_data['potential']['epsilon_depth']
        sigma_r0 = inp_data['potential']['sigma_r0']
        L = LJLagrangian(free_idx, x0, atom_mass, epsilon_depth, sigma_r0)
    else:
        print(f"Unsupported potential type: {inp_data['potential']['type']}")
        return

    # Nonconservative forces (constant for now)
    F_resultant = jnp.array(inp_data['applied_forces'])
    Qnc_resultant = L.x_to_q(F_resultant)
    Qnc = lambda t, q, qd: Qnc_resultant

    # Simulation parameters
    ts = jnp.linspace(0, t1, Nt)
    # Note the generalized coordinates q do not include the constrained DOF
    q0 = L.x_to_q(x0)
    Nq = q0.size
    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])

    mod = AutoEL(L, Qnc)

    # Run simulation
    toc(times, "Setup")
    sol = simulate(mod, ts, y0, tol=tol, max_steps=max_steps)
    toc(times, "Simulation")
    t_wallclock = times[-1] - times[-2]

    # Save state vector at each timestep
    ys = np.array(sol.ys)
    #np.savetxt(out_path, ys, delimiter=',', header='x,y,z', comments='')

    # Animate with visualize
    #xs = np.reshape(ys[:, 0:3*Nx], (Nt,Nx,3))
    qs = ys[:,0:Nq]
    #xs = q_to_x(qs, free_idx, x0)
    xs = np.array([L.q_to_x(qi) for qi in qs])

    return ys, xs, t_wallclock



def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run a simulation on a pre-generated crystal lattice."
    )
    
    # Positional argument (Required)
    parser.add_argument(
        "inp_file", 
        type=str, 
        help="Path to the HDF5 input file (e.g., sim001.inp.h5)"
    )
    
    # Optional flag argument
    parser.add_argument(
        "-o", "--out_file", 
        type=str, 
        default=None,
        help="Path to save the output HDF5 results (e.g., sim001.res.h5)"
    )

    # Parse what the user typed in the terminal
    args = parser.parse_args()

    # Verify the file actually exists before trying to open it
    msg = f"ERROR: The file '{args.inp_file}' does not exist."
    assert os.path.exists(args.inp_file), msg

    # If not provided, build the output path automatically
    if args.out_file is None:
        if args.inp_file[-3:] == ".h5":
            if args.inp_file[-7:] == ".inp.h5":
                inp_stem = args.inp_file[0:-7]
            else:
                inp_stem = args.inp_file[0:-3]
        else:
            print("WARNING: The input file doesn't end in .h5")
            inp_stem = args.inp_file
        args.out_file = f"{inp_stem}.res.h5"
    else:
        if args.out_file[-7:] != ".res.h5":
            print("WARNING: By convention, simulation result files should end"
                " in '.res.h5'")

    # Pull out the essential information from the input file
    print(f"Loading input file: {args.inp_file}")
    inp_data = load_inp(args.inp_file)
    Nx = inp_data['x0'].shape[0]
    print(f"\tNx = {Nx}, t_end = {inp_data['t1']}")

    # Run simulation
    ys, xs, t_wallclock = run_simulation(inp_data)

    # Save the results
    save_res(args.inp_file, args.out_file, ys, xs, t_wallclock)

    # Animate the results
    app_plot_sol(xs)


if __name__ == "__main__":
    main()