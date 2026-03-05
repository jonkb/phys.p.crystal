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


def load_inp(input_filepath):
    """ Read the contents of the provided hdf5 file
    Return the essential information for the simulation
    """
    inp_data = {}
    with h5py.File(input_filepath, 'r') as f:
        assert f.attrs['file_type'] == "simulation_input", msg_load_err

        # Pull out atomic coordinates
        inp_data['x0'] = np.array(f['lattice']['coordinates'])

        # Pull out body forces
        grp_fbody = f['forces']['body']
        inp_data['atom_mass'] = grp_fbody.attrs['atom_mass']
        # TODO: gravity

        # Pull out interatomic potential
        inp_data['potential'] = dict(f['forces']['interatomic'].attrs)

        # TODO: Tractions & Constraints

        # Simulation options
        grp_sim = f['simulation']
        inp_data['t1'] = grp_sim['time'].attrs['t1']
        inp_data['Nt'] = grp_sim['time'].attrs['Nt']
        inp_data['tol'] = grp_sim['options'].attrs['tol']
        inp_data['max_steps'] = grp_sim['options'].attrs['max_steps']
    return inp_data

def save_res(inp_file, out_file, ys, xs, t_wallclock):
    """Save simulation results
        HDF Hierarchy
        -------------
        /root
            /input -- Copy of input HDF (see design_lattice.save_inp)
            /
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
    epsilon_depth = inp_data['potential']['epsilon_depth']
    sigma_r0 = inp_data['potential']['sigma_r0']
    t1 = inp_data['t1']
    Nt = inp_data['Nt']
    tol = inp_data['tol']
    max_steps = inp_data['max_steps']

    # Array specifying which DOF are constrained
    #   TODO: Load con
    con = np.zeros_like(x0).astype(bool)
    #con[0,:] = 1    # Constrain atom 0 fully
    left_face = x0[:,0] < -2.75
    con[left_face,0] = 1
    con = jnp.array(con)

    free_idx = jnp.where(~con)

    # Nonconservative forces (None for now)
    Qnc = None

    # Lagrangian
    #   TODO: Other potential functions
    L = LJLagrangian(free_idx, x0, atom_mass, epsilon_depth, sigma_r0)

    # Remove the constrained DOF from the simulation
    q0 = L.x_to_q(x0)
    Nq = q0.size

    # Simulation parameters
    ts = jnp.linspace(0, t1, Nt)
    #q0 = jnp.reshape(x0, (-1,))

    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])

    mod = AutoEL(L, Qnc)

    # Run simulation
    toc(times, "Setup")
    sol = simulate(mod, ts, y0, tol=1e-6, max_steps=int(1e6))
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

    # Run simulation
    ys, xs, t_wallclock = run_simulation(inp_data)

    # Save the results
    save_res(args.inp_file, args.out_file, ys, xs, t_wallclock)

    # Animate the results
    app_plot_sol(xs)


if __name__ == "__main__":
    main()