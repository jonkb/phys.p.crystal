""" sim.py
Run a simulation from an input file (*.inp.h5)
"""

import os
import argparse
import h5py
import numpy as np
import sympy as sp
import jax.numpy as jnp
import jax
from jax_md import space, simulate

from util import tic, toc, isonow
from lattice_energy import get_energy_and_neighbor_fns

msg_load_err = "Invalid input file"

def in_region(limits, x0):
    """Return a boolean mask for whether each point in x0 is in limits"""
    in_x = (x0[:, 0] >= limits[0, 0]) & (x0[:, 0] <= limits[0, 1])
    in_y = (x0[:, 1] >= limits[1, 0]) & (x0[:, 1] <= limits[1, 1])
    in_z = (x0[:, 2] >= limits[2, 0]) & (x0[:, 2] <= limits[2, 1])
    return in_x & in_y & in_z

def parse_applied_forces(force_dict, x0):
    """ 
    Parses applied forces and returns a JAX-compatible function F_ext(t) 
    that calculates the (N,3) force matrix for any given time t.
    """
    regions = []
    t_sym = sp.Symbol('t')
    
    for region_name, group in force_dict.items():
        limits = np.array(group['limits'])
        
        # Find which atoms fall within the bounding box
        mask = in_region(limits, x0)
        mask_3d = jnp.broadcast_to(mask[:, None], x0.shape)
        
        # Parse and compile the three SymPy strings
        funcs = []
        # Make sure this list matches ForcesPanel.fi_lbls
        for lbl in ['f_x', 'f_y', 'f_z']:
            expr_str = group.attrs[lbl]
            try:
                expr = sp.sympify(expr_str)
                fn = sp.lambdify(t_sym, expr, modules='jax')
                funcs.append(fn)
            except Exception as e:
                raise ValueError(f"Failed to compile {lbl} force expression '{expr_str}' in region '{region_name}': {e}")
        
        # Bundle them for this region
        fx_fn, fy_fn, fz_fn = funcs
        
        # Test-fire the functions to ensure JIT compatibility
        try:
            _ = jax.jit(fx_fn)(jnp.array(0.0))
            _ = jax.jit(fy_fn)(jnp.array(0.0))
            _ = jax.jit(fz_fn)(jnp.array(0.0))
        except Exception as e:
            raise RuntimeError(f"JAX JIT compilation failed for forces in region '{region_name}': {e}")
            
        regions.append((mask_3d, fx_fn, fy_fn, fz_fn))

    # Define the actual time-dependent force function for the solver loop
    def F_ext_fn(t):
        F_resultant = jnp.zeros_like(x0, dtype=float)
        
        for mask_3d, fx_fn, fy_fn, fz_fn in regions:
            # Evaluate the three functions at time t
            # Explicitly cast to float to prevent dtype mismatches if SymPy returns integers (e.g., from "0")
            current_vector = jnp.array([fx_fn(t), fy_fn(t), fz_fn(t)], dtype=float)
            
            # Add the force vector to the atoms inside this region's mask
            F_resultant = jnp.where(mask_3d, F_resultant + current_vector, F_resultant)
            
        return F_resultant

    return F_ext_fn

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
        #inp_data['tol'] = float(grp_sim['options'].attrs['tol'])
        #inp_data['max_steps'] = int(grp_sim['options'].attrs['max_steps'])
        inp_data['irad'] = float(grp_sim['options'].attrs.get('irad', 5.0))
    return inp_data

def save_res(inp_file, out_file, xs, t_wallclock):
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
    # Array specifying which DOF are constrained
    con = jnp.array(inp_data['constraints'])
    F_appl = inp_data['applied_forces']   # Function returning (N,3) force array at time t
    r_cutoff = inp_data['irad']  # Interaction cutoff radius

    Nx = x0.shape[0]
    dt = t1 / Nt
    print("Simulation parameters:")
    print(f"\tParticles: {Nx}")
    print(f"\tEnding time: {t1}, Steps: {Nt}, Timestep: {dt:.3e}")

    # 1. Setup Space (Free boundary conditions)
    displacement_fn, shift_fn = space.free()

    # 2. Setup Neighbor List and Interatomic Energy
    box_size = jnp.max(x0) - jnp.min(x0) + 2.0 * r_cutoff
    neighbor_fn, interatomic_energy_fn = get_energy_and_neighbor_fns(
        displacement_fn, 
        box_size, 
        inp_data['potential'], 
        r_cutoff
    )

    # 3. Define Total Energy (Interatomic + Applied External Forces)
    def total_energy_fn(R, neighbor, t, **kwargs):
        # Fix constrained DOF to initial positions
        R_fixed = jnp.where(con, x0, R)

        # Sparse nearest-neighbor interatomic potential
        V_inter = interatomic_energy_fn(R_fixed, neighbor=neighbor)
        
        # Potential representation of the applied force: U = -F * x
        F_ext = F_appl(t)
        V_applied = -jnp.sum(F_ext * R_fixed)
        
        return V_inter + V_applied

    # 4. Initialize Symplectic Integrator (Velocity Verlet)
    init_fn, apply_fn = simulate.nve(total_energy_fn, shift_fn, dt=dt)
    
    # 5. Define the Simulation Step for lax.scan
    @jax.jit
    def step_fn(carry, t):
        state, nbrs = carry
        
        # Update neighbor list (only rebuilds if particles moved outside the buffer)
        nbrs = nbrs.update(state.position)
        
        # Step the NVE integrator
        state = apply_fn(state, neighbor=nbrs, t=t)
        
        return (state, nbrs), state.position

    # Allocate initial state and neighbor list
    nbrs = neighbor_fn.allocate(x0)
    key = jax.random.PRNGKey(0) 
    state = init_fn(key, x0, kT=0.0, mass=atom_mass, neighbor=nbrs, t=0.0)

    # 6. Run the Simulation Loop
    ts = jnp.linspace(0, t1, Nt)
    toc(times, "Setup")
    (_, final_nbrs), xs = jax.lax.scan(step_fn, (state, nbrs), ts)
    toc(times, "Simulation")
    t_wallclock = times[-1] - times[-2]

    if final_nbrs.did_buffer_overflow:
        print("WARNING: Neighbor list buffer overflowed! Increase capacity_multiplier.")

    # Return trajectories
    xs = np.array(xs)
   
    return xs, t_wallclock


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Run a simulation on a pre-generated crystal lattice."
    )
    parser.add_argument("inp_file", type=str, 
        help="Path to the HDF5 input file (e.g., sim001.inp.h5)"
    )
    parser.add_argument("--out_file", type=str, default=None, 
        help="Path to save the output HDF5 results (e.g., sim001.res.h5)"
    )
    parser.add_argument("--no_gui", action="store_true", 
        help="Don't show the animated solution once the simulation is done"
    )
    args = parser.parse_args()

    # Verify the file actually exists before trying to open it
    msg = f"ERROR: The file '{args.inp_file}' does not exist."
    assert os.path.exists(args.inp_file), msg

    # If not provided, build the output path automatically
    if args.out_file is None:
        if args.inp_file.endswith(".inp.h5"):
            inp_stem = args.inp_file[:-7]
        elif args.inp_file.endswith(".h5"):
            inp_stem = args.inp_file[:-3]
        else:
            print("WARNING: The input file doesn't end in .h5")
            inp_stem = args.inp_file
        args.out_file = f"{inp_stem}.res.h5"
    elif not args.out_file.endswith(".res.h5"):
        print("WARNING: By convention, simulation result files should end in '.res.h5'")

    print(f"Loading input file: {args.inp_file}")
    inp_data = load_inp(args.inp_file)

    xs, t_wallclock = run_simulation(inp_data)
    
    print(f"Simulation completed in {t_wallclock:.2f} seconds.")

    # Save the results
    save_res(args.inp_file, args.out_file, xs, t_wallclock)

    # Animate the results
    if not args.no_gui:
        from ui.visualize import app_plot_sol
        app_plot_sol(xs)
