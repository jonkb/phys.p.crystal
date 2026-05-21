""" lattice_energy.py
Contains functions for the potential energy of a crystal using jax-md
"""
from jax_md import energy

def get_energy_and_neighbor_fns(displacement_fn, box_size, potential_info, r_cutoff=10.0):
    """
    Returns a JAX-MD neighbor_fn and energy_fn that natively use sparse neighbor lists.
    """
    pot_type = potential_info['type']
    
    # Smoothly truncate the energy to zero starting 1 unit before the cutoff
    r_onset = r_cutoff - 1.0 
    
    if pot_type == "Lennard-Jones":
        neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
            displacement_fn,
            box_size=box_size,
            sigma=potential_info['sigma_r0'],
            epsilon=potential_info['epsilon_depth'],
            r_onset=r_onset,
            r_cutoff=r_cutoff
        )
        return neighbor_fn, energy_fn
        
    elif pot_type == "Morse":
        neighbor_fn, energy_fn = energy.morse_neighbor_list(
            displacement_fn,
            box_size=box_size,
            sigma=potential_info['req'],
            epsilon=potential_info['De_depth'],
            alpha=potential_info['a_slope'],
            r_onset=r_onset,
            r_cutoff=r_cutoff
        )
        return neighbor_fn, energy_fn
        
    else:
        raise ValueError(f"Unsupported potential type: {pot_type}")