# phys.p.crystal
Simulate atoms in a crystal lattice

## Quick Start
1. Run design_lattice.py to create a positions file
2. Run sim.py to run the simulation
3. Run visualize.py to view the results

NOTE: For now, many things are still hard-coded in sim.py & visualize.py

## Installation
See requirements.txt
For instructions on installing JAX, see https://docs.jax.dev/en/latest/installation.html

## Data storage
HDF5 files are used for storing two kinds of data:
1. Lattice designer / simulation input files (*.inp.h5)
2. Simulation results (*.res.h5)
    For reproducibility, the entire contents of the input file are saved in as a group (/input/)

To inspect the contents of these files without opening the GUI, consider [hdf5view](https://tgwoodcock.github.io/hdf5view/index.html)

## GPU acceleration
autoDyn uses JAX, which supports GPU acceleration on many systems

NOTE: It is possible to install JAX with GPU support on some AMD systems, but it is significantly harder. See https://docs.jax.dev/en/latest/installation.html#install-amd-gpu