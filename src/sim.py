""" sim.py
Run a simulation from given initial conditions
"""

import os
import numpy as np
import jax.numpy as jnp

from autoDyn import AutoEL, simulate
from visualize import app_plot_sol

# Constants
t1 = 1
Nt = 100
src_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(src_dir, "..", "data")
in_path = os.path.join(data_dir, "positions_20260303-100733.csv")
out_path = os.path.join(data_dir, "sol_20260303-100733.csv")
atom_mass = 1


def potential(r):
    """Interatomic potential function"""
    # Use where to handle r=0
    return jnp.where(r>0, -2/r + 1/r**2, 0.0)

def L(q, qd):
    """Lagrangian L=T-V"""
    T = jnp.sum(atom_mass*qd**2)/2
    # Interatomic distances matrix
    x = jnp.reshape(q, (-1,3))
    # TODO: CHECK THIS
    dx = jnp.reshape(x, (-1,3,1)) - jnp.reshape(x.T, (1,3,-1))
    R = jnp.sqrt(jnp.sum(jnp.square(dx), axis=1))
    V = jnp.sum(potential(R))
    return T-V

Qnc = None

# Equinox module to be simulated with diffrax
mod = AutoEL(L, Qnc)

# Load initial conditions from file
x0 = np.loadtxt(in_path, delimiter=',', skiprows=1)
Nx = x0.shape[0]

# Simulation parameters
ts = jnp.linspace(0, t1, Nt)
q0 = jnp.reshape(x0, (-1,))

qd0 = jnp.zeros_like(q0)
x0 = jnp.concatenate([q0, qd0])

sol = simulate(mod, ts, x0, tol=1e-6, max_steps=int(1e6))

# Save state vector at each timestep
ys = np.array(sol.ys)
np.savetxt(out_path, ys, delimiter=',', header='x,y,z', comments='')

# Animate with visualize
xs = np.reshape(ys[:, 0:3*Nx], (Nt,Nx,3))
app_plot_sol(xs, t1)

