""" sim.py
Run a simulation from given initial conditions
"""

import os
import numpy as np
import jax.numpy as jnp
from jax import jit, lax

from autoDyn import AutoEL, simulate
from visualize import app_plot_sol

# Constants
t1 = 3
Nt = 50
src_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(src_dir, "..", "data")
in_path = os.path.join(data_dir, "positions_20260303-195122.csv")
out_path = os.path.join(data_dir, "sol_20260303-195122.csv")
atom_mass = 1


def potential(r):
    """Interatomic potential function"""
    # Use where to handle r=0
    s = 0.89 / jnp.sqrt(2)
    u = 0.1
    lj = lambda r: 4*u*((s/r)**12 - (s/r)**6)
    #return jnp.where(r>0, -2/r + 1/r**2, 0.0)
    return jnp.where(r>0, lj(r), 0.0)


@jit
def V_total_lax(q):
    """System potential energy function
    Implemented with lax so the for loops run quickly

    Pythonic for loop equivalent:
        V = 0
        for i in range(Nx):
            for j in range(i+1,Nx):
                dx = x[i,:] - x[j,:]
                r = jnp.sqrt(jnp.sum(jnp.square(dx)))
                V += potential(r)
    """
    x = jnp.reshape(q, (-1,3))
    Nx = x.shape[0]
    
    def inner_loop(j, carry):
        V, x, i = carry
        dx = x[i,:] - x[j,:]
        r = jnp.sqrt(jnp.sum(jnp.square(dx)))
        Vij = potential(r)
        return (V+Vij, x, i)
    def outer_loop(i, carry):
        V, x = carry
        Vi, _, _ = lax.fori_loop(i+1, Nx, inner_loop, (0.0, x, i))
        return (V+Vi, x)

    V, _ = lax.fori_loop(0, Nx, outer_loop, (0.0, x))
    # 2* to count both atoms in each bond
    return 2*V

@jit
def V_total_jnp(q):
    """System potential energy function
    Requires high ram because it creates an array of shape (Nx,3,Nx)
    """
    x = jnp.reshape(q, (-1,3))
    # Interatomic distances matrix
    dx = jnp.reshape(x, (-1,3,1)) - jnp.reshape(x.T, (1,3,-1))
    R = jnp.sqrt(jnp.sum(jnp.square(dx), axis=1))
    return jnp.sum(potential(R))

def L(q, qd):
    """Lagrangian L=T-V"""
    T = jnp.sum(atom_mass*qd**2)/2
    V = V_total_lax(q)
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

""" #TESTING
q0 = q0[0:9]
print(q0)
print("V_total_lax(q0): ", V_total_lax(q0))
print("V_total_jnp(q0): ", V_total_jnp(q0))
"""


qd0 = jnp.zeros_like(q0)
y0 = jnp.concatenate([q0, qd0])

sol = simulate(mod, ts, y0, tol=1e-6, max_steps=int(1e6))

# Save state vector at each timestep
ys = np.array(sol.ys)
np.savetxt(out_path, ys, delimiter=',', header='x,y,z', comments='')

# Animate with visualize
xs = np.reshape(ys[:, 0:3*Nx], (Nt,Nx,3))
app_plot_sol(xs)

