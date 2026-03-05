""" sim.py
Run a simulation from given initial conditions
"""

import os
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from functools import partial
#from jax.tree_util import Partial as partial # Throws a really verbose warning

from autoDyn import AutoEL, simulate
from visualize import app_plot_sol
from util import tic, toc

# Constants
t1 = 2
Nt = 50
src_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.realpath(os.path.join(src_dir, "..", "data"))
in_path = os.path.join(data_dir, "positions_20260303-195122.csv")
out_path = os.path.join(data_dir, "sol_20260303-195122.csv")
atom_mass = 1


def q_to_x(q, free_idx, x0):
    """Given the generalized coordinates, return the atomic positions
    q: Generalized coordinates
    free_idx: free DOF
    x0: Initial positions
    """
    return x0.at[free_idx].set(q)

def qd_to_xd(qd, free_idx, x0):
    """Given the generalized velocities, return the atomic velocities
    qd: Generalized velocities
    free_idx: free DOF
    con: boolean mask for x, indicating constrained DOF
    Constrained DOF always have zero velocity
    """
    xd = jnp.zeros_like(x0)
    return xd.at[free_idx].set(qd)

def x_to_q(x, free_idx):
    """Given the positions x, return the generalized coordinates
    x: Atomic positions
    con: boolean mask for x, indicating constrained DOF
    """
    return x[free_idx]

def potential(r):
    """Interatomic potential function"""
    # Use where to handle r=0
    s = 0.89 / jnp.sqrt(2)
    u = 0.1
    lj = lambda r: 4*u*((s/r)**12 - (s/r)**6)
    #return jnp.where(r>0, -2/r + 1/r**2, 0.0)
    return jnp.where(r>0, lj(r), 0.0)

#@jit
def V_total_lax(x):
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

#@jit
def V_total_jnp(x):
    """System potential energy function
    Requires high ram because it creates an array of shape (Nx,3,Nx)
    """
    # Interatomic distances matrix
    dx = jnp.reshape(x, (-1,3,1)) - jnp.reshape(x.T, (1,3,-1))
    R = jnp.sqrt(jnp.sum(jnp.square(dx), axis=1))
    return jnp.sum(potential(R))

# Decorator to specify that con & x0 are constants
#@jit(static_argnums=(2,3))
def L(q, qd, free_idx, x0):
    """Lagrangian L=T-V"""
    # Recover physical position & velocity, including constrained DOF
    x = q_to_x(q, free_idx, x0)
    xd = qd_to_xd(qd, free_idx, x0)
    T = atom_mass/2*jnp.sum(xd**2)
    V = V_total_lax(x)
    return T-V

#def Qnc(t, q, qd):
#    """Non-conservative forces"""
#    # TEST: Constant force on DOF 1
#    f = jnp.zeros_like(q)
#    return f.at[0].set(3.0)

# TODO: Constrain DOF
#   That DOF is no longer a generalized coordinate
#   It is a constant value, needed to calculate energies
#   It should be re-inserted into the final array of positions

if __name__ == "__main__":

    # Start timing
    times = tic()

    # Load initial conditions from file
    x0 = np.loadtxt(in_path, delimiter=',', skiprows=1)
    x0 = jnp.array(x0)
    Nx = x0.shape[0]
    print(f"Loaded initial positions of {Nx} atoms from:")
    print(in_path)

    # Array specifying which DOF are constrained
    con = np.zeros_like(x0).astype(bool)
    #con[0,:] = 1    # Constrain atom 0 fully
    left_face = x0[:,0] < -2.75
    con[left_face,0] = 1
    con = jnp.array(con)

    free_idx = jnp.where(~con)

    # Nonconservative forces (None for now)
    Qnc = None

    # Remove the constrained DOF from the simulation
    q0 = x_to_q(x0, free_idx)
    Nq = q0.size

    # Simulation parameters
    ts = jnp.linspace(0, t1, Nt)
    #q0 = jnp.reshape(x0, (-1,))

    """ #TESTING
    q0 = q0[0:9]
    print(q0)
    print("V_total_lax(q0): ", V_total_lax(q0))
    print("V_total_jnp(q0): ", V_total_jnp(q0))
    """

    qd0 = jnp.zeros_like(q0)
    y0 = jnp.concatenate([q0, qd0])

    # Equinox module to be simulated with diffrax
    #fL = lambda q,qd: L(q, qd, con, x0)
    fL = jit(partial(L, free_idx=free_idx, x0=x0),
            static_argnames=("free_idx", "x0"))

    mod = AutoEL(fL, Qnc)

    """ TESTING
    print(171, L(q0, qd0, free_idx, x0))
    print(172, fL(q0, qd0))
    print(173, mod.L(q0, qd0))
    """

    # Run simulation
    toc(times, "Setup")
    sol = simulate(mod, ts, y0, tol=1e-6, max_steps=int(1e6))
    toc(times, "Simulation")

    # Save state vector at each timestep
    ys = np.array(sol.ys)
    #np.savetxt(out_path, ys, delimiter=',', header='x,y,z', comments='')

    # Animate with visualize
    #xs = np.reshape(ys[:, 0:3*Nx], (Nt,Nx,3))
    qs = ys[:,0:Nq]
    #xs = q_to_x(qs, free_idx, x0)
    xs = np.array([q_to_x(qi, free_idx, x0) for qi in qs])
    app_plot_sol(xs)

