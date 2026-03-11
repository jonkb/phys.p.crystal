""" lattice_energy.py
Contains functions for the kinetic and potential energy of a crystal
"""

import jax.numpy as jnp
import jax
from jax import lax
import equinox as eqx

def potential_lj(r, s, u):
    """Lennard-Jones interatomic potential function"""
    #lj = lambda r: 4*u*((s/r)**12 - (s/r)**6)
    lj = 4*u*((s/r)**12 - (s/r)**6)
    #return jnp.where(r>0, -2/r + 1/r**2, 0.0)
    return jnp.where(r>0, lj, 0.0)
    #return jnp.where(r>0, lj(r), 0.0)

def potential_morse(r, De, a, req):
    """Morse interatomic potential function"""
    arg = -a*(r - req)
    morse = De*(jnp.exp(2*arg) - 2*jnp.exp(arg))
    return morse

class LatticeLagrangian(eqx.Module):
    # Constants
    free_idx: tuple # Non-constrained DOF
    x0: jax.Array # Initial positions, to use for constrained DOF
    atom_mass: float

    def q_to_x(self, q):
        """Given the generalized coordinates, return the atomic positions
        q: Generalized coordinates
        free_idx: free DOF
        x0: Initial positions
        """
        return self.x0.at[self.free_idx].set(q)

    def qd_to_xd(self, qd):
        """Given the generalized velocities, return the atomic velocities
        qd: Generalized velocities
        free_idx: free DOF
        Constrained DOF always have zero velocity
        """
        xd = jnp.zeros_like(self.x0)
        return xd.at[self.free_idx].set(qd)

    def x_to_q(self, x):
        """Given the positions x, return the generalized coordinates
        x: Atomic positions
        free_idx: free DOF
        """
        return x[self.free_idx]

    def potential(self, r):
        """Interatomic potential function"""
        return 1/r

    def V_total_lax(self, x):
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
            Vij = self.potential(r)
            return (V+Vij, x, i)
        def outer_loop(i, carry):
            V, x = carry
            Vi, _, _ = lax.fori_loop(i+1, Nx, inner_loop, (0.0, x, i))
            return (V+Vi, x)

        V, _ = lax.fori_loop(0, Nx, outer_loop, (0.0, x))
        # 2* to count both atoms in each bond
        return 2*V

    @jax.jit
    def __call__(self, q, qd):
        """Lagrangian L=T-V"""
        # Recover physical position & velocity, including constrained DOF
        x = self.q_to_x(q)
        xd = self.qd_to_xd(qd)
        T = self.atom_mass/2*jnp.sum(xd**2)
        V = self.V_total_lax(x)
        return T-V

class LJLagrangian(LatticeLagrangian):
    epsilon_depth: float
    sigma_r0: float

    def potential(self, r):
        """Interatomic potential function"""
        return potential_lj(r, self.sigma_r0, self.epsilon_depth)

class MorseLagrangian(LatticeLagrangian):
    De_depth: float
    a_slope: float
    req: float

    def potential(self, r):
        """Interatomic potential function"""
        return potential_lj(r, self.De_depth, self.a_slope, self.req)
