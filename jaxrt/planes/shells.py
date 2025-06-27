"""
JAX-based particle binning onto spherical shells for ray tracing.

This module provides functionality to bin dark matter particles from N-body
simulations onto spherical shells and convert them to HEALPix maps for
gravitational lensing ray tracing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import healpy as hp
from typing import Optional, Tuple, Union
from astropy.cosmology import FLRW
import astropy.units as u


@jax.jit
def _compute_distances(positions: jnp.ndarray, observer_position: jnp.ndarray) -> jnp.ndarray:
    """Compute distances from observer to particles."""
    delta = positions - observer_position
    return jnp.sqrt(jnp.sum(delta**2, axis=1))


@jax.jit
def _bin_particles_to_shell(
    positions: jnp.ndarray,
    masses: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_distance: float,
    shell_thickness: float,
    nside: int,
) -> jnp.ndarray:
    """Bin particles to a single spherical shell."""
    # Compute distances
    distances = _compute_distances(positions, observer_position)
    
    # Find particles in shell
    shell_min = shell_distance - shell_thickness / 2
    shell_max = shell_distance + shell_thickness / 2
    in_shell = (distances >= shell_min) & (distances <= shell_max)
    
    # Get shell particles
    shell_positions = positions[in_shell]
    shell_masses = masses[in_shell]
    
    if shell_positions.shape[0] == 0:
        # No particles in shell
        return jnp.zeros(12 * nside**2)
    
    # Convert to angular positions from observer
    shell_vectors = shell_positions - observer_position
    shell_distances = jnp.sqrt(jnp.sum(shell_vectors**2, axis=1))
    shell_unit_vectors = shell_vectors / shell_distances[:, None]
    
    # Convert to theta, phi coordinates
    theta = jnp.arccos(jnp.clip(shell_unit_vectors[:, 2], -1, 1))
    phi = jnp.arctan2(shell_unit_vectors[:, 1], shell_unit_vectors[:, 0])
    
    # Convert to HEALPix pixel indices
    ipix = hp.ang2pix(nside, theta, phi, nest=False)
    
    # Bin masses into pixels using scatter-add
    npix = 12 * nside**2
    surface_density = jnp.zeros(npix)
    surface_density = surface_density.at[ipix].add(shell_masses)
    
    # Convert to surface density (mass per unit area)
    # Shell area = 4π * r²
    shell_area = 4 * jnp.pi * shell_distance**2
    pixel_area = shell_area / npix
    surface_density = surface_density / pixel_area
    
    return surface_density


def create_density_shells_from_particles(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_distances: jnp.ndarray,
    shell_thickness: float,
    nside: int,
    cosmology: Optional[FLRW] = None,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Create density shells from N-body simulation particles.
    
    Parameters
    ----------
    particle_positions : jnp.ndarray
        Particle positions with shape (N_particles, 3) in Mpc/h
    particle_masses : jnp.ndarray
        Particle masses with shape (N_particles,) in Msun/h
    observer_position : jnp.ndarray
        Observer position with shape (3,) in Mpc/h
    shell_distances : jnp.ndarray
        Shell distances from observer in Mpc/h
    shell_thickness : float
        Thickness of each shell in Mpc/h
    nside : int
        HEALPix resolution parameter
    cosmology : astropy.cosmology.FLRW, optional
        Cosmology object for distance-redshift conversion
        
    Returns
    -------
    shell_maps : jnp.ndarray
        Surface density maps for each shell with shape (n_shells, 12*nside**2)
        in units of Msun/h per (Mpc/h)²
    shell_redshifts : jnp.ndarray, optional
        Redshifts corresponding to shell distances (if cosmology provided)
    """
    n_shells = len(shell_distances)
    npix = 12 * nside**2
    
    # Initialize output arrays
    shell_maps = jnp.zeros((n_shells, npix))
    
    # Process each shell
    for i, shell_distance in enumerate(shell_distances):
        shell_map = _bin_particles_to_shell(
            particle_positions,
            particle_masses,
            observer_position,
            shell_distance,
            shell_thickness,
            nside,
        )
        shell_maps = shell_maps.at[i].set(shell_map)
    
    # Convert distances to redshifts if cosmology provided
    shell_redshifts = None
    if cosmology is not None:
        # Convert comoving distance to redshift
        distances_with_units = shell_distances * u.Mpc / cosmology.h
        shell_redshifts = jnp.array([
            cosmology.z_at_value(cosmology.comoving_distance, d)
            for d in distances_with_units
        ])
    
    return shell_maps, shell_redshifts


def create_shells_from_lightcone(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    particle_redshifts: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    redshift_thickness: float,
    nside: int,
    cosmology: FLRW,
) -> jnp.ndarray:
    """
    Create density shells from lightcone simulation particles using redshift binning.
    
    Parameters
    ----------
    particle_positions : jnp.ndarray
        Particle positions with shape (N_particles, 3) in Mpc/h
    particle_masses : jnp.ndarray
        Particle masses with shape (N_particles,) in Msun/h
    particle_redshifts : jnp.ndarray
        Particle redshifts with shape (N_particles,)
    observer_position : jnp.ndarray
        Observer position with shape (3,) in Mpc/h
    shell_redshifts : jnp.ndarray
        Shell redshifts for binning
    redshift_thickness : float
        Thickness of each redshift shell
    nside : int
        HEALPix resolution parameter
    cosmology : astropy.cosmology.FLRW
        Cosmology object for distance calculations
        
    Returns
    -------
    shell_maps : jnp.ndarray
        Surface density maps for each shell with shape (n_shells, 12*nside**2)
    """
    n_shells = len(shell_redshifts)
    npix = 12 * nside**2
    shell_maps = jnp.zeros((n_shells, npix))
    
    for i, shell_z in enumerate(shell_redshifts):
        # Find particles in redshift shell
        z_min = shell_z - redshift_thickness / 2
        z_max = shell_z + redshift_thickness / 2
        in_shell = (particle_redshifts >= z_min) & (particle_redshifts <= z_max)
        
        if jnp.sum(in_shell) == 0:
            continue
            
        shell_positions = particle_positions[in_shell]
        shell_masses = particle_masses[in_shell]
        
        # Compute shell distance from cosmology
        shell_distance = cosmology.comoving_distance(shell_z).to(u.Mpc).value * cosmology.h
        
        # Create shell map
        shell_map = _bin_particles_to_shell(
            shell_positions,
            shell_masses,
            observer_position,
            shell_distance,
            redshift_thickness * shell_distance / shell_z,  # Approximate thickness in Mpc/h
            nside,
        )
        shell_maps = shell_maps.at[i].set(shell_map)
    
    return shell_maps


def convert_shells_to_convergence(
    shell_maps: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    source_redshift: float,
    cosmology: FLRW,
) -> jnp.ndarray:
    """
    Convert surface density shells to convergence maps for weak lensing.
    
    Parameters
    ----------
    shell_maps : jnp.ndarray
        Surface density maps with shape (n_shells, npix)
    shell_redshifts : jnp.ndarray
        Redshifts of the shells
    source_redshift : float
        Redshift of source galaxies
    cosmology : astropy.cosmology.FLRW
        Cosmology object
        
    Returns
    -------
    convergence_maps : jnp.ndarray
        Convergence maps with shape (n_shells, npix)
    """
    # Critical surface density
    c = 299792458  # m/s
    G = 6.67430e-11  # m³ kg⁻¹ s⁻²
    
    # Convert units and compute critical surface density
    critical_density = 3 * cosmology.H0**2 / (8 * jnp.pi * G)  # in proper units
    
    # Lensing efficiency
    d_source = cosmology.angular_diameter_distance(source_redshift)
    
    convergence_maps = jnp.zeros_like(shell_maps)
    
    for i, shell_z in enumerate(shell_redshifts):
        if shell_z >= source_redshift:
            continue
            
        d_lens = cosmology.angular_diameter_distance(shell_z)
        d_lens_source = cosmology.angular_diameter_distance_z1z2(shell_z, source_redshift)
        
        # Lensing efficiency
        lensing_efficiency = d_lens * d_lens_source / d_source
        
        # Convert surface density to convergence
        sigma_crit = critical_density * d_source / (d_lens * d_lens_source)
        convergence_maps = convergence_maps.at[i].set(
            shell_maps[i] / sigma_crit * lensing_efficiency
        )
    
    return convergence_maps


def save_shells_to_fits(
    shell_maps: jnp.ndarray,
    filename: str,
    shell_redshifts: Optional[jnp.ndarray] = None,
    nest: bool = False,
) -> None:
    """
    Save shell maps to FITS files.
    
    Parameters
    ----------
    shell_maps : jnp.ndarray
        Shell maps with shape (n_shells, npix)
    filename : str
        Base filename (shell index will be appended)
    shell_redshifts : jnp.ndarray, optional
        Redshifts for each shell
    nest : bool
        Whether maps use NEST ordering
    """
    n_shells = shell_maps.shape[0]
    
    for i in range(n_shells):
        shell_filename = f"{filename}_shell_{i:03d}.fits"
        
        # Create header with metadata
        header = {}
        if shell_redshifts is not None:
            header['REDSHIFT'] = float(shell_redshifts[i])
        header['SHELL_ID'] = i
        
        # Save map
        hp.write_map(
            shell_filename,
            np.array(shell_maps[i]),
            nest=nest,
            coord='C',
            extra_header=header,
        )