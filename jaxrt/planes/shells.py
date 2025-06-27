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
import jax_cosmo as jc


@jax.jit
def _compute_distances(positions: jnp.ndarray, observer_position: jnp.ndarray) -> jnp.ndarray:
    """Compute distances from observer to particles."""
    delta = positions - observer_position
    return jnp.sqrt(jnp.sum(delta**2, axis=1))


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
    
    # Convert to HEALPix pixel indices - handle JAX/NumPy conversion properly
    theta_np = np.array(theta)
    phi_np = np.array(phi)
    ipix = hp.ang2pix(nside, theta_np, phi_np, nest=False)
    ipix = jnp.array(ipix)  # Convert back to JAX array
    
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


def _validate_inputs(
    particle_masses: jnp.ndarray,
    nside: int,
    shell_distances: jnp.ndarray,
) -> None:
    """Validate input parameters."""
    # Check for negative masses
    if jnp.any(particle_masses < 0):
        raise ValueError("Particle masses must be non-negative")
    
    # Check nside is power of 2
    if nside <= 0 or (nside & (nside - 1)) != 0:
        raise ValueError("nside must be a positive power of 2")
    
    # Check for non-positive shell distances
    if jnp.any(shell_distances <= 0):
        raise ValueError("Shell distances must be positive")


def _bin_all_shells(
    positions: jnp.ndarray,
    masses: jnp.ndarray, 
    observer: jnp.ndarray,
    distances: jnp.ndarray,
    thickness: float,
    nside: int,
) -> jnp.ndarray:
    """Vectorized binning of particles to all shells using jax.vmap."""
    return jax.vmap(
        lambda d: _bin_particles_to_shell(positions, masses, observer, d, thickness, nside)
    )(distances)


def create_density_shells_from_particles(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_distances: jnp.ndarray,
    shell_thickness: float,
    nside: int,
    cosmology: Optional[jc.Cosmology] = None,
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
        HEALPix resolution parameter (must be power of 2)
    cosmology : jax_cosmo.Cosmology, optional
        Cosmology object for distance-redshift conversion
        
    Returns
    -------
    shell_maps : jnp.ndarray
        Surface density maps for each shell with shape (n_shells, 12*nside**2)
        in units of Msun/h per (Mpc/h)²
    shell_redshifts : jnp.ndarray, optional
        Redshifts corresponding to shell distances (if cosmology provided)
    """
    # Input validation
    _validate_inputs(particle_masses, nside, shell_distances)
    
    # Use vectorized shell creation for better performance
    shell_maps = _bin_all_shells(
        particle_positions,
        particle_masses,
        observer_position, 
        shell_distances,
        shell_thickness,
        nside,
    )
    
    # Convert distances to redshifts if cosmology provided
    shell_redshifts = None
    if cosmology is not None:
        # Convert comoving distance to redshift using jax-cosmo
        # Distances are in Mpc/h, jax-cosmo expects Mpc
        distances_mpc = shell_distances / cosmology.h
        shell_redshifts = jax.vmap(
            lambda d: jc.background.z_at_chi(cosmology, d)
        )(distances_mpc)
    
    return shell_maps, shell_redshifts


def _create_single_lightcone_shell(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    particle_redshifts: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_z: float,
    redshift_thickness: float,
    nside: int,
    shell_distance: float,
) -> jnp.ndarray:
    """Create a single shell from lightcone particles."""
    # Find particles in redshift shell
    z_min = shell_z - redshift_thickness / 2
    z_max = shell_z + redshift_thickness / 2
    in_shell = (particle_redshifts >= z_min) & (particle_redshifts <= z_max)
    
    # Handle case with no particles
    n_in_shell = jnp.sum(in_shell)
    
    # Use conditional to avoid issues with empty arrays
    shell_map = jnp.where(
        n_in_shell > 0,
        _bin_particles_to_shell(
            particle_positions[in_shell], 
            particle_masses[in_shell],
            observer_position,
            shell_distance,
            redshift_thickness * shell_distance / jnp.maximum(shell_z, 1e-6),  # Avoid division by zero
            nside,
        ),
        jnp.zeros(12 * nside**2)
    )
    
    return shell_map


def create_shells_from_lightcone(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    particle_redshifts: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    redshift_thickness: float,
    nside: int,
    cosmology: jc.Cosmology,
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
        HEALPix resolution parameter (must be power of 2)
    cosmology : jax_cosmo.Cosmology
        Cosmology object for distance calculations
        
    Returns
    -------
    shell_maps : jnp.ndarray
        Surface density maps for each shell with shape (n_shells, 12*nside**2)
    """
    # Input validation
    _validate_inputs(particle_masses, nside, jnp.array([1.0]))  # Basic validation
    
    # Compute shell distances from cosmology
    shell_distances = jax.vmap(
        lambda z: jc.background.radial_comoving_distance(cosmology, 1.0 / (1.0 + z)) * cosmology.h
    )(shell_redshifts)
    
    # Vectorized shell creation
    shell_maps = jax.vmap(
        lambda z, d: _create_single_lightcone_shell(
            particle_positions, particle_masses, particle_redshifts,
            observer_position, z, redshift_thickness, nside, d
        )
    )(shell_redshifts, shell_distances)
    
    return shell_maps


def _compute_single_convergence(
    shell_map: jnp.ndarray,
    shell_z: float,
    source_redshift: float,
    cosmology: jc.Cosmology,
) -> jnp.ndarray:
    """Compute convergence for a single shell."""
    # Skip shells at or behind source
    def compute_convergence():
        # Angular diameter distances (in Mpc)
        a_lens = 1.0 / (1.0 + shell_z)
        a_source = 1.0 / (1.0 + source_redshift)
        d_lens = jc.background.angular_diameter_distance(cosmology, a_lens)
        d_source = jc.background.angular_diameter_distance(cosmology, a_source)
        d_lens_source = jc.background.angular_diameter_distance_z1z2(
            cosmology, a_lens, a_source
        )
        
        # Critical surface density using jax-cosmo constants
        # σ_crit = c² / (4πG) * D_s / (D_l * D_ls)
        c_light = 299792458.0  # m/s (exact)
        G_newton = 6.67430e-11  # m³ kg⁻¹ s⁻² (2018 CODATA)
        
        # Convert to proper units: Msun/h per (Mpc/h)²
        # Factor includes: unit conversions + h factors
        Msun_kg = 1.989e30  # kg
        Mpc_m = 3.086e22  # m
        
        # σ_crit in units of Msun/h per (Mpc/h)²
        sigma_crit_factor = (c_light**2 / (4 * jnp.pi * G_newton)) * (Mpc_m**2 / Msun_kg)
        sigma_crit = sigma_crit_factor * d_source / (d_lens * d_lens_source)
        
        # Account for h factors: distances from jax-cosmo don't include h
        # shell_map is in Msun/h per (Mpc/h)², distances in Mpc
        # Need to multiply by h for correct units
        sigma_crit *= cosmology.h  
        
        # Convergence = surface density / critical surface density
        return shell_map / sigma_crit
    
    # Only compute convergence for shells in front of source
    return jnp.where(
        shell_z < source_redshift,
        compute_convergence(),
        jnp.zeros_like(shell_map)
    )


def convert_shells_to_convergence(
    shell_maps: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    source_redshift: float,
    cosmology: jc.Cosmology,
) -> jnp.ndarray:
    """
    Convert surface density shells to convergence maps for weak lensing.
    
    Parameters
    ----------
    shell_maps : jnp.ndarray
        Surface density maps with shape (n_shells, npix) in Msun/h per (Mpc/h)²
    shell_redshifts : jnp.ndarray
        Redshifts of the shells
    source_redshift : float
        Redshift of source galaxies
    cosmology : jax_cosmo.Cosmology
        Cosmology object
        
    Returns
    -------
    convergence_maps : jnp.ndarray
        Convergence maps with shape (n_shells, npix) (dimensionless)
    """
    # Vectorized convergence computation
    convergence_maps = jax.vmap(
        lambda shell_map, shell_z: _compute_single_convergence(
            shell_map, shell_z, source_redshift, cosmology
        )
    )(shell_maps, shell_redshifts)
    
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