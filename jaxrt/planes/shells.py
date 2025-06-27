"""
Spherical shell creation utilities for gravitational lensing ray tracing.

This module provides functions to bin dark matter particles from n-body simulations
onto spherical shells and convert them to HEALPix maps for efficient ray tracing.
"""

import jax
import jax.numpy as jnp
import healpy as hp
import numpy as np
from typing import Optional, Tuple, Union
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


def create_density_shells_from_particles(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    observer_position: jnp.ndarray,
    shell_distances: jnp.ndarray,
    shell_thickness: float,
    nside: int = 512,
    cosmology: Optional[FlatLambdaCDM] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create density shells from dark matter particles by binning them onto spherical shells.
    
    Args:
        particle_positions: Particle positions in comoving coordinates [N, 3] (Mpc/h)
        particle_masses: Particle masses [N] (Msun/h)
        observer_position: Observer position [3] (Mpc/h)
        shell_distances: Radial distances to shell centers [n_shells] (Mpc/h)
        shell_thickness: Thickness of each shell (Mpc/h)
        nside: HEALPix nside parameter (must be power of 2)
        cosmology: Astropy cosmology object (optional)
        
    Returns:
        Tuple of (shell_maps, shell_redshifts)
        - shell_maps: HEALPix maps for each shell [n_shells, 12*nside**2]
        - shell_redshifts: Redshifts corresponding to each shell [n_shells]
    """
    n_shells = len(shell_distances)
    n_pixels = hp.nside2npix(nside)
    
    # Initialize shell maps
    shell_maps = jnp.zeros((n_shells, n_pixels))
    
    # Compute particle distances from observer
    particle_positions_rel = particle_positions - observer_position
    particle_distances = jnp.linalg.norm(particle_positions_rel, axis=1)
    
    # Convert to redshifts if cosmology is provided
    if cosmology is not None:
        # Convert comoving distances to redshifts
        shell_redshifts = jnp.array([
            cosmology.z_at_value(cosmology.comoving_distance, dist * u.Mpc).value
            for dist in shell_distances
        ])
    else:
        # Simple approximation: z ≈ H0 * d / c
        H0 = 70.0  # km/s/Mpc
        c = 299792.458  # km/s
        shell_redshifts = H0 * shell_distances / c
    
    # Bin particles into shells
    shell_maps_list = []
    for i, shell_dist in enumerate(shell_distances):
        # Select particles within shell
        shell_min = shell_dist - shell_thickness / 2
        shell_max = shell_dist + shell_thickness / 2
        
        in_shell = (particle_distances >= shell_min) & (particle_distances < shell_max)
        shell_particles = particle_positions_rel[in_shell]
        shell_masses = particle_masses[in_shell]
        
        if len(shell_particles) > 0:
            # Convert to spherical coordinates
            shell_map = _particles_to_healpix_map(
                shell_particles, shell_masses, nside, shell_dist
            )
        else:
            shell_map = jnp.zeros(n_pixels)
        
        shell_maps_list.append(shell_map)
    
    shell_maps = jnp.array(shell_maps_list)
    
    return shell_maps, shell_redshifts


@jax.jit
def _particles_to_healpix_map(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    nside: int,
    shell_radius: float
) -> jnp.ndarray:
    """
    Convert particle positions to HEALPix map using JAX operations.
    
    Args:
        particle_positions: Particle positions relative to observer [N, 3]
        particle_masses: Particle masses [N]
        nside: HEALPix nside parameter
        shell_radius: Radius of the shell for surface density calculation
        
    Returns:
        HEALPix map of surface density [12*nside**2]
    """
    # Convert Cartesian to spherical coordinates
    x, y, z = particle_positions[:, 0], particle_positions[:, 1], particle_positions[:, 2]
    
    # Compute theta (colatitude) and phi (azimuth)
    r = jnp.sqrt(x**2 + y**2 + z**2)
    theta = jnp.arccos(z / r)  # colatitude [0, π]
    phi = jnp.arctan2(y, x)    # azimuth [-π, π]
    
    # Convert to HEALPix pixel indices
    # Note: We need to use numpy here since healpy doesn't support JAX
    theta_np = np.array(theta)
    phi_np = np.array(phi)
    pixel_indices = hp.ang2pix(nside, theta_np, phi_np)
    
    # Bin masses into pixels
    n_pixels = hp.nside2npix(nside)
    pixel_map = jnp.zeros(n_pixels)
    
    # Use JAX scatter_add for efficient binning
    pixel_map = pixel_map.at[pixel_indices].add(particle_masses)
    
    # Convert to surface density (mass per unit area)
    pixel_area = 4 * jnp.pi * shell_radius**2 / n_pixels  # Area per pixel
    surface_density_map = pixel_map / pixel_area
    
    return surface_density_map


def create_shells_from_lightcone(
    particle_positions: jnp.ndarray,
    particle_masses: jnp.ndarray,
    particle_redshifts: jnp.ndarray,
    observer_position: jnp.ndarray,
    target_redshifts: jnp.ndarray,
    redshift_thickness: float = 0.1,
    nside: int = 512,
    cosmology: Optional[FlatLambdaCDM] = None
) -> jnp.ndarray:
    """
    Create density shells from a lightcone simulation by binning particles by redshift.
    
    Args:
        particle_positions: Particle positions [N, 3] (Mpc/h)
        particle_masses: Particle masses [N] (Msun/h)
        particle_redshifts: Redshifts of particles [N]
        observer_position: Observer position [3] (Mpc/h)
        target_redshifts: Target redshifts for shell centers [n_shells]
        redshift_thickness: Thickness of redshift bins
        nside: HEALPix nside parameter
        cosmology: Astropy cosmology object (optional)
        
    Returns:
        HEALPix maps for each redshift shell [n_shells, 12*nside**2]
    """
    n_shells = len(target_redshifts)
    n_pixels = hp.nside2npix(nside)
    
    # Convert redshifts to comoving distances if cosmology provided
    if cosmology is not None:
        shell_distances = jnp.array([
            cosmology.comoving_distance(z).to(u.Mpc).value
            for z in target_redshifts
        ])
    else:
        # Simple approximation
        H0 = 70.0  # km/s/Mpc
        c = 299792.458  # km/s
        shell_distances = c * target_redshifts / H0
    
    shell_maps_list = []
    
    for i, target_z in enumerate(target_redshifts):
        # Select particles within redshift range
        z_min = target_z - redshift_thickness / 2
        z_max = target_z + redshift_thickness / 2
        
        in_shell = (particle_redshifts >= z_min) & (particle_redshifts < z_max)
        shell_particles = particle_positions[in_shell] - observer_position
        shell_masses = particle_masses[in_shell]
        
        if len(shell_particles) > 0:
            shell_map = _particles_to_healpix_map(
                shell_particles, shell_masses, nside, shell_distances[i]
            )
        else:
            shell_map = jnp.zeros(n_pixels)
        
        shell_maps_list.append(shell_map)
    
    return jnp.array(shell_maps_list)


def convert_shells_to_convergence(
    shell_maps: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    source_redshift: float,
    cosmology: FlatLambdaCDM,
    critical_density: float = 2.77536627e11  # Critical density in h^2 Msun/Mpc^3
) -> jnp.ndarray:
    """
    Convert surface density shells to convergence maps for lensing calculations.
    
    Args:
        shell_maps: Surface density maps [n_shells, n_pixels] (Msun/h / Mpc^2)
        shell_redshifts: Redshifts of shells [n_shells]
        source_redshift: Redshift of background sources
        cosmology: Astropy cosmology object
        critical_density: Critical density of the universe
        
    Returns:
        Convergence maps [n_shells, n_pixels]
    """
    n_shells, n_pixels = shell_maps.shape
    
    # Compute lensing efficiency for each shell
    lensing_efficiency = jnp.zeros(n_shells)
    
    for i, z_lens in enumerate(shell_redshifts):
        if z_lens < source_redshift:
            # Angular diameter distances
            D_l = cosmology.angular_diameter_distance(z_lens).to(u.Mpc).value
            D_s = cosmology.angular_diameter_distance(source_redshift).to(u.Mpc).value
            D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, source_redshift).to(u.Mpc).value
            
            # Lensing efficiency
            efficiency = (D_l * D_ls) / D_s
            lensing_efficiency = lensing_efficiency.at[i].set(efficiency)
    
    # Convert surface density to convergence
    # κ = Σ / Σ_crit, where Σ_crit = c^2 / (4πG) * D_s / (D_l * D_ls)
    G = 4.301e-9  # Gravitational constant in Mpc/Msun * (km/s)^2
    c = 299792.458  # km/s
    
    critical_surface_density = (c**2 / (4 * jnp.pi * G)) / lensing_efficiency[:, None]
    convergence_maps = shell_maps / critical_surface_density
    
    return convergence_maps


def save_shells_to_fits(
    shell_maps: jnp.ndarray,
    shell_redshifts: jnp.ndarray,
    output_prefix: str,
    nside: int
) -> None:
    """
    Save shell maps to FITS files using HEALPix format.
    
    Args:
        shell_maps: Shell maps [n_shells, n_pixels]
        shell_redshifts: Redshifts of shells [n_shells]
        output_prefix: Prefix for output filenames
        nside: HEALPix nside parameter
    """
    for i, (shell_map, z) in enumerate(zip(shell_maps, shell_redshifts)):
        filename = f"{output_prefix}_shell_{i:03d}_z_{z:.3f}.fits"
        hp.write_map(filename, np.array(shell_map), nside=nside, overwrite=True)
        print(f"Saved shell {i} (z={z:.3f}) to {filename}")