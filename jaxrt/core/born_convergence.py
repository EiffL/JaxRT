"""
Born convergence implementation in JAX.

This module implements the Born approximation for gravitational lensing
convergence calculations, following the KISS principle.
"""

import jax
import jax.numpy as jnp
from typing import List, Optional
import jax_cosmo as jc


@jax.jit
def lensing_kernel(chi: jnp.ndarray, chi_source: float) -> jnp.ndarray:
    """
    Compute the lensing kernel W(chi, chi_source).
    
    The lensing kernel is: W(chi, chi_source) = (1 - chi/chi_source)
    
    Args:
        chi: Comoving distances to lens planes [Mpc]
        chi_source: Comoving distance to source [Mpc]
        
    Returns:
        Lensing kernel values
    """
    return jnp.where(chi < chi_source, 1.0 - chi / chi_source, 0.0)


@jax.jit
def _interpolate_density(
    density_map: jnp.ndarray,
    positions: jnp.ndarray,
    map_size_rad: float
) -> jnp.ndarray:
    """
    Bilinear interpolation of density map at given positions.
    
    Args:
        density_map: 2D density map [resolution, resolution]
        positions: Angular positions [2, n_rays] in radians
        map_size_rad: Angular size of the map in radians
        
    Returns:
        Interpolated density values [n_rays]
    """
    resolution = density_map.shape[0]
    
    # Convert to pixel coordinates (centered, periodic boundaries)
    pixel_coords = (positions + map_size_rad / 2) * resolution / map_size_rad
    pixel_coords = pixel_coords % resolution
    
    # Get integer and fractional parts
    i_coords = jnp.floor(pixel_coords).astype(int)
    f_coords = pixel_coords - i_coords
    
    x_i, y_i = i_coords[0], i_coords[1]
    x_f, y_f = f_coords[0], f_coords[1]
    
    # Next pixel coordinates (periodic)
    x_i1 = (x_i + 1) % resolution
    y_i1 = (y_i + 1) % resolution
    
    # Bilinear interpolation
    f00 = density_map[y_i, x_i]
    f10 = density_map[y_i, x_i1]
    f01 = density_map[y_i1, x_i]
    f11 = density_map[y_i1, x_i1]
    
    return (f00 * (1 - x_f) * (1 - y_f) + 
            f10 * x_f * (1 - y_f) + 
            f01 * (1 - x_f) * y_f + 
            f11 * x_f * y_f)


@jax.jit
def born_convergence(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_distances: jnp.ndarray,
    source_distance: float,
    map_size_rad: float
) -> jnp.ndarray:
    """
    Compute Born convergence from density planes.
    
    The Born convergence is computed as:
    κ(θ) = ∫ δ(χ, θ) * W(χ, χs) dχ
    
    where W(χ, χs) = (1 - χ/χs) is the lensing kernel.
    
    Args:
        ray_positions: Angular positions [2, n_rays] in radians
        density_planes: List of density maps
        plane_distances: Comoving distances to planes [n_planes] in Mpc
        source_distance: Comoving distance to source in Mpc
        map_size_rad: Angular size of density maps in radians
        
    Returns:
        Convergence values at ray positions [n_rays]
    """
    # Basic input validation
    n_planes = len(density_planes)
    if n_planes == 0:
        raise ValueError("No density planes provided")
    if len(plane_distances) != n_planes:
        raise ValueError(f"Mismatch: {n_planes} planes, {len(plane_distances)} distances")
    
    # Convert to array for vectorized operations
    density_array = jnp.array(density_planes)
    
    # Compute lensing kernels
    kernels = lensing_kernel(plane_distances, source_distance)
    
    # Compute distance intervals (simple forward differences)
    if n_planes == 1:
        d_chi = jnp.array([plane_distances[0] / 10.0])  # Default interval
    else:
        d_chi = jnp.concatenate([
            plane_distances[1:] - plane_distances[:-1],  # Forward differences
            [plane_distances[-1] - plane_distances[-2]]   # Last interval
        ])
    
    # Vectorized computation over all planes
    def compute_plane_contribution(plane_data):
        density_map, kernel, interval = plane_data
        density_at_rays = _interpolate_density(density_map, ray_positions, map_size_rad)
        return density_at_rays * kernel * interval
    
    # Apply to all planes and sum
    plane_data = (density_array, kernels, d_chi)
    contributions = jax.vmap(compute_plane_contribution, in_axes=(0,))(plane_data)
    
    return jnp.sum(contributions, axis=0)


def born_convergence_from_cosmology(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_redshifts: jnp.ndarray,
    source_redshift: float,
    map_size_rad: float,
    cosmology: Optional[jc.Cosmology] = None
) -> jnp.ndarray:
    """
    Compute Born convergence using cosmological distances.
    
    Args:
        ray_positions: Angular positions [2, n_rays] in radians
        density_planes: List of density maps
        plane_redshifts: Redshifts of each plane [n_planes]
        source_redshift: Source redshift
        map_size_rad: Angular size of density maps in radians
        cosmology: JAX-Cosmo cosmology (default: Planck18)
        
    Returns:
        Convergence values at ray positions [n_rays]
    """
    if cosmology is None:
        cosmology = jc.Planck18()
    
    # Compute comoving distances
    plane_distances = jc.background.radial_comoving_distance(
        cosmology, 1.0 / (1.0 + plane_redshifts)
    )
    source_distance = jc.background.radial_comoving_distance(
        cosmology, 1.0 / (1.0 + source_redshift)
    )
    
    return born_convergence(
        ray_positions, density_planes, plane_distances, 
        source_distance, map_size_rad
    )