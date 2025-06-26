"""
Born convergence implementation in JAX.

This module implements the Born approximation for gravitational lensing
convergence calculations, following the mathematical framework used in LensTools.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
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
def _interpolate_density_at_positions(
    density_map: jnp.ndarray,
    positions: jnp.ndarray,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Interpolate density values at given angular positions using bilinear interpolation.
    
    Args:
        density_map: 2D density map [map_resolution, map_resolution]
        positions: Angular positions [2, n_rays] in radians
        map_size_rad: Physical size of the map in radians
        map_resolution: Resolution of the density map
        
    Returns:
        Interpolated density values [n_rays]
    """
    # Convert angular positions to pixel coordinates
    # positions are in radians, ranging from 0 to map_size_rad
    pixel_coords = positions * (map_resolution - 1) / map_size_rad
    
    # Ensure coordinates are within bounds
    pixel_coords = jnp.clip(pixel_coords, 0, map_resolution - 1)
    
    # Get integer and fractional parts
    i_coords = jnp.floor(pixel_coords).astype(int)
    f_coords = pixel_coords - i_coords
    
    # Extract x and y coordinates
    x_i, y_i = i_coords[0], i_coords[1]
    x_f, y_f = f_coords[0], f_coords[1]
    
    # Ensure we don't go out of bounds for i+1
    x_i1 = jnp.minimum(x_i + 1, map_resolution - 1)
    y_i1 = jnp.minimum(y_i + 1, map_resolution - 1)
    
    # Bilinear interpolation
    # Get the four corner values
    f00 = density_map[y_i, x_i]
    f10 = density_map[y_i, x_i1]
    f01 = density_map[y_i1, x_i]
    f11 = density_map[y_i1, x_i1]
    
    # Interpolate
    result = (
        f00 * (1 - x_f) * (1 - y_f) +
        f10 * x_f * (1 - y_f) +
        f01 * (1 - x_f) * y_f +
        f11 * x_f * y_f
    )
    
    return result


@jax.jit
def born_convergence(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_distances: jnp.ndarray,
    source_distance: float,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Compute Born convergence from a set of density planes.
    
    The Born convergence is computed as:
    κ(θ) = ∫ δ(χ, θ) * W(χ, χs) dχ
    
    where W(χ, χs) = (1 - χ/χs) is the lensing kernel.
    
    Args:
        ray_positions: Angular positions of rays [2, n_rays] in radians
        density_planes: List of density maps, each [map_resolution, map_resolution]
        plane_distances: Comoving distances to each plane [n_planes] in Mpc
        source_distance: Comoving distance to source in Mpc
        map_size_rad: Angular size of each density map in radians
        map_resolution: Resolution of each density map
        
    Returns:
        Convergence values at ray positions [n_rays]
    """
    n_rays = ray_positions.shape[1]
    n_planes = len(density_planes)
    
    # Initialize convergence array
    convergence = jnp.zeros(n_rays)
    
    # Compute lensing kernel for all planes
    kernels = lensing_kernel(plane_distances, source_distance)
    
    # Integrate over all planes
    for i in range(n_planes):
        # Interpolate density at ray positions
        density_at_rays = _interpolate_density_at_positions(
            density_planes[i], ray_positions, map_size_rad, map_resolution
        )
        
        # Add contribution to convergence
        # For discrete planes, we need to multiply by the distance interval
        if i < n_planes - 1:
            d_chi = plane_distances[i + 1] - plane_distances[i]
        else:
            # For the last plane, use the same interval as the previous one
            d_chi = plane_distances[i] - plane_distances[i - 1] if i > 0 else 1.0
            
        convergence += density_at_rays * kernels[i] * d_chi
    
    return convergence


def born_convergence_from_cosmology(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_redshifts: jnp.ndarray,
    source_redshift: float,
    map_size_rad: float,
    map_resolution: int,
    cosmology: Optional[jc.Cosmology] = None
) -> jnp.ndarray:
    """
    Compute Born convergence using cosmological distances.
    
    Args:
        ray_positions: Angular positions of rays [2, n_rays] in radians
        density_planes: List of density maps, each [map_resolution, map_resolution]
        plane_redshifts: Redshifts of each plane [n_planes]
        source_redshift: Source redshift
        map_size_rad: Angular size of each density map in radians
        map_resolution: Resolution of each density map
        cosmology: JAX-Cosmo cosmology object (default: Planck18)
        
    Returns:
        Convergence values at ray positions [n_rays]
    """
    if cosmology is None:
        cosmology = jc.Planck18()
    
    # Compute comoving distances
    plane_distances = jc.background.radial_comoving_distance(cosmology, 1.0 / (1.0 + plane_redshifts))
    source_distance = jc.background.radial_comoving_distance(cosmology, 1.0 / (1.0 + source_redshift))
    
    return born_convergence(
        ray_positions, 
        density_planes, 
        plane_distances, 
        source_distance,
        map_size_rad,
        map_resolution
    )