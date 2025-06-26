"""
Born convergence implementation in JAX.

This module implements the Born approximation for gravitational lensing
convergence calculations, following the mathematical framework used in LensTools.
"""

import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
import jax_cosmo as jc
from ..utils import interpolate_density_at_positions


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


# Interpolation function moved to utils.interpolation module


def _validate_born_convergence_inputs(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_distances: jnp.ndarray,
    source_distance: float,
    map_size_rad: float,
    map_resolution: int
) -> None:
    """
    Validate inputs for born_convergence function.
    
    Args:
        ray_positions: Angular positions of rays [2, n_rays] in radians
        density_planes: List of density maps
        plane_distances: Comoving distances to each plane [n_planes] in Mpc
        source_distance: Comoving distance to source in Mpc
        map_size_rad: Angular size of each density map in radians
        map_resolution: Resolution of each density map
        
    Raises:
        ValueError: If inputs are invalid
    """
    if ray_positions.ndim != 2 or ray_positions.shape[0] != 2:
        raise ValueError(f"ray_positions must have shape [2, n_rays], got {ray_positions.shape}")
    
    if len(density_planes) == 0:
        raise ValueError("density_planes cannot be empty")
    
    if len(density_planes) != len(plane_distances):
        raise ValueError(
            f"Number of density_planes ({len(density_planes)}) must match "
            f"number of plane_distances ({len(plane_distances)})"
        )
    
    if source_distance <= 0:
        raise ValueError(f"source_distance must be positive, got {source_distance}")
    
    if map_size_rad <= 0:
        raise ValueError(f"map_size_rad must be positive, got {map_size_rad}")
    
    if map_resolution <= 0:
        raise ValueError(f"map_resolution must be positive, got {map_resolution}")
    
    # Check that all density planes have the correct shape
    for i, plane in enumerate(density_planes):
        expected_shape = (map_resolution, map_resolution)
        if plane.shape != expected_shape:
            raise ValueError(
                f"density_planes[{i}] has shape {plane.shape}, expected {expected_shape}"
            )
    
    # Check that plane distances are sorted and positive
    if jnp.any(plane_distances <= 0):
        raise ValueError("All plane_distances must be positive")
    
    if jnp.any(plane_distances[1:] <= plane_distances[:-1]):
        raise ValueError("plane_distances must be sorted in ascending order")
    
    if jnp.any(plane_distances >= source_distance):
        raise ValueError("All plane_distances must be less than source_distance")


@jax.jit
def _compute_distance_intervals(plane_distances: jnp.ndarray) -> jnp.ndarray:
    """
    Compute distance intervals between planes with proper edge handling.
    
    Args:
        plane_distances: Comoving distances to each plane [n_planes] in Mpc
        
    Returns:
        Distance intervals [n_planes] in Mpc
    """
    n_planes = len(plane_distances)
    
    # For multiple planes, use forward differences for interior points
    # and backward difference for the last point
    intervals = jnp.zeros(n_planes)
    
    if n_planes == 1:
        # For single plane, use a default interval based on typical lensing distances
        # This is more physically meaningful than arbitrary 1.0
        intervals = intervals.at[0].set(plane_distances[0] / 10.0)
    else:
        # Forward differences for all but the last plane
        forward_diffs = plane_distances[1:] - plane_distances[:-1]
        intervals = intervals.at[:-1].set(forward_diffs)
        
        # For the last plane, use the same interval as the previous one
        intervals = intervals.at[-1].set(forward_diffs[-1])
    
    return intervals


@jax.jit 
def _vectorized_born_convergence_core(
    ray_positions: jnp.ndarray,
    density_planes_array: jnp.ndarray,
    plane_distances: jnp.ndarray,
    source_distance: float,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Vectorized core computation for Born convergence.
    
    This function is fully vectorized and JIT-compilable.
    
    Args:
        ray_positions: Angular positions of rays [2, n_rays] in radians
        density_planes_array: Density maps [n_planes, map_resolution, map_resolution]
        plane_distances: Comoving distances to each plane [n_planes] in Mpc
        source_distance: Comoving distance to source in Mpc
        map_size_rad: Angular size of each density map in radians
        map_resolution: Resolution of each density map
        
    Returns:
        Convergence values at ray positions [n_rays]
    """
    n_planes, _, _ = density_planes_array.shape
    n_rays = ray_positions.shape[1]
    
    # Compute lensing kernel for all planes
    kernels = lensing_kernel(plane_distances, source_distance)
    
    # Compute distance intervals  
    d_chi = _compute_distance_intervals(plane_distances)
    
    # Vectorized interpolation for all planes at once
    def interpolate_single_plane(plane_data):
        plane, kernel, interval = plane_data
        density_at_rays = interpolate_density_at_positions(
            plane, ray_positions, map_size_rad, map_resolution
        )
        return density_at_rays * kernel * interval
    
    # Use vmap to vectorize over planes
    plane_data = (density_planes_array, kernels, d_chi)
    contributions = jax.vmap(interpolate_single_plane, in_axes=(0,))(plane_data)
    
    # Sum contributions from all planes
    convergence = jnp.sum(contributions, axis=0)
    
    return convergence


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
    
    This version uses vectorized operations for improved performance
    and proper distance interval calculation to avoid edge case bugs.
    
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
    # Convert list to array for vectorized operations
    density_planes_array = jnp.array(density_planes)
    
    return _vectorized_born_convergence_core(
        ray_positions, density_planes_array, plane_distances,
        source_distance, map_size_rad, map_resolution
    )


def born_convergence_safe(
    ray_positions: jnp.ndarray,
    density_planes: List[jnp.ndarray],
    plane_distances: jnp.ndarray,
    source_distance: float,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Safe version of born_convergence with comprehensive input validation.
    
    This version includes full input validation but cannot be JIT compiled.
    Use this for debugging or when you need validation, otherwise use the JIT version.
    
    Args:
        ray_positions: Angular positions of rays [2, n_rays] in radians
        density_planes: List of density maps, each [map_resolution, map_resolution]
        plane_distances: Comoving distances to each plane [n_planes] in Mpc
        source_distance: Comoving distance to source in Mpc
        map_size_rad: Angular size of each density map in radians
        map_resolution: Resolution of each density map
        
    Returns:
        Convergence values at ray positions [n_rays]
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_born_convergence_inputs(
        ray_positions, density_planes, plane_distances,
        source_distance, map_size_rad, map_resolution
    )
    
    return born_convergence(
        ray_positions, density_planes, plane_distances,
        source_distance, map_size_rad, map_resolution
    )


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