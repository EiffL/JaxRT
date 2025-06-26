"""
Interpolation utilities for JaxRT.

This module provides shared interpolation functions used across the JaxRT package.
"""

import jax
import jax.numpy as jnp


def _validate_interpolation_inputs(
    density_map: jnp.ndarray,
    positions: jnp.ndarray,
    map_size_rad: float,
    map_resolution: int
) -> None:
    """
    Validate inputs for interpolation functions.
    
    Args:
        density_map: 2D density map
        positions: Angular positions [2, n_rays] in radians
        map_size_rad: Physical size of the map in radians
        map_resolution: Resolution of the density map
        
    Raises:
        ValueError: If inputs are invalid
    """
    if density_map.ndim != 2:
        raise ValueError(f"density_map must be 2D, got shape {density_map.shape}")
    
    if density_map.shape[0] != density_map.shape[1]:
        raise ValueError(f"density_map must be square, got shape {density_map.shape}")
    
    if density_map.shape[0] != map_resolution:
        raise ValueError(
            f"density_map size {density_map.shape[0]} doesn't match map_resolution {map_resolution}"
        )
    
    if positions.ndim != 2 or positions.shape[0] != 2:
        raise ValueError(f"positions must have shape [2, n_rays], got {positions.shape}")
    
    if map_size_rad <= 0:
        raise ValueError(f"map_size_rad must be positive, got {map_size_rad}")
    
    if map_resolution <= 0:
        raise ValueError(f"map_resolution must be positive, got {map_resolution}")
    
    # Check for NaN/inf values
    if jnp.any(jnp.isnan(density_map)) or jnp.any(jnp.isinf(density_map)):
        raise ValueError("density_map contains NaN or infinite values")
    
    if jnp.any(jnp.isnan(positions)) or jnp.any(jnp.isinf(positions)):
        raise ValueError("positions contains NaN or infinite values")


@jax.jit
def interpolate_density_at_positions(
    density_map: jnp.ndarray,
    positions: jnp.ndarray,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Interpolate density values at given angular positions using bilinear interpolation.
    
    This function uses proper boundary handling with periodic boundary conditions
    instead of edge duplication to avoid interpolation artifacts.
    
    Args:
        density_map: 2D density map [map_resolution, map_resolution]
        positions: Angular positions [2, n_rays] in radians
        map_size_rad: Physical size of the map in radians
        map_resolution: Resolution of the density map
        
    Returns:
        Interpolated density values [n_rays]
    """
    # Convert angular positions to pixel coordinates
    # positions are in radians, centered at map_size_rad/2
    pixel_coords = (positions + map_size_rad / 2) * map_resolution / map_size_rad
    
    # Use periodic boundary conditions instead of clipping
    pixel_coords = pixel_coords % map_resolution
    
    # Get integer and fractional parts
    i_coords = jnp.floor(pixel_coords).astype(int)
    f_coords = pixel_coords - i_coords
    
    # Extract x and y coordinates
    x_i, y_i = i_coords[0], i_coords[1]
    x_f, y_f = f_coords[0], f_coords[1]
    
    # Handle periodic boundary conditions properly
    x_i1 = (x_i + 1) % map_resolution
    y_i1 = (y_i + 1) % map_resolution
    
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


# Non-JIT version that includes input validation
def interpolate_density_at_positions_safe(
    density_map: jnp.ndarray,
    positions: jnp.ndarray,
    map_size_rad: float,
    map_resolution: int
) -> jnp.ndarray:
    """
    Safe version of interpolate_density_at_positions with input validation.
    
    This version includes comprehensive input validation but cannot be JIT compiled.
    Use this for debugging or when you need validation, otherwise use the JIT version.
    
    Args:
        density_map: 2D density map [map_resolution, map_resolution]
        positions: Angular positions [2, n_rays] in radians
        map_size_rad: Physical size of the map in radians
        map_resolution: Resolution of the density map
        
    Returns:
        Interpolated density values [n_rays]
        
    Raises:
        ValueError: If inputs are invalid
    """
    _validate_interpolation_inputs(density_map, positions, map_size_rad, map_resolution)
    return interpolate_density_at_positions(density_map, positions, map_size_rad, map_resolution)