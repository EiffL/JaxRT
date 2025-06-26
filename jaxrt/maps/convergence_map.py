"""
Convergence map utilities for gravitational lensing.

This module provides tools for creating and manipulating convergence maps
resulting from gravitational lensing calculations.
"""

import jax.numpy as jnp
from typing import Optional, Tuple
import jax_cosmo as jc


class ConvergenceMap:
    """
    A convergence map for gravitational lensing.
    
    This class represents a 2D convergence field computed from ray tracing
    through gravitational lensing simulations.
    """
    
    def __init__(
        self,
        convergence_map: jnp.ndarray,
        map_size_rad: float,
        source_redshift: float,
        cosmology: Optional[jc.Cosmology] = None
    ):
        """
        Initialize a convergence map.
        
        Args:
            convergence_map: 2D convergence field [n_pix, n_pix]
            map_size_rad: Physical angular size of the map in radians
            source_redshift: Source redshift
            cosmology: JAX-Cosmo cosmology (default: Planck18)
        """
        self.convergence_map = convergence_map
        self.map_size_rad = map_size_rad
        self.source_redshift = source_redshift
        self.resolution = convergence_map.shape[0]
        
        if cosmology is None:
            cosmology = jc.Planck18()
        self.cosmology = cosmology
        
        # Compute source comoving distance
        self.source_comoving_distance = jc.background.radial_comoving_distance(
            cosmology, 1.0 / (1.0 + source_redshift)
        )
    
    @property
    def pixel_size_rad(self) -> float:
        """Pixel size in radians."""
        return self.map_size_rad / self.resolution
    
    @property 
    def pixel_size_arcmin(self) -> float:
        """Pixel size in arcminutes."""
        return self.pixel_size_rad * 180.0 * 60.0 / jnp.pi
    
    @property
    def pixel_size_arcsec(self) -> float:
        """Pixel size in arcseconds."""
        return self.pixel_size_arcmin * 60.0
    
    def statistics(self) -> dict:
        """
        Compute basic statistics of the convergence map.
        
        Returns:
            Dictionary with mean, std, min, max, rms
        """
        return {
            'mean': float(jnp.mean(self.convergence_map)),
            'std': float(jnp.std(self.convergence_map)),
            'min': float(jnp.min(self.convergence_map)),
            'max': float(jnp.max(self.convergence_map)),
            'rms': float(jnp.sqrt(jnp.mean(self.convergence_map**2)))
        }
    
    def create_ray_grid(self) -> jnp.ndarray:
        """
        Create a regular grid of ray positions covering the map.
        
        Returns:
            Ray positions [2, n_rays] in radians
        """
        # Create coordinate arrays
        coords_1d = jnp.linspace(0, self.map_size_rad, self.resolution)
        xx, yy = jnp.meshgrid(coords_1d, coords_1d, indexing='ij')
        
        # Flatten and stack to get [2, n_rays] array
        ray_positions = jnp.stack([xx.flatten(), yy.flatten()], axis=0)
        
        return ray_positions
    
    @classmethod
    def from_ray_positions(
        cls,
        ray_positions: jnp.ndarray,
        convergence_values: jnp.ndarray,
        map_size_rad: float,
        source_redshift: float,
        cosmology: Optional[jc.Cosmology] = None
    ) -> 'ConvergenceMap':
        """
        Create a ConvergenceMap from ray positions and convergence values.
        
        Args:
            ray_positions: Ray positions [2, n_rays] in radians
            convergence_values: Convergence values at ray positions [n_rays]
            map_size_rad: Angular size of the map in radians
            source_redshift: Source redshift
            cosmology: JAX-Cosmo cosmology
            
        Returns:
            ConvergenceMap instance
        """
        # Determine map resolution from ray positions
        # Assume rays are on a regular grid
        n_rays = ray_positions.shape[1]
        resolution = int(jnp.sqrt(n_rays))
        
        # Reshape convergence values to 2D map
        convergence_map = convergence_values.reshape(resolution, resolution)
        
        return cls(convergence_map, map_size_rad, source_redshift, cosmology)


def create_ray_grid(resolution: int, map_size_rad: float) -> jnp.ndarray:
    """
    Create a regular grid of ray positions.
    
    Args:
        resolution: Number of rays per side (total rays = resolution^2)
        map_size_rad: Angular size of the field in radians
        
    Returns:
        Ray positions [2, resolution^2] in radians
    """
    # Create coordinate arrays
    coords_1d = jnp.linspace(0, map_size_rad, resolution)
    xx, yy = jnp.meshgrid(coords_1d, coords_1d, indexing='ij')
    
    # Flatten and stack to get [2, n_rays] array
    ray_positions = jnp.stack([xx.flatten(), yy.flatten()], axis=0)
    
    return ray_positions