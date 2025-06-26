"""
Density plane utilities for gravitational lensing.

This module provides tools for creating and manipulating density planes
used in gravitational lensing ray tracing calculations.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import jax_cosmo as jc


class DensityPlane:
    """
    A density plane for gravitational lensing calculations.
    
    This class represents a 2D density field at a specific redshift,
    used for ray tracing through gravitational lensing simulations.
    """
    
    def __init__(
        self,
        density_map: jnp.ndarray,
        map_size_rad: float,
        redshift: float,
        cosmology: Optional[jc.Cosmology] = None
    ):
        """
        Initialize a density plane.
        
        Args:
            density_map: 2D density field [n_pix, n_pix]
            map_size_rad: Physical angular size of the map in radians
            redshift: Redshift of the plane
            cosmology: JAX-Cosmo cosmology (default: Planck18)
        """
        self.density_map = density_map
        self.map_size_rad = map_size_rad
        self.redshift = redshift
        self.resolution = density_map.shape[0]
        
        if cosmology is None:
            cosmology = jc.Planck18()
        self.cosmology = cosmology
        
        # Compute comoving distance
        self.comoving_distance = jc.background.radial_comoving_distance(
            cosmology, 1.0 / (1.0 + redshift)
        )
    
    @property
    def pixel_size_rad(self) -> float:
        """Pixel size in radians."""
        return self.map_size_rad / self.resolution
    
    @property 
    def pixel_size_arcmin(self) -> float:
        """Pixel size in arcminutes."""
        return self.pixel_size_rad * 180.0 * 60.0 / jnp.pi
    
    def interpolate_at_positions(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Interpolate density values at given angular positions.
        
        Args:
            positions: Angular positions [2, n_rays] in radians
            
        Returns:
            Interpolated density values [n_rays]
        """
        return _interpolate_density_at_positions(
            self.density_map, positions, self.map_size_rad, self.resolution
        )


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
    f00 = density_map[y_i, x_i]
    f10 = density_map[y_i, x_i1]
    f01 = density_map[y_i1, x_i]
    f11 = density_map[y_i1, x_i1]
    
    result = (
        f00 * (1 - x_f) * (1 - y_f) +
        f10 * x_f * (1 - y_f) +
        f01 * (1 - x_f) * y_f +
        f11 * x_f * y_f
    )
    
    return result


def generate_gaussian_density_plane(
    resolution: int,
    map_size_rad: float,
    power_spectrum_amplitude: float = 1e-3,
    power_spectrum_index: float = -2.0,
    random_key: Optional[jax.random.PRNGKey] = None
) -> jnp.ndarray:
    """
    Generate a Gaussian random density field with a given power spectrum.
    
    Args:
        resolution: Map resolution (n_pix x n_pix)
        map_size_rad: Angular size of the map in radians
        power_spectrum_amplitude: Amplitude of the power spectrum
        power_spectrum_index: Power law index (default: -2.0 for scale-invariant)
        random_key: JAX random key (if None, creates a new one)
        
    Returns:
        Gaussian density field [resolution, resolution]
    """
    if random_key is None:
        random_key = jax.random.PRNGKey(42)
    
    # Create frequency grids
    freq = jnp.fft.fftfreq(resolution, d=map_size_rad/resolution)
    kx, ky = jnp.meshgrid(freq, freq, indexing='ij')
    k = jnp.sqrt(kx**2 + ky**2)
    
    # Avoid division by zero at k=0
    k = jnp.where(k == 0, 1.0, k)
    
    # Power spectrum: P(k) = A * k^n
    power_spectrum = power_spectrum_amplitude * k**power_spectrum_index
    
    # Set DC component to zero
    power_spectrum = power_spectrum.at[0, 0].set(0.0)
    
    # Generate Gaussian random field in Fourier space
    # Split key for real and imaginary parts
    key1, key2 = jax.random.split(random_key)
    
    # Generate complex Gaussian noise
    real_part = jax.random.normal(key1, (resolution, resolution))
    imag_part = jax.random.normal(key2, (resolution, resolution))
    noise_fourier = real_part + 1j * imag_part
    
    # Apply power spectrum
    density_fourier = noise_fourier * jnp.sqrt(power_spectrum / 2.0)
    
    # Ensure Hermitian symmetry for real output
    # This is needed for jnp.fft.ifft2 to produce real output
    density_fourier = density_fourier.at[0, 0].set(0.0)  # DC component
    
    # Transform back to real space
    density_field = jnp.real(jnp.fft.ifft2(density_fourier))
    
    return density_field


def create_density_planes_sequence(
    n_planes: int,
    redshift_range: Tuple[float, float],
    resolution: int,
    map_size_rad: float,
    power_spectrum_amplitude: float = 1e-3,
    power_spectrum_index: float = -2.0,
    random_seed: int = 42
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a sequence of Gaussian density planes at different redshifts.
    
    Args:
        n_planes: Number of density planes
        redshift_range: (z_min, z_max) redshift range
        resolution: Map resolution for each plane
        map_size_rad: Angular size of each map in radians
        power_spectrum_amplitude: Power spectrum amplitude
        power_spectrum_index: Power spectrum index
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (density_planes, redshifts)
        - density_planes: List of density maps [n_planes, resolution, resolution]
        - redshifts: Array of redshifts [n_planes]
    """
    # Create redshift array
    z_min, z_max = redshift_range
    redshifts = jnp.linspace(z_min, z_max, n_planes)
    
    # Generate density planes
    main_key = jax.random.PRNGKey(random_seed)
    keys = jax.random.split(main_key, n_planes)
    
    density_planes = []
    for i in range(n_planes):
        density_map = generate_gaussian_density_plane(
            resolution=resolution,
            map_size_rad=map_size_rad,
            power_spectrum_amplitude=power_spectrum_amplitude,
            power_spectrum_index=power_spectrum_index,
            random_key=keys[i]
        )
        density_planes.append(density_map)
    
    return jnp.array(density_planes), redshifts