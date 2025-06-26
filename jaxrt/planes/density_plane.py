"""
Density plane utilities for gravitational lensing.

This module provides tools for creating and manipulating density planes
used in gravitational lensing ray tracing calculations.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple
import jax_cosmo as jc
from ..utils import interpolate_density_at_positions


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
        return interpolate_density_at_positions(
            self.density_map, positions, self.map_size_rad, self.resolution
        )


# Interpolation function moved to utils.interpolation module


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
    
    # Ensure proper Hermitian symmetry for real output
    # This is needed for jnp.fft.ifft2 to produce real output
    # Set DC component to zero (real)
    density_fourier = density_fourier.at[0, 0].set(0.0)
    
    # Ensure Hermitian symmetry: F[k] = F*[-k] for real output
    # Use jnp.fft.rfft2 approach: only generate half the frequencies
    # and let the inverse FFT handle the symmetry automatically
    # For now, just set problematic frequencies to be real
    
    # Make sure edge frequencies are real to avoid complex artifacts
    # Set first row to be real (k_y = 0)
    density_fourier = density_fourier.at[0, :].set(jnp.real(density_fourier[0, :]))
    
    # Set first column to be real (k_x = 0) 
    density_fourier = density_fourier.at[:, 0].set(jnp.real(density_fourier[:, 0]))
    
    # If even resolution, set Nyquist frequencies to be real
    if resolution % 2 == 0:
        nyquist = resolution // 2
        density_fourier = density_fourier.at[nyquist, :].set(jnp.real(density_fourier[nyquist, :]))
        density_fourier = density_fourier.at[:, nyquist].set(jnp.real(density_fourier[:, nyquist]))
    
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