"""
Density plane utilities for gravitational lensing.

This module provides functions for generating Gaussian density planes
used in gravitational lensing ray tracing calculations.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple


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
    key1, key2 = jax.random.split(random_key)
    
    # Generate complex Gaussian noise
    real_part = jax.random.normal(key1, (resolution, resolution))
    imag_part = jax.random.normal(key2, (resolution, resolution))
    noise_fourier = real_part + 1j * imag_part
    
    # Apply power spectrum
    density_fourier = noise_fourier * jnp.sqrt(power_spectrum / 2.0)
    
    # Ensure proper Hermitian symmetry for real output
    density_fourier = density_fourier.at[0, 0].set(0.0)  # DC component real
    
    # Make edge frequencies real to avoid complex artifacts
    density_fourier = density_fourier.at[0, :].set(jnp.real(density_fourier[0, :]))
    density_fourier = density_fourier.at[:, 0].set(jnp.real(density_fourier[:, 0]))
    
    # Nyquist frequencies (if even resolution)
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
        - density_planes: Array of density maps [n_planes, resolution, resolution]
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