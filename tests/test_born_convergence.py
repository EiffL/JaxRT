"""
Test Born convergence implementation against LensTools reference.

This module provides comprehensive tests comparing the JAX implementation
of Born convergence with the reference implementation in LensTools.
"""

import numpy as np
import jax.numpy as jnp
import jax
import pytest
from astropy.units import deg, Mpc

# Import our JAX implementation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jaxrt.core.born_convergence import born_convergence_from_cosmology
from jaxrt.planes.density_plane import create_density_planes_sequence
from jaxrt.maps.convergence_map import create_ray_grid, ConvergenceMap

# Import LensTools for reference
try:
    from lenstools.simulations.raytracing import RayTracer, DensityPlane
    from lenstools import ConvergenceMap as LensToolsConvergenceMap
    import lenstools
    LENSTOOLS_AVAILABLE = True
except ImportError:
    LENSTOOLS_AVAILABLE = False
    
import jax_cosmo as jc


class TestBornConvergence:
    """Test Born convergence implementation."""
    
    @pytest.fixture
    def test_parameters(self):
        """Standard test parameters."""
        return {
            'resolution': 64,  # Smaller for faster tests
            'map_size_degrees': 1.0,
            'n_planes': 10,
            'redshift_range': (0.1, 2.0),
            'source_redshift': 2.0,
            'power_spectrum_amplitude': 1e-4,
            'random_seed': 42
        }
    
    @pytest.fixture
    def density_planes_data(self, test_parameters):
        """Generate test density planes."""
        params = test_parameters
        map_size_rad = params['map_size_degrees'] * np.pi / 180.0
        
        density_planes, redshifts = create_density_planes_sequence(
            n_planes=params['n_planes'],
            redshift_range=params['redshift_range'],
            resolution=params['resolution'],
            map_size_rad=map_size_rad,
            power_spectrum_amplitude=params['power_spectrum_amplitude'],
            random_seed=params['random_seed']
        )
        
        return {
            'density_planes': density_planes,
            'redshifts': redshifts,
            'map_size_rad': map_size_rad,
            'resolution': params['resolution']
        }
    
    def test_born_convergence_basic(self, test_parameters, density_planes_data):
        """Test basic Born convergence functionality."""
        params = test_parameters
        data = density_planes_data
        
        # Create ray grid
        ray_positions = create_ray_grid(data['resolution'], data['map_size_rad'])
        
        # Compute Born convergence
        convergence = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=params['source_redshift'],
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution']
        )
        
        # Basic sanity checks
        assert convergence.shape == (data['resolution']**2,)
        assert jnp.isfinite(convergence).all()
        assert not jnp.isnan(convergence).any()
        
        # Check convergence is reasonable (not too large)
        assert jnp.abs(convergence).max() < 1.0
        
        # Check RMS is reasonable for our test amplitude
        rms = jnp.sqrt(jnp.mean(convergence**2))
        assert 1e-6 < rms < 1e-2
    
    def test_convergence_map_creation(self, test_parameters, density_planes_data):
        """Test ConvergenceMap creation and utilities."""
        params = test_parameters
        data = density_planes_data
        
        # Create ray grid
        ray_positions = create_ray_grid(data['resolution'], data['map_size_rad'])
        
        # Compute Born convergence
        convergence = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=params['source_redshift'],
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution']
        )
        
        # Create ConvergenceMap
        conv_map = ConvergenceMap.from_ray_positions(
            ray_positions=ray_positions,
            convergence_values=convergence,
            map_size_rad=data['map_size_rad'],
            source_redshift=params['source_redshift']
        )
        
        # Test properties
        assert conv_map.resolution == data['resolution']
        assert conv_map.source_redshift == params['source_redshift']
        assert conv_map.map_size_rad == data['map_size_rad']
        
        # Test statistics
        stats = conv_map.statistics()
        assert 'mean' in stats
        assert 'std' in stats
        assert 'rms' in stats
        assert jnp.isfinite(stats['mean'])
        assert stats['std'] > 0
    
    @pytest.mark.skipif(not LENSTOOLS_AVAILABLE, reason="LensTools not available")
    def test_against_lenstools_reference(self, test_parameters):
        """Test our implementation against LensTools reference."""
        params = test_parameters
        map_size_rad = params['map_size_degrees'] * np.pi / 180.0
        
        # Create reproducible density planes
        np.random.seed(params['random_seed'])
        
        # Create density planes for both implementations
        density_planes_jax, redshifts = create_density_planes_sequence(
            n_planes=params['n_planes'],
            redshift_range=params['redshift_range'],
            resolution=params['resolution'],
            map_size_rad=map_size_rad,
            power_spectrum_amplitude=params['power_spectrum_amplitude'],
            random_seed=params['random_seed']
        )
        
        # Convert to numpy for LensTools
        density_planes_np = [np.array(plane) for plane in density_planes_jax]
        
        # Setup LensTools RayTracer
        tracer = RayTracer(lens_mesh_size=params['resolution'])
        
        # Add density planes to tracer
        from astropy.cosmology import Planck18
        cosmo = Planck18
        
        for i, (density_map, z) in enumerate(zip(density_planes_np, redshifts)):
            # Create LensTools DensityPlane
            angle = params['map_size_degrees'] * deg
            plane = DensityPlane(
                data=density_map,
                angle=angle,
                redshift=float(z),
                cosmology=cosmo,
                unit=lenstools.rad**2  # dimensionless
            )
            tracer.addLens(plane)
        
        # Create ray positions
        ray_positions_jax = create_ray_grid(params['resolution'], map_size_rad)
        
        # Convert ray positions for LensTools (expects shape [2, n_rays] in degrees)
        ray_positions_lenstools = np.array(ray_positions_jax) * 180.0 / np.pi
        ray_positions_lenstools = ray_positions_lenstools.T  # LensTools expects [n_rays, 2]
        ray_positions_lenstools = ray_positions_lenstools * deg
        
        # Compute convergence with LensTools
        convergence_lenstools = tracer.convergenceBorn(
            ray_positions_lenstools, 
            z=params['source_redshift'],
            save_intermediate=False
        )
        
        # Compute convergence with our JAX implementation
        convergence_jax = born_convergence_from_cosmology(
            ray_positions=ray_positions_jax,
            density_planes=list(density_planes_jax),
            plane_redshifts=redshifts,
            source_redshift=params['source_redshift'],
            map_size_rad=map_size_rad,
            map_resolution=params['resolution']
        )
        
        # Convert to numpy for comparison
        convergence_jax_np = np.array(convergence_jax)
        
        # Compare results
        # The implementations might have slightly different conventions,
        # so we compare statistical properties rather than exact values
        
        # Check means are close
        mean_lenstools = np.mean(convergence_lenstools)
        mean_jax = np.mean(convergence_jax_np)
        assert abs(mean_lenstools - mean_jax) < 0.1 * abs(mean_lenstools)
        
        # Check RMS are close
        rms_lenstools = np.sqrt(np.mean(convergence_lenstools**2))
        rms_jax = np.sqrt(np.mean(convergence_jax_np**2))
        assert abs(rms_lenstools - rms_jax) < 0.2 * rms_lenstools
        
        # Check correlation is high
        correlation = np.corrcoef(convergence_lenstools.flatten(), 
                                 convergence_jax_np.flatten())[0, 1]
        assert correlation > 0.8, f"Correlation too low: {correlation}"
        
        print(f"Comparison results:")
        print(f"  LensTools mean: {mean_lenstools:.6f}, JAX mean: {mean_jax:.6f}")
        print(f"  LensTools RMS: {rms_lenstools:.6f}, JAX RMS: {rms_jax:.6f}")
        print(f"  Correlation: {correlation:.6f}")
    
    def test_cosmology_consistency(self, test_parameters, density_planes_data):
        """Test that cosmology parameters are handled consistently."""
        params = test_parameters
        data = density_planes_data
        
        # Create ray grid
        ray_positions = create_ray_grid(data['resolution'], data['map_size_rad'])
        
        # Test with different cosmologies
        cosmo1 = jc.Planck18()
        cosmo2 = jc.Planck15()
        
        # Compute convergence with different cosmologies
        conv1 = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=params['source_redshift'],
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution'],
            cosmology=cosmo1
        )
        
        conv2 = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=params['source_redshift'],
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution'],
            cosmology=cosmo2
        )
        
        # Results should be different but correlated
        correlation = jnp.corrcoef(conv1, conv2)[0, 1]
        assert correlation > 0.9  # Should be highly correlated
        assert not jnp.allclose(conv1, conv2)  # But not identical
    
    def test_source_redshift_scaling(self, test_parameters, density_planes_data):
        """Test that convergence scales properly with source redshift."""
        params = test_parameters
        data = density_planes_data
        
        # Create ray grid
        ray_positions = create_ray_grid(data['resolution'], data['map_size_rad'])
        
        # Test with different source redshifts
        z_source_1 = 1.0
        z_source_2 = 2.0
        
        conv1 = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=z_source_1,
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution']
        )
        
        conv2 = born_convergence_from_cosmology(
            ray_positions=ray_positions,
            density_planes=list(data['density_planes']),
            plane_redshifts=data['redshifts'],
            source_redshift=z_source_2,
            map_size_rad=data['map_size_rad'],
            map_resolution=data['resolution']
        )
        
        # Higher source redshift should give higher convergence
        assert jnp.mean(jnp.abs(conv2)) > jnp.mean(jnp.abs(conv1))
        
        # Should be highly correlated
        correlation = jnp.corrcoef(conv1, conv2)[0, 1]
        assert correlation > 0.95


if __name__ == "__main__":
    # Run a quick test
    test = TestBornConvergence()
    
    # Test parameters
    params = {
        'resolution': 32,
        'map_size_degrees': 1.0,
        'n_planes': 5,
        'redshift_range': (0.1, 2.0),
        'source_redshift': 2.0,
        'power_spectrum_amplitude': 1e-4,
        'random_seed': 42
    }
    
    # Generate density planes
    map_size_rad = params['map_size_degrees'] * np.pi / 180.0
    density_planes, redshifts = create_density_planes_sequence(
        n_planes=params['n_planes'],
        redshift_range=params['redshift_range'],
        resolution=params['resolution'],
        map_size_rad=map_size_rad,
        power_spectrum_amplitude=params['power_spectrum_amplitude'],
        random_seed=params['random_seed']
    )
    
    data = {
        'density_planes': density_planes,
        'redshifts': redshifts,
        'map_size_rad': map_size_rad,
        'resolution': params['resolution']
    }
    
    # Run basic test
    test.test_born_convergence_basic(params, data)
    print("✓ Basic Born convergence test passed")
    
    # Run convergence map test
    test.test_convergence_map_creation(params, data)
    print("✓ Convergence map creation test passed")
    
    print("All tests passed!")