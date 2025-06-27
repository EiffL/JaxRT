"""
Tests for shell creation functionality.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import FlatLambdaCDM

from jaxrt.planes.shells import (
    create_density_shells_from_particles,
    create_shells_from_lightcone,
    convert_shells_to_convergence
)


class TestShells:
    """Test shell creation functions."""
    
    def setup_method(self):
        """Set up test data."""
        # Create mock particle data
        self.n_particles = 1000
        self.rng = np.random.default_rng(42)
        
        # Random particle positions in a box [0, 100] Mpc/h
        self.particle_positions = jnp.array(
            self.rng.uniform(0, 100, (self.n_particles, 3))
        )
        
        # Random particle masses
        self.particle_masses = jnp.array(
            self.rng.uniform(1e10, 1e12, self.n_particles)
        )
        
        # Observer at origin
        self.observer_position = jnp.array([0.0, 0.0, 0.0])
        
        # Shell parameters
        self.shell_distances = jnp.array([50.0, 75.0, 100.0])
        self.shell_thickness = 10.0
        self.nside = 32  # Small nside for fast testing
        
        # Cosmology
        self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    
    def test_create_density_shells_basic(self):
        """Test basic shell creation functionality."""
        shell_maps, shell_redshifts = create_density_shells_from_particles(
            particle_positions=self.particle_positions,
            particle_masses=self.particle_masses,
            observer_position=self.observer_position,
            shell_distances=self.shell_distances,
            shell_thickness=self.shell_thickness,
            nside=self.nside
        )
        
        # Check output shapes
        expected_n_pixels = 12 * self.nside**2
        assert shell_maps.shape == (len(self.shell_distances), expected_n_pixels)
        assert shell_redshifts.shape == (len(self.shell_distances),)
        
        # Check that maps are non-negative
        assert jnp.all(shell_maps >= 0)
        
        # Check that at least some pixels have non-zero values
        assert jnp.sum(shell_maps) > 0
    
    def test_create_density_shells_with_cosmology(self):
        """Test shell creation with cosmology."""
        shell_maps, shell_redshifts = create_density_shells_from_particles(
            particle_positions=self.particle_positions,
            particle_masses=self.particle_masses,
            observer_position=self.observer_position,
            shell_distances=self.shell_distances,
            shell_thickness=self.shell_thickness,
            nside=self.nside,
            cosmology=self.cosmology
        )
        
        # Check that redshifts are reasonable
        assert jnp.all(shell_redshifts > 0)
        assert jnp.all(shell_redshifts < 1.0)  # Should be reasonable for nearby shells
    
    def test_create_shells_from_lightcone(self):
        """Test lightcone shell creation."""
        # Create mock redshift data
        particle_redshifts = jnp.array(
            self.rng.uniform(0.1, 0.5, self.n_particles)
        )
        target_redshifts = jnp.array([0.2, 0.3, 0.4])
        
        shell_maps = create_shells_from_lightcone(
            particle_positions=self.particle_positions,
            particle_masses=self.particle_masses,
            particle_redshifts=particle_redshifts,
            observer_position=self.observer_position,
            target_redshifts=target_redshifts,
            redshift_thickness=0.05,
            nside=self.nside
        )
        
        # Check output shape
        expected_n_pixels = 12 * self.nside**2
        assert shell_maps.shape == (len(target_redshifts), expected_n_pixels)
        assert jnp.all(shell_maps >= 0)
    
    def test_convert_shells_to_convergence(self):
        """Test conversion to convergence maps."""
        # Create shell maps
        shell_maps, shell_redshifts = create_density_shells_from_particles(
            particle_positions=self.particle_positions,
            particle_masses=self.particle_masses,
            observer_position=self.observer_position,
            shell_distances=self.shell_distances,
            shell_thickness=self.shell_thickness,
            nside=self.nside,
            cosmology=self.cosmology
        )
        
        # Convert to convergence
        source_redshift = 1.0
        convergence_maps = convert_shells_to_convergence(
            shell_maps=shell_maps,
            shell_redshifts=shell_redshifts,
            source_redshift=source_redshift,
            cosmology=self.cosmology
        )
        
        # Check output shape
        assert convergence_maps.shape == shell_maps.shape
        
        # Convergence can be positive or negative, but should be finite
        assert jnp.all(jnp.isfinite(convergence_maps))
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with no particles in shell
        far_shell_distances = jnp.array([1000.0])  # Far from all particles
        
        shell_maps, shell_redshifts = create_density_shells_from_particles(
            particle_positions=self.particle_positions,
            particle_masses=self.particle_masses,
            observer_position=self.observer_position,
            shell_distances=far_shell_distances,
            shell_thickness=self.shell_thickness,
            nside=self.nside
        )
        
        # Should return zero map
        assert jnp.sum(shell_maps) == 0
        
        # Test with single particle
        single_position = jnp.array([[50.0, 0.0, 0.0]])
        single_mass = jnp.array([1e11])
        
        shell_maps, _ = create_density_shells_from_particles(
            particle_positions=single_position,
            particle_masses=single_mass,
            observer_position=self.observer_position,
            shell_distances=jnp.array([50.0]),
            shell_thickness=self.shell_thickness,
            nside=self.nside
        )
        
        # Should have non-zero map
        assert jnp.sum(shell_maps) > 0


if __name__ == "__main__":
    # Simple test runner for development
    test_instance = TestShells()
    test_instance.setup_method()
    
    print("Running basic shell creation test...")
    test_instance.test_create_density_shells_basic()
    print("✓ Basic shell creation test passed")
    
    print("Running cosmology test...")
    test_instance.test_create_density_shells_with_cosmology()
    print("✓ Cosmology test passed")
    
    print("Running lightcone test...")
    test_instance.test_create_shells_from_lightcone()
    print("✓ Lightcone test passed")
    
    print("Running convergence conversion test...")
    test_instance.test_convert_shells_to_convergence()
    print("✓ Convergence conversion test passed")
    
    print("Running edge cases test...")
    test_instance.test_edge_cases()
    print("✓ Edge cases test passed")
    
    print("\nAll tests passed! 🎉")