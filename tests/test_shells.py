"""Tests for particle-to-shell binning functionality."""

import jax.numpy as jnp
import numpy as np
import pytest
import jax_cosmo as jc

from jaxrt.planes.shells import (
    create_density_shells_from_particles,
    create_shells_from_lightcone,
    convert_shells_to_convergence,
    _compute_distances,
    _bin_particles_to_shell,
)


@pytest.fixture
def mock_particles():
    """Create mock particle data for testing."""
    # Create a simple distribution of particles
    n_particles = 1000
    rng = np.random.RandomState(42)
    
    # Random positions in a cube
    positions = rng.uniform(-100, 100, (n_particles, 3))
    
    # Random masses
    masses = rng.uniform(1e10, 1e12, n_particles)
    
    return jnp.array(positions), jnp.array(masses)


@pytest.fixture
def mock_cosmology():
    """Create mock cosmology for testing."""
    return jc.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0
    )


def test_compute_distances():
    """Test distance computation."""
    positions = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    observer = jnp.array([0, 0, 0])
    
    distances = _compute_distances(positions, observer)
    expected = jnp.array([0, 1, 1, 1])
    
    assert jnp.allclose(distances, expected)


def test_bin_particles_to_shell(mock_particles):
    """Test binning particles to a single shell."""
    positions, masses = mock_particles
    observer = jnp.array([0, 0, 0])
    
    # Create a shell at distance 50 with thickness 10
    shell_map = _bin_particles_to_shell(
        positions, masses, observer, 50.0, 10.0, nside=8
    )
    
    # Check output shape
    assert shell_map.shape == (12 * 8**2,)
    
    # Check that some pixels have non-zero values
    assert jnp.sum(shell_map > 0) > 0
    
    # Check that total mass is conserved (approximately)
    distances = _compute_distances(positions, observer)
    in_shell = (distances >= 45) & (distances <= 55)
    expected_total_mass = jnp.sum(masses[in_shell])
    
    # Total mass in shell map (need to account for surface density conversion)
    shell_area = 4 * jnp.pi * 50.0**2
    pixel_area = shell_area / (12 * 8**2)
    total_mass_in_map = jnp.sum(shell_map) * pixel_area
    
    assert jnp.allclose(total_mass_in_map, expected_total_mass, rtol=1e-10)


def test_create_density_shells_from_particles(mock_particles):
    """Test creating density shells from particles."""
    positions, masses = mock_particles
    observer = jnp.array([0, 0, 0])
    shell_distances = jnp.array([30.0, 60.0, 90.0])
    
    shell_maps, shell_redshifts = create_density_shells_from_particles(
        positions, masses, observer, shell_distances, 10.0, nside=8
    )
    
    # Check output shapes
    assert shell_maps.shape == (3, 12 * 8**2)
    assert shell_redshifts is None  # No cosmology provided
    
    # Check that all shells have some content
    for i in range(3):
        assert jnp.sum(shell_maps[i] > 0) > 0


def test_create_density_shells_with_cosmology(mock_particles, mock_cosmology):
    """Test creating density shells with cosmology."""
    positions, masses = mock_particles
    observer = jnp.array([0, 0, 0])
    shell_distances = jnp.array([50.0, 100.0])
    
    shell_maps, shell_redshifts = create_density_shells_from_particles(
        positions, masses, observer, shell_distances, 10.0, nside=8,
        cosmology=mock_cosmology
    )
    
    # Check that redshifts are computed
    assert shell_redshifts is not None
    assert len(shell_redshifts) == 2
    assert jnp.all(shell_redshifts > 0)


def test_create_shells_from_lightcone(mock_cosmology):
    """Test creating shells from lightcone particles."""
    # Create mock lightcone data
    n_particles = 500
    rng = np.random.RandomState(42)
    
    positions = jnp.array(rng.uniform(-50, 50, (n_particles, 3)))
    masses = jnp.array(rng.uniform(1e10, 1e12, n_particles))
    redshifts = jnp.array(rng.uniform(0.1, 2.0, n_particles))
    
    observer = jnp.array([0, 0, 0])
    shell_redshifts = jnp.array([0.5, 1.0, 1.5])
    
    shell_maps = create_shells_from_lightcone(
        positions, masses, redshifts, observer, shell_redshifts, 0.1, nside=8, 
        cosmology=mock_cosmology
    )
    
    # Check output shape
    assert shell_maps.shape == (3, 12 * 8**2)


def test_convert_shells_to_convergence(mock_cosmology):
    """Test converting shells to convergence."""
    # Create mock shell data
    n_shells = 3
    nside = 8
    npix = 12 * nside**2
    
    shell_maps = jnp.ones((n_shells, npix)) * 1e12  # Surface density in Msun/h per (Mpc/h)²
    shell_redshifts = jnp.array([0.3, 0.6, 0.9])
    source_redshift = 1.2
    
    convergence_maps = convert_shells_to_convergence(
        shell_maps, shell_redshifts, source_redshift, mock_cosmology
    )
    
    # Check output shape
    assert convergence_maps.shape == shell_maps.shape
    
    # Check that shells behind source have zero convergence
    assert jnp.all(convergence_maps[shell_redshifts >= source_redshift] == 0)
    
    # Check that shells in front of source have non-zero convergence
    in_front = shell_redshifts < source_redshift
    assert jnp.all(convergence_maps[in_front] > 0)


def test_empty_shell_handling():
    """Test handling of empty shells."""
    # Create particles all at one distance
    positions = jnp.array([[10, 0, 0], [10, 1, 0], [10, 0, 1]])
    masses = jnp.array([1e10, 1e10, 1e10])
    observer = jnp.array([0, 0, 0])
    
    # Create shell that doesn't contain any particles
    shell_map = _bin_particles_to_shell(
        positions, masses, observer, 50.0, 1.0, nside=8
    )
    
    # Should be all zeros
    assert jnp.all(shell_map == 0)


def test_shell_thickness_effect(mock_particles):
    """Test that shell thickness affects particle selection."""
    positions, masses = mock_particles
    observer = jnp.array([0, 0, 0])
    
    # Create shells with different thicknesses
    thin_shell = _bin_particles_to_shell(
        positions, masses, observer, 50.0, 1.0, nside=8
    )
    thick_shell = _bin_particles_to_shell(
        positions, masses, observer, 50.0, 20.0, nside=8
    )
    
    # Thick shell should capture more particles
    assert jnp.sum(thick_shell) > jnp.sum(thin_shell)


@pytest.mark.parametrize("nside", [4, 8, 16])
def test_different_nside_values(mock_particles, nside):
    """Test that different nside values work correctly."""
    positions, masses = mock_particles
    observer = jnp.array([0, 0, 0])
    
    shell_map = _bin_particles_to_shell(
        positions, masses, observer, 50.0, 10.0, nside=nside
    )
    
    # Check correct output size
    expected_npix = 12 * nside**2
    assert shell_map.shape == (expected_npix,)
    
    # Check that we have some non-zero pixels
    assert jnp.sum(shell_map > 0) > 0


def test_observer_position_effect():
    """Test that observer position affects the results."""
    # Create a simple particle distribution
    positions = jnp.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]])
    masses = jnp.array([1e10, 1e10, 1e10])
    
    # Two different observer positions
    observer1 = jnp.array([0, 0, 0])
    observer2 = jnp.array([5, 0, 0])
    
    shell_map1 = _bin_particles_to_shell(
        positions, masses, observer1, 20.0, 5.0, nside=8
    )
    shell_map2 = _bin_particles_to_shell(
        positions, masses, observer2, 20.0, 5.0, nside=8
    )
    
    # Results should be different
    assert not jnp.allclose(shell_map1, shell_map2)


def test_input_validation():
    """Test input validation functions."""
    from jaxrt.planes.shells import _validate_inputs
    
    # Test negative masses
    with pytest.raises(ValueError, match="non-negative"):
        _validate_inputs(jnp.array([-1, 1]), 8, jnp.array([10.0]))
    
    # Test invalid nside
    with pytest.raises(ValueError, match="power of 2"):
        _validate_inputs(jnp.array([1, 1]), 7, jnp.array([10.0]))
    
    with pytest.raises(ValueError, match="power of 2"):
        _validate_inputs(jnp.array([1, 1]), 0, jnp.array([10.0]))
    
    # Test non-positive distances
    with pytest.raises(ValueError, match="positive"):
        _validate_inputs(jnp.array([1, 1]), 8, jnp.array([0.0]))
    
    with pytest.raises(ValueError, match="positive"):
        _validate_inputs(jnp.array([1, 1]), 8, jnp.array([-1.0]))
    
    # Test valid inputs (should not raise)
    _validate_inputs(jnp.array([1, 1]), 8, jnp.array([10.0]))