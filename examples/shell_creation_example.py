"""
Example: Creating density shells from N-body simulation particles

This example demonstrates how to use JaxRT to bin dark matter particles 
from an N-body simulation onto spherical shells and convert them to HEALPix maps.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import healpy as hp

# Import JaxRT functions
from jaxrt import (
    create_density_shells_from_particles,
    create_shells_from_lightcone,
    convert_shells_to_convergence,
    save_shells_to_fits
)


def generate_mock_nbody_data(n_particles=10000, box_size=200.0, seed=42):
    """
    Generate mock N-body simulation data for demonstration.
    
    Args:
        n_particles: Number of dark matter particles
        box_size: Size of simulation box in Mpc/h
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (positions, masses) arrays
    """
    rng = np.random.default_rng(seed)
    
    # Generate random particle positions in simulation box
    positions = rng.uniform(0, box_size, (n_particles, 3))
    
    # Generate particle masses with some scatter around mean
    mean_mass = 1e11  # Msun/h
    masses = rng.lognormal(np.log(mean_mass), 0.5, n_particles)
    
    return jnp.array(positions), jnp.array(masses)


def main():
    """Main example function."""
    print("JaxRT Shell Creation Example")
    print("=" * 40)
    
    # Set up cosmology
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)
    print(f"Using cosmology: {cosmology}")
    
    # Generate mock N-body data
    print("\n1. Generating mock N-body simulation data...")
    particle_positions, particle_masses = generate_mock_nbody_data(
        n_particles=50000, box_size=200.0
    )
    print(f"   Generated {len(particle_positions)} particles")
    print(f"   Simulation box size: 200 Mpc/h")
    print(f"   Total mass: {jnp.sum(particle_masses):.2e} Msun/h")
    
    # Set observer position (e.g., at one corner of the box)
    observer_position = jnp.array([0.0, 0.0, 0.0])
    
    # Define shells at different distances
    shell_distances = jnp.array([50.0, 75.0, 100.0, 125.0, 150.0])  # Mpc/h
    shell_thickness = 20.0  # Mpc/h
    nside = 256  # HEALPix resolution parameter
    
    print(f"\n2. Creating {len(shell_distances)} density shells...")
    print(f"   Shell distances: {shell_distances} Mpc/h")
    print(f"   Shell thickness: {shell_thickness} Mpc/h")
    print(f"   HEALPix nside: {nside} (npix = {12*nside**2})")
    
    # Create density shells
    shell_maps, shell_redshifts = create_density_shells_from_particles(
        particle_positions=particle_positions,
        particle_masses=particle_masses,
        observer_position=observer_position,
        shell_distances=shell_distances,
        shell_thickness=shell_thickness,
        nside=nside,
        cosmology=cosmology
    )
    
    print(f"   Created shells at redshifts: {shell_redshifts}")
    
    # Print some statistics
    print("\n3. Shell statistics:")
    for i, (z, shell_map) in enumerate(zip(shell_redshifts, shell_maps)):
        total_mass = jnp.sum(shell_map) * (4 * np.pi * shell_distances[i]**2) / len(shell_map)
        mean_density = jnp.mean(shell_map)
        std_density = jnp.std(shell_map)
        print(f"   Shell {i}: z={z:.3f}, total_mass={total_mass:.2e} Msun/h")
        print(f"            mean_density={mean_density:.2e}, std_density={std_density:.2e}")
    
    # Convert to convergence maps for lensing
    print("\n4. Converting to convergence maps...")
    source_redshift = 1.0
    convergence_maps = convert_shells_to_convergence(
        shell_maps=shell_maps,
        shell_redshifts=shell_redshifts,
        source_redshift=source_redshift,
        cosmology=cosmology
    )
    
    print(f"   Converted for source redshift z_s = {source_redshift}")
    
    # Print convergence statistics
    print("\n5. Convergence statistics:")
    for i, (z, conv_map) in enumerate(zip(shell_redshifts, convergence_maps)):
        mean_conv = jnp.mean(conv_map)
        std_conv = jnp.std(conv_map)
        max_conv = jnp.max(conv_map)
        print(f"   Shell {i}: z={z:.3f}, κ_mean={mean_conv:.4f}, κ_std={std_conv:.4f}, κ_max={max_conv:.4f}")
    
    # Visualize one of the maps
    print("\n6. Creating visualization...")
    shell_idx = 2  # Middle shell
    
    # Create a simple Mollweide projection plot
    plt.figure(figsize=(12, 6))
    
    # Surface density map
    plt.subplot(1, 2, 1)
    hp.mollview(
        np.array(shell_maps[shell_idx]), 
        title=f'Surface Density Shell {shell_idx} (z={shell_redshifts[shell_idx]:.3f})',
        unit='Msun/h/Mpc²',
        cmap='viridis'
    )
    
    # Convergence map
    plt.subplot(1, 2, 2)
    hp.mollview(
        np.array(convergence_maps[shell_idx]),
        title=f'Convergence Shell {shell_idx} (z={shell_redshifts[shell_idx]:.3f})',
        unit='κ',
        cmap='RdBu_r'
    )
    
    plt.tight_layout()
    plt.savefig('shell_maps_example.png', dpi=150, bbox_inches='tight')
    print("   Saved visualization to 'shell_maps_example.png'")
    
    # Save shells to FITS files (optional)
    save_fits = False  # Set to True to save FITS files
    if save_fits:
        print("\n7. Saving shells to FITS files...")
        save_shells_to_fits(
            shell_maps=shell_maps,
            shell_redshifts=shell_redshifts,
            output_prefix="example_shell",
            nside=nside
        )
    
    print("\n✓ Example completed successfully!")
    print("\nThe shell creation functionality provides:")
    print("  - Efficient binning of N-body particles onto spherical shells")
    print("  - HEALPix map output for fast angular operations")
    print("  - Automatic conversion to convergence for lensing calculations")
    print("  - Support for both distance-based and redshift-based binning")
    print("  - JAX acceleration for high-performance computing")


if __name__ == "__main__":
    main()