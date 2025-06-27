#!/usr/bin/env python3
"""
Example: Creating density shells from N-body simulation particles

This example demonstrates how to use JaxRT to bin dark matter particles
from N-body simulations onto spherical shells and convert them to HEALPix maps
for gravitational lensing ray tracing.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.cosmology import FlatLambdaCDM

from jaxrt import create_density_shells_from_particles, convert_shells_to_convergence


def create_mock_simulation_data(n_particles=10000, box_size=200.0):
    """
    Create mock N-body simulation data for demonstration.
    
    Parameters
    ----------
    n_particles : int
        Number of dark matter particles
    box_size : float
        Size of simulation box in Mpc/h
        
    Returns
    -------
    positions : jnp.ndarray
        Particle positions in Mpc/h
    masses : jnp.ndarray
        Particle masses in Msun/h
    """
    rng = np.random.RandomState(42)
    
    # Create clustered particle distribution
    n_clusters = 20
    cluster_centers = rng.uniform(-box_size/2, box_size/2, (n_clusters, 3))
    cluster_sizes = rng.uniform(5, 15, n_clusters)
    particles_per_cluster = n_particles // n_clusters
    
    positions = []
    masses = []
    
    for i in range(n_clusters):
        # Generate particles around cluster center
        cluster_pos = rng.normal(
            cluster_centers[i], 
            cluster_sizes[i], 
            (particles_per_cluster, 3)
        )
        positions.append(cluster_pos)
        
        # Mass follows rough power law
        cluster_masses = rng.lognormal(
            mean=np.log(1e11), 
            sigma=0.5, 
            size=particles_per_cluster
        )
        masses.append(cluster_masses)
    
    # Add remaining particles randomly
    remaining = n_particles - n_clusters * particles_per_cluster
    if remaining > 0:
        random_pos = rng.uniform(-box_size/2, box_size/2, (remaining, 3))
        random_masses = rng.lognormal(mean=np.log(1e10), sigma=0.3, size=remaining)
        positions.append(random_pos)
        masses.append(random_masses)
    
    positions = jnp.array(np.vstack(positions))
    masses = jnp.array(np.concatenate(masses))
    
    return positions, masses


def main():
    """Main example execution."""
    print("JaxRT Shell Creation Example")
    print("=" * 40)
    
    # Set up cosmology
    cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    print(f"Using cosmology: {cosmology}")
    
    # Create mock simulation data
    print("\n1. Creating mock N-body simulation data...")
    positions, masses = create_mock_simulation_data(n_particles=50000, box_size=300.0)
    print(f"   Created {len(positions)} particles")
    print(f"   Position range: [{jnp.min(positions):.1f}, {jnp.max(positions):.1f}] Mpc/h")
    print(f"   Mass range: [{jnp.min(masses):.2e}, {jnp.max(masses):.2e}] Msun/h")
    
    # Set observer position (typically at simulation box edge)
    observer_position = jnp.array([-150.0, 0.0, 0.0])  # Mpc/h
    print(f"   Observer position: {observer_position} Mpc/h")
    
    # Define shell parameters
    shell_distances = jnp.array([50.0, 100.0, 150.0, 200.0, 250.0])  # Mpc/h
    shell_thickness = 20.0  # Mpc/h
    nside = 128  # HEALPix resolution (higher = more detailed)
    
    print(f"\n2. Creating {len(shell_distances)} density shells...")
    print(f"   Shell distances: {shell_distances} Mpc/h")
    print(f"   Shell thickness: {shell_thickness} Mpc/h") 
    print(f"   HEALPix nside: {nside} (npix = {12 * nside**2})")
    
    # Create density shells
    shell_maps, shell_redshifts = create_density_shells_from_particles(
        particle_positions=positions,
        particle_masses=masses,
        observer_position=observer_position,
        shell_distances=shell_distances,
        shell_thickness=shell_thickness,
        nside=nside,
        cosmology=cosmology
    )
    
    print(f"   Successfully created shells with shape: {shell_maps.shape}")
    print(f"   Shell redshifts: {shell_redshifts}")
    
    # Analyze shell properties
    print(f"\n3. Analyzing shell properties...")
    for i, (dist, z) in enumerate(zip(shell_distances, shell_redshifts)):
        shell_map = shell_maps[i]
        total_mass = jnp.sum(shell_map) * (4 * jnp.pi * dist**2) / (12 * nside**2)
        n_pixels_with_mass = jnp.sum(shell_map > 0)
        max_density = jnp.max(shell_map)
        
        print(f"   Shell {i}: z={z:.3f}, total_mass={total_mass:.2e} Msun/h")
        print(f"            {n_pixels_with_mass}/{12*nside**2} pixels have mass")
        print(f"            max surface density={max_density:.2e} Msun/h/(Mpc/h)²")
    
    # Convert to convergence maps for lensing
    source_redshift = 1.0
    print(f"\n4. Converting to convergence for source at z = {source_redshift}...")
    
    convergence_maps = convert_shells_to_convergence(
        shell_maps, shell_redshifts, source_redshift, cosmology
    )
    
    for i, z in enumerate(shell_redshifts):
        if z < source_redshift:
            max_kappa = jnp.max(convergence_maps[i])
            print(f"   Shell {i} (z={z:.3f}): max convergence = {max_kappa:.4f}")
        else:
            print(f"   Shell {i} (z={z:.3f}): behind source, convergence = 0")
    
    # Visualization
    print(f"\n5. Creating visualizations...")
    
    # Plot 2 shells for demonstration
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('JaxRT Shell Creation Example', fontsize=16)
    
    for i in range(min(2, len(shell_maps))):
        # Surface density map
        hp.mollview(
            np.array(shell_maps[i]), 
            title=f'Shell {i}: Surface Density (z={shell_redshifts[i]:.3f})',
            unit='Msun/h/(Mpc/h)²',
            sub=(2, 2, 2*i + 1),
            cmap='viridis'
        )
        
        # Convergence map
        if shell_redshifts[i] < source_redshift:
            hp.mollview(
                np.array(convergence_maps[i]),
                title=f'Shell {i}: Convergence (z={shell_redshifts[i]:.3f})',
                unit='κ',
                sub=(2, 2, 2*i + 2),
                cmap='RdBu_r'
            )
        else:
            # Empty plot for shells behind source
            ax = plt.subplot(2, 2, 2*i + 2)
            ax.text(0.5, 0.5, f'Shell behind source\n(z={shell_redshifts[i]:.3f} > {source_redshift})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    
    # Save results
    print(f"\n6. Saving results...")
    
    # Save plots
    plot_filename = 'shell_creation_example.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"   Saved plot: {plot_filename}")
    
    # Save shell data (optional - commented out to avoid large files)
    # from jaxrt import save_shells_to_fits
    # save_shells_to_fits(shell_maps, 'density_shells', shell_redshifts)
    # print(f"   Saved FITS files: density_shells_shell_*.fits")
    
    print(f"\n✓ Example completed successfully!")
    print(f"  Created {len(shell_maps)} density shells from {len(positions)} particles")
    print(f"  Generated HEALPix maps with nside={nside} ({12*nside**2} pixels each)")
    print(f"  Computed convergence maps for source at z={source_redshift}")
    
    return shell_maps, shell_redshifts, convergence_maps


if __name__ == "__main__":
    # Set up matplotlib for non-interactive use
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    # Run example
    try:
        shell_maps, shell_redshifts, convergence_maps = main()
        print("\nExample data available in variables: shell_maps, shell_redshifts, convergence_maps")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required packages: pip install -e .")
    except Exception as e:
        print(f"Error running example: {e}")
        raise