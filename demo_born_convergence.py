"""
Demo script for JaxRT Born convergence implementation.

This script demonstrates how to use the JAX-based Born convergence
implementation and compares it with the LensTools reference.
"""

import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from astropy.units import deg
import time

# Import our JAX implementation
from jaxrt.core.born_convergence import born_convergence_from_cosmology
from jaxrt.planes.density_plane import create_density_planes_sequence
from jaxrt.maps.convergence_map import create_ray_grid, ConvergenceMap

# Try to import LensTools for comparison
try:
    from lenstools.simulations.raytracing import RayTracer, DensityPlane
    from lenstools import ConvergenceMap as LensToolsConvergenceMap
    import lenstools
    from astropy.cosmology import Planck18
    LENSTOOLS_AVAILABLE = True
    print("LensTools available - will run comparison")
except ImportError:
    LENSTOOLS_AVAILABLE = False
    print("LensTools not available - running JAX implementation only")

import jax_cosmo as jc


def main():
    """Run the Born convergence demo."""
    print("=" * 60)
    print("JaxRT Born Convergence Demo")
    print("=" * 60)
    
    # Test parameters
    params = {
        'resolution': 128,
        'map_size_degrees': 2.0,
        'n_planes': 20,
        'redshift_range': (0.1, 2.0),
        'source_redshift': 2.0,
        'power_spectrum_amplitude': 5e-4,
        'random_seed': 42
    }
    
    print(f"Parameters:")
    print(f"  Resolution: {params['resolution']}x{params['resolution']}")
    print(f"  Map size: {params['map_size_degrees']}°")
    print(f"  Number of planes: {params['n_planes']}")
    print(f"  Redshift range: {params['redshift_range']}")
    print(f"  Source redshift: {params['source_redshift']}")
    print()
    
    # Convert map size to radians
    map_size_rad = params['map_size_degrees'] * np.pi / 180.0
    
    # Generate density planes
    print("Generating Gaussian density planes...")
    start_time = time.time()
    
    density_planes, redshifts = create_density_planes_sequence(
        n_planes=params['n_planes'],
        redshift_range=params['redshift_range'],
        resolution=params['resolution'],
        map_size_rad=map_size_rad,
        power_spectrum_amplitude=params['power_spectrum_amplitude'],
        random_seed=params['random_seed']
    )
    
    plane_generation_time = time.time() - start_time
    print(f"  Generated {params['n_planes']} planes in {plane_generation_time:.3f}s")
    
    # Create ray grid
    print("Creating ray grid...")
    ray_positions = create_ray_grid(params['resolution'], map_size_rad)
    print(f"  Created {ray_positions.shape[1]} rays")
    
    # Compute Born convergence with JAX
    print("\nComputing Born convergence with JAX...")
    start_time = time.time()
    
    convergence_jax = born_convergence_from_cosmology(
        ray_positions=ray_positions,
        density_planes=list(density_planes),
        plane_redshifts=redshifts,
        source_redshift=params['source_redshift'],
        map_size_rad=map_size_rad,
        map_resolution=params['resolution']
    )
    
    jax_time = time.time() - start_time
    print(f"  JAX computation time: {jax_time:.3f}s")
    
    # Create ConvergenceMap
    conv_map_jax = ConvergenceMap.from_ray_positions(
        ray_positions=ray_positions,
        convergence_values=convergence_jax,
        map_size_rad=map_size_rad,
        source_redshift=params['source_redshift']
    )
    
    # Print statistics
    stats_jax = conv_map_jax.statistics()
    print(f"  JAX Convergence statistics:")
    print(f"    Mean: {stats_jax['mean']:.6f}")
    print(f"    RMS: {stats_jax['rms']:.6f}")
    print(f"    Std: {stats_jax['std']:.6f}")
    print(f"    Range: [{stats_jax['min']:.6f}, {stats_jax['max']:.6f}]")
    
    # Compare with LensTools if available
    convergence_lenstools = None
    lenstools_time = None
    
    if LENSTOOLS_AVAILABLE:
        print("\nComputing Born convergence with LensTools...")
        start_time = time.time()
        
        # Convert density planes to numpy
        density_planes_np = [np.array(plane) for plane in density_planes]
        
        # Setup LensTools RayTracer
        tracer = RayTracer(lens_mesh_size=params['resolution'])
        cosmo = Planck18
        
        # Add density planes to tracer
        for i, (density_map, z) in enumerate(zip(density_planes_np, redshifts)):
            angle = params['map_size_degrees'] * deg
            plane = DensityPlane(
                data=density_map,
                angle=angle,
                redshift=float(z),
                cosmology=cosmo,
                unit=lenstools.rad**2
            )
            tracer.addLens(plane)
        
        # Convert ray positions for LensTools
        ray_positions_lenstools = np.array(ray_positions) * 180.0 / np.pi
        ray_positions_lenstools = ray_positions_lenstools.T * deg
        
        # Compute convergence
        convergence_lenstools = tracer.convergenceBorn(
            ray_positions_lenstools, 
            z=params['source_redshift'],
            save_intermediate=False
        )
        
        lenstools_time = time.time() - start_time
        print(f"  LensTools computation time: {lenstools_time:.3f}s")
        
        # LensTools statistics
        mean_lt = np.mean(convergence_lenstools)
        rms_lt = np.sqrt(np.mean(convergence_lenstools**2))
        std_lt = np.std(convergence_lenstools)
        
        print(f"  LensTools Convergence statistics:")
        print(f"    Mean: {mean_lt:.6f}")
        print(f"    RMS: {rms_lt:.6f}")
        print(f"    Std: {std_lt:.6f}")
        print(f"    Range: [{convergence_lenstools.min():.6f}, {convergence_lenstools.max():.6f}]")
        
        # Compare results
        correlation = np.corrcoef(
            convergence_lenstools.flatten(), 
            np.array(convergence_jax).flatten()
        )[0, 1]
        
        print(f"\n  Comparison:")
        print(f"    Correlation: {correlation:.6f}")
        print(f"    Mean difference: {abs(mean_lt - stats_jax['mean']):.6f}")
        print(f"    RMS difference: {abs(rms_lt - stats_jax['rms']):.6f}")
        print(f"    Speed ratio (LensTools/JAX): {lenstools_time/jax_time:.2f}x")
    
    # Create visualization
    print("\nCreating visualization...")
    
    if LENSTOOLS_AVAILABLE:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        axes = [axes]
    
    # JAX convergence map
    conv_map_2d = conv_map_jax.convergence_map
    extent = [0, params['map_size_degrees'], 0, params['map_size_degrees']]
    
    im1 = axes[0].imshow(conv_map_2d, extent=extent, origin='lower', cmap='RdBu_r')
    axes[0].set_title('JAX Born Convergence')
    axes[0].set_xlabel('Angle [degrees]')
    axes[0].set_ylabel('Angle [degrees]')
    plt.colorbar(im1, ax=axes[0], label='κ')
    
    if LENSTOOLS_AVAILABLE:
        # LensTools convergence map
        conv_map_lt = convergence_lenstools.reshape(params['resolution'], params['resolution'])
        im2 = axes[1].imshow(conv_map_lt, extent=extent, origin='lower', cmap='RdBu_r')
        axes[1].set_title('LensTools Born Convergence')
        axes[1].set_xlabel('Angle [degrees]')
        axes[1].set_ylabel('Angle [degrees]')
        plt.colorbar(im2, ax=axes[1], label='κ')
        
        # Difference map
        diff_map = np.array(conv_map_2d) - conv_map_lt
        im3 = axes[2].imshow(diff_map, extent=extent, origin='lower', cmap='RdBu_r')
        axes[2].set_title('Difference (JAX - LensTools)')
        axes[2].set_xlabel('Angle [degrees]')
        axes[2].set_ylabel('Angle [degrees]')
        plt.colorbar(im3, ax=axes[2], label='Δκ')
    
    plt.tight_layout()
    plt.savefig('born_convergence_demo.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization as 'born_convergence_demo.png'")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print(f"  Plane generation: {plane_generation_time:.3f}s")
    print(f"  JAX Born convergence: {jax_time:.3f}s")
    if LENSTOOLS_AVAILABLE:
        print(f"  LensTools Born convergence: {lenstools_time:.3f}s")
        print(f"  JAX speedup: {lenstools_time/jax_time:.2f}x")
    
    print(f"\nDemo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()