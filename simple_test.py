"""
Simple test script for JaxRT without external dependencies.
This script tests the basic functionality using mock JAX functions.
"""

import numpy as np
import sys
import os

# Add our package to the path
sys.path.insert(0, '.')

def mock_jax_test():
    """Test our implementation using mock JAX functions."""
    print("Testing JaxRT implementation with mock functions...")
    
    # Mock JAX numpy
    class MockJNP:
        @staticmethod
        def array(x):
            return np.array(x)
        
        @staticmethod
        def stack(arrays, axis=0):
            return np.stack(arrays, axis=axis)
        
        @staticmethod
        def linspace(start, stop, num):
            return np.linspace(start, stop, num)
        
        @staticmethod
        def meshgrid(x, y, indexing='xy'):
            return np.meshgrid(x, y, indexing=indexing)
        
        @staticmethod
        def zeros(shape):
            return np.zeros(shape)
        
        @staticmethod
        def where(condition, x, y):
            return np.where(condition, x, y)
        
        @staticmethod
        def clip(a, a_min, a_max):
            return np.clip(a, a_min, a_max)
        
        @staticmethod
        def floor(x):
            return np.floor(x)
        
        @staticmethod
        def minimum(x, y):
            return np.minimum(x, y)
        
        @staticmethod
        def sqrt(x):
            return np.sqrt(x)
        
        @staticmethod
        def mean(x):
            return np.mean(x)
        
        @staticmethod
        def fft():
            return np.fft
        
        @staticmethod
        def real(x):
            return np.real(x)
        
        @staticmethod
        def pi():
            return np.pi
    
    # Mock JAX random
    class MockRandom:
        @staticmethod
        def PRNGKey(seed):
            np.random.seed(seed)
            return seed
        
        @staticmethod
        def split(key, num=2):
            return [key + i for i in range(num)]
        
        @staticmethod
        def normal(key, shape):
            return np.random.normal(size=shape)
    
    # Mock JAX functions
    class MockJax:
        random = MockRandom()
        
        @staticmethod
        def jit(func):
            return func
    
    # Replace imports in our modules
    import jaxrt.core.born_convergence as bc_module
    import jaxrt.planes.density_plane as dp_module
    import jaxrt.maps.convergence_map as cm_module
    
    # Replace jax.numpy with our mock
    bc_module.jnp = MockJNP()
    dp_module.jnp = MockJNP()
    cm_module.jnp = MockJNP()
    
    # Replace jax with our mock
    dp_module.jax = MockJax()
    
    # Test basic functionality
    print("✓ Module imports successful")
    
    # Test density plane generation
    try:
        density_map = dp_module.generate_gaussian_density_plane(
            resolution=32,
            map_size_rad=0.1,
            power_spectrum_amplitude=1e-4,
            random_key=42
        )
        print(f"✓ Generated density plane: {density_map.shape}")
    except Exception as e:
        print(f"❌ Density plane generation failed: {e}")
        return False
    
    # Test ray grid creation
    try:
        ray_positions = cm_module.create_ray_grid(32, 0.1)
        print(f"✓ Created ray grid: {ray_positions.shape}")
    except Exception as e:
        print(f"❌ Ray grid creation failed: {e}")
        return False
    
    # Test lensing kernel
    try:
        chi = np.array([100, 200, 300, 400])  # Mpc
        chi_source = 500  # Mpc
        kernel = bc_module.lensing_kernel(chi, chi_source)
        expected = 1.0 - chi / chi_source
        print(f"✓ Lensing kernel test: {kernel} vs {expected}")
        assert np.allclose(kernel, expected)
    except Exception as e:
        print(f"❌ Lensing kernel test failed: {e}")
        return False
    
    print("All basic tests passed!")
    return True

def test_project_structure():
    """Test that all expected files exist."""
    print("Testing project structure...")
    
    expected_files = [
        'jaxrt/__init__.py',
        'jaxrt/core/__init__.py',
        'jaxrt/core/born_convergence.py',
        'jaxrt/planes/__init__.py',
        'jaxrt/planes/density_plane.py',
        'jaxrt/maps/__init__.py',
        'jaxrt/maps/convergence_map.py',
        'jaxrt/utils/__init__.py',
        'tests/__init__.py',
        'tests/test_born_convergence.py',
        'pyproject.toml',
        'requirements.txt',
        'README.md',
        'CLAUDE.md',
        'demo_born_convergence.py'
    ]
    
    missing_files = []
    for file in expected_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✓ All expected files present")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("JaxRT Simple Test Suite")
    print("=" * 50)
    
    # Test project structure
    if not test_project_structure():
        print("❌ Project structure test failed")
        return False
    
    # Test mock implementation
    if not mock_jax_test():
        print("❌ Mock implementation test failed")
        return False
    
    print("=" * 50)
    print("✅ All tests passed!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("1. Install JAX: pip install jax jaxlib")
    print("2. Install jax-cosmo: pip install jax-cosmo")
    print("3. Run full tests: python tests/test_born_convergence.py")
    print("4. Run demo: python demo_born_convergence.py")
    
    return True

if __name__ == "__main__":
    main()