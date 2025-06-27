# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaxRT is a pure JAX reimplementation of gravitational lensing ray tracing techniques found in LensTools. The goal is to leverage JAX's GPU acceleration and automatic differentiation capabilities to create a high-performance ray tracing library for weak gravitational lensing analysis.

## Current Development Status

**Early Stage**: The repository currently contains only basic project structure. No JAX implementation exists yet - this needs to be built from scratch.

## Reference Implementation

The `extern/LensTools/` directory contains the original implementation that serves as the reference for reimplementation:

**Key modules to study for reimplementation:**
- `extern/LensTools/lenstools/simulations/raytracing.py` - Core ray tracing algorithms
- `extern/LensTools/lenstools/image/convergence.py` - Convergence map operations  
- `extern/LensTools/lenstools/image/shear.py` - Shear map operations
- `extern/LensTools/lenstools/scripts/raytracing.py` - Ray tracing workflow scripts

**Key concepts from LensTools to reimplement in JAX:**
- Lens plane operations and stacking
- Ray deflection calculations
- Convergence and shear map generation
- Multi-plane ray tracing
- Born approximation ray tracing

## Development Setup (To Be Created)

The following project structure should be established:

```
jaxrt/                 # Main package directory
├── __init__.py
├── core/              # Core ray tracing algorithms
├── planes/            # Lens plane operations
├── maps/              # Convergence/shear map handling
└── utils/             # Utility functions

pyproject.toml         # Project configuration
requirements.txt       # Dependencies (jax, jaxlib, numpy, etc.)
```

## Dependencies (To Be Defined)

Primary dependencies will include:
- `jax` and `jaxlib` for GPU acceleration and autodiff
- `numpy` for array operations compatibility
- `scipy` for scientific computing utilities
- Additional astronomy packages as needed (e.g., `astropy`)

## Development Approach

1. **Study LensTools implementation** - Understand algorithms and data structures
2. **Design JAX architecture** - Plan how to leverage JAX's functional programming model
3. **Implement core ray tracing** - Start with basic single-plane ray tracing
4. **Add multi-plane support** - Extend to multiple lens planes
5. **Optimize for GPU** - Ensure efficient vectorization and memory usage
6. **Add automatic differentiation** - Enable gradient-based optimization

## Key JAX Considerations

- Use `jax.numpy` instead of `numpy` for array operations
- Design with vectorization in mind (`jax.vmap`)
- Consider JIT compilation (`jax.jit`) for performance
- Plan for automatic differentiation (`jax.grad`, `jax.jacobian`)
- Design for functional programming (pure functions, immutable data)