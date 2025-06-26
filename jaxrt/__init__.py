"""
JaxRT: JAX-based gravitational lensing ray tracing library

A pure JAX reimplementation of gravitational lensing ray tracing techniques
for high-performance GPU acceleration and automatic differentiation.
"""

__version__ = "0.1.0"

from .core.born_convergence import (
    born_convergence,
    born_convergence_safe,
    born_convergence_from_cosmology,
    lensing_kernel
)
from .planes.density_plane import DensityPlane
from .maps.convergence_map import ConvergenceMap

__all__ = [
    "born_convergence",
    "born_convergence_safe", 
    "born_convergence_from_cosmology",
    "lensing_kernel",
    "DensityPlane", 
    "ConvergenceMap",
]