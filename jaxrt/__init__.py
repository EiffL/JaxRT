"""
JaxRT: JAX-based gravitational lensing ray tracing library

A pure JAX reimplementation of gravitational lensing ray tracing techniques
for high-performance GPU acceleration and automatic differentiation.
"""

__version__ = "0.1.0"

from .core.born_convergence import (
    born_convergence,
    born_convergence_from_cosmology,
    lensing_kernel
)
from .maps.convergence_map import ConvergenceMap
from .planes.shells import (
    create_density_shells_from_particles,
    create_shells_from_lightcone,
    convert_shells_to_convergence,
    save_shells_to_fits
)

__all__ = [
    "born_convergence",
    "born_convergence_from_cosmology",
    "lensing_kernel",
    "ConvergenceMap",
    "create_density_shells_from_particles",
    "create_shells_from_lightcone",
    "convert_shells_to_convergence", 
    "save_shells_to_fits",
]