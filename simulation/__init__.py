"""
Celestium 2.0 - Simulation Package
"""

from .physics import (
    GravityModel,
    MU_EARTH,
    MU_MOON,
    R_EARTH,
    R_MOON,
    hohmann_delta_v,
    calculate_tli_velocity,
    vis_viva_velocity,
    G0
)

from .integrator import OrbitalIntegrator, AdaptiveIntegrator

from .spacecraft import Vehicle, VEHICLES

# optimization과 trajectory는 나중에 import (순환 참조 방지)
__version__ = "2.0.0"
