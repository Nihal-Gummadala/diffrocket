import jax
import jax.numpy as jnp

import diffrocket
from diffrocket import constants


def test_version():
    assert isinstance(diffrocket.__version__, str)


def test_sea_level_density_consistent():
    # Ideal gas law: rho = p / (R T)
    rho = constants.SEA_LEVEL_PRESSURE / (constants.R_AIR * constants.SEA_LEVEL_TEMPERATURE)
    assert abs(rho - constants.SEA_LEVEL_DENSITY) < 1e-3


def test_jax_grad_works():
    # Sanity check that JAX autodiff is available in CI.
    f = lambda x: jnp.sin(x) ** 2
    assert jnp.isclose(jax.grad(f)(0.3), jnp.sin(0.6))
