# diffrocket

**A differentiable rocket flight simulator built on JAX.**

> 🚧 In active development. Target: v0.1 on PyPI by December 2026.

Most rocket simulators answer *"how high will this rocket go?"*
diffrocket also answers *"how should I change the rocket to make it go where I want?"*
Because every step of the flight is written in JAX, you can take gradients of
any flight outcome (apogee, landing point, stability margin) with respect to
any design parameter, and use them to:

- **Optimize designs.** Find the fin size or dry mass that hits a target apogee.
- **Fit the model to real flights.** Recover drag coefficient and thrust from an altimeter log.
- **Run fast Monte Carlo.** `vmap` thousands of dispersed flights at once.

## Roadmap

- [ ] 1D vertical ascent (gravity, thrust curves, drag, standard atmosphere)
- [ ] First gradient: d(apogee)/d(Cd), checked against finite differences
- [ ] 3-DOF with launch rail and wind
- [ ] Differentiable recovery events (apogee detection, parachute)
- [ ] 6-DOF rigid body with quaternions
- [ ] Aerodynamic moments (Barrowman center of pressure, damping)
- [ ] Simulated sensors and a Kalman filter
- [ ] Validation against OpenRocket and RocketPy
- [ ] v0.1 release on PyPI

## Install (development)

```bash
git clone https://github.com/YOUR-USERNAME/diffrocket.git
cd diffrocket
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
