# Copilot Instructions for TFG - Earth-Moon Trajectory Simulation

## Project Overview

This is a **numerical simulation project for computing Earth-Moon-Projectile trajectories** (Trabajo Fin de Grado - TFG). The codebase models projectile motion under gravitational influence of both Earth and Moon using numerical integration (RK45 method via scipy).

**Key Objective**: Analyze optimal initial velocities and Moon position angles to achieve specific trajectory outcomes (e.g., moon impact, escape, orbital mechanics).

## Architecture & Core Concepts

### Two Main Physics Models

1. **Modelo_1.py** - Simplified 1D model
   - Single dimension: distance `r` from Earth center to projectile
   - State: `[r(t), v(t)]` (position, velocity)
   - Simpler physics, faster computation
   - Good for baseline velocity calculations

2. **Modelo_2_*.py** (Advanced 2D models)
   - 2D coordinates: `[x, y, vx, vy]` (position and velocity components)
   - Moon orbits Earth at angular velocity `w = 2.662e-6 rad/s`
   - Two variants:
     - `_con_posicion_inicial_Luna.py`: Moon position configurable via `theta_0` parameter
     - `_sin_posicion_inicial_Luna.py`: Moon at fixed position (simplified)
   - Realistic gravity from both bodies

### Constant Values (SI Units)
All files share these fundamental constants defined at module level:
- `G = 6.67430e-11` (gravitational constant)
- `Mt = 5.972e24` (Earth mass, kg)
- `Ml = 7.34767309e22` (Moon mass, kg)
- `Rt = 6.371e6` (Earth radius, m)
- `Rl = 1.737e6` (Moon radius, m)
- `S = 3.844e8` (Earth-Moon center distance, m)
- `w = 2.662e-6` (Moon angular velocity, rad/s)

> **Note**: Constants vary slightly between files; use values from the file you're actively modifying.

## Critical Functions & Patterns

### State Representation
```python
# 1D Model (Modelo_1.py)
y = [r, v]  # radius, velocity along line to moon

# 2D Model (Modelo_2_*.py)
state = [x, y, vx, vy]  # 2D position and velocity components
```

### ODE System Definition
```python
def system_equations(t, state):
    """Returns [dx/dt, dy/dt, dvx/dt, dvy/dt]"""
    # Calculate gravitational accelerations
    # Return derivatives as list
```

### Event Functions (Collision Detection)
Events terminate integration when conditions met:
- `collision_moon(t, state)`: Triggered when projectile reaches Moon surface
- `collision_earth(t, state)`: Triggered when projectile re-enters Earth atmosphere
- Both marked with `.terminal = True` to stop integration

**Pattern**: Return signed distance (negative = collision), not boolean.

### Trajectory Simulation Pattern
```python
state0 = [Rt, 0, v0, 0]  # Start at Earth surface, horizontal launch
t_span = (0, 5*24*3600)  # 5 days max
sol = solve_ivp(
    system_equations, 
    t_span, 
    state0,
    method='RK45',
    rtol=1e-8,            # High precision for stability
    atol=1e-8,
    events=[collision_moon, collision_earth]
)
```

## File Organization & Purpose

| File | Purpose |
|------|---------|
| `Modelo_1.py` | 1D simplified model, velocity calculation examples |
| `Modelo_2_con_posicion_inicial_Luna.py` | Advanced 2D with configurable Moon position (`theta_0`) |
| `Modelo_2_sin_posicion_inicial_Luna.py` | Advanced 2D with fixed Moon position |
| `trayectorias_modelo_2_1.py` | Visualization of multiple trajectories with different `v0` values |
| `trayectorias_dsitintas_v0_modelo_2.py` | Trajectory comparison utility |
| `Notebook_resultados_modelos.ipynb` | Analysis & results aggregation |
| `optimization.csv`, `solution.csv` | Output from trajectory optimizations |

## Simulation Workflow

1. **Define initial velocity** `v0` (m/s) and Moon angle `theta_0` (radians)
2. **Set up ODE problem** with state, time span, and event functions
3. **Solve with `solve_ivp`** using RK45 (Runge-Kutta 4/5)
4. **Extract trajectory** from `sol.t` (time) and `sol.y` (state components)
5. **Analyze results**: Check for collisions, compute minimum distances, plot

## Project-Specific Conventions

### Angle Conventions
- Moon angle: `theta_0 = -np.pi/4` is common starting position (~-45°)
- Range for exploration: `np.linspace(-np.pi/6, -np.pi/12, N)` (±30° to ±15°)

### Velocity Ranges
- Baseline: ~11184 m/s (escape velocity + refinements)
- Exploration: typically ±10% around baseline (e.g., 11100 to 11200 m/s)

### Integration Parameters
- Always use `rtol=1e-8, atol=1e-8` for high precision
- Method: Always `RK45` (adaptive Runge-Kutta)
- Time: Usually 5 days max = `5*24*3600 = 432000` seconds

### Visualization Patterns
- **3D surfaces**: `ax.plot_surface()` for velocity vs. angle vs. distance metrics
- **2D trajectories**: Multi-line plots with different colors per velocity
- **Celestial bodies**: Scaled circles for Earth/Moon for visibility
- **Moon orbit**: Dashed circle at radius `S`
- **Units in plots**: Convert to km/s or km for readability

## Development Patterns to Follow

1. **Modularity**: Extract helper functions (`moon_position()`, `calculate_distances()`) so main flow is clear
2. **Parameterization**: Allow `v0`, `theta_0`, `t_max` as function parameters, not hardcoded
3. **Error events**: Always include collision detection in `solve_ivp`
4. **Documentation**: Docstrings describe what physical quantities are computed
5. **Naming**: Use Spanish descriptors where present (e.g., `Luna`, `Tierra`) but keep function names in English

## Key Dependencies

- **numpy**: Numerical arrays and calculations
- **scipy.integrate.solve_ivp**: ODE integration (RK45 method)
- **matplotlib.pyplot**: All visualizations
- **Optional**: pandas for CSV handling in `optimization.csv`

## When Extending the Codebase

- Add new models to existing Modelo_N.py pattern
- Reuse constant definitions (or centralize in a `constants.py` if standardizing)
- Event functions must return scalar signed distance, with `.terminal = True` set
- All simulations expect `[x, y, vx, vy]` state format for Modelo_2 variants
- Trajectory analysis should compute minimum distance to Moon surface, not just collision binary
