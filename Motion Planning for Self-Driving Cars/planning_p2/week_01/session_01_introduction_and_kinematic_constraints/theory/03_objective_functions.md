# Objective Functions for Autonomous Driving

## 1. Multi-Objective Optimization Formulation

The motion planner solves an optimization problem to find the optimal trajectory $\tau^* = \{\mathbf{x}(t), \mathbf{u}(t)\}_{t=0}^T$ that minimizes a composite cost functional:

$$J(\tau) = w_{safety} J_{safety} + w_{comfort} J_{comfort} + w_{progress} J_{progress} + w_{rules} J_{rules}$$

Subject to:
- Kinematic differential equations: $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$
- Initial boundary condition: $\mathbf{x}(0) = \mathbf{x}_{current}$
- Actuation & kinematic bounds: $|\delta| \le \delta_{max}, |a| \le a_{max}$
- Collision avoidance constraints: $\mathcal{V}(\mathbf{x}(t)) \cap \mathcal{O}(t) = \emptyset$

---

## 2. Mathematical Definition of Cost Terms (Modern Spatiotemporal Domain)

### A. Safety Cost ($J_{safety}$)
Penalizes proximity to road edges, obstacles, and vulnerable road users (VRUs) using smooth potential fields:

$$J_{safety} = \int_{0}^T \sum_{i \in \text{obstacles}} \exp\left( - \frac{d(\mathbf{x}(t), \mathcal{O}_i(t))^2}{2 \sigma_{safe}^2} \right) dt$$

Where $d(\mathbf{x}, \mathcal{O}_i)$ is the Euclidean distance from the vehicle boundary to obstacle $i$.

### B. Comfort Cost ($J_{comfort}$)
Passenger comfort depends directly on minimizing acceleration, lateral acceleration, and jerk (time derivative of acceleration):

$$J_{comfort} = \int_{0}^T \left( w_a a(t)^2 + w_{alat} (v(t)^2 \kappa(t))^2 + w_j \dot{a}(t)^2 + w_{\dot{\kappa}} \dot{\kappa}(t)^2 \right) dt$$

Where:
- $\dot{a}(t) = \frac{da}{dt}$ is longitudinal jerk ($m/s^3$).
- $\dot{\kappa}(t) = \frac{d\kappa}{dt}$ is the curvature rate (proportional to steering wheel rotational velocity).
- $v(t)^2 \kappa(t) = a_{lat}(t)$ is lateral acceleration ($m/s^2$).

### C. Progress & Efficiency Cost ($J_{progress}$)
Encourages reaching the destination efficiently and matching desired cruising speed $v_{target}$:

$$J_{progress} = \int_{0}^T \left( (v(t) - v_{target})^2 + w_{ref} d_{ref}(\mathbf{x}(t))^2 \right) dt$$

Where $d_{ref}(\mathbf{x})$ is the lateral cross-track error from the lane centerline.

### D. Traffic Rule Adherence Cost ($J_{rules}$)
Penalizes driving over speed limits, crossing solid lane markers, or failing to stop at stop lines using differentiable penalty formulations:

$$J_{rules} = \int_{0}^T \max(0, v(t) - v_{limit})^2 dt + w_{solid\_line} \cdot \mathbb{I}_{\text{lane crossing}}$$

---

## 3. Comparison: Classical Spatial ($s$) vs. Modern Spatiotemporal ($t$) Formulations

Why do some classical literature and course materials use different objective functions?

Motion planning has historically evolved from **decoupled spatial planning** (Path-Velocity Decomposition) to **unified spatiotemporal trajectory optimization** (Model Predictive Control & Variational Trajectory Optimization).

### Summary Comparison Table

| Category                   | Classical Decoupled Spatial Formulation ($s \in [0, s_f]$)                               | Modern Unified Spatiotemporal Formulation ($t \in [0, T]$)                    | Rationale for Difference                                                                                                                                                                                      |
| :------------------------- | :--------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Domain**                 | Parameterized along path arc-length $s$: $\int_0^{s_f} (\dots) ds$                       | Parameterized over continuous time $t$: $\int_0^T (\dots) dt$                 | Unifies path planning and velocity profile into a single dynamic state-space optimization.                                                                                                                    |
| **Shortest Path**          | Arc-length minimization: $s_f = \int_{x_i}^{x_f}\sqrt{1+\left(\frac{dy}{dx}\right)^2}dx$ | Implicitly handled via reference tracking ($d_{ref}^2$) and tracking velocity | Direct integration along fixed geometric paths is decoupled from time dynamics.                                                                                                                               |
| **Travel Time / Progress** | Direct time integral: $T_f = \int_0^{s_f}\frac{1}{v(s)}ds$                               | Tracking target cruising speed: $\int_0^T (v(t) - v_{target})^2 dt$           | $\frac{1}{v(s)}$ has a singularity at $v=0$ (stopped vehicle). Velocity tracking is well-conditioned at all speeds.                                                                                           |
| **Curvature & Bending**    | Static curvature bending energy: $\int_0^{s_f} \|\kappa(s)\|^2 ds$                       | Dynamic lateral acceleration & steer rate: $(v^2\kappa)^2 + \dot{\kappa}^2$   | Curvature at 5 km/h is comfortable, but the same curvature at 100 km/h is dangerous. Dynamic acceleration captures true physics.                                                                              |
| **Smoothness**             | Spatial jerk: $\int_0^{s_f} \|\dddot{x}(s)\|^2 ds$                                       | Time-domain jerk & control effort: $\dot{a}(t)^2 + \dot{\delta}(t)^2$         | Directly relates to human vestibular comfort and actuator motor strain.                                                                                                                                       |
| **Speed Limit Adherence**  | Linear Hinge Loss: $\int_0^{s_f} (v(s) - v_{limit})_+ ds$                                | Squared Hinge Loss: $\int_0^T \max(0, v(t) - v_{limit})^2 dt$                 | **Differentiability ($C^1$ smoothness)**: Linear hinge has a non-differentiable kink at $v=v_{limit}$, causing numerical gradient solvers (QP, SQP, CasADi) to oscillate. Squared hinge is smooth and convex. |

---

## 4. Key Takeaways for Optimization

1. **Decoupled Planning (Classical):**
   - Stage 1: Solve geometric path $\mathbf{r}(s) = [x(s), y(s)]^T$ minimizing arc length and curvature $\int \kappa(s)^2 ds$.
   - Stage 2: Solve 1D velocity profile $v(s)$ subject to lateral acceleration limits $v(s) \le \sqrt{a_{lat,max}/\kappa(s)}$.
   - *Limitation:* Can produce conservative or infeasible trajectories in dynamic traffic with moving obstacles.

2. **Unified Trajectory Optimization (Modern):**
   - Simultaneously optimizes state trajectory $\mathbf{x}(t)$ and control inputs $\mathbf{u}(t) = [a(t), \dot{\delta}(t)]^T$ over time horizon $T$.
   - Directly handles dynamic obstacles, passenger comfort (jerk $\dot{a}$), and actuator limits within a smooth optimization program.
