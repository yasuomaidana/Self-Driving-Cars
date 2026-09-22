# Objective Functions for Autonomous Driving

## 1. Multi-Objective Optimization Formulation

The motion planner solves an optimization problem to find the best trajectory $\tau^* = \{\mathbf{x}(t), \mathbf{u}(t)\}_{t=0}^T$ that minimizes a composite cost functional:

$$J(\tau) = w_{safety} J_{safety} + w_{comfort} J_{comfort} + w_{progress} J_{progress} + w_{rules} J_{rules}$$

Subject to:
- Kinematic differential equations: $\dot{\mathbf{x}} = f(\mathbf{x}, \mathbf{u})$
- Initial boundary condition: $\mathbf{x}(0) = \mathbf{x}_{current}$
- Actuation & kinematic bounds: $|\delta| \le \delta_{max}, |a| \le a_{max}$
- Collision avoidance constraints: $\mathcal{V}(\mathbf{x}(t)) \cap \mathcal{O}(t) = \emptyset$

---

## 2. Mathematical Definition of Cost Terms

### A. Safety Cost ($J_{safety}$)
Penalizes proximity to road edges, obstacles, and vulnerable road users (VRUs):

$$J_{safety} = \int_{0}^T \sum_{i \in \text{obstacles}} \exp\left( - \frac{d(\mathbf{x}(t), \mathcal{O}_i(t))^2}{2 \sigma_{safe}^2} \right) dt$$

Where $d(\mathbf{x}, \mathcal{O}_i)$ is the Euclidean distance from the vehicle boundary to obstacle $i$.

### B. Comfort Cost ($J_{comfort}$)
Passenger comfort depends strongly on minimizing accelerations, lateral accelerations, and jerk (time derivative of acceleration):

$$J_{comfort} = \int_{0}^T \left( w_a a(t)^2 + w_{alat} (v(t)^2 \kappa(t))^2 + w_j \dot{a}(t)^2 + w_{\dot{\kappa}} \dot{\kappa}(t)^2 \right) dt$$

Where:
- $\dot{a}(t) = \frac{da}{dt}$ is the longitudinal jerk ($m/s^3$).
- $\dot{\kappa}(t) = \frac{d\kappa}{dt}$ is the curvature rate (related to steering wheel rotational speed).

### C. Progress & Efficiency Cost ($J_{progress}$)
Encourages reaching the destination efficiently and matching desired target speed $v_{target}$:

$$J_{progress} = \int_{0}^T \left( (v(t) - v_{target})^2 + w_{ref} d_{ref}(\mathbf{x}(t))^2 \right) dt$$

Where $d_{ref}(\mathbf{x})$ is the lateral deviation from the lane centerline.

### D. Traffic Rule Adherence Cost ($J_{rules}$)
Penalizes driving over speed limits, crossing solid lane markers, or failing to come to a complete stop before stop lines:

$$J_{rules} = \int_{0}^T \max(0, v(t) - v_{limit})^2 dt + w_{solid\_line} \cdot \mathbb{I}_{\text{lane crossing}}$$
