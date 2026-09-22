# Frenet-Serret Frame Transformations

## 1. The Road-Aligned Frenet Coordinate Frame

Let $\mathbf{r}(s) = [x_r(s), y_r(s)]^T$ be the reference path parameterized by arc length $s$, with unit tangent $\mathbf{t}_r(s) = [\cos(\theta_r(s)), \sin(\theta_r(s))]^T$ and unit normal $\mathbf{n}_r(s) = [-\sin(\theta_r(s)), \cos(\theta_r(s))]^T$.

Any Cartesian position $\mathbf{p} = [x, y]^T$ is decomposed into:
- $s$: Arc length of the closest point along the reference path.
- $d$: Perpendicular signed lateral offset from the reference path.

$$\mathbf{p}(s, d) = \mathbf{r}(s) + d \cdot \mathbf{n}_r(s)$$

In scalar components:
$$x(s, d) = x_r(s) - d \sin(\theta_r(s))$$
$$y(s, d) = y_r(s) + d \cos(\theta_r(s))$$

---

## 2. Heading and Curvature Transformations

The vehicle heading $\theta$ in global Cartesian space relates to the reference heading $\theta_r$ and lateral slope $d' = \frac{\partial d}{\partial s}$:

$$\theta = \theta_r(s) + \arctan\left(\frac{d'(s)}{1 - d(s) \kappa_r(s)}\right)$$

Where $\kappa_r(s)$ is the curvature of the reference curve.
