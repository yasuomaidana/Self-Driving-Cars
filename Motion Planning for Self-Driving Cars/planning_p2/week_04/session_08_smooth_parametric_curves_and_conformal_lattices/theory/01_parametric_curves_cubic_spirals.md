# Parametric Curves & Continuous Curvature Planning

## 1. The Need for Continuous Curvature ($C^2$ Smoothness)

A trajectory is $C^1$ smooth if its tangent (heading angle $\theta$) is continuous. However, instantaneous changes in curvature $\kappa$ require instantaneous changes in steering angle $\delta = \arctan(L \kappa)$, demanding infinite steering velocity $\dot{\delta} \to \infty$ and producing unbounded lateral jerk $\dot{a}_{lat} \to \infty$.

Therefore, autonomous vehicle paths must be **$C^2$ smooth** (continuous curvature $\kappa(s)$).

---

## 2. Cubic Spirals & Quintic Splines

### A. Cubic Spirals
Curvature $\kappa(s)$ is parameterized as a cubic polynomial of arc length $s \in [0, S]$:
$$\kappa(s) = a s^3 + b s^2 + c s + d$$
Heading and position are recovered by Fresnel-type numerical integration:
$$\theta(s) = \theta_0 + \int_0^s \kappa(u) du$$
$$x(s) = x_0 + \int_0^s \cos(\theta(u)) du, \quad y(s) = y_0 + \int_0^s \sin(\theta(u)) du$$

### B. Quintic Polynomial Paths (Frenet Lateral Profile)
A 5th-order polynomial $d(s) = a_0 + a_1 s + a_2 s^2 + a_3 s^3 + a_4 s^4 + a_5 s^5$ uniquely matches 6 boundary conditions:
- Initial state: Position $d_0$, slope $d'_0$, curvature $d''_0$ at $s = 0$.
- Target state: Position $d_f$, slope $d'_f = 0$, curvature $d''_f = 0$ at $s = S$.
