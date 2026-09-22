# Nonlinear Path Optimization

## 1. Mathematical Formulation of Path Optimization

Given an initial coarse or discrete sequence of waypoints $\mathbf{P}_{ref} = [p_1, p_2, \dots, p_N]$, path optimization seeks an optimal smoothed continuous path $\mathbf{p}(s) = [x(s), y(s)]$ that minimizes a multi-term cost function:

$$\min_{\mathbf{p}} J(\mathbf{p}) = w_{smooth} \int_0^S \kappa(s)^2 ds + w_{ref} \sum_{i=1}^N \|\mathbf{p}(s_i) - p_{ref, i}\|^2 + w_{curv\_rate} \int_0^S \dot{\kappa}(s)^2 ds$$

Subject to:
- Initial boundary pose: $x(0) = x_0, y(0) = y_0, \theta(0) = \theta_0, \kappa(0) = \kappa_0$
- Final boundary pose: $x(S) = x_f, y(S) = y_f, \theta(S) = \theta_f, \kappa(S) = \kappa_f$
- Curvature limits: $|\kappa(s)| \le \kappa_{max} \quad \forall s \in [0, S]$
- Corridor clearance: $d(\mathbf{p}(s), \text{obstacle}) \ge d_{safe}$

---

## 2. Numerical Solution via Nonlinear Solvers

In Python, optimization is solved using Sequential Least Squares Programming (`SLSQP`) or Interior Point methods in `scipy.optimize.minimize`.

The path is parameterized by decision variables $\mathbf{z} = [a, b, c, d, S]$ defining a cubic spiral $\kappa(s) = as^3 + bs^2 + cs + d$.
The solver iteratively adjusts $\mathbf{z}$ until gradient tolerances and boundary constraints are satisfied.
