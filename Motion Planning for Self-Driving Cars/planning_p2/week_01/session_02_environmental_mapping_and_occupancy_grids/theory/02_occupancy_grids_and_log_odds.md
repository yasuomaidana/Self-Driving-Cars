# Occupancy Grids & The Log-Odds Formulation

## 1. 2D Occupancy Grid Mapping

An **Occupancy Grid Map** discretizes the continuous continuous 2D Cartesian world into a tessellated grid of cells:
$$\mathbf{m} = \{m_i\}, \quad i \in \{1, 2, \dots, N\}$$

Each grid cell $m_i$ represents a small square area (e.g. $0.1m \times 0.1m$ or $0.2m \times 0.2m$) with a binary state:
- $m_i = 1$: Cell is occupied by an obstacle.
- $m_i = 0$: Cell is free drivable space.

Under the assumption that individual cell states are conditionally independent given measurements $z_{1:t}$ and vehicle poses $x_{1:t}$:

$$P(\mathbf{m} \mid z_{1:t}, x_{1:t}) = \prod_{i} P(m_i \mid z_{1:t}, x_{1:t})$$

---

## 2. The Log-Odds Update Rule

Directly updating probabilities $P(m_i \mid z_{1:t}, x_{1:t})$ via Bayes' rule requires continuous multiplications and normalization, leading to floating-point underflow or rounding errors near $0$ and $1$.

To eliminate these issues, we define the **Log-Odds ratio** $l(m_i)$:

$$l(m_i) = \log \left( \frac{P(m_i)}{1 - P(m_i)} \right) = \operatorname{logit}(P(m_i))$$

The inverse transformation to recover the probability from log-odds is the standard logistic sigmoid:

$$P(m_i) = 1 - \frac{1}{1 + \exp(l(m_i))} = \frac{1}{1 + \exp(-l(m_i))}$$

### Recursive Binary Bayes Update Equation
Applying Bayes' theorem and taking logarithms yields the simple additive update rule:

$$l_{t}(m_i) = l_{t-1}(m_i) + \operatorname{inverse\_sensor\_model}(m_i, z_t, x_t) - l_0$$

Where:
- $l_t(m_i)$: Posterior log-odds of cell $i$ at time $t$.
- $l_{t-1}(m_i)$: Prior log-odds from time $t-1$.
- $\operatorname{inverse\_sensor\_model}(m_i, z_t, x_t) = \log \left( \frac{P(m_i \mid z_t, x_t)}{1 - P(m_i \mid z_t, x_t)} \right)$
- $l_0 = \log\left(\frac{P_{prior}(m_i)}{1 - P_{prior}(m_i)}\right)$ is the prior log-odds (for uniform uninformative prior $P_0 = 0.5 \implies l_0 = 0$).

### Clamping Log-Odds for Dynamic Adaptability
To prevent log-odds from growing infinitely (which would make it impossible to clear an obstacle that moved away), we clamp $l_t$:

$$l_{min} \le l_t(m_i) \le l_{max}$$
Typical values: $l_{min} = -4.0$ ($P \approx 0.018$), $l_{max} = +4.0$ ($P \approx 0.982$).
