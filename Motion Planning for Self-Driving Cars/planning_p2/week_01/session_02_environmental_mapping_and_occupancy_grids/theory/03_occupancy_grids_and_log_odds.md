# Occupancy Grids & The Log-Odds Formulation

## 1. Occupancy Grid Representations: 2D vs. 3D

An **Occupancy Grid Map** discretizes continuous Cartesian space into a tessellated grid of discrete cells or volumetric elements:
$$\mathbf{m} = \{m_i\}, \quad i \in \{1, 2, \dots, N\}$$

Depending on the operational design domain (ODD) and computational budget, occupancy grids are implemented in either **2D** or **3D**:

### A. 2D Occupancy Grids (Planar)
* **Geometry:** Discretizes the 2D ground plane $(x, y)$ into square pixels (e.g., $0.1\,\text{m} \times 0.1\,\text{m}$ or $0.2\,\text{m} \times 0.2\,\text{m}$).
* **Usage:** Standard on-road autonomous driving where vertical height differences can be simplified after ground removal and height clipping.
* **Pros:** Highly computationally efficient, low memory footprint ($O(N^2)$), and enables fast 2D convolution/distance transforms for costmaps.
* **Cons:** Cannot represent multi-level structures (e.g., multi-story parking garages, driving under overpasses/bridges, tunnels).

### B. 3D Occupancy Grids (Volumetric Voxels & OctoMaps)
* **Geometry:** Discretizes 3D space $(x, y, z)$ into cubic voxels (e.g., $0.1\,\text{m} \times 0.1\,\text{m} \times 0.1\,\text{m}$), often managed hierarchically using **Octrees** (such as *OctoMap*) or GPU-accelerated sparse voxel hash tables.
* **Usage:** Drones/UAVs, off-road autonomous vehicles, warehouse robots with vertical manipulators, and complex multi-level urban environments.
* **Pros:** Fully captures 3D overhangs, bridges, tunnels, tree canopies, and sloping terrain.
* **Cons:** High memory consumption ($O(N^3)$ dense), higher ray-casting complexity ($O(N^3)$ 3D Bresenham/voxel traversal), and slower inflation.

### Cell State Definition:
Each cell or voxel $m_i$ maintains a binary state:
- $m_i = 1$: Cell is occupied by an obstacle.
- $m_i = 0$: Cell is free drivable/navigable space.

### Core Assumptions:
1. **Static Environment:** The physical obstacle map does not change during the update step (dynamic obstacles are filtered out beforehand).
2. **Cell Independence:** The state of cell $m_i$ is conditionally independent of all other cells $m_j$ given the sensor measurements $z_{1:t}$ and vehicle poses $x_{1:t}$:
   $$P(\mathbf{m} \mid z_{1:t}, x_{1:t}) = \prod_{i=1}^N P(m_i \mid z_{1:t}, x_{1:t})$$
3. **Known Vehicle Pose:** The vehicle trajectory $x_{1:t}$ is known with high certainty (from GNSS/IMU/localization).

---

## 2. Derivation of the Recursive Binary Bayes Filter

To update the probability of a single cell $m_i$ being occupied given history of measurements $z_{1:t}$ and poses $x_{1:t}$ (omitting $x$ for brevity):

### Step 1: Apply Bayes' Rule to the Measurement History
Separating the most recent measurement $z_t$ from historical measurements $z_{1:t-1}$:

$$P(m_i \mid z_{1:t}) = \frac{P(z_t \mid m_i, z_{1:t-1}) \cdot P(m_i \mid z_{1:t-1})}{P(z_t \mid z_{1:t-1})}$$

Using the **Markov assumption** ($z_t$ depends only on the current state $m_i$, not prior measurements):
$$P(z_t \mid m_i, z_{1:t-1}) = P(z_t \mid m_i)$$

Therefore:
$$P(m_i \mid z_{1:t}) = \frac{P(z_t \mid m_i) \cdot P(m_i \mid z_{1:t-1})}{P(z_t \mid z_{1:t-1})}$$

### Step 2: Apply Bayes' Rule to the Measurement Model $P(z_t \mid m_i)$
$$P(z_t \mid m_i) = \frac{P(m_i \mid z_t) \cdot P(z_t)}{P(m_i)}$$

Substituting back:
$$P(m_i \mid z_{1:t}) = \frac{P(m_i \mid z_t) \cdot P(z_t) \cdot P(m_i \mid z_{1:t-1})}{P(m_i) \cdot P(z_t \mid z_{1:t-1})}$$

### Step 3: Compute the Probability of the Opposite State $P(\neg m_i \mid z_{1:t})$
Similarly, for the cell being free ($\neg m_i$):
$$P(\neg m_i \mid z_{1:t}) = \frac{P(\neg m_i \mid z_t) \cdot P(z_t) \cdot P(\neg m_i \mid z_{1:t-1})}{P(\neg m_i) \cdot P(z_t \mid z_{1:t-1})}$$

### Step 4: Divide the Odds Ratios
Dividing $P(m_i \mid z_{1:t})$ by $P(\neg m_i \mid z_{1:t})$, the difficult marginal probability $P(z_t \mid z_{1:t-1})$ and measurement probability $P(z_t)$ completely cancel out!

$$\frac{P(m_i \mid z_{1:t})}{P(\neg m_i \mid z_{1:t})} = \frac{P(m_i \mid z_t)}{1 - P(m_i \mid z_t)} \cdot \frac{P(m_i \mid z_{1:t-1})}{1 - P(m_i \mid z_{1:t-1})} \cdot \frac{1 - P(m_i)}{P(m_i)}$$

---

## 3. The Log-Odds Representation

Taking the natural logarithm of both sides transforms the multiplications into simple, fast **additions**:

$$\log \left(\frac{P(m_i \mid z_{1:t})}{1 - P(m_i \mid z_{1:t})}\right) = \log\left(\frac{P(m_i \mid z_t)}{1 - P(m_i \mid z_t)}\right) + \log\left(\frac{P(m_i \mid z_{1:t-1})}{1 - P(m_i \mid z_{1:t-1})}\right) - \log\left(\frac{P(m_i)}{1 - P(m_i)}\right)$$

Defining the **Log-Odds** $l(m_i) = \operatorname{logit}(P(m_i)) = \log\left(\frac{P(m_i)}{1 - P(m_i)}\right)$:

$$l_t(m_i) = l_{t-1}(m_i) + \operatorname{inverse\_sensor\_model}(m_i, z_t, x_t) - l_0$$

Where:
- $l_t(m_i)$: Posterior log-odds of cell $i$ at time step $t$.
- $l_{t-1}(m_i)$: Prior log-odds accumulated up to time $t-1$.
- $\operatorname{inverse\_sensor\_model}(m_i, z_t, x_t) = \operatorname{logit}(P(m_i \mid z_t, x_t))$
- $l_0 = \operatorname{logit}(P_{\text{prior}}(m_i))$ (for uniform uninformative prior $P_0 = 0.5 \implies l_0 = 0$).

### Probability Recovery (Logistic Sigmoid):
$$P(m_i) = \frac{1}{1 + \exp(-l_t(m_i))}$$

---

## 4. Why Log-Odds is Superior in Practice

1. **Numerical Stability:** Avoids floating-point underflow when multiplying microscopic probabilities (e.g. $10^{-9}$).
2. **Speed:** Additive updates are orders of magnitude faster on CPU and GPU.
3. **Clamping & Dynamic Responsiveness:** To allow the map to clear an obstacle that has moved or handle transient noise, log-odds are clamped:
   $$l_{\min} \le l_t(m_i) \le l_{\max}$$
   Typical bounds: $l_{\min} = -4.0$ ($P \approx 0.018$) and $l_{\max} = +4.0$ ($P \approx 0.982$).
