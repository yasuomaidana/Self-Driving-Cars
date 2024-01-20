# Pose Estimation from LIDAR Data

What motion of the car best aligns the two point clouds?
![State Estimation](./State%20Estimation.png)
![Estmiation Problem](./Estimation%20problem.jpg)

## The Iterative Closest Point (ICP) Algorithm

**Intuition:** When the optimal motion is found, corresponding points will be closer to each other than to other points

**Heuristic:** For each point, the best candidate for a
corresponding point is the point that is closest to it right now

### ICP Algorithm

1. Get an initial guess for the transformation $\left\{  \check{\bm{C}}_{S'S},\check{\bm{r}}_S^{S'S}\right\}$
2. Associate each point in $\bm{P}_{s'}$ (blue). with the nearest point in $\bm{P}_s$ (red)
![Ipc Algo](./IPC%20Algo.jpg)
3. Solve for the optimal transformation $\left\{  \hat{\bm{C}}_{S'S},\hat{\bm{r}}_S^{S'S}\right\}$
4. Repeat until convergence

#### Solving for the Optimal Transformation

Using Least-Squares
$$\left\{  \hat{\bm{C}}_{S'S},\hat{\bm{r}}_S^{S'S}\right\}=\argmin_{\left\{  {\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right\}} \mathcal{L}_{LS}\left({\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right)$$
$$\mathcal{L}_{LS}\left({\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right)=\sum_{j=1}^{n} \left\lVert \bm{C}_{S'S}\left(\bm{p}_s^{(j)}-\bm{r}_S^{S'S}\right) - \bm{p}_{S'}^{(j)} \right\rVert_2^2$$
> **Careful:** Rotations need special treatment
because they don't behave like vectors!

## Solving for the Optimal Transformation
