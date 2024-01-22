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
2. Associate each point in $\bm{P}_{S'}$ (blue). with the nearest point in $\bm{P}_S$ (red)
![Ipc Algo](./IPC%20Algo.jpg)
3. Solve for the optimal transformation $\left\{  \hat{\bm{C}}_{S'S},\hat{\bm{r}}_S^{S'S}\right\}$
4. Repeat until convergence

#### Solving for the Optimal Transformation

Using Least-Squares
$$\left\{  \hat{\bm{C}}_{S'S},\hat{\bm{r}}_S^{S'S}\right\}=\argmin_{\left\{  {\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right\}} \mathcal{L}_{LS}\left({\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right)$$
$$\mathcal{L}_{LS}\left({\bm{C}}_{S'S},{\bm{r}}_S^{S'S}\right)=\sum_{j=1}^{n} \left\lVert \bm{C}_{S'S}\left(\bm{p}_S^{(j)}-\bm{r}_S^{S'S}\right) - \bm{p}_{S'}^{(j)} \right\rVert_2^2$$
> **Careful:** Rotations need special treatment
because they don't behave like vectors!

## Solving for the Optimal Transformation (Algorithm)

1. Compute the centroids of each point cloud
$$\bm{\mu}_S=\frac{1}{n}\sum_{j=1}^n\bm{p}_S^{(j)} \quad\quad\quad \bm{\mu}_{S'}=\frac{1}{n}\sum_{j=1}^n\bm{p}_{S'}^{(j)}$$
2. Compute a matrix capturing the spread of the two point clouds
$$\bm{W}_{S'S}=\frac{1}{n}\sum_{j=1}^{n}\left(\bm{p}_S^{(j)}-\bm{\mu}\right)\left(\bm{p}_{S'}^{(j)}-\bm{\mu}_{S'} \right)^T$$
3. Use the singular value decomposition of the matrix to get the optimal rotation $$\bm{US}\bm{V}^T=\bm{W}_{S'S}\quad\quad\quad\hat{\bm{C}}_{S'S}=\bm{U}\begin{bmatrix}1&&0&&0\\0&&1&&0\\0&&0&&\det{\bm{U}}\det{\bm{V}} \end{bmatrix}\bm{V}^T$$
    * Where $\bm{U}$ and $\bm{V}$ are rotation matrix, $\bm{U}^T\bm{U} = \bm{V}^T\bm{V} = 1$
    * $\bm{S}$ is the singular values (scaling) matrix
    * $\det{\bm{U}}\det{\bm{V}}$ This term ensures that we get a proper rotation without any reflection
4. Use the optimal rotation to get the optimal translation by *aligning the centroids*
$$\hat{\bm{r}}_S^{S'S}=\bm{\bm{\mu}}_S-\hat{\bm{C}}_{S'S}^T\bm{\mu}_{S'}$$

## Estimating Uncertainty

We can obtain an estimate of the covariance matrix of the ICP solution using this formula:
$$\text{cov}(\hat{\bm{x}})\simeq\left[\left(\frac{\partial^2\mathcal{L}}{\partial\bm{x}^2}\right)^{-1}\frac{\partial^2\mathcal{L}}{\partial\bm{z}\partial\bm{x}}\text{cov}(\bm{z})\frac{\partial^2\mathcal{L}}{\partial\bm{z}\partial\bm{x}}^T\left(\frac{\partial^2\mathcal{L}}{\partial\bm{x}^2}\right)^{-1}\right]_{\bm{x}=\hat{\bm{x}}}$$

Where

* $\text{cov}(\hat{\bm{x}})$ is the covariance of estimated motion parameters
* $\text{cov}(\bm{x})$ is the covariance of all measurements (both point clouds)

## Variants

**Point-to-point** ICP minimizes the *Euclidean distance* between each point in $P_{S'}$ (blue) and the *nearest point* in $P_S$(red)

![point to point](./Point%20to%20Point.jpg)

**Point-to-plane** ICP minimizes the *perpendicular
distance* between each point in $P_{S'}$ (blue) and the *nearest plane* in $P_S$ (red)

![point to plane](./Point%20to%20Plane.jpg)

* This tends to work well in structured
environments like cities

## Robust Loss Functions

Error term
$$\bm{e}^{(j)}=\bm{C}_{S'S}\left(\bm{p}_S^{(j)}-\bm{r}_S^{S'S}\right)-\bm{p}_{S'}^{(j)}$$

Error term
$$\mathcal{L}=\sum_{j=1}^{n}{\bm{e}^{(j)}}^T\bm{e}^{(j)}$$

![Error functions](./Error%20functions.jpg)

## Additional material

* Read Chapter 8, Section 1.3 of [Timothy D. Barfoot, State Estimation for Robotics (2016)](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf).

* Read the Wikipedia articles on [point set registration](https://en.wikipedia.org/wiki/Point_set_registration) and [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point).

* Examine a method to produce an accurate [closed-form estimate of  ICP's covariance](https://ieeexplore.ieee.org/document/4209579) from Andrea Censi of the University of Rome "La Sapienza" (now at ETH Zurich).

* Read a research paper on [LIDAR and Inertial Fusion for Pose Estimation by Non-linear Optimization](https://arxiv.org/abs/1710.07104), available for free on arXiv.

*Review the original papers by [Yang Chen and Gerard Medioni (1991)](https://ieeexplore.ieee.org/document/132043), and [Paul Besl and Neil McKay (1992)](https://ieeexplore.ieee.org/document/121791)
, that first described (variations of the) iterative closest point (ICP) algorithm.
