# The Nonlinear Kalman Filter

## The Extended Kalman Filter

![kalman filter recap](./kalman%20filter%20recap.jpg)

## Linearizing a nonlinear system

For the Extended Kalman Filter (EKF), we choose the operating point to be our most recent state estimate, our known input, and zero noise

Linearized motion model:
$$\bm{x}_k=\bm{f}_{k-1}\left(\bm{x}_{k-1},\bm{u}_{k-1},\bm{w}_{k-1}\right)\approx \bm{f}_{k-1} \left(\hat{\bm{x}}_{k-1},\bm{u}_{k-1},0\right)+\left. \frac{\partial \bm{f}_{k-1}}{\partial\bm{x}_{k-1}} \right|_{\hat{\bm{x}}_{k-1},\bm{u}_{k-1},0} \left(\bm{x}_{k-1}-\hat{\bm{x}}_{k-1} \right)+
\left. \frac{\partial \bm{f}_{k-1}}{\partial\bm{w}_{k-1}} \right|_{\hat{\bm{x}}_{k-1},\bm{u}_{k-1},0} \bm{w}_{k-1}
$$
$$\left. \frac{\partial \bm{f}_{k-1}}{\partial\bm{x}_{k-1}} \right|_{\hat{\bm{x}}_{k-1},\bm{u}_{k-1},0} =\bm{F}_{k-1} \quad\quad
\left. \frac{\partial \bm{f}_{k-1}}{\partial\bm{w}_{k-1}} \right|_{\hat{\bm{x}}_{k-1},\bm{u}_{k-1},0} =\bm{L}_{k-1} $$

Linearized measurement model
$$\bm{y}_k=\bm{h}_{k}\left(\bm{x}_{k},\bm{v}_{k}\right)\approx \bm{h}_{k} \left(\check{\bm{x}}_{k},0\right)+\left. \frac{\partial \bm{h}_{k}}{\partial\bm{x}_{k}} \right|_{\check{\bm{x}}_{k},0} \left(\bm{x}_{k}-\check{\bm{x}}_{k} \right)+
\left. \frac{\partial \bm{h}_{k}}{\partial\bm{v}_{k}} \right|_{\check{\bm{x}}_{k},0} \bm{v}_{k}
$$
$$\left. \frac{\partial \bm{h}_{k}}{\partial\bm{x}_{k}} \right|_{\check{\bm{x}}_{k},0} =\bm{H}_{k} \quad\quad
\left. \frac{\partial \bm{h}_{k}}{\partial\bm{v}_{k}} \right|_{\check{\bm{x}}_{k},0} =\bm{M}_{k} $$

We now have a linear system in state-space! The matrices $\bm{F}_{k-1}$, $\bm{L}_{k-1}$, $\bm{H}_{k}$, and $\bm{M}_{k}$ are
called the Jacobian matrices of the system.

Intuitively, the Jacobian matrix tells you how fast each output of the function is changing along each input dimension.

---

### Understanding the Noise Jacobians ($\bm{L}$ and $\bm{M}$)

In standard linear Kalman filtering, noise is assumed to be **purely additive** ($\bm{w}$ added directly to state derivatives, $\bm{v}$ added directly to measurements). In real-world robotic and self-driving systems, noise often enters **nonlinearly** through control inputs (actuators) or coordinate transformations.

* **$\bm{L}_{k-1}$ (Process Noise Jacobian):** Maps process / input noise $\bm{w}_{k-1}$ into the state space. It appears in the covariance prediction step:
  $$\check{\bm{P}}_k = \bm{F}_{k-1} \hat{\bm{P}}_{k-1} \bm{F}_{k-1}^T + \bm{L}_{k-1} \bm{Q}_{k-1} \bm{L}_{k-1}^T$$
* **$\bm{M}_k$ (Measurement Noise Jacobian):** Maps raw measurement noise $\bm{v}_k$ into the measurement space. It appears in the innovation covariance / Kalman gain:
  $$\bm{S}_k = \bm{H}_k \check{\bm{P}}_k \bm{H}_k^T + \bm{M}_k \bm{R}_k \bm{M}_k^T$$

---

### Example 1: Derivation of $\bm{L}_{k-1}$ (Process Noise Jacobian)

#### Case A: Additive Process Noise
If noise enters additively as $\bm{x}_k = \bm{f}_{k-1}(\bm{x}_{k-1}, \bm{u}_{k-1}) + \bm{w}_{k-1}$, where $\bm{w}_{k-1} \in \mathbb{R}^n$:
$$\bm{L}_{k-1} = \left. \frac{\partial \bm{f}_{k-1}}{\partial \bm{w}_{k-1}} \right|_{\hat{\bm{x}}_{k-1}, \bm{u}_{k-1}, \bm{0}} = \bm{I}_{n \times n}$$

#### Case B: Non-Additive Actuator/Control Noise (2D Vehicle Kinematics)
Consider a vehicle state with 2D position and heading:
$$\bm{x}_{k-1} = \begin{bmatrix} p_x \\ p_y \\ \theta \end{bmatrix}_{k-1}$$

The control inputs are linear velocity $v_{k-1}$ and yaw rate $\omega_{k-1}$, corrupted by input noise $\bm{w}_{k-1} = \begin{bmatrix} w_v \\ w_\omega \end{bmatrix}_{k-1} \sim \mathcal{N}(\bm{0}, \bm{Q}_{k-1})$. Over a sampling time $\Delta t$, the nonlinear motion equations are:

$$\bm{f}_{k-1}(\bm{x}_{k-1}, \bm{u}_{k-1}, \bm{w}_{k-1}) = \begin{bmatrix} f_1 \\ f_2 \\ f_3 \end{bmatrix} = \begin{bmatrix} p_{x, k-1} + (v_{k-1} + w_{v, k-1}) \cos(\theta_{k-1}) \Delta t \\ p_{y, k-1} + (v_{k-1} + w_{v, k-1}) \sin(\theta_{k-1}) \Delta t \\ \theta_{k-1} + (\omega_{k-1} + w_{\omega, k-1}) \Delta t \end{bmatrix}$$

To compute $\bm{L}_{k-1}$, take the partial derivatives with respect to each component of noise $\bm{w} = \begin{bmatrix} w_v & w_\omega \end{bmatrix}^T$:

$$\bm{L}_{k-1} = \left. \frac{\partial \bm{f}_{k-1}}{\partial \bm{w}_{k-1}} \right|_{\hat{\bm{x}}_{k-1}, \bm{u}_{k-1}, \bm{w}=\bm{0}} = \begin{bmatrix} \frac{\partial f_1}{\partial w_v} & \frac{\partial f_1}{\partial w_\omega} \\ \frac{\partial f_2}{\partial w_v} & \frac{\partial f_2}{\partial w_\omega} \\ \frac{\partial f_3}{\partial w_v} & \frac{\partial f_3}{\partial w_\omega} \end{bmatrix}_{\hat{\bm{x}}_{k-1}, \bm{0}}$$

Evaluating each term:
1. $\frac{\partial f_1}{\partial w_v} = \cos(\hat{\theta}_{k-1}) \Delta t$, $\quad \frac{\partial f_1}{\partial w_\omega} = 0$
2. $\frac{\partial f_2}{\partial w_v} = \sin(\hat{\theta}_{k-1}) \Delta t$, $\quad \frac{\partial f_2}{\partial w_\omega} = 0$
3. $\frac{\partial f_3}{\partial w_v} = 0$, $\quad \frac{\partial f_3}{\partial w_\omega} = \Delta t$

Thus:
$$\bm{L}_{k-1} = \begin{bmatrix} \cos(\hat{\theta}_{k-1}) \Delta t & 0 \\ \sin(\hat{\theta}_{k-1}) \Delta t & 0 \\ 0 & \Delta t \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

> **Key takeaway:** $\bm{L}_{k-1}$ maps the $2\text{D}$ actuator noise into the $3\text{D}$ state space, projecting velocity noise along the vehicle's heading direction $\hat{\theta}_{k-1}$.

---

### Example 2: Derivation of $\bm{M}_k$ (Measurement Noise Jacobian)

#### Case A: Additive Measurement Noise
If the measurement model is $\bm{y}_k = \bm{h}_k(\bm{x}_k) + \bm{v}_k$, where $\bm{v}_k \in \mathbb{R}^m$:
$$\bm{M}_k = \left. \frac{\partial \bm{h}_k}{\partial \bm{v}_k} \right|_{\check{\bm{x}}_k, \bm{0}} = \bm{I}_{m \times m}$$

#### Case B: Non-Additive Sensor Noise (Polar Sensor Converted to Cartesian Measurement)
Suppose a LiDAR or Radar measures range $r$ and bearing $\phi$ with sensor noise $\bm{v}_k = \begin{bmatrix} v_r \\ v_\phi \end{bmatrix} \sim \mathcal{N}(\bm{0}, \bm{R}_k)$, but the measurement pre-processor converts them into Cartesian coordinates $\bm{y}_k = \begin{bmatrix} y_x \\ y_y \end{bmatrix}$:

$$\bm{y}_k = \bm{h}_k(\bm{x}_k, \bm{v}_k) = \begin{bmatrix} h_1 \\ h_2 \end{bmatrix} = \begin{bmatrix} (r_k(\bm{x}_k) + v_r) \cos(\phi_k(\bm{x}_k) + v_\phi) \\ (r_k(\bm{x}_k) + v_r) \sin(\phi_k(\bm{x}_k) + v_\phi) \end{bmatrix}$$

To compute $\bm{M}_k$, take the partial derivatives with respect to $\bm{v}_k = \begin{bmatrix} v_r & v_\phi \end{bmatrix}^T$ at $\bm{v} = \bm{0}$:

$$\bm{M}_k = \left. \frac{\partial \bm{h}_k}{\partial \bm{v}_k} \right|_{\check{\bm{x}}_k, \bm{v}=\bm{0}} = \begin{bmatrix} \frac{\partial h_1}{\partial v_r} & \frac{\partial h_1}{\partial v_\phi} \\ \frac{\partial h_2}{\partial v_r} & \frac{\partial h_2}{\partial v_\phi} \end{bmatrix}_{\check{\bm{x}}_k, \bm{0}}$$

Evaluating each term using the chain rule and evaluating at $\bm{v} = \bm{0}$:
1. $\left. \frac{\partial h_1}{\partial v_r} \right|_{\bm{v}=\bm{0}} = \cos(\check{\phi}_k)$
2. $\left. \frac{\partial h_1}{\partial v_\phi} \right|_{\bm{v}=\bm{0}} = -(\check{r}_k + 0) \sin(\check{\phi}_k + 0) = -\check{r}_k \sin(\check{\phi}_k)$
3. $\left. \frac{\partial h_2}{\partial v_r} \right|_{\bm{v}=\bm{0}} = \sin(\check{\phi}_k)$
4. $\left. \frac{\partial h_2}{\partial v_\phi} \right|_{\bm{v}=\bm{0}} = (\check{r}_k + 0) \cos(\check{\phi}_k + 0) = \check{r}_k \cos(\check{\phi}_k)$

Thus:
$$\bm{M}_k = \begin{bmatrix} \cos(\check{\phi}_k) & -\check{r}_k \sin(\check{\phi}_k) \\ \sin(\check{\phi}_k) & \check{r}_k \cos(\check{\phi}_k) \end{bmatrix}$$

> **Key takeaway:** The angular noise $v_\phi$ produces Cartesian uncertainty that grows proportionally with the range distance $\check{r}_k$.

#### Case C: Scale-Factor & Calibration Multiplicative Noise
Suppose a sensor measuring position $p_k$ has an unknown scaling/calibration error $v_s$ and offset $v_b$ such that $\bm{v}_k = \begin{bmatrix} v_s \\ v_b \end{bmatrix}$:
$$y_k = h_k(p_k, \bm{v}_k) = p_k (1 + v_s) + v_b$$

The Jacobian $\bm{M}_k$ with respect to noise parameters $\bm{v}_k = \begin{bmatrix} v_s & v_b \end{bmatrix}^T$ is:
$$\bm{M}_k = \left. \begin{bmatrix} \frac{\partial y_k}{\partial v_s} & \frac{\partial y_k}{\partial v_b} \end{bmatrix} \right|_{\check{p}_k, \bm{0}} = \begin{bmatrix} \check{p}_k & 1 \end{bmatrix}$$

## Putting it all together

With our linearized models and Jacobians, we can now use the Kalman Filter equations!

![all together](./all%20together.jpg)

## Example

![example p1](./example%20p1.jpg)
![example p2](./example%202.jpg)
![example p3](./example%20p3.jpg)

Using the Extended Kalman Filter equations, what is our updated position?
$$\hat{p}_1$$
![result](./result.jpg)
![result 2](./result2.jpg)

## An Improvement EKF - The Error State Extended Kalman Filter

What's in a State?

We can think Of the vehicle state as composed Of two parts:

$$\bm{x}=\bm{\hat{x}}+\delta\bm{x}$$

* $\bm{x}$: True State
* $\bm{\hat{x}}$: Nominal State ("Large")
* $\delta\bm{x}$: Error State ("Small")

![whats in a state](./whats%20in%20a%20state.jpg)

* We can continuously update the nominal state by integrating the motion model
* Modelling errors and process noise accumulate into the error state

### The Error-State Extended Kalman Filter

The Error-State Extended Kalman Filter estimates the error state directly and uses it as a correction to the nominal state:

**Linearized motion model**
$$\bm{x}_k = \bm{f}_{k-1}(\bm{\hat{x}}_{k-1},\bm{u}_{k-1},\bm{0})+\bm{F}_{k-1}(\bm{x}_{k-1}-\bm{\hat{x}}_{k-1})+\bm{L}_{k-1}\\\downarrow\\
\bm{x}_k - \bm{f}_{k-1}(\bm{\hat{x}}_{k-1},\bm{u}_{k-1},\bm{0})=\bm{F}_{k-1}(\bm{x}_{k-1}-\bm{\hat{x}}_{k-1})+\bm{L}_{k-1}\\ \downarrow \\
\delta\bm{x}_k= \bm{x}_k - \bm{f}_{k-1}(\bm{\hat{x}}_{k-1},\bm{u}_{k-1},\bm{0}) \quad  \delta\bm{x}_{k-1}= \bm{x}_{k-1} - \bm{\hat{x}}_{k-1}$$

**Linearized measurement model**
$$\bm{y}_k= \bm{h}_k(\bm{\check{x},0})+\bm{H}_k(\bm{x}_k-\check{\bm{x}}_k)+\bm{M}_k\bm{v}_k\\\downarrow\\
\bm{x}_k-\check{\bm{x}}_k=\delta\bm{x}_k$$

> Where $\delta\bm{x}_k$ and $\delta\bm{x}_{k-1}$ are Error States

Loop:

1. Update nominal state with motion model:
$$\check{\bm{x}}_k=\bm{f}_{k-1}(\bm{x}_{k-1},\bm{u}_{k-1},\bm{0}) \quad\quad \bm{x}_{k-1} \text{ could be also } \bm{\check{x}}_{k-1} \text{ or } \bm{\hat{x}}_{k-1}$$
2. Propagate uncertainity
$$\bm{\check{P}}_k=\bm{F}_{k-1}\bm{P}_{k-1}\bm{F}_{k-1}^T+\bm{L}_{k-1}\bm{Q}_{k-1}\bm{L}_{k-1}^T \quad\quad \bm{P}_{k-1} \text{ could be also } \bm{\check{P}}_{k-1} \text{ or } \bm{\hat{P}}_{k-1}$$
3. If a measurement is available:
   1. Compute Kalman Gain
   $$\bm{K}_k= \bm{\check{P}}_{k}\bm{H}_k^T(\bm{H}_k\bm{\check{P}}_{k}\bm{H}_k^T+\bm{R})^{-1}$$
   2. Compute error state $$\delta\bm{\hat{x}}_k=\bm{K}_k(\bm{y}_k-\bm{h}_k(\check{\bm{x}}_k,0))$$
   3. Correct nominal state $$\bm{\hat{x}}_k=\check{\bm{x}}_k+\delta\bm{\hat{x}}_k$$
   4. Update state covariance $$\hat{\bm{P}}_k=(\bm{1}-\bm{K}_k\bm{H}_k)\check{\bm{P}}_k$$

## Why use the ES-EKF?

**Better performance compared to the vanilla EKF**</br>
The "small" error state is more amenable to linear filtering than the "large" nominal state, which can be integrated nonlinearly

**Easy to work with constrained quantities (e.g., rotations in 3D)** </br>
We can also break down the state using a generalized composition operator
$$\bm{x}=\hat{\bm{x}}\oplus\delta\bm{x}$$
>$\bm{x}$ true state</br>
>$\bm{\hat{x}}$ Nominal State (Overparamatrized, constrained)</br>
>$\delta\bm{x}$ Error State (Minimal parametrization, unsconstrained)

## Limitation of the EKF

The EKF works by linearizing the nonlinear motion and measurement models to update the mean and covariance of the state.

The difference between the linear approximation and the nonlinear function is called linearization error

In general, linearization error depends on:

1. How nonlinear the function is
2. How far away from the operating point the linear approximation is being used

![linearized error](./linearized%20error.jpg)
$$f(x)\approx f(a)+\left. \frac{\partial f(x)}{\partial x} \right|_{x=a} (x-a)$$

The EKF is prone to linearization error when:
1. The system dynamics are highly nonlinear
2. The sensor sampling time is slow relative how fast the system is evolving

This has two important consequences:
1. The estimated mean state can become very different from the true state
2. The estimated state covariance can fail to capture the true uncertainty in the state

> Linearization error can cause the estimator to be
overconfident in a wrong answer!

Computing Jacobian matrices for complicated nonlinear functions is also a common source of error in EKF implementations!

* Analytical differentiation is prone to human error
* Numerical differentiation can be slow and unstable
* Automatic differentiation (e.g., at compile time) can also behave unpredictably

## An Alternative to the EKF - The Unscented Kalman Filter

"It is easier to approximate a probability distribution than it is to approximate an arbitrary nonlinear function" <cite>— S.Julier, J. Uhlmann, and H. Durrant-Whyte (2000)</cite>
![unscented kalman filter](./unscented%20kf.jpg)

### Choosing sigmapoints

For an $N$-dimensional $\mathcal{N}(\bm{\mu_x},\bm{\Sigma_{xx}})$ PDF we need $2N+ 1$ sigma points:

1. Compute the Cholesky Decomposition of the covariance matrix
$$\bm{L}\bm{L}^T=\bm{\Sigma_{xx}} \quad (\bm{L} \text{ lower triangular})$$
2. Calculate the sigma-points
$$\bm{x}_0=\bm{\mu}_x\\
\bm{x}_i=\bm{\mu}_x+\sqrt{N+\kappa} \ col_i\bm{L} \quad i=1,\dots N\\
\bm{x}_{i+N}=\bm{\mu}_x-\sqrt{N+\kappa} \ col_i\bm{L} \quad i=1,\dots N
$$

> Where $\kappa=3-N$ for partial PDFs

#### Transforming and recombining

Next we pass each of our $2N+ 1$ sigma points through the nonlinear function $\bm{h}(\bm{x})$
$$\bm{y}_i=\bm{h}(\bm{x_i}) \quad i=0,\dots,2N$$

And finally compute the mean and covariance of the output PDF

$$\text{Mean: }\bm{\mu}_{y}=\sum_{i=0}^{2N}\alpha_i\bm{y}_i$$
$$\text{Covariance: }\bm{\Sigma}_{yy}=\sum_{i=0}^{2N}\alpha_i(\bm{y}_i-\bm{\mu}_{y})(\bm{y}_i-\bm{\mu}_{y})^T$$

$$\text{Weights: } \bm{\alpha}_i=\left\{
  \begin{aligned}
  \frac{\kappa}{N+\kappa} && i =0 \\
  \frac{1}{2}\frac{1}{N+\kappa} && \text{otherwise}
  \end{aligned}
\right.$$

## The Unscented Kalman Filter (UKF)
We can easily use the Unscented Transform in our Kalman Filtering framework with nonlinear models:

**Nonlinear motion model** $$\bm{x}_k=\bm{f}_{k-1}(\bm{x}_{k-1},\bm{u}_{k-1},\bm{w}_{k-1})\\\bm{w}_k\sim\mathcal{N}(\bm{0},\bm{Q}_k)$$
**Nonlinear measurement model** $$\bm{y}_k=\bm{h}_k(\bm{x}_k,\bm{v}_k)\\\bm{v}_k\sim\mathcal{N}(\bm{0},\bm{R}_k)$$

Instead of approximating the system equations by linearizing, we will calculate sigma points and use the Unscented Transform to approximate the PDFs directly!

### Prediction step
To propagate the state from time $(k— 1)$ to time $k$, apply the Unscented Transform using the current best guess for the mean and covariance

1. Compute sigma points
$$\bm{\hat{L}}_{k-1}\bm{\hat{L}}_{k-1}^T=\bm{\hat{P}}_{k-1}$$
$$\bm{\hat{x}}_{k-1}^{(0)}=\bm{\hat{x}}_{k-1}$$
$$\bm{\hat{x}}_{k-1}^{(i)}=\bm{\hat{x}}_{k-1}+\sqrt{N+\kappa} \ col_i\bm{\hat{L}}_{k-1} \quad i=1,\dots N$$
$$
\bm{\hat{x}}_{k-1}^{(i+N)}=\bm{\hat{x}}_{k-1}-\sqrt{N+\kappa} \ col_i\bm{\hat{L}}_{k-1} \quad i=1,\dots N
$$

2. Propagate sigma points
$$\bm{\check{x}}_k^{(i)}=\bm{f}_{k-1}(\bm{\hat{x}}_{k-1}^{(i)},\bm{u}_{k-1},\bm{0}) \quad i=0,\dots 2N$$

3. Compute predicted mean and covariance
$$\bm{\alpha}^{(i)}=\left\{
  \begin{aligned}
  \frac{\kappa}{N+\kappa} && i =0 \\
  \frac{1}{2}\frac{1}{N+\kappa} && \text{otherwise}
  \end{aligned}
\right.$$
$$\bm{\check{x}}_{k}=\sum_{i=0}^{2N}\alpha^{(i)}\bm{\check{x}}_k^{(i)}$$
$$\bm{\check{P}}_{k}=\sum_{i=0}^{2N}\alpha^{(i)}(\bm{\check{x}}_k^{(i)}-\bm{\check{x}}_{k})(\bm{\check{x}}_k^{(i)}-\bm{\check{x}}_{k})^T+\bm{Q}_{k-1}$$

> $\bm{Q}_{k-1}$ Additive process noise

### Correction step

To correct the state estimate using measurements at time k, use the nonlinear measurement model and the sigma points from the prediction step to predict the measurements

1. Predict measurements from propagated sigma points
$$\bm{\hat{y}}_k^{(i)}=\bm{h}_k(\bm{\check{x}}_k^{(i),\bm{0}})\quad i=0,\dots2N$$
2. Estimate mean and covariance of predicted measurements
$$\bm{\hat{y}}_{k}=\sum_{i=0}^{2N}\alpha^{(i)}\bm{\hat{y}}_k^{(i)}$$
$$\bm{P}_{y}=\sum_{i=0}^{2N}\alpha^{(i)}(\bm{\hat{y}}_k^{(i)}-\bm{\hat{y}}_{k})(\bm{\hat{y}}_k^{(i)}-\bm{\hat{y}}_{k})^T+\bm{R}_{k}$$

3. Compute cross-covariance and Kalman gain
$$\bm{P}_{xy}=\sum_{i=0}^{2N}\alpha^{(i)}(\bm{\check{x}}_k^{(i)}-\bm{\check{x}}_{k})(\bm{\hat{y}}_k^{(i)}-\bm{\hat{y}}_{k})^T$$
$$\bm{K}_k=\bm{P}_{xy}\bm{P}_{y}^{-1}$$

4. Compute corrected mean and covariance
$$\bm{\hat{x}}_k = \check{\bm{x}}_k + \bm{K}_k(\bm{y}_k - \bm{\hat{y}}_k)$$
$$\bm{\hat{P}}_k = \check{\bm{P}}_k - \bm{K}_k \bm{P}_y \bm{K}_k^T$$

> $\bm{R}_{k}$: Additive measurement noise covariance matrix, $\bm{P}_y = \bm{S}_k$: Innovation covariance matrix.

---

### Detailed Step-by-Step UKF Example: 2D Radar Landmark Tracking

To understand every computation in practice, let's walk through a concrete numerical example of the UKF measurement update step for a 2D position state observed by a polar radar sensor.

#### 1. Problem Formulation
* **State vector ($L = 2$):** $\bm{x} = \begin{bmatrix} x \\ y \end{bmatrix}$ (Position in Cartesian coordinates).
* **Prior state estimate & covariance:**
  $$\check{\bm{x}}_k = \begin{bmatrix} 10.0 \\ 0.0 \end{bmatrix} \text{ m}, \quad \check{\bm{P}}_k = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$
* **Nonlinear measurement model $\bm{h}(\bm{x})$ (Range & Bearing from sensor at origin):**
  $$\bm{y} = \bm{h}(\bm{x}) + \bm{v} = \begin{bmatrix} \sqrt{x^2 + y^2} \\ \operatorname{atan2}(y, x) \end{bmatrix} + \bm{v}$$
* **Measurement noise covariance $\bm{R}$:**
  $$\bm{R} = \begin{bmatrix} 0.01\text{ m}^2 & 0 \\ 0 & 0.001\text{ rad}^2 \end{bmatrix}$$
* **Actual sensor measurement received:**
  $$\bm{y}_k = \begin{bmatrix} 10.2\text{ m} \\ 0.05\text{ rad} \end{bmatrix}$$

---

#### 2. UT Parameters and Weights
For state dimension $L = 2$, choose standard parameter $\kappa = 3 - L = 1$, $\gamma = \sqrt{L + \kappa} = \sqrt{3} \approx 1.73205$.
Number of sigma points: $2L + 1 = 5$.

Weights:
* Center weight: $\alpha^{(0)} = \frac{\kappa}{L + \kappa} = \frac{1}{3} \approx 0.3333$
* Outer weights ($i = 1, \dots, 4$): $\alpha^{(i)} = \frac{1}{2(L + \kappa)} = \frac{1}{6} \approx 0.1667$

---

#### 3. Generating the Sigma Points
1. **Cholesky decomposition:** $\bm{L}\bm{L}^T = \check{\bm{P}}_k \implies \bm{L} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \end{bmatrix}$
2. **Sigma points $\boldsymbol{\mathcal{X}}^{(i)}$:**
   * $\boldsymbol{\mathcal{X}}^{(0)} = \check{\bm{x}}_k = \begin{bmatrix} 10.0 \\ 0.0 \end{bmatrix}$
   * $\boldsymbol{\mathcal{X}}^{(1)} = \check{\bm{x}}_k + \sqrt{3} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 11.732 \\ 0.0 \end{bmatrix}$
   * $\boldsymbol{\mathcal{X}}^{(2)} = \check{\bm{x}}_k + \sqrt{3} \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 10.0 \\ 3.464 \end{bmatrix}$
   * $\boldsymbol{\mathcal{X}}^{(3)} = \check{\bm{x}}_k - \sqrt{3} \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 8.268 \\ 0.0 \end{bmatrix}$
   * $\boldsymbol{\mathcal{X}}^{(4)} = \check{\bm{x}}_k - \sqrt{3} \begin{bmatrix} 0.0 \\ 2.0 \end{bmatrix} = \begin{bmatrix} 10.0 \\ -3.464 \end{bmatrix}$

---

#### 4. Propagating Sigma Points Through Sensor Model $\bm{h}(\cdot)$
Pass each sigma point through $\bm{y}^{(i)} = \begin{bmatrix} \sqrt{x_i^2 + y_i^2} \\ \operatorname{atan2}(y_i, x_i) \end{bmatrix}$:

1. $\boldsymbol{\mathcal{Y}}^{(0)} = \bm{h}\left(\begin{bmatrix} 10.0 \\ 0.0 \end{bmatrix}\right) = \begin{bmatrix} 10.000 \\ 0.0000 \end{bmatrix}$
2. $\boldsymbol{\mathcal{Y}}^{(1)} = \bm{h}\left(\begin{bmatrix} 11.732 \\ 0.0 \end{bmatrix}\right) = \begin{bmatrix} 11.732 \\ 0.0000 \end{bmatrix}$
3. $\boldsymbol{\mathcal{Y}}^{(2)} = \bm{h}\left(\begin{bmatrix} 10.0 \\ 3.464 \end{bmatrix}\right) = \begin{bmatrix} \sqrt{10^2 + 3.464^2} \\ \operatorname{atan2}(3.464, 10) \end{bmatrix} = \begin{bmatrix} 10.583 \\ 0.3335 \end{bmatrix}$
4. $\boldsymbol{\mathcal{Y}}^{(3)} = \bm{h}\left(\begin{bmatrix} 8.268 \\ 0.0 \end{bmatrix}\right) = \begin{bmatrix} 8.268 \\ 0.0000 \end{bmatrix}$
5. $\boldsymbol{\mathcal{Y}}^{(4)} = \bm{h}\left(\begin{bmatrix} 10.0 \\ -3.464 \end{bmatrix}\right) = \begin{bmatrix} 10.583 \\ -0.3335 \end{bmatrix}$

---

#### 5. Computing Predicted Measurement Mean and Innovation Covariance
* **Predicted measurement mean $\hat{\bm{y}}_k$:**
  $$\hat{\bm{y}}_k = \sum_{i=0}^4 \alpha^{(i)} \boldsymbol{\mathcal{Y}}^{(i)} = \frac{1}{3}\begin{bmatrix} 10.0 \\ 0 \end{bmatrix} + \frac{1}{6}\begin{bmatrix} 11.732 + 10.583 + 8.268 + 10.583 \\ 0 + 0.3335 + 0 - 0.3335 \end{bmatrix} = \begin{bmatrix} 10.194 \\ 0.0000 \end{bmatrix}$$
  *(Notice how the non-zero cross-axis variance naturally increases predicted range mean to $10.194\text{ m}$, capturing non-linear curvature without truncation!)*

* **Innovation Covariance $\bm{P}_y$ (or $\bm{S}_k$):**
  $$\bm{P}_y = \sum_{i=0}^4 \alpha^{(i)} (\boldsymbol{\mathcal{Y}}^{(i)} - \hat{\bm{y}}_k)(\boldsymbol{\mathcal{Y}}^{(i)} - \hat{\bm{y}}_k)^T + \bm{R}$$
  $$\bm{P}_y = \begin{bmatrix} 1.037 & 0.0 \\ 0.0 & 0.0381 \end{bmatrix}$$

* **Cross-Covariance $\bm{P}_{xy}$:**
  $$\bm{P}_{xy} = \sum_{i=0}^4 \alpha^{(i)} (\boldsymbol{\mathcal{X}}^{(i)} - \check{\bm{x}}_k)(\boldsymbol{\mathcal{Y}}^{(i)} - \hat{\bm{y}}_k)^T = \begin{bmatrix} 1.000 & 0.0 \\ 0.0 & 0.385 \end{bmatrix}$$

---

#### 6. Kalman Gain and State Update
* **Kalman Gain $\bm{K}_k$:**
  $$\bm{K}_k = \bm{P}_{xy} \bm{P}_y^{-1} = \begin{bmatrix} \frac{1.000}{1.037} & 0 \\ 0 & \frac{0.385}{0.0381} \end{bmatrix} \approx \begin{bmatrix} 0.964 & 0.0 \\ 0.0 & 10.10 \end{bmatrix}$$

* **State Correction $\hat{\bm{x}}_k$:**
  $$\bm{y}_k - \hat{\bm{y}}_k = \begin{bmatrix} 10.20 \\ 0.05 \end{bmatrix} - \begin{bmatrix} 10.194 \\ 0.00 \end{bmatrix} = \begin{bmatrix} 0.006 \\ 0.050 \end{bmatrix}$$
  $$\hat{\bm{x}}_k = \check{\bm{x}}_k + \bm{K}_k(\bm{y}_k - \hat{\bm{y}}_k) = \begin{bmatrix} 10.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.964(0.006) \\ 10.10(0.050) \end{bmatrix} = \begin{bmatrix} 10.006 \\ 0.505 \end{bmatrix}\text{ m}$$

* **Updated Covariance $\hat{\bm{P}}_k$:**
  $$\hat{\bm{P}}_k = \check{\bm{P}}_k - \bm{K}_k \bm{P}_y \bm{K}_k^T = \begin{bmatrix} 0.036 & 0.0 \\ 0.0 & 0.111 \end{bmatrix}$$

## Summary

| EKF                     | ES-EKF                                            | UKF                                               |                     |
| ----------------------- | ------------------------------------------------- | ------------------------------------------------- | ------------------- |
| **Operating Principle** | <center>Linearization </br> (Full State)</center> | <center>Linearization </br>(Error State)</center> | Unscented Transform |
| **Accuracy**            | Good                                              | Better                                            | Best                |
| **Jacobians**           | Required                                          | Required                                          | Not required        |
| **Speed**               | Slightly faster                                   | Slightly faster                                   | Slightly slower     |

## Additional Resources

* To learn more about nonlinear Kalman filtering, check out [this article](https://www.embedded.com/using-nonlinear-kalman-filtering-to-estimate-signals/) by Dan Simon (available for free).

* A detailed explanation of linearization and how it relates to the EKF can be found in Chapter 13,  Sections 1 and 2 of Dan Simon, Optimal State Estimation (2006)

* Review *[Circumventing dynamic modeling: evaluation of the error-state Kalman filter applied to mobile robot localization](https://ieeexplore.ieee.org/document/772597)* by Stergios Roumeliotis et al. on the use of the error-state Kalman filter for mobile robot localization. This paper deals with the important case of aided localization.

* Read Section 5 of a technical report by Joan Solà, [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/pdf/1711.02508.pdf), 2017

* A research paper on [The Unscented Kalman Filters for Nonlinear Estimation](https://groups.seas.harvard.edu/courses/cs281/papers/unscented.pdf) by the Oregon Graduate Institute of Science & Technology

* Read a tutorial on the [Unscented Kalman Filter](./TutorialUKF-e.pdf)
 by Gabriel A. Terejanu.

* Dig into the original article *["A New Extension of the Kalman Filter to Nonlinear Systems"](https://www.cs.unc.edu/~welch/kalman/media/pdf/Julier1997_SPIE_KF.pdf)* by Simon Julier and Jeffrey Uhlmann that introduced the unscented Kalman filter.
