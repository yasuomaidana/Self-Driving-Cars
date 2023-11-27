# Linear Recursive Estimator

1. Suppose we have an optimal estimate, $\hat{\bm{x}}_{k-1}$, of our unknown parameters at
time $k—1$
2. Then we obtain a new measurement at time $k:\bm{y}_k=\bm{H}_k\bm{x}+\bm{v}_k$

> **Goal**: compute $\hat{\bm{x}}_k$ as a function of $\bm{y}_k$ and $\hat{\bm{x}}_{k-1}$

We can use a *linear recursive update*:
$$\bm{\hat{x}}_k = \bm{\hat{x}}_{k-1} + \bm{K}_k\left(\bm{y}_k - \bm{H}_k\bm{\hat{x}}_{k-1}) \right)$$

We update our new state as a linear combination of the previous best guess and the current measurement residual (or error), weighted by a gain matrix $\bm{K}_k$

## Recursive Least Squares

But what is the gain matrix $\bm{K}_k$?

We can compute it by minimizing a similar least squares criterion, but this time
we'll use a probabilistic formulation.

We wish to minimize the **expected value of the sum of squared errors** of our current
estimate at time step $k$:
$$\mathcal{L}_{RLS} = \mathbb{E}[(x_k-\hat{x}_k)^2]=\sigma_k^2$$

If we have $n$ unknown parameters at time step k ,we generalize this to
$$\mathcal{L}_{RLS} = \mathbb{E}[(x_{1k}-\hat{x}_{1k})^2+\dots+(x_{nk}-\hat{x}_{nk})^2]=Trace(\bm{P}_k)$$

Where $\bm{P}_k$ is the estimator **covariance**

Using our linear recursive formulation, we can express covariance as a function of $\bm{K}_k$
$$\bm{P}_k = (1-\bm{K}_k\bm{H}_k)\bm{P}_{k-1}(1-\bm{K}_k\bm{H}_k)^T+\bm{K}_k\bm{R}_k\bm{K}_k^T$$

We can show (through matrix calculus) that this is minimized when:
$$\bm{K}_k = \bm{P}_{k-1}\bm{H}_k^T\left(\bm{H}_k\bm{P}_{k-1}\bm{H}_k^T+\bm{R}_k\right)^{-1}$$

With this expression, we can also simplify our expression for $\bm{P}_k$:
$$\bm{P}_k = \bm{P}_{k-1}-\bm{K}_k\bm{H}_k\bm{P}_{k-1}\\=(1-\bm{K}_k\bm{H}_k)\bm{P}_{k-1}$$

> Our covariance *'shrinks'* with each measurement

## Algorithm

1. Initialize the estimator:
$$\bm{\hat{x}}_0=\mathbb{E}[\bm{x}]$$
$$\bm{P}_0=\mathbb{E}[(\bm{x}-\bm{\hat{x}}_0)(\bm{x}-\bm{\hat{x}}_0)^T]$$
2. Set up the measurement model, defining the Jacobian and the measurement
covariance matrix:
$$\bm{y}_k=\bm{H}_k\bm{x}+\bm{v}_k$$
3. Update the estimate of $\bm{\hat{x}}_k$ and the covariance $\bm{\hat{P}}_k$ using:
$$\bm{K}_k = \bm{P}_{k-1}\bm{H}_k^T\left(\bm{H}_k\bm{P}_{k-1}\bm{H}_k^T+\bm{R}_k\right)^{-1}$$
$$\bm{\hat{x}}_k = \bm{\hat{x}}_{k-1} + \bm{K}_k\left(\bm{y}_k - \bm{H}_k\bm{\hat{x}}_{k-1}) \right)$$
$$\bm{P}_k =(1-\bm{K}_k\bm{H}_k)\bm{P}_{k-1}$$
