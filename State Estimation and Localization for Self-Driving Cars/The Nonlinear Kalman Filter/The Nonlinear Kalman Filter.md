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
$$\bm{y}_k=\bm{h}_{k}\left(\bm{x}_{k},\bm{v}_{k}\right)\approx \bm{h}_{k} \left(\check{\bm{x}}_{k},0\right)+\left. \frac{\partial \bm{h}_{k}}{\partial\bm{x}_{k-1}} \right|_{\check{\bm{x}}_{k},0} \left(\bm{x}_{k}-\check{\bm{x}}_{k} \right)+
\left. \frac{\partial \bm{h}_{k}}{\partial\bm{v}_{k}} \right|_{\hat{\bm{x}}_{k},0} \bm{v}_{k}
$$
$$\left. \frac{\partial \bm{h}_{k}}{\partial\bm{x}_{k-1}} \right|_{\check{\bm{x}}_{k},0} =\bm{H}_{k} \quad\quad
\left. \frac{\partial \bm{h}_{k}}{\partial\bm{v}_{k}} \right|_{\hat{\bm{x}}_{k},0} =\bm{M}_{k} $$

We now have a linear system in state-space! The matrices $\bm{F}_{k-1}$, $\bm{L}_{k-1}$, $\bm{H}_{k}$, and $\bm{M}_{k}$ are
called the Jacobian matrices of the system.

Intuitively, the Jacobian matrix tells you how fast each output of the function is changing along each input dimension

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
   $$\bm{K}_k= \bm{\check{P}}_{k}\bm{H}_k^T(\bm{H}_k\bm{\check{P}}_{k}\bm{H}_k^T+\bm{R})^-1$$
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

## Additional Resources

* To learn more about nonlinear Kalman filtering, check out [this article](https://www.embedded.com/using-nonlinear-kalman-filtering-to-estimate-signals/) by Dan Simon (available for free).

* A detailed explanation of linearization and how it relates to the EKF can be found in Chapter 13,  Sections 1 and 2 of Dan Simon, Optimal State Estimation (2006)

* Review an important paper by Stergios Roumeliotis et al. on the use of the error-state Kalman filter for mobile robot localization. This paper deals with the important case of aided localization.

* Read Section 5 of a technical report by Joan Solà, [Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/pdf/1711.02508.pdf), 2017
