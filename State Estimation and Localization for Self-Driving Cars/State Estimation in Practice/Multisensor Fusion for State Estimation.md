# Multisensor Fusion for State Estimation

## Why use GNSS with IMU & LIDAR?

* Error dynamics are completely different
and uncorrelated
* IMU provides 'smoothing' of GNSS, fill-in
during outages due to jamming or
maneuvering
* Wheel odometry is also possible (if only
2D position orientation is desired)
* GNSS provides absolute positioning
information to mitigate IMU drift
* LIDAR provides accurate local positioning within known maps

## Tightly vs. Loosely Coupled

![EKF coupling](./EKF%20Coupling.jpg)

## Extended Kalman Filter | IMU + GSS + LIDAR

![Sensor System](./Sensor%20System.jpg)

### Some preliminaries

**Vehicle state** consists of position, velocity and parametrization of orientation using a unit quaternion:
$$\bm{x}_k=\begin{bmatrix} \bm{p}_k \\ \bm{v}_k \\ \bm{q}_k \end{bmatrix} \in R^{10} $$

**Motion model input** will consist of specific force and rotational rates from our IMU:
$$\bm{u}_k=\begin{bmatrix} \bm{f}_k \\ \bm{\omega}_k \end{bmatrix} \in R^{6} $$

#### Motion model

* *Position* $\quad\bm{p}_k=\bm{p}_{k-1}+\Delta t \bm{p}_{v-1} + \frac{\Delta t^2}{2}\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$
* *Velocity* $\quad\bm{v}_k=\bm{v}_{k-1}+\Delta t\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$
* *Orientation* $\quad\bm{q}_k=\bm{q}_{k-1}\otimes \bm{q}\left(\bm{\omega}_{k-1}\Delta t\right) = \bm{\Omega}\left(\bm{q}(\bm{\omega}_{k-1}\Delta t)\right) \bm{q}_{k_-1}$

Where:
$$\bm{C}_{ns}=\bm{C}_{ns}(\bm{q}_{k-1})$$
$$\Omega\left(\begin{bmatrix} q_w\\ \bm{q}_v \end{bmatrix}\right) =  q_w\bm{1}+\begin{bmatrix}0&&-\bm{q}_v^T\\\bm{q}_v&&-[\bm{q}_v]_\times\end{bmatrix}$$
$$\bm{q}(\bm{\theta})=\begin{bmatrix} \cos{\frac{|\theta|}{2}}\\\frac{\bm{\theta}}{|\bm{\theta}|} \sin{\frac{|\theta|}{2}} \end{bmatrix}$$

#### Linearized Motion Model

Error state:
$$\delta\bm{x}_k=\begin{bmatrix}\delta{\bm{p}}_k\\
\delta{\bm{v}}_k\\\delta{\bm{\phi}}_k\end{bmatrix}\in R^9$$
> $\delta{\bm{\phi}}_k$ is a $3\times1$ rotation error

Error Dynamics:
$$\delta\bm{x}_k = \bm{F}_{k-1}\delta\bm{x}_{k-1}+\bm{L}_{k-1}\bm{n}_{k-1}$$
> $\bm{n}_{k-1}$ measurement noise

Where:
$$\bm{F}_{k-1}=\begin{bmatrix}
\bm{1} &\bm{1}\Delta t&0\\
0&\bm{1}&-\left[\bm{C}_{ns}\bm{f}_{k-1}\right]_{\times}\Delta t \\
0&0&\bm{1}
\end{bmatrix}$$
$$\bm{L}_{k-1} = \begin{bmatrix}0&0\\\bm{1}&0\\0&\bm{1}\end{bmatrix} \quad\quad \bm{n}_k \sim\mathcal{N}\left(\bm{0},\bm{Q}_k\right) \sim \mathcal{N}\left(\bm{0} ,\Delta t^2\begin{bmatrix}\sigma_{acc}^2&\\&\sigma_{gyro}^2 \end{bmatrix}\right)$$
> $\bm{1}$ is the $3\times3$ identity matrix

#### Measurement Model | GNSS

$$\bm{y}_k=\bm{h}(\bm{x}_k)+\bm{v}_k\\=\bm{H}_k\bm{x}_k+\bm{v}_k = \begin{bmatrix}\bm{1} & \bm{0} & \bm{0}\end{bmatrix}\bm{x}_k + \bm{v}_k\\=\bm{p}_k+\bm{v}_k$$
$$\bm{v}_k \sim \mathcal{N}(\bm{0},\bm{R}_{\text{GNSS}})$$

#### Measurement Model | Lidar

$$\bm{y}_k=\bm{h}(\bm{x}_k)+\bm{v}_k\\=\bm{H}_k\bm{x}_k+\bm{v}_k = \begin{bmatrix}\bm{1} & \bm{0} & \bm{0}\end{bmatrix}\bm{x}_k + \bm{v}_k\\=\bm{p}_k+\bm{v}_k$$
$$\bm{v}_k \sim \mathcal{N}(\bm{0},\bm{R}_{\text{LIDAR}})$$

## EKF | IMU + GNSS + LIDAR

Loop:
1. Update state with IMU INPUTS:
$$\check{\bm{x}}_k=\begin{bmatrix}
\check{\bm{p}}_k\\\check{\bm{v}}_k\\\check{\bm{q}}_k
\end{bmatrix}$$
$$\check{\bm{p}}_k=\bm{p}_{k-1}+\Delta t \bm{p}_{v-1} + \frac{\Delta t^2}{2}\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$$
$$\check{\bm{v}}_k=\bm{v}_{k-1}+\Delta t\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$$
$$\check{\bm{q}}_k=\bm{\Omega}\left(\bm{q}(\bm{\omega}_{k-1}\Delta t)\right) \bm{q}_{k_-1}$$
> Where:$$\bm{p}_{k-1} \text{,}\bm{v}_{k-1} \text{,}\bm{q}_{k-1} \text{,}$$can be either be corrected or uncorrected depending on whether or not there was a
GNSS/LIDAR measurement at time step $k- 1$

2. Propagate uncertainty
$$\check{P}_k=\bm{F}_{k-1}\bm{P}_{k-1}\bm{F}_{k-1}^T + \bm{L}_{k-1}\bm{Q}_{k-1}\bm{L}_{k-1}^T $$ 
>Where $\bm{p}_{k-1}$ can be either $\check{P}_k$ or $\hat{P}_k$

3. If **GNSS** or **LIDAR** position available
   1. Compute Kalman Gain
   $$\bm{K}_k=\bm{\check{P}}_k\bm{H}_k^T\left(\bm{H}_k\bm{\check{P}}_k\bm{H}_k^T \right)^{-1}$$
> $\bm{R}$ one of $\bm{R}_{\text{GNSS}}$ or $\bm{R}_{\text{Lidar}}$
  2. Compute error state $$\delta\bm{x}_k=\bm{K}_k\left(\bm{y}_k-\check{\bm{p}}_k\right)$$
  3. Correct predicted state
  $$\hat{\bm{p}}_k=\check{\bm{p}}_k + \delta\bm{p}_k$$
  $$\hat{\bm{v}}_k=\check{\bm{v}}_k + \delta\bm{v}_k$$
  $$\hat{\bm{q}}_k= \bm{q}(\delta\bm{\phi})\otimes\check{\bm{q}}_k \leftarrow \text{global orientation error}$$
  4. Computed corrected covariance
  $$\hat{\bm{P}}_k=\left(\bm{1} - \bm{K}_k\bm{H}_k \right)\check{\bm{P}}_k$$

## Additional Resources

* Read Sections 5.1-5.4 and Section 6.1 of a technical report by
[Joan Solà, Quaternion kinematics for the error-state Kalman filter, 2017](https://arxiv.org/pdf/1711.02508.pdf).

* Read a
[research paper](https://www.sciencedirect.com/science/article/pii/S2405896317323674)
 by Jay Farrell and Paul Roysdon that provides a tutorial for autonomous driving state estimation.

* Read a
[Medium article](https://medium.com/@wilburdes/sensor-fusion-algorithms-for-autonomous-driving-part-1-the-kalman-filter-and-extended-kalman-a4eab8a833dd)
 about sensor fusion algorithms for autonomous driving (Kalman filters and extended Kalman filters).

* Review an
[article](https://www.technologyreview.com/s/608321/this-image-is-why-self-driving-cars-come-loaded-with-many-types-of-sensors/)
 from MIT Technology Review that explains the need for sensor fusion to enable robust autonomous driving
 