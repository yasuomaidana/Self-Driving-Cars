# Solution

## Error-State EKF| IMU+GNSS+LIDAR

![sensor system][sensor system]

[sensor system]: ../State%20Estimation%20in%20Practice/Sensor%20System.jpg

## Preliminaries

### Vehicle State Initialization

The vehicle state at each time step consists of position, velocity, and
orientation (parameterized by a unit quaternion); the inputs to the motion
model are the IMU specific force and angular rate measurements:

$$ \bm{x}_k=\begin{bmatrix} \bm{p}_k \\ \bm{v}_k \\ \bm{q}_k \end{bmatrix} \in R^{10} \quad\quad
\bm{u}_k=\begin{bmatrix} \bm{f}_k \\ \bm{\omega}_k \end{bmatrix} \in R^{6} $$

```python
## Initialization

p_est = np.zeros([imu_f.data.shape[0], 3])  # position estimates
v_est = np.zeros([imu_f.data.shape[0], 3])  # velocity estimates
q_est = np.zeros([imu_f.data.shape[0], 4])  # orientation estimates as quaternions
p_cov = np.zeros([imu_f.data.shape[0], 9, 9])  # covariance matrices at each timestep

# Set initial values.
p_est[0] = gt.p[0]
v_est[0] = gt.v[0]
q_est[0] = Quaternion(euler=gt.r[0]).to_numpy()
p_cov[0] = np.zeros(9)  # covariance of estimate
gnss_i = 0
lidar_i = 0
```

## EKF
### Prediction Step Using IMU Data

The main filter loop operate by first predicting the next state (vehicle pose
and velocity), by integrating the high-rate IMU measurements:

* *Position* $\quad\bm{p}_k=\bm{p}_{k-1}+\Delta t \bm{p}_{v-1} + \frac{\Delta t^2}{2}\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$
* *Velocity* $\quad\bm{v}_k=\bm{v}_{k-1}+\Delta t\left( \bm{C}_{ns}\bm{f}_{k-1}+\bm{g}\right)$
* *Orientation* $\quad\bm{q}_k=\bm{q}_{k-1}\otimes\bm{q}\left(\bm{\omega}_{k-1}\Delta t\right) = \bm{\Omega}\left(\bm{q}(\bm{\omega}_{k-1}\Delta t)\right) \bm{q}_{k_-1}$

Where:
$$\bm{C}_{ns}=\bm{C}_{ns}(\bm{q}_{k-1})$$
$$\Omega\left(\begin{bmatrix} q_w\\ \bm{q}_v \end{bmatrix}\right) =  q_w\bm{1}+\begin{bmatrix}0&&-\bm{q}_v^T\\\bm{q}_v&&-[\bm{q}_v]_\times\end{bmatrix}$$
$$\bm{q}(\bm{\theta})=\begin{bmatrix} \cos{\frac{|\theta|}{2}}\\\frac{\bm{\theta}}{|\bm{\theta}|} \sin{\frac{|\theta|}{2}} \end{bmatrix}$$

```python
def prediction(p: np.ndarray, v: np.ndarray, q: np.ndarray, omega_sensor: np.ndarray, acc_sensor: np.ndarray,
               delta_t_: float,
               k_: int):
    """
    :param p: position
    :param v: velocity
    :param q: quaterion configuration
    :param omega_sensor: Angular velocity sensor
    :param acc_sensor:
    :param delta_t_: time step
    :param k_:
    :return: p_check, v_check, q_check predicted state
    """
    c_ns = Quaternion(*q[k_ - 1]).to_mat()  # Quaterion as transformation matrix
    c_ns_dot_acc_prev = c_ns @ acc_sensor.data[k_ - 1]
    acceleration = (c_ns_dot_acc_prev + g)

    p_check = p[k_ - 1] + delta_t_ * v_est[k_ - 1] + delta_t_ ** 2 * acceleration / 2
    v_check = v[k_ - 1] + delta_t_ * acceleration

    q_from_w = Quaternion(axis_angle=omega_sensor.data[k - 1] * delta_t_)
    q_check = q_from_w.quat_mult_right(q[k_ - 1])
    return p_check, v_check, q_check, c_ns_dot_acc_prev


# 1. Update state with IMU inputs
p_check_, v_check_, q_check_, acc_prev_xyz = prediction(p_est, v_est, q_est, imu_w, imu_f, delta_t, k)
p_est[k] = p_check_
v_est[k] = v_check_
q_est[k] = q_check_
```

### Error State Linearization

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

#### Linearization

```python
# 1.1 Linearize the motion model and compute Jacobians
f_jac_km = np.eye(9)
f_jac_km[0:3, 3:6] = np.eye(3) * delta_t
f_jac_km[3:6, 6:9] = -skew_symmetric(acc_prev_xyz) * delta_t
```

### Uncertainty Propagation

The uncertainty in the state is captured by the state covariance (uncertainty)
matrix; note that the uncertainty grows over time (until a measurement
arrives):

$$\check{P}_k=\bm{F}_{k-1}\bm{P}_{k-1}\bm{F}_{k-1}^T + \bm{L}_{k-1}\bm{Q}_{k-1}\bm{L}_{k-1}^T $$
>Where $\bm{p}_{k-1}$ can be either $\check{P}_k$ or $\hat{P}_k$

```python
# 2. Propagate uncertainty
q_cov_km = np.eye(6) * delta_t**2
q_cov_km[:3, :3] *= var_imu_f
q_cov_km[3:, 3:] *= var_imu_w
p_cov_check = f_jac_km @ p_cov[k - 1] @ f_jac_km.T + l_jac @ q_cov_km @ l_jac.T
p_cov[k] = p_cov_check
```

### Measurement Availability

IMU data arrive at a much faster rate than either GNSS or LIDAR sensor
measurements; when a measurement arrives, a measurement update is
applied:

```python
# 3. Check availability of GNSS and LIDAR measurements
if lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] <= imu_f.t[k - 1]:
    p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(
        var_lidar, p_cov_check, lidar.data[lidar_i].T, p_est[k], v_est[k], q_est[k])
    lidar_i += 1

if gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] <= imu_f.t[k - 1]:
    p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(
        var_gnss, p_cov_check, gnss.data[gnss_i].T, p_est[k], v_est[k], q_est[k])
    gnss_i += 1
```

### Measurement Update

Measurements are processed sequentially by the EKF as they arrive; in our case, both the GNSS receiver and the LIDAR provide position updates:

1. Compute Kalman Gain
   $$\bm{K}_k=\bm{\check{P}}_k\bm{H}_k^T\left(\bm{H}_k\bm{\check{P}}_k\bm{H}_k^T \right)^{-1}$$

2. Compute error state $$\delta\bm{x}_k=\bm{K}_k\left(\bm{y}_k-\check{\bm{p}}_k\right)$$

3. Correct predicted state
  $$\hat{\bm{p}}_k=\check{\bm{p}}_k + \delta\bm{p}_k$$
  $$\hat{\bm{v}}_k=\check{\bm{v}}_k + \delta\bm{v}_k$$
  $$\hat{\bm{q}}_k= \bm{q}(\delta\bm{\phi})\otimes\check{\bm{q}}_k \leftarrow \text{global orientation error}$$

4. Computed corrected covariance
  $$\hat{\bm{P}}_k=\left(\bm{1} - \bm{K}_k\bm{H}_k \right)\check{\bm{P}}_k$$

$$\bm{y}_k=\bm{h}(\bm{x}_k)+\bm{v}_k\\=\bm{H}_k\bm{x}_k+\bm{v}_k = \begin{bmatrix}\bm{1} & \bm{0} & \bm{0}\end{bmatrix}\bm{x}_k + \bm{v}_k\\=\bm{p}_k+\bm{v}_k$$
$$\bm{v}_k \sim \mathcal{N}(\bm{0},\bm{R}_{\text{GNSS}}) \quad\text{or} \quad \bm{v}_k \sim \mathcal{N}(\bm{0},\bm{R}_{\text{LIDAR}})$$
> $$\bm{K}_k= \bm{\check{P}}_{k}\bm{H}_k^T(\bm{H}_k\bm{\check{P}}_{k}\bm{H}_k^T+\bm{R})^{-1}$$
> $\bm{R}$ is one of $\bm{R}_{\text{GNSS}}$ or $\bm{R}_{\text{LIDAR}}$

```python
def measurement_update(sensor_var: float, p_cov_check_: np.ndarray, y_k, p_check: np.ndarray, v_check, q_check):
    """

    :param sensor_var:
    :param p_cov_check_:
    :param y_k:
    :param p_check:
    :param v_check:
    :param q_check:
    :return: p_check, v_check, q_check, p_cov_check
    """
    # 3.1 Compute Kalman Gain
    r_cov_sensor = np.eye(3) * sensor_var
    to_invert = h_jac @ p_cov_check_ @ h_jac.T + r_sensor
    if np.linalg.det(to_invert) == 0:
        raise "Singular matrix"
    k_k = p_cov_check_.dot(h_jac.T).dot(np.linalg.inv(to_invert))

    # 3.2 Compute error state
    error_state = k_k @ (y_k - p_check)

    # 3.3 Correct predicted state
    p_hat = p_check + error_state[0:3]
    v_hat = v_check + error_state[3:6]
    q_hat = Quaternion(euler=error_state[6:9]).quat_mult_left(q_check)

    # 3.4 Compute corrected covariance
    p_cov_hat = (np.identity(9) - k_k @ h_jac) @ p_cov_check_
    return p_hat, v_hat, q_hat, p_cov_hat
```
