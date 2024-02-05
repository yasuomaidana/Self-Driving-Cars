import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

# 1. Data

with open('data/pt1_data.pkl', 'rb') as file:
    data = pickle.load(file)

gt = data['gt']
imu_f = data['imu_f']
imu_w = data['imu_w']
gnss = data['gnss']
lidar = data['lidar']

file.close()
# transform the LIDAR data to the IMU frame using our
# known extrinsic calibration rotation matrix C_li and translation vector t_i_li.

C_li = np.array([
    [0.99376, -0.09722, 0.05466],
    [0.09971, 0.99401, -0.04475],
    [-0.04998, 0.04992, 0.9975]
])

t_i_li = np.array([0.5, 0.1, 0.5])

# Transform from the LIDAR frame to the vehicle (IMU) frame.
lidar.data = (C_li @ lidar.data.T).T + t_i_li

var_imu_f = 0.10
var_imu_w = 0.25
var_gnss = 0.01
var_lidar = 1.00

R_GNSS = np.identity(3) * var_gnss  # covariance matrix related to GNSS
R_Lidar = np.identity(3) * var_lidar  # covariance matrix related to Lidar

g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

# Initialization

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


# 4. Measurement Update

def measurement_update(sensor_var: float, p_cov_check_: np.ndarray, y_k, p_check: np.ndarray, v_check, q_check):
    """

    :param sensor_var:
    :param p_cov_check_:
    :param y_k:
    :param p_check:
    :param v_check:
    :param q_check:
    :return: p_hat, v_hat, q_hat, p_cov_hat
    """
    # 3.1 Compute Kalman Gain
    r_cov_sensor = np.eye(3) * sensor_var
    to_invert = h_jac @ p_cov_check_ @ h_jac.T + r_cov_sensor
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


for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    p_check_, v_check_, q_check_, acc_prev_xyz = prediction(p_est, v_est, q_est, imu_w, imu_f, delta_t, k)
    p_est[k] = p_check_
    v_est[k] = v_check_
    q_est[k] = q_check_

    # 1.1 Linearize the motion model and compute Jacobians
    f_jac_km = np.eye(9)
    f_jac_km[0:3, 3:6] = np.eye(3) * delta_t
    f_jac_km[3:6, 6:9] = -skew_symmetric(acc_prev_xyz) * delta_t

    # 2. Propagate uncertainty
    q_cov_km = np.eye(6) * delta_t ** 2
    q_cov_km[:3, :3] *= var_imu_f
    q_cov_km[3:, 3:] *= var_imu_w
    p_cov_check = f_jac_km @ p_cov[k - 1] @ f_jac_km.T + l_jac @ q_cov_km @ l_jac.T
    p_cov[k] = p_cov_check

    # 3. Check availability of GNSS and LIDAR measurements
    if lidar_i < lidar.t.shape[0] and lidar.t[lidar_i] <= imu_f.t[k - 1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(
            var_lidar, p_cov_check, lidar.data[lidar_i].T, p_est[k], v_est[k], q_est[k])
        lidar_i += 1

    if gnss_i < gnss.t.shape[0] and gnss.t[gnss_i] <= imu_f.t[k - 1]:
        p_est[k], v_est[k], q_est[k], p_cov[k] = measurement_update(
            var_gnss, p_cov_check, gnss.data[gnss_i].T, p_est[k], v_est[k], q_est[k])
        gnss_i += 1

# 6. Results and Analysis ###################################################################

################################################################################################
# Now that we have state estimates for all of our sensor data, let's plot the results. This plot
# will show the ground truth and the estimated trajectories on the same plot. Notice that the
# estimated trajectory continues past the ground truth. This is because we will be evaluating
# your estimated poses from the part of the trajectory where you don't have ground truth!
################################################################################################
est_traj_fig = plt.figure()
ax: Axes3D = est_traj_fig.add_subplot(111, projection='3d')
ax.plot(p_est[:, 0], p_est[:, 1], p_est[:, 2], label='Estimated')
ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label='Ground Truth')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_zlabel('Up [m]')
ax.set_title('Ground Truth and Estimated Trajectory')
ax.set_xlim(0, 200)
ax.set_ylim(0, 200)
ax.set_zlim(-2, 2)
ax.set_xticks([0, 50, 100, 150, 200])
ax.set_yticks([0, 50, 100, 150, 200])
ax.set_zticks([-2, -1, 0, 1, 2])
ax.legend(loc=(0.62, 0.77))
ax.view_init(elev=45, azim=-50)
plt.show()

################################################################################################
# We can also plot the error for each of the 6 DOF, with estimates for our uncertainty
# included. The error estimates are in blue, and the uncertainty bounds are red and dashed.
# The uncertainty bounds are +/- 3 standard deviations based on our uncertainty (covariance).
################################################################################################
error_fig, ax = plt.subplots(2, 3)
error_fig.suptitle('Error Plots')
num_gt = gt.p.shape[0]
p_est_euler = []
p_cov_euler_std = []

# Convert estimated quaternions to euler angles
for i in range(len(q_est)):
    qc = Quaternion(*q_est[i, :])
    p_est_euler.append(qc.to_euler())

    # First-order approximation of RPY covariance
    J = rpy_jacobian_axis_angle(qc.to_axis_angle())
    p_cov_euler_std.append(np.sqrt(np.diagonal(J @ p_cov[i, 6:, 6:] @ J.T)))

p_est_euler = np.array(p_est_euler)
p_cov_euler_std = np.array(p_cov_euler_std)

# Get uncertainty estimates from P matrix
p_cov_std = np.sqrt(np.diagonal(p_cov[:, :6, :6], axis1=1, axis2=2))

titles = ['Easting', 'Northing', 'Up', 'Roll', 'Pitch', 'Yaw']
for i in range(3):
    ax[0, i].plot(range(num_gt), gt.p[:, i] - p_est[:num_gt, i])
    ax[0, i].plot(range(num_gt), 3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].plot(range(num_gt), -3 * p_cov_std[:num_gt, i], 'r--')
    ax[0, i].set_title(titles[i])
ax[0, 0].set_ylabel('Meters')

for i in range(3):
    ax[1, i].plot(range(num_gt), angle_normalize(gt.r[:, i] - p_est_euler[:num_gt, i]))
    ax[1, i].plot(range(num_gt), 3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].plot(range(num_gt), -3 * p_cov_euler_std[:num_gt, i], 'r--')
    ax[1, i].set_title(titles[i + 3])
ax[1, 0].set_ylabel('Radians')
plt.show()
