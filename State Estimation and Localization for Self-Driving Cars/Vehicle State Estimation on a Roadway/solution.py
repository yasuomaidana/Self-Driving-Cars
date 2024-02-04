import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rotations import angle_normalize, rpy_jacobian_axis_angle, skew_symmetric, Quaternion

## 1. Data

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

g = np.array([0, 0, -9.81])  # gravity
l_jac = np.zeros([9, 6])
l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
h_jac = np.zeros([3, 9])
h_jac[:, :3] = np.eye(3)  # measurement model jacobian

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

    q_from_w = Quaternion(axis_angle=omega_sensor.data[k-1] * delta_t_)
    q_check = q_from_w.quat_mult_right(q[k_ - 1])
    return p_check, v_check, q_check, c_ns_dot_acc_prev


for k in range(1, imu_f.data.shape[0]):  # start at 1 b/c we have initial prediction from gt
    delta_t = imu_f.t[k] - imu_f.t[k - 1]

    # 1. Update state with IMU inputs
    p_check_, v_check_, q_check_, acc_prev_xyz = prediction(p_est, v_est, q_est, imu_w, imu_f, delta_t, k)
    p_est[k] = p_check_
    v_est[k] = v_check_
    q_est[k] = q_check_
