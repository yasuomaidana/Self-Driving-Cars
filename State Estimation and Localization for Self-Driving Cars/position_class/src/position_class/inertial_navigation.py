"""3D Inertial Navigation, Quaternion Kinematics, and GNSS Positioning."""

from typing import List, Optional, Tuple, Union
import numpy as np


class Quaternion:
    """Hamiltonian Unit Quaternion representation [q_w, q_x, q_y, q_z]."""

    def __init__(self, q: Union[List[float], np.ndarray]):
        arr = np.asarray(q, dtype=np.float64).flatten()
        if arr.shape[0] != 4:
            raise ValueError("Quaternion must have 4 elements: [w, x, y, z]")
        norm = np.linalg.norm(arr)
        if norm < 1e-12:
            raise ValueError("Zero-norm quaternion is invalid")
        self.q = arr / norm

    @property
    def w(self) -> float:
        return float(self.q[0])

    @property
    def x(self) -> float:
        return float(self.q[1])

    @property
    def y(self) -> float:
        return float(self.q[2])

    @property
    def z(self) -> float:
        return float(self.q[3])

    @property
    def vec(self) -> np.ndarray:
        return self.q[1:4]

    def conjugate(self) -> "Quaternion":
        """Returns the quaternion conjugate."""
        return Quaternion([self.w, -self.x, -self.y, -self.z])

    def multiply(self, other: "Quaternion") -> "Quaternion":
        """Hamiltonian product: q_result = self * other."""
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion([w, x, y, z])

    def to_rotation_matrix(self) -> np.ndarray:
        """Converts quaternion to 3x3 direction cosine matrix C_nb (body to nav)."""
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x**2 + z**2), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x**2 + y**2)]
        ], dtype=np.float64)

    def to_euler_angles(self) -> Tuple[float, float, float]:
        """Converts quaternion to Euler angles (roll, pitch, yaw) in radians."""
        w, x, y, z = self.w, self.x, self.y, self.z
        
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return float(roll), float(pitch), float(yaw)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> "Quaternion":
        """Constructs a quaternion from axis and angle (radians)."""
        axis_arr = np.asarray(axis, dtype=np.float64).flatten()
        norm = np.linalg.norm(axis_arr)
        if norm < 1e-12:
            return Quaternion([1.0, 0.0, 0.0, 0.0])
        u = axis_arr / norm
        half = angle / 2.0
        return Quaternion([np.cos(half), u[0] * np.sin(half), u[1] * np.sin(half), u[2] * np.sin(half)])

    @staticmethod
    def from_rotvec(rotvec: np.ndarray) -> "Quaternion":
        """Constructs quaternion from rotation vector (omega * dt)."""
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-12:
            return Quaternion([1.0, 0.0, 0.0, 0.0])
        return Quaternion.from_axis_angle(rotvec, angle)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Returns 3x3 skew symmetric matrix [v]_x."""
    arr = np.asarray(v, dtype=np.float64).flatten()
    return np.array([
        [0.0, -arr[2], arr[1]],
        [arr[2], 0.0, -arr[0]],
        [-arr[1], arr[0], 0.0]
    ], dtype=np.float64)


class StrapdownIMUIntegrator:
    r"""Strapdown Inertial Navigation Dead Reckoning System.
    
    Integrates 3-axis accelerometer and 3-axis gyroscope measurements in 3D:
    p_{k} = p_{k-1} + v_{k-1} dt + 0.5 * (C_nb * f_b + g) dt^2
    v_{k} = v_{k-1} + (C_nb * f_b + g) dt
    q_{k} = q_{k-1} \otimes Exp(0.5 * omega_b * dt)
    """

    def __init__(
        self,
        init_pos: np.ndarray = np.zeros(3),
        init_vel: np.ndarray = np.zeros(3),
        init_quat: Optional[Quaternion] = None,
        gravity: np.ndarray = np.array([0.0, 0.0, -9.81])
    ):
        self.p = np.asarray(init_pos, dtype=np.float64).copy()
        self.v = np.asarray(init_vel, dtype=np.float64).copy()
        self.q = init_quat if init_quat is not None else Quaternion([1.0, 0.0, 0.0, 0.0])
        self.g = np.asarray(gravity, dtype=np.float64).copy()

    def step(self, f_b: np.ndarray, omega_b: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, Quaternion]:
        """Propagate state forward by dt using specific force and angular velocity."""
        # 1. Orientation integration
        rotvec = omega_b * dt
        delta_q = Quaternion.from_rotvec(rotvec)
        self.q = self.q.multiply(delta_q)
        
        # 2. Transform specific force to navigation frame
        C_nb = self.q.to_rotation_matrix()
        a_nav = C_nb @ f_b + self.g

        # 3. Position and velocity integration
        self.p = self.p + self.v * dt + 0.5 * a_nav * (dt**2)
        self.v = self.v + a_nav * dt

        return self.p.copy(), self.v.copy(), self.q


def solve_gnss_trilateration(
    satellite_positions: np.ndarray,
    pseudoranges: np.ndarray,
    initial_guess: Optional[np.ndarray] = None,
    max_iters: int = 15,
    tolerance: float = 1e-4
) -> Tuple[np.ndarray, float]:
    """Computes receiver 3D position and clock bias using GNSS Pseudorange Least Squares (Gauss-Newton).
    
    Args:
        satellite_positions: (N, 3) matrix of satellite coordinates [x_s, y_s, z_s].
        pseudoranges: (N,) vector of measured pseudoranges rho_i.
        initial_guess: (4,) initial state [x, y, z, c*dt_rx].
        max_iters: Maximum Gauss-Newton iterations.
        tolerance: Convergence threshold for state norm update.
        
    Returns:
        receiver_position: (3,) estimated receiver coordinates [x, y, z].
        clock_bias: estimated receiver clock bias (in meters).
    """
    N = satellite_positions.shape[0]
    if N < 4:
        raise ValueError("At least 4 satellites are required for 3D GNSS positioning.")

    if initial_guess is None:
        state = np.zeros(4, dtype=np.float64)
    else:
        state = np.asarray(initial_guess, dtype=np.float64).copy()

    for _ in range(max_iters):
        rx_pos = state[:3]
        c_dt = state[3]
        
        # Predicted ranges and geometry matrix
        diffs = rx_pos - satellite_positions  # (N, 3)
        geometric_ranges = np.linalg.norm(diffs, axis=1)  # (N,)
        
        pred_pseudoranges = geometric_ranges + c_dt
        residuals = pseudoranges - pred_pseudoranges  # (N,)

        # Jacobian H: rows are [ (rx - sat_i)/range_i, 1 ]
        unit_vectors = diffs / geometric_ranges[:, np.newaxis]
        H = np.column_stack([unit_vectors, np.ones(N)])

        # Normal equations update: delta_x = (H^T H)^{-1} H^T residuals
        delta_x = np.linalg.solve(H.T @ H, H.T @ residuals)
        state += delta_x

        if np.linalg.norm(delta_x) < tolerance:
            break

    return state[:3], float(state[3])
