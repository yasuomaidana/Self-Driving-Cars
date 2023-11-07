# Lateral Dynamics of a Bicycle

Lateral acceleration: $a_y=\ddot{y} + \omega^2R = V(\dot{\beta}+\dot{\psi})$

Force equation: $mV(\dot{\beta}+\dot{\psi}) = F_{yf} + F_{yr}$

Moment of inertia equation: $I_z\ddot{\psi} = l_fF_{yf} - l_rF_{yr}$

![Dynamic model](./Lateral%20Dynamics.jpg)

## Tire Slip angles

* Many different tire slip models
* For small tire slip angles, the lateral tire forces are approximated as a linear function of tire slip angle
* Tire variables:
  * Front tire slip angle: $\alpha_f$
  * Rear tire slip angle: $\alpha_r$
![Slip angles](./Slip%20Angles.jpg)

## Front and Rear Tire Forces

![tire forces](./Tire%20forces.jpg)

* $C_f$: linearized cornering stiffness of the front wheel
$$F_{yf} = C_f\alpha_f = C_f \left(\delta-\beta-\frac{l_f\dot{\psi}}{V}\right)$$
* $C_r$: linearized cornering stiffness of the rear wheel
$$F_{yr} = C_r\alpha_r = C_r \left(-\beta-\frac{l_f\dot{\psi}}{V}\right)$$

## Lateral and yaw Dynamics

$$\dot{\beta} = \frac{-\left({C_r+C_f}\right)}{mV}\beta + \left(\frac{C_rl_r-C_fl_f}{mV^2}-1\right)\dot{\psi} + \frac{C_f}{mV}\delta$$

$$\ddot{\psi} = \frac{C_rl_r-C_fl_f}{I_z}\beta - \frac{C_rl_r^2+C_fl_f^2}{I_zV} \dot{\psi} + \frac{C_fl_f}{I_z}\delta$$

## Standard State Space Representation

* State Vector: $X_{lat} = \begin{bmatrix}y&&\beta&&\psi&&\dot{\psi}\end{bmatrix}^T$

$$A_{lat}=\begin{bmatrix} 
0 && V && V && 0 \\
0 && \frac{-\left({C_r+C_f}\right)}{mV} && 0 && \frac{C_rl_r-C_fl_f}{mV^2}-1 \\
0 && 0 && 0 && 1 \\
0 && \frac{C_rl_r-C_fl_f}{I_z} && 0 && - \frac{C_rl_r^2+C_fl_f^2}{I_zV} 
\end{bmatrix}$$

$$B_{lat}=\begin{bmatrix} 
0 \\
\frac{C_f}{mV}\\
0\\
\frac{C_fl_f}{I_z}
\end{bmatrix}$$

$$\dot{X}=A_{lat}X_{lat} + B_{lat}\delta$$

## Additional resources

Read more about the Lateral Dynamics of Bicycle Model (pages 27-44) in :

R. Rajamani (2012), "Lateral Vehicle Dynamics" In: Vehicle Dynamics and Control, Mechanical Engineering Series.
