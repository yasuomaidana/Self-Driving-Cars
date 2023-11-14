# Advanced Steering Control - MPC

Model predictive control (MPC)

* Numerically solving an optimization problem at each time step.
* Receding horizon approach

Advantages of MPC

* Straightforward formulation
* Explicitly handles constraints
* Applicable to linear or nonlinear models

Disadvantages of MPC

* Computationally expensive

## Receding horizon Control

![Receding horizon Control](./Receding%20horizon%20Control.jpg)
Receding Horizon Control Algorithm

* Pick receding horizon length $(T)$
* For each time step, $t$
* Set initial state to predicted state, $x_t$
  * Perform optimization over finite horizon $t$ to $T$ while traveling from $x_{t-1}$ to $x_t$
  * Apply first control command, $u_t$, from optimization at time $t$

## MPC structure

![MPC structure](./MPC%20structure.jpg)

### Linear MPC formulation

Linear time-invariant discrete time model:
$$x_{t+1}= Ax_t + Bu_t$$

MPC seeks to find control policy $U$
$$U = \{u_{t|t},u_{t+1|t},u_{t+2|t},...\}$$
Objective function - regulation:
$$J(x(t), U) =
\sum_{j=t}^{t+T-1}x_{j|t}^TQx_{j|t}+u_{j|t}^TRu_{j|t}
$$
Objective function - tracking:
$$\delta x_{j|t} = x_{j|t,des} - x_{j|t} \\ J(x(t), U) =
\sum_{j=t}^{t+T-1}\delta x_{j|t}^TQ\delta x_{j|t}+u_{j|t}^TRu_{j|t}$$

### Linear MPC SOLUTION
Unconstrained, finite horizon, discrete time problem formulation:
$$U \overset{\Delta}{=} \overset{min}{\{u_{t|t},u_{t+1|t},\dots \}} \ J(x(t), U) =
x_{t+T|t}^TQ_fx_{t+T|t} +
\sum_{j=t}^{t+T-1}x_{j|t}^TQx_{j|t}+u_{j|t}^TRu_{j|t}$$
$$s.t. \ \ \ \ \ \ x_{j+1|t}= Ax_{t|t} + Bu_{t|t} , \ t\leq j \leq t+T-1$$

Linear quadratic regulator, provides a closed form solution
* Full state feedback: $u_t = —Kx_t$
* Control gain K is a matrix
* Refer to supplemental materials

### (Non)Linear MPC formulation
Constrained (non)linear finite horizon discrete time case
![non linear formulation](./non%20linear%20formulation.jpg)
No closed form solution, must be solved numerically

## Vehicle Lateral Control

![vehicle lateral control](./vehicle%20lateral%20control.jpg)

## Model predictive controller

Cost Function - Minimize

* Deviation from desired trajectory
* Minimization of control command magnitude

Constraints - Subject to

* Longitudinal and lateral dynamic models
* Tire force limits

Can incorporate low level controller, adding
constraints for:

* Engine map
* Full dynamic vehicle model
* Actuator models
* Tire force models

## Additional material

Falcone, P. et al., "Predictive Active Steering Control for Autonomous Vehicle Systems", IEEE (2007).
