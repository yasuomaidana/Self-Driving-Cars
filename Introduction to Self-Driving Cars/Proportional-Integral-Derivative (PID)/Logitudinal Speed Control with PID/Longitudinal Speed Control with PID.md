# Longitudinal Speed Control with PID

## Architecture of Vehicle Control Strategy

![Architecture of Vehicle Control Strategy](./Architecture%20of%20Vehicle%20Control%20Strategy.jpg)

## Longitudinal Speed Control

![Longitudinal Control System](./Longitudinal%20Control%20System.png)
Cruise control:

Speed of the vehicle is controlled (by throttling and braking) to be kept at the reference speed

## Upper Level Controller

Determines the desired acceleration for the vehicle
(based on the reference and actual velocity).
$$\ddot{x}_{des}=K_P(\dot{x}_{ref} - \dot{x}) + K_I\int_0^t(\dot{x}_{ref} - \dot{x})dt + K_D\frac{d(\dot{x}_{ref} - \dot{x})}{dt}$$

* $\ddot{x}_{des}$: Desired Acceleration
* $\dot{x}_{ref}$: Reference Velocity
* $\dot{x}$: Vehicle Velocity
* $K_P, K_I, K_P$: PID Gains

## Lower Level Controller

![Lower Level Controller](./Lower%20Level%20Controller.jpg)
Throttle input is calculated such that the vehicle track the desired acceleration determined by the upper level controller

* Assumptions:
  * Only throttle actuations is considered (no braking)
  * The torque converter is locked (gear 3+)
  * The tire slip is small (gentle longitudinal maneuvers)

$$T_{engine} = \frac{J_e}{(r_{eff}(GR))}\ddot{x}_{des} + T_{Load}$$
