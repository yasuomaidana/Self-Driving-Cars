# Vehicle Slip Angle

![slip angle](./Slip%20angle.jpg)

## Vehicle (Bicycle) Slip Angle

* Slip angle $$\beta=\tan^{-1}\frac{V_y}{V_x} = \tan^{-1}\frac{\dot{y}}{\dot{x}}$$
* Using small angle assumption: $$\beta \approx \frac{\dot{y}}{\dot{x}}$$

![bicycle slip anlge](./bicycle%20slip%20anlge.jpg)

## Tire slip angles

Tire slip angle is the angle between the direction in which a wheel
is pointing and the direction in which it is actually travelling
![tire slip angle](./tire%20slip%20angles.jpg)
$$\alpha_r = -\beta + \frac{l_r \dot{\psi}}{V}$$
$$\alpha_f = \delta -\beta + \frac{l_f \dot{\psi}}{V}$$

## Slip ration (longitudinal slip)

$$s = \frac{wr_e - V}{V}$$
Where:

* $w$ is wheel angular speed
* $r_e$ tire effective radius
* $V$ vehicle forward velocity

### Sleep ratios

* $wr_e<V$: Wheels are skidding, this happens during deceleration of the vehicle, during normal braking
* $wr_e>V$: Wheels are spinning, this happens during acceleration, especially in low friction driving (icy road)
* $wr_e=0$: Wheels are locked, this happens during heavy or panic braking where the vehicle loses its desired traction

## Parametized tire models

* Linear Tire Model

Assumption: the relationship between slip angle and
force is linear
![linear tire model](./linear%20tire%20model.jpg)

* Pacejka Tire Model

Widely used in model-based control development.

![Pacejka Tire Model](./Pacejka%20Tire%20Model.jpg)
$$F\left(x,F_z\right) = D \sin{\left(C \tan^-1\left( Bx - E \left(BX\right)\right)\right)}\mu F_z$$
Where:

* $x$: could be either slip ratio or slip angle (in tire modeling)
* $\mu$: road friction coefficient
* $F_z$: tire vertical force

## Additional resources

Moad Kissai, Bruno Monsuez, Adriana Tapus, Didier Martinez. ["A new linear tire model with varying parameters"](https://hal.archives-ouvertes.fr/hal-01690792/). 2017 2nd IEEE International Conference on Intelligent Transportation Engineering (ICITE), Sep 2017, Singapore, Singapore. IEEE, Intelligent Transportation Engineering (ICITE), 2017 2nd IEEE International Conference on. <10.1109/ICITE.2017.8056891>.