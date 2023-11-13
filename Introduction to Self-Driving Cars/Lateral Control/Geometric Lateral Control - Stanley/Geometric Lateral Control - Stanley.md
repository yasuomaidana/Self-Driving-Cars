# Geometric Lateral Control - Stanley

Stanley method is the path tracking approach used by Stanford University's Darpa Grand Challenge team.

* Uses the center of the front axle as a reference point
* Look at both the error in heading and the error in position relative to the closest point on the path
* Define an intuitive steering law to:
  * Correct heading error
  * Correct position error
  * Obey max steering angle bounds

## Heading control law

![heading control law](./Heading%20control%20law.jpg)

* Steer to align heading with desired heading (proportional to heading error) $$\delta(t) = \psi(t)$$
* Steer to eliminate crosstrack error
  * Essentially proportional to error
  * Inversely proportional to speed
  * Limit effect for large errors with inverse tan
  * Gain k determined experimentally
  $$\delta(t) = \tan^-1\left(\frac{ke(t)}{v_f(t)} \right)$$
* Maximum and minimum steering angles $$\delta(t) \in [\delta_{min},\delta_{max}]$$

## Combined steering law

$$\delta(t) = \psi(t) + \tan^-1\left(\frac{ke(t)}{v_f(t)} \right), \ \delta(t)\in[\delta_{min}, \delta_{max}]$$

For large heading error, steer in opposite direction
![heading error](./heading%20error.jpg)

* The larger the heading error, the larger the steering correction
* Fixed at limit beyond maximum steering angle, assuming no crosstrack error

For large positive crosstrack error:
![crosstrack error](./crosstrack%20error.jpg)
$$\tan^{-1}\left(\frac{ke(t)}{v_f(rt)} \right) \approx \frac{\pi}{2} \rightarrow \delta(t) \approx \psi(t) + \frac{\pi}{2} $$

* As heading changes due to steering angle, the heading correction counteracts the crosstrack correction, and drives the steering angle back to zero
* The vehicle approaches the path, crosstrack error drops, and steering command starts to correct heading alignment.

## Error Dynamics

The error dynamics when not at maximum steering angle are:

$$\dot{e}(t) = -v_f(t)\sin\left(\psi(t) - \delta(t) \right) = -v_f(t)\sin\left(\tan^{-1}\left(\frac{ke(t)}{v_f(t)} \right)\right) \\= \frac{-ke(t)}{\sqrt{1+\left(\frac{ke(t)}{v_f} \right)^2}}$$

For small crosstrack errors, leads to exponential
decay characteristics
$$\dot{e}(t) \approx -ke(t)$$

## Adjustments

Low speed operation

* Inverse speed can cause numerical instability
* Add softening constant to controller $k_s$
$$\delta(t) = \psi(t) + \tan^{-1}\left(\frac{ke(t)}{k_s + v_f(t)} \right)$$

Extra damping on heading

* Becomes an issue at higher speeds in real vehicle

Steer into constant radius curves

* Improves tracking on curves by adding a feedforward term on
heading

## Additional resources

Snider, J. M., ["Automatic Steering Methods for Autonomous Automobile Path Tracking"](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf), Robotics Institute, Carnegie Mellon University, Pittsburg (February 2009).

Hoffmann, G. et al., ["Autonomous Automobile Trajectory Tracking for Off-Road Driving: Controller Design, Experimental Validation and Racing"](http://ai.stanford.edu/~gabeh/papers/hoffmann_stanley_control07.pdf), Stanford University, (2007).
