# Intrductin t Lateral Vehicle Cntrl

## Lateral Cntrl Design

Lateral cntrl fr an autmbile

* Define errr relative t desired path
* Select a cntrl law that drives errrs t zer and satisfies input cnstraints
* Add dynamic cnsideratins t manage frces and mments acting n vehicle

## The reference path

![path references](./Path%20References.jpg)
Track:

* Straight line segments
* Waypints
* Parameterized curves

Main goals:

* Heading path alignment
* Elimination of offset to path.

## Two Types of Control Design

Geometric Controllers

* Pure pursuit (carrot following)
* Stanley

Dynamic Controllers

* MPC control
* Other control systems
  * Sliding mode, feedback linearization

## Plant model

![plant model](./Plant%20model.jpg)
Vehicle (bicycle) model & parameters

* All states variables and inputs defined relative to the centre of front axle

## Driving controllers

Controller error terms
$$\dot{\psi}_{des(t)} -\dot{\psi} = \frac{v_f(t)\sin{\delta(t)}}{L}$$

* Heading error
![heading error](./heading%20error.jpg)
  * Component of velocity perpendicular to trajectory divided by ICR radius
  * Desired heading is zero

$$\dot{\psi} = \frac{v_f(t)\sin{\delta(t)}}{L}$$

* Crosstrack error $(e)$ :
![crosstrack error](./crosstrack%20error.jpg)
  * Distance from center of front axle to the closest point on path.
  * Rate of change of crosstrack error $(\dot{e})$
$$\dot{e}(t) = v_f(t)\sin(\psi(t)-\delta(t))$$

## Additional material

J. Jiang and A. Astolfi, ["Lateral Control of an Autonomous Vehicle,"](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8286943&isnumber=8363076) in IEEE Transactions on Intelligent Vehicles, vol. 3, no. 2, pp. 228-237, June 2018.

To compute the minimum distance to a curved path defined by a spline:

Wang, H., Kearney, J., & Atkinson, K. (2002, June). [Robust and efficient computation of the closest point on a spline curve](http://homepage.divms.uiowa.edu/~kearney/pubs/CurvesAndSurfaces_ClosestPoint.pdf). In Proceedings of the 5th International Conference on Curves and Surfaces (pp. 397-406).
