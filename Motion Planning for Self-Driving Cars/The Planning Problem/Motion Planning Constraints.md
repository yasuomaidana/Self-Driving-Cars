# Motion Planning Constraints

## Bicycle Model

![bicycle model](/bicycle%20model.jpg)

$$\dot{\theta}=\frac{V\tan{(\delta)}}{L}$$
$$|\kappa|\le \kappa_{\text{max}}$$

* Kinematics simplified to bicycle model
* Bicycle model imposes curvature constraint on our path planning process
* Curvature constraint is non-holonomic
  * Non-holonomic[^1] constraints reduce the directions a mobile robot can travel at any point
  * Makes motion planning challenging

## Curvature $k$

$$k = \frac{1}{r} = \frac{x'y''-y'x''}{(x'^2+y'^2)^{3/2}}$$

## Vehicle Dynamics

Recall: friction ellipse denotes maximum magnitude of tire forces before stability loss.

![friction ellipse](./friction%20ellipse.jpg)

Friction forces are extreme limit; more useful constraint is accelerations tolerable by passengers.

* Given by "comfort rectangle" range of lateral and longitudinal accelerations

> The friction ellipse denotes the maximum magnitude of the friction forces that can be generated between a car tire and the road. If the applied forces of the car's engine exceed the friction forces of the tire, the tires will slip on the road surface.f

### Dynamics and Curvature

Friction limits and comfort restrict lateral acceleration $a_{\text{lat}}$.

$$a_{\text{lat}}=\frac{v^2}{r}, \quad a_{\text{lat}} \leq a_{\text{lat}_\text{max}}$$

* Lateral acceleration is a function of instantaneous turning radius of path and velocity

$$v^2\leq \frac{a_{\text{lat}_\text{max}}}{k}$$

## Static Obstacles

* Static obstacles block portions of workspace
  * Occupancy grid encoding stores obstacle locations
* Static obstacle constraints satisfied by performing collision checking
  * Can check for collisions using the swath of the vehicle's path
  * Can also check for closest obstacle along ego vehicle's path

## Rules of the Road and Regulatory Elements

* Lane constraints restrict path locations
* Signs, traffic lights influence vehicle behaviour

[^1]: This means that the constraint doesn't depend on the only the state of the robot, but also on how the robot got to its current state. Non-holonomic constraints reduce the number of directions a robot can take at any given point in its workspace. In general, non-holonomic constraints make the planning problem much more complex.
