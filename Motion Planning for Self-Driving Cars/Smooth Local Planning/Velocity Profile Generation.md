# Velocity Profile Generation

## Behavioural Planner Reference Velocity

* Need to compute reference velocity
* Can use the speed limit of the road as a starting point
* Behavioural planner maneuver will also influence reference velocity
  * E.g. a stopping maneuver requires us to stop

## Dynamic Obstacles

<img alt="Dynamic Obstacles" src="./Dynamic Obstacles.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* Lead dynamic obstacles regulate our speed to prevent collisions
* Time to collision is an important metric to preserve when driving with lead vehicles
* Need to reach the red point at lead vehicle speed to ensure there is no collision

$$\text{TTC}=\frac{v_\text{ego}-v_\text{lead}}S$$

## Curvature and Lateral Acceleration

<img alt="Curvature and Lateral Acceleration" src="./Curvature and Lateral Acceleration.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* Curvature recorded at intermediate points, $k_i$
* Velocity bounded by maximum lateral acceleration from comfort rectangle

$$v_k\leq\sqrt{\frac{a_\text{lat}}{k_i}}$$
$$v_f=\min\left(v_\text{ref},v_\text{lead},v_k\right)$$

## Linear Ramp

### Profile

* Simplest shape is a linear ramp to our desired velocity
* We know the total arc length of our path $s$ and our initial and final speed $v_0$ an $v_f$

### Acceleration Calculation

* Can calculate acceleration using initial and final velocity as well as path arc length
  * Need to be sure acceleration values don't exceed our comfort rectangle
* If we clamp our acceleration, we can recompute the final velocity using the clamped acceleration for $a$

$$\frac{v_f^2-v_0^2}{2s}=a \quad \sqrt{2as+v_0^2}=v_f$$

### Velocity Calculation

<img alt="Velocity Calculation" src="./Velocity Calculation.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* For a given acceleration, we can then compute the velocity at each point by using the accumulated arc length $s_i$ up to that point

$$\sqrt{2as_i+v_0^2}=v_{f_i}$$

## Trapezoidal Profile

* Alternative profile is trapezoidal, car decelerates to slower speed before stopping
  * Useful for stop sign scenarios
* Deceleration chosen to be well within comfort rectangle to maximize passenger comfort

### First Segment

* First step is to determine distance required to reach transit velocity $v_t$ using gentle deceleration $a_0$
* Can then compute linear deceleration for all points up to point of reaching transit speed

$$\frac{v_t^2-v_0^2}{2a_0}=s_a$$
$$\sqrt{2a_0s_i+v_i^2}=v_{f_i}$$

### Third segment

* Can repeat a similar process to reach a stop from $v_t$ using gentle deceleration $a_0$
* Need to first find point of initial deceleration $s_b$
* The points in between 1st and 3rd segment have constant
velocity $v_t$

$$\frac{0-v_t^2}{2a_0}=s_f-sb$$
$$\sqrt{2a_0(s_i-s_b)+v_i^2}=v_{f_i}$$

### All Segments

<img alt="Trapezoidal Profile" src="./Trapezoidal Profile.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

Bringing in the constant velocity transit section, we have our full trapezoidal velocity profile

$$v_{f_i}=
\begin{cases}
\sqrt{2a_0s_i+v_i^2},&s_i\leq s_a\\
v_t, &s_a\leq s_i \leq s_b\\
\sqrt{2a_0(s_i-s_b)+v_i^2}, &s_b\leq s_i \leq s_f
\end{cases}$$
