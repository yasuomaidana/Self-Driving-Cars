# Objective Functions for Autonomous Driving

## Efficiency

**Path length**: Minimize the arc length of a path to
generate the shortest path to the goal

$$s_f=\int_{x_i}^{x_f}\sqrt{1+\left(\frac{dy}{dx}\right)^2}dx$$

**Travel time**: Minimize time to destination while following
the planned path

$$T_f=\int_0^{s_f}\frac{1}{v(s)}ds$$

## Reference Tracking

Penalize deviation from the reference path or speed profile.

$$\int_0^{s_f}||x(s)-x_{\text{ref}}(s)||ds$$
$$\int_0^{s_f}||v(s)-v_{\text{ref}}(s)||ds$$

For velocity: Hinge loss to penalize speed limit
violations severely

$$\int_0^{s_f}(v(s)-v_{\text{ref}}(s))_+ds$$

>Essentially, this term is only active when the velocity profile exceeds the speed limit, that is when the difference between our speed and the speed limit is positive, otherwise this term is zero. This allows us to penalize speed limit violations more harshly while still allowing us to encourage the autonomous vehicle to reach its required reference velocity.

## Smoothness

$$\int_0^{s_f}||\overset{...}{x}(s)||^2ds$$

Where $\overset{...}{x}$ is the jerk. Jerk is the rate of change of acceleration with respect to time, or the third derivative of position. The jerk along the car's trajectory greatly impacts the user's comfort while in the car.

## Curvature

$$\int_0^{s_f}||k(s)||^2ds$$
