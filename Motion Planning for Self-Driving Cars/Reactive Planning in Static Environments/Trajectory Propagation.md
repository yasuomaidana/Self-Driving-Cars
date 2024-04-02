# Trajectory Propagation

## Kinematic vs Dynamic Model

|Particle Kinematic Model|Particle Dynamic Model|
|:-:|:-:|
|$$\ddot{x}=a$$|$$M\ddot{x}+B\dot{x}=F$$|
|Disregards mass and inertia of the robot|Takes mass and inertia into consideration|
|Uses linear and angular velocities (and/orderivatives) as input|Uses forces and torques as inputs|

## Recall: Kinematic Bicycle Model

<img alt="Bycicle Model" src="./Bycicle Model.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* $x$ and y correspond to base link
position of the robot
* $\theta$ corresponds to heading of the
chassis with respect to x-axis
* $\delta$ is the steering angle input, v is the
velocity input

$$\dot{x} = v\cos({\theta}) \\\dot{y} = v\sin({\theta})\\
\dot{\theta} = \frac{v\tan({\delta})}{L}\\
\quad\delta_{\min} \leq \delta \leq \delta_{\max}\\ \quad
v_{\min} \leq v \leq v_{\max}$$

## Kinematic Model Discretization

Discretization of differential equations allows for efficient computation of trajectories

Recursive definition saves computation time

$$x_n=\sum_{i=0}^{n-1}v_i\cos\left(\theta_i\right)\Delta t= x_{n-1} + v_{n-1} \cos\left(\theta_{n-1}\right)\Delta t$$
$$y_n=\sum_{i=0}^{n-1}v_i\sin\left(\theta_i\right)\Delta t= y_{n-1} + v_{n-1} \sin\left(\theta_{n-1}\right)\Delta t$$
$$\theta_n = \sum_{i=0}^{n-1}\frac{v_i\tan\left(\delta_i\right)}{L} \Delta t = \theta_{n-1} + \frac{v_i\tan\left(\delta_{n-1}\right)}{L} \Delta t $$

## Constant Velocity and Steering Angle Example

For a given control sequence, we can compute the vehicle's trajectory

Useful for prediction as well

## Varying Input for Obstacle Avoidance

To avoid obstacles, we require more complex maneuvers

We can vary the steering input according to a steering function to navigate complex scenarios

Main objective of local planning is to compute the control inputs (or trajectory) required to navigate to goal point without collision
