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
