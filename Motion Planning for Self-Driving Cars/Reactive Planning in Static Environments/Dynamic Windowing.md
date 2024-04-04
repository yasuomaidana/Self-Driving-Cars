# Dynamic Windowing

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

### Bicycle Model + Acceleration Constraints

* Higher order terms handled by adding constraints
* More comfort for passengers, but less maneuverability

$$\ddot{\theta}_\text{min} \leq\ddot{\theta}\leq \ddot{\theta}_\text{max}$$
$$\ddot{x}_\text{min} \leq\ddot{x}\leq \ddot{x}_\text{max}$$
<!-- $$\ddot{y}_\text{min} \leq\ddot{y}\leq \ddot{y}_\text{max}$$ -->

## Constraint in Terms of Steering Angle

* Angular acceleration constraint may prevent us from selecting certain maneuvers based on current angular velocity
* Change in steering angle between planning cycles is bounded
* Similar logic applies for changes in linear velocity inputs between planning cycles

$$\dot{\theta} = \frac{v\tan(\delta)}{L}$$
$$\lvert\ddot{\theta}\rvert = \left\vert \frac{\dot{\theta}_2-\dot{\theta}_1}{\Delta t}\right\rvert$$
$$\lvert \tan(\delta_2)- \tan(\delta_1)\rvert\leq\frac{\ddot{\theta}_{\max}L\Delta t}{v}$$

## Example

Given current steering angle and the angular acceleration bound, which candidate trajectories do not violate that bound?

$$v=\frac{1m}s$$
$$\delta_1=\frac{\pi}8$$
$$\delta_{\min}=\frac{-\pi}4 \quad \delta_{\max}=\frac{\pi}4$$
$$\Delta t = 1s \quad L=1m \quad \lvert \ddot{\theta}\rvert\leq0.6s^{-2}$$

## Comparing Trajectories

<img alt="Dynamic Windowing Example" src="./Dynamic Windowing Example.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Trajectories that exceed the angular acceleration constraint are coloured red
* Added constraints reduce manoeuvrability of the robot
