# Trajectory Rollout Algorithm

## Trajectory Rollout Planner

<img alt="Trajectory Planner Example" src="./Trajectory Planner Example.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Uses trajectory propagation to generate candidate set of trajectories
* Among collision-free trajectories, select trajectory that makes the most progress to goal

## Trajectory Set Generation

* Each trajectory corresponds to a fixed control input
to our model
  * Typically uniformly sampled across range of possible inputs
* More sampled trajectories leads to more maneuverability
* Fewer sampled trajectories improves computation time

## Trajectory Propagation

Holding the velocity constant and varying the steering angle gives candidate set of trajectories

$$x_n=\sum_{i=0}^{n-1}v_i\cos\left(\theta_i\right)\Delta t= x_{n-1} + v_{n-1} \cos\left(\theta_{n-1}\right)\Delta t$$
$$y_n=\sum_{i=0}^{n-1}v_i\sin\left(\theta_i\right)\Delta t= y_{n-1} + v_{n-1} \sin\left(\theta_{n-1}\right)\Delta t$$
$$\theta_n = \sum_{i=0}^{n-1}\frac{v_i\tan\left(\delta_i\right)}{L} \Delta t = \theta_{n-1} + \frac{v_i\tan\left(\delta_{i}\right)}{L} \Delta t $$

## Swath Based Collision Checking

* Swath is generated for each candidate trajectory
* Collision checking is performed for each point in the swath using the occupancy grid

$$S=\cup_{p\in P}F\left(x\left(p\right),y\left(p\right),\theta\left(p\right) \right)$$

## Objective Function

* Rewarding progress to a goal point is the ultimate goal of motion planning

$$J=\lVert x_n - x_{\text{goal}}\rVert$$
$$J=\alpha_1\lVert x_n - x_{\text{goal}}\rVert +\alpha_2\sum_{i=1}^nk_i^2+\alpha_3\sum_{i=1}^n\lVert x_i - P_{\text{center}}(x_i)\rVert$$
Where:

* $\lVert x_n - x_{\text{goal}}\rVert$ : Distance to Goal
* $k_i$ : Curvature
* $\lVert x_i - P_{\text{center}}(x_i)\rVert$ : Deviation from centerline

## Example

<img alt="Example Start" src="./Example Start.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Steering angle bounded by $|\delta|\leq\frac{\pi}{4}$
* Time discretization of $0.1s$, with a $2s$ planning horizon
* Want to reach the gold region while avoiding red obstacles

* First trajectory corresponds to $\delta=-\frac{\pi}{4}$
* Second trajectory corresponds to $\delta=-\frac{\pi}{8}$
* Straight line trajectory corresponds to $\delta=0$
* Positive steering angle trajectories are symmetrical with the negative steering angle trajectories

<img alt="Example 2" src="./Example 2.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Using the vehicle footprint, we compute the swath for each trajectory
* Any trajectory whose swath intersects an obstacle is coloured red to mark a collision
* Green denotes a collision-free trajectory

<img alt="Example 3" src="./Example 3.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Objective is evaluated across all collision-free trajectories
* The black path makes the most progress towards the goal region, so it is selected

<img alt="Example 4" src="./Example 4.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

* Planning cycle is shorter than trajectory length
* The orange portion of the trajectory is not executed before the next planning cycle

<img alt="Example 5" src="./Example 5.jpg" style="height:30vh;margin: 1em auto; display:block;"/>

• This process is continued until goal is reached
• This planner is greedy and sub-optimal, but is fast enough to allow for online planning

<img alt="Example 6" src="./Example 6.jpg" style="height:30vh;margin: 1em auto; display:block;"/>
### Receding Horizon Example

* Only Is of 2s trajectory is executed at each planning iteration
* Planning horizon end time recedes towards time point when goal is reached
