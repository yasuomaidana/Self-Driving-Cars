# Conformal Lattice Planning

## Conformal Lattice

<img alt="Conformal Lattice" src="./Conformal Lattice.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Goal is to plan a feasible collision-free path to goal
* Conformal lattice exploits road structure to speed up planning
* Lattice paths are laterally offset from a goal point along road

## Goal Horizon

<img alt="Goal Horizon" src="./Goal Horizon.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Short lookahead improves computation time, but reduces ability to avoid obstacles
* Goal point is dynamically calculated based on speed and other factors
* Endpoints are sampled laterally offset from goal according to the heading along the road

## Generating Spirals

* Can then compute cubic spirals to each goal point
* Focus on kinematic feasibility for now, collision-checking comes later
* If a goal point cannot be reached with a spiral under the kinematic constraints, discard that goal point

$$\min f_{be}(a_0,a_1,a_2,a_3,s_f) +\alpha\left(x_s(p_4)-x_f\right)
+\beta\left(y_s(p_4)-y_f\right)
+\gamma\left(\theta_s(p_4)-\theta_f\right)$$
$$\text{s.t.}\begin{cases} |p_1|\leq k_\text{max} \\
|p_2|\leq k_\text{max}\end{cases}$$

## Getting Spiral Parameters

* Convert optimization variables back into spiral parameters
* Can then use spiral coefficients to sample points along the spiral

$$a_0=p_0$$
$$a_1=-\frac{11\frac{p_0}{2}-9p_1+9\frac{p_2}{2}-p_3}{p_4}$$
$$a_2=\frac{9p_0-45\frac{p_1}{2}+18p_2-9\frac{p_3}{2}}{p_4^2}$$
$$a_3=\frac{9\frac{p_0}{2} - 27\frac{p_1}{2}+ 27\frac{p_2}{2}- 9\frac{p_3}{2}}{p_4^3}$$

## Trapezoidal Rule Integration

<img alt="Trapezoidal" src="./Trapezoidal Rule.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Use numerical integration to generate positions along path
* Trapezoidal rule is faster for generating entire path than Simpson's rule
* Discrete representation generated using ```cumulative_trapezoid()``` function in Python

$$
\begin{array}{lr}
  x(s) = x_0 + \int_0^s \cos(\theta(s')) ds' \\\\
  y(s) = y_0 + \int_0^s \sin(\theta(s')) ds'
\end{array} \Rightarrow \int_0^sf(x)dx\approx\sum_{i=1}^{N-1}\frac{f(x_{i+1})+f(x_i)}{2}(x_{i+1}-x_i)$$

## Collision Checking

<img alt="Collision Checking" src="./Collision Checking.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Can do this through circle-based or swath-based collision
checking for each point along each path
* Parked vehicle (in red) represents obstacle, paths that
would result in a collision with it are marked red

## Path Selection

<img alt="Path Selection" src="./Path Selection.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Need to select best path among collision-free paths
* Objective function for selection is a design choice
* Can reward paths that track the center of the road, and
penalize paths that come too close to obstacles
* Best path highlighted in blue

## Full Path

<img alt="Full Path" src="./Full Path.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Can repeat this process for each planning step as car moves along road
* Path will converge to the centerline of the road, even when obstacles are present
