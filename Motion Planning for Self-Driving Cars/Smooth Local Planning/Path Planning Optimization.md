# Path Planning Optimization

## Cubic Spiral and Boundary Conditions

<img alt="Boundary Conditions" src="./Boundary Conditions.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Boundary conditions specify starting state and required ending state
* Spiral end position lacks closed form solution, requires numerical approximation

$$k(s)=a_3s^3+a_2s^2+a_1s+a_0$$
$$x(s)=x_0+\int_0^s\left(\cos(\theta(s'))\right)ds'$$
$$y(s)=y_0+\int_0^s\left(\sin(\theta(s'))\right)ds'$$

## Position Integrals and Simpson's Rule

<img alt="Position Integrals and Simpson's Rule" src="./Simpson rule.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* Simpson's rule has improved accuracy over other methods
* Divides the integration interval into $n$ regions, and evaluates the function at each region boundary

$$\int_0^sf(s')ds'\approx\frac{s}{3n}\left(f(0)+4f\left(\frac{s}n\right)+2f\left(\frac{2s}n\right)+\dots+f(s)\right)$$

## Boundary Conditions via Simpson's Rule

<img alt="Boundary Conditions" src="./Boundary Conditions.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Using our Simpson's approximations, we can now write out the full boundary conditions in terms of spiral parameters
* Can now generate a spiral that satisfies boundary conditions by optimizing its spiral parameters and its length, $s_f$

$$x_s(s_f)=x_f \quad y_s(s_f)=y_f\\
\theta(s_f)=\theta_f \quad k(s_f)=k_f$$

## Approximate Curvature Constraints

<img alt="Kinematic Constraints" src="./Kinematic Constraints.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Want to apply curvatureconstraints to path so it is drivable by the vehicle
* Curvature constraints correspond to minimum vehicle turning radius
* Can constrain sampled points along the path due to well-behaved nature of spiral's curvature

$$\left\lvert k\frac{s_f}3\right\rvert\leq k_\text{max}$$
$$\left\lvert k\frac{2s_f}3\right\rvert\leq k_\text{max}$$

## Bending Energy Objective

$$f_{be}(a_0,a_1,a_2,a_3,s_f)=\int_0^{s_f}(a_0+a_1+a_2+a_3)^2ds$$

* Bending energy distributes curvature evenly along spiral to promote comfort
  * Equal to integral of square curvature along path, which has closed form for spirals
* Gradient also has a closed form solution
  * Has many terms, so best left to a symbolic solver

## Initial Optimization Problem

* Can bring constraints and objective together to form the full optimization problem
* Can perform optimization in the vehicle's body attached frame to set starting boundary condition to zero

$$\min f_{be}(a_0,a_1,a_2,a_3,s_f) \text{ s.t.}\begin{cases}
\left\lvert k\frac{s_f}3\right\rvert\leq k_\text{max}, \quad \left\lvert k\frac{2s_f}3\right\rvert\leq k_\text{max}\\
x_s(0)=x_0, \quad x_s(s_f) = x_f\\
y_s(0)=y_0, \quad y_s(s_f) = y_f\\
\theta_s(0)=\theta_0, \quad \theta_s(s_f) = \theta_f\\
k(0)=k_0, \quad k(s_f) = k_f
\end{cases}$$

## Soft Constraints

* Challenging for optimizer to satisfy constraints exactly
* Can soften equality constraints by penalizing deviation heavily in the objective function
* We also assume initial curvature is known, which corresponds to $a_0$

$$\min f_{be}(a_0,a_1,a_2,a_3,s_f) +\alpha\left(x_s(s_f)-x_f\right)
+\beta\left(y_s(s_f)-y_f\right)
+\gamma\left(\theta_s(s_f)-\theta_f\right)$$
$$\text{s.t}\begin{cases}
\left\lvert k\frac{s_f}3\right\rvert\leq k_\text{max} \\\left\lvert k\frac{2s_f}3\right\rvert\leq k_\text{max}\\
k(s_f) = k_f
\end{cases}$$

## Parameter Remapping

<img alt="Parameter Remapping" src="./Parameter Remapping.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Can remap spiral parameters
* $p_0$ to $p_3$ corresponds to curvature at 4 points equally spaced along path
* $p_4$ corresponds to the arc length of the spiral
* Since initial and final curvature are known, $p_0$ and $p_3$ eliminated from optimization, reducing dimensionality

$$a_0=p_0$$
$$a_1=-\frac{11\frac{p_0}{2}-9p_1+9\frac{p_2}{2}-p_3}{p_4}$$
$$a_2=\frac{9p_0-45\frac{p_1}{2}+18p_2-9\frac{p_3}{2}}{p_4^2}$$
$$a_3=\frac{9\frac{p_0}{2} - 27\frac{p_1}{2}+ 27\frac{p_2}{2}- 9\frac{p_3}{2}}{p_4^3}$$

## Final Optimization Problem

* Replacing spiral parameters with new parameters leads to new optimization formulation
* Curvature constraints correspond directly to new parameters
* Boundary conditions handled by soft constraints and constant $p_0$ and $p_3$

$$\min f_{be}(a_0,a_1,a_2,a_3,s_f) +\alpha\left(x_s(p_4)-x_f\right)
+\beta\left(y_s(p_4)-y_f\right)
+\gamma\left(\theta_s(p_4)-\theta_f\right)$$
$$\text{s.t.}\begin{cases} |p_1|\leq k_\text{max} \\
|p_2|\leq k_\text{max}\end{cases}$$
