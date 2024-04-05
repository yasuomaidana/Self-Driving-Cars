# Parametric Curves

## Boundary Conditions

<img alt="Boundary Conditions" src="./Boundary Conditions.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* Boundary conditions must hold on either endpoint of a path
  * The starting and ending conditions of the path

## Kinematic Constraints

<img alt="Kinematic Constraints" src="./Kinematic Constraints.jpg" style="max-height:30vh;margin: 1em auto; display:block;"/>

* Maximum curvature along path cannot be exceeded
* Ensures that car can drive along path

## Parametric Curves Definitions

* Parametric curve $\bm{r}$ can be described by a set of parameterized equations $$\bm{r}(u) = \langle x(u),y(u)\rangle \quad u\in[0,1]$$
* Parameter denotes path traversal, can be arc length or unitless
* Example: Cubic spline formulation for $x$ and $y$ $$x(u)=\alpha_3u^3+\alpha_2u^2+\alpha_1u+\alpha_0 \\ y(u)=\beta_3u^3+\beta_2u^2+\beta_1u+\beta_0$$

## Path Optimization

* Want to optimize path according to cost functional $f$
* Parametric curves allow for optimizing over parameter space, which simplifies optimization formulation

$$\min{f(\bm{r}(u))}\ \text{s.t}\begin{cases}
c(\bm{r}(u))\leq\alpha, \quad\forall u \in[0,1]\\
\bm{r}(0) = \beta_0\\
\bm{r}(u_f) = \beta_f
\end{cases}$$

## Non-Parametric Path

<img alt="Trajectory Planner Example" src="./Trajectory Planner Example.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Reactive planner used non-parametric paths underlying each trajectory
  * Path was represented as a sequence of points rather than parameterized curves

## Path Parameterization Examples

<img alt="Quintic Splina and Cubic Spiral" src="./Quintic Splina and Cubic Spiral.jpg" style="max-height:40vh;margin: 1em auto; display:block;"/>

* Two common parameterized curves are quintic splines and cubic spirals
* Both allow us to satisfy boundary conditions, and can be optimized parametrically

## Quintic Splines

* $x$ and $y$ are defined by 5th order splines
* Closed form solution available for $(x, y, \theta, k)$ boundary conditions
$$x(u)=\alpha_5u^5+\alpha_4u^4+\alpha_3u^3+\alpha_2u^2+\alpha_1u+\alpha_0$$
$$y(u)=\beta_5u^5+\beta_4u^4+\beta_3u^3+\beta_2u^2+\beta_1u+\beta_0$$
$$u\in[0,1]$$

## Quintic Splines Curvature

* Challenging to constrain curvature due to nature of spline's curvature
  * Due to potential discontinuities in curvature or its derivatives

$$k(u)=\frac{x'(u)y''(u)-y'(u)x''(u)}{(x'(u)^2+y'(u)^2)^{\frac{3}2}}$$

## Polynomial Spirals

* Spirals are defined by their curvature as a function of arc
length
$$k(s)=a_3s^3+a_2s^2+a_1s+a_0$$
$$\theta(s)=\theta_0+\int_0^s\left(a_3s'^3+a_2s'^2+a_1s'+a_0\right) ds' \\
= \theta_0+a_3\frac{s^4}{4}+a_2\frac{s^3}{3}+a_1\frac{s^2}{2}+a_0s$$
* Closed form curvature definition allows for simple curvature
constraint checking
  * Curvature is well-behaved between sampled points as well due to polynomial formulation

$$x(s)=x_0+\int_0^s\left(\cos(\theta(s'))\right)ds'$$
$$y(s)=y_0+\int_0^s\left(\sin(\theta(s'))\right)ds'$$

## Polynomial Spiral Position
* Spiral position does not have a closed form solution
* Fresnel integrals need to be evaluated numerically
  * This can be done using Simpson's rule

$$x(s)=x_0+\int_0^s\left(\cos(\theta(s'))\right)ds'$$
$$y(s)=y_0+\int_0^s\left(\sin(\theta(s'))\right)ds'$$

> $$\int_0^sf(s')ds'\approx\frac{s}{3n}\left(f(0)+4f\left(\frac{s}n\right)+2f\left(\frac{2s}n\right)+\dots+f(s)\right)$$
> Simpson's Rule
