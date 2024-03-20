# Populating Occupancy Grids from LIDAR Scan Data

## Bayesian Update Of The Occupancy Grid - Summary

Bayes' theorem is applied for at each update step for each
cell.

$$bel_t(m^i) =n p(y_t|m^i)bel_{t-1}(m^i)$$

* $p(y_t|m^i)$: **Current measurement**
* $bel_{t-1}(m^i)$: **Previous belief map**
* $n$: **Normalizer constant**

## Issue With Standard Bayesian Update

Update a single unoccupied grid cell

|$bel_{t-1}(m)$| $p(y_t\mid m^i)$ |new $bel_t(m)$|
|--|--|--|
|0.000638|0.000012|0.000000008|

Multiplication of numbers close to zero is hard for computers

Store the log odds ratio rather than probability
$$bel_t(m)\rightarrow(-\infty,\infty) \quad \log{\left(\frac{p}{1-p}\right)}$$

### Conversion

![conversion](./Conversion.jpg)

$$\text{logit}(p) = \log{\left(\frac{p}{1-p}\right)} \\ p=\frac{e^{\text{logit}(p)}}{1+e^{\text{logit}(p)}}$$

## Bayesian Log Odds Single Cell Update Derivation

Applying Bayes' rule:$$p(m^i\mid y_{1:t})=\frac{p(y_{t}\mid y_{1:t-1},m^i)p(m^i\mid y_{1:t-1})}{p(y_{t}\mid y_{1:t-1})}$$

* $m^i$: Current map cell
* $y_{1:t}$: Sensor measurement for
given cell
* $p(y_{t}\mid y_{1:t-1},m^i)$: Probability of getting measurement $y_t$ given the cell state at all previous measurements $y_{1:t-1},m^i$. **Pulling out current measurement** $y_t$ from past measurements $y_{1:t-1}$
* $p(m^i\mid y_{1:t-1})$: Probability a cell is occupied given all measurements to time $t-1$
* $p(y_{t}\mid y_{1:t-1})$: Probability of getting the measurements $y_t$ given all previous measurements up to time $t-1$. **Pulling out current measurement** $y_t$ from past measurements $y_{1:t-1}$

**Applying the Markov assumption**:

$$p(m^i\mid y_{1:t})=\frac{p(y_{t}\mid m^i)p(m^i\mid y_{1:t-1})}{p(y_{t}\mid y_{1:t-1})}$$

Applying Bayes' rule to measurement model:

$$p(y_t\mid m^i)=\frac{p(m^i \mid y_{t})p(y_{t})}{p(m^i)}$$

Yields:

$$p(m^i\mid y_{1:t})=\frac{p(m^i \mid y_{t})p(y_{t})p(m^i\mid y_{1:t-1})}{p(m^i)p(y_{t}\mid y_{1:t-1})}$$

* Denominator: $1-p$

$$p(\neg m^i\mid y_{1:t})=1-p(m^i\mid y_{1:t})=\frac{p(\neg m^i \mid y_{t})p(y_{t})p(m^i\mid y_{1:t-1})}{p(\neg m^i)p(y_{t}\mid y_{1:t-1})}$$

* Logit function $\text{logit}(p) = \log{\left(\frac{p}{1-p}\right)}$

$$\frac{p(m^i\mid y_{1:t})}{p(\neg m^i\mid y_{1:t})}= \frac{\frac{p(m^i \mid y_{t})p(y_{t})p(m^i\mid y_{1:t-1})}{p(m^i)p(y_{t}\mid y_{1:t-1})}}{\frac{p(\neg m^i \mid y_{t})p(y_{t})p(m^i\mid y_{1:t-1})}{p(\neg m^i)p(y_{t}\mid y_{1:t-1})}}$$

$$\frac{p(m^i\mid y_{1:t})}{p(\neg m^i\mid y_{1:t})}= \frac{p(m^i \mid y_{t})p(\neg m^i)p(y_{t})p(m^i\mid y_{1:t-1})}{p(\neg m^i \mid y_{t})p(m^i)p(\neg m^i\mid y_{1:t-1})}$$

Rewriting

$$\frac{p(m^i\mid y_{1:t})}{p(\neg m^i\mid y_{1:t})}= \frac{p(m^i \mid y_{t})(1-p(m^i))p(y_{t})p(m^i\mid y_{1:t-1})}{(1-p(m^i \mid y_{t}))p(m^i)(1-p(m^i\mid y_{1:t-1}))}$$

Applying logit

$$\text{logit}(p(m^i\mid y_{1:t}))= \text{logit}(p(m^i\mid y_{t}))+\text{logit}(p(m^i\mid y_{1:t-1}))-\text{logit}(p(m^i))$$

Shorthand version

$$l_{t,i}=\text{logit}(p(m^i\mid y_{t}))+l_{t-1,i}-l_{0,i}$$

Where:

* $p(m^i\mid y_{t})$: Inverse Measurement Model
* $l_{t-1,i}$: Previous belief
* $l_{0,i}$: Initial belief

### Advantages

* Numerically stable
* Computationally efficient

## Inverse Measurement Module

$$l_{t,i}=\text{logit}(p(m^i\mid y_{t}))+l_{t-1,i}-l_{0,i}$$

Remembering $p(m^i\mid y_{t})$: is the state of the occupancy grid given a measurement

So far we have only seen the following measurement model:

$$p(y_t\mid m^i)$$

Which describe the state of the occupancy grid given a measurement

Then a inverse measurement model is needed!

### Inverse Measurement Module — To be improved

Scanner bearing:

$$\phi^s = \begin{bmatrix}-\phi_{max}^s & \dots & \phi_{max}^s\end{bmatrix} \quad \phi_j^s\in\phi^s$$

Scanner ranges:
$$r^s = \begin{bmatrix}-r_{1}^s & \dots & r_{j}^s\end{bmatrix} \quad r_j^s\in r_{max}^s$$

## Inverse Measurement Module Model

![inverse measurement module](./Inverse%20Measurement%20Module.jpg)

![grid model](./grid%20model.jpg)

![model with probabilities](./model%20with%20probabilities.jpg)

![to be fixed](./to%20be%20fixed.jpg)

Relative range:

$$r^i=\sqrt{\left(m_x^i-x_{1,t} \right)^2+\left(m_y^i-x_{2,t} \right)^2}$$

Relative bearing:
$$\phi^i=\tan^{-1}\left(\frac{m_y^i-x_{2,t}}{m_x^i-x_{1,t}}\right)-x_{3,t}$$

Closest relative bearing:

$$k=\argmin\left(|\phi^i-\phi_{j}^s|\right)$$

![inverse model](./inverse%20model.jpg)

$\alpha$: defines the affected range for high probability

$\beta$: defines the affected angle for low and high probability

### Algorithm

* No information $$\text{if }r^i>\min\left(r_{max}^s\right) \quad \text{or} \quad |\phi^i-\phi_k^s|>\beta/2$$
* High probability $$\text{if }r_k^s<r_{max}^s \quad \text{and} \quad |r^i-r_k^s|>\alpha/2$$
* Low probability: $$\text{if }r_k^s<r_{max}^s$$

$$\begin{cases} \text{No information} & r^i>\min\left(r_{max}^s\right) \quad \text{or} \quad |\phi^i-\phi_k^s|>\beta/2\\
\text{High probability} & r_k^s<r_{max}^s \quad \text{and} \quad |r^i-r_k^s|>\alpha/2 \\
\text{Low probability} & r_k^s<r_{max}^s
\end{cases}$$

#### Example

![example](./example.jpg)

Example — red cells denote high probability of occupied, given measurement denoted by red x.

## Inverse Measurement Module With Ray Tracing

Ray tracing algorithm using Bresenham's line algorithm:

* Rasterized line algorithm
* Uses very cheap fixed point operations for
fast calculations

Perform update on each beam from the LIDAR rather then each cell on the grid

* Preforms far fewer updates (ignores no information zone)
* Much cheaper per operation
