# Optimization in Python

## Minimize Function

```python linenums
result = minimize(objective_function, x_0, method='L-BFGS-B', jac=jacobian, constraints=bounds, options={'disp':True})
```

* Many optimization algorithms available
  * Specific one chosen depends on "method" parameter
* Model Jacobian passed in through "jac"
* Model constraints passed in through "constraints"
  * Different forms of constraints available
* $x_0$ gives initial guess for optimizer

## Objective Function and Jacobian

* BFGS requires objective function as input, as well as a function to evaluate the Jacobian
* These functions will take a vector of the optimization variables as input

```python linenums
def objective_function(x):
    return x[0]**2 + 4*x[0]*x[1]

def objective_jacobian(x):
    return [2*x[0] + 4*x[1], 4*x[0]]
```

## Result

```python
result = minimize(objective_function, x_0, method='L-BFGS-B', jac=jacobian,constraints=bounds, options={'disp':True})

print(result.x)
```

* Upon completion, optimization returns a result variable
* The *"x"* member variable gives the final vector of optimized variables where the local minimum has been reached

## Bounds

```python
bounds = [[-10.0,5.0],[-3.0,4.0]]
```

$$-10\leq x_0 \leq 5\\-3\leq x_1 \leq 4$$

* Simplest constraints are inequality constraints on optimization variables, denoted as "bounds"
* The $i^{th}$ sub-list of the list denotes the upper and lower bounds for the $i^{th}$ optimization variable
* Bounds are passed to "constraints" parameter in minimize function

### Other Constraints

```python
linear_constraints = LinearConstraints([[1,2],[2,1]],[-5,1],[2,4])
```

$$\begin{bmatrix} -5\\1\end{bmatrix} \leq \begin{bmatrix}1 &2\\ 2&1 \end{bmatrix} \begin{bmatrix}x_0\\x_1\end{bmatrix} \leq \begin{bmatrix}2\\4\end{bmatrix}$$

* Can also pass linear constraints and nonlinear constraints to optimizer depending on optimization algorithm
  * More details in SciPy documentation
* Can also combine different constraint methods in a list of constraints
