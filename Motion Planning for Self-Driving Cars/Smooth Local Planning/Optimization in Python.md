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
def objective_function (x):
    return x[0]**2 + 4*x[0]*x[1]
def objective_jacobian(x):
    return [2*x[0]** + 4*x[1], 4*x[0]]
```
