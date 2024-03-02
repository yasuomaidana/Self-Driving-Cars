# Feed Forward Neural Networks

A Feedforward Neural Network defines a mapping
from input $x$ to output $y$ as:
$$y = f(x;\theta)$$
Where $\theta$ is the learned parameters and $x$ is our input data

An $N$ layer FNN is represented as the function
composition:
$$f(x;\theta) = f^{N}\left( f^{N-1}\left(\dots f^{2}\left( f^{1}\left( x \right)\right) \right) \right)$$

## Example

![Feed forward neural networks](https://learnopencv.com/wp-content/uploads/2017/10/mlp-diagram.jpg)

### Functions to estimate for Self Driving Cars

* Object Classification: Image->Label
* Object Detection: Image->Label+Location
* Depth Estimation: Image-> Depth for every pixel
* Semantic Segmentation: Image->Label for every pixel

## Mode Of Action Of Neural Networks

* **Training**: Give neural network **examples of** $f^{*} (\bm{x})$. for a wide variation of the input $\bm{x}$. Then, optimize its parameters $\bm{\theta}$ to force $f(\bm{x}; \bm{\theta}) \cong f^{*} (\bm{x})$
* Pairs of $\bm{x}$ and $f^{*} (\bm{x})$ are called **training data**
* Only **output** is specified by training data! Network is
free to do anything with its **hidden layers**

## Hidden units

$$h_n = g\left(\bm{W}^Th_{n-1} + \bm{b} \right)$$

* Activation function $g$
* Input $h_{n-1}$
* Weight matrix $\bm{W}$
* Bias $\bm{b}$
* Parameters $\bm{\theta}$ are the weights and biases of all the layers of the network
* Transformed parameters passed through activation function $g$

> $g$ is a non linear function

### Additional Resources

* Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep Learning (Vol. 1)](https://www.deeplearningbook.org/). Cambridge: MIT press. Read sections 6.1, 6.3.
