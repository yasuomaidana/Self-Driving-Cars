# Neural Network Training with Gradient Descent

## Neural Network Loss Functions

Thousands of training example pairs $[\bm{x},f^*(\bm{x})]$

The Loss function computed over all $\bm{N}$ training examples is termed the Training Loss and can be written as: $$J(\theta)=\frac{1}{N}\sum_{i=1}^NL[f(x_i,\bm{\theta}),f^*(x_i)]$$

The gradient of the training loss with respect to the parameters $\bm{\theta}$ can be written as:
$$\nabla_{\bm{\theta}}J(\theta)=\nabla_{\bm{\delta}}[\frac{1}{N}\sum_{i=1}^NL[f(x_i,\bm{\theta}),f^*(x_i)]=\frac{1}{N}\sum_{i=1}^N\nabla_{\bm{\delta}}L[f(x_i,\bm{\theta}),f^*(x_i)$$

## Batch Gradient Descent

Batch Gradient Descent is an iterative first order optimization procedure

### Algorithm

1. Initialize parameters $\bm{\theta}$
2. While Stopping Condition is Not Met:
   1. Compute gradient of loss function over all training examples:$$\nabla_{\bm{\theta}}J(\theta)=\frac{1}{N}\sum_{i=1}^N\nabla_{\bm{\delta}}L[f(x_i,\bm{\theta}),f^*(x_i)$$
   2. Update parameters according to: $$\bm{\theta}\leftarrow\bm{\theta}-\epsilon\nabla_{\bm{\theta}}J(\theta)$$

### Parameter Initialization and Stopping Conditions

Parameter Initialization:

* Weights: initialized by randomly sampling from a standard normal distribution
* Biases: initialized to 0
* Other heuristics exist

Stopping Conditions:

* Number of iterations: How many training iterations the neural network has performed
* Change In $\bm{\theta}$ value: Stop if $\bm{\theta}_{\text{new}} - \bm{\theta}_{\text{old}} <$ Threshold
* Change In $J(\bm{\theta})$ value: Stop if $J(\bm{\theta}_{\text{new}}) - J(\bm{\theta}_{\text{old}}) <$ Threshold

### Drawbacks

**Backpropagation** used to compute $\nabla_{\bm{\theta}}J(\theta)$ is very **expensive** to compute over the whole training dataset.

Standard error of the mean estimated from $N$ samples is $\frac{\sigma}{\sqrt{N}}$ where $\sigma$ is the standard deviation of the value of the samples

Using all samples to estimate the gradient results in less than linear return in accuracy of this estimate

Use a small subsample (Minibatch) of the training data to estimate the gradient

## Stochastic (minibatch) Gradient Descent

1. Initialize parameters $\bm{\theta}$
2. While Stopping Condition is Not Met:
   1. Sample a preset number $N'$ of examples (minibatch) from the training data
   2. Compute gradient of loss function over all training examples:$$\nabla_{\bm{\theta}}J(\theta)=\frac{1}{N'}\sum_{i=1}^{N'}\nabla_{\bm{\delta}}L[f(x_i,\bm{\theta}),f^*(x_i)$$
   3. Update parameters according to: $$\bm{\theta}\leftarrow\bm{\theta}-\epsilon\nabla_{\bm{\theta}}J(\theta)$$

### What Minibatch Size To Use ?

GPUs work better with powers of 2 batch sizes

Large batch sizes > 256:

* Hardware underutilized with very small batch sizes.
* More accurate estimate of the gradient, but with less than linear returns

Small batch size < 64

* Small batches can offer a regularizing effect. The best
generalization error is often achieved with batch size of 1.

Always make sure dataset is shuffled before sampling
minibatch

## SGD Variations

* Momentum SGD, Nestrove Momentum SGD
* Ada-Grad, RMS-Prop
* ADAM (Adaptive Moment Estimation)

> ADAM: Implemented in most deep neural network libraries, fairly robust to the choice of the learning rate and other
hyperparameters
