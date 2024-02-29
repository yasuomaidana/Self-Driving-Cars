# Output Layers and Loss Functions

## Types of Tasks

* Classification: Given input $\bm{x}$ map it to one of $\bm{k}$ classes or categories.
  * Image classification, semantic segmentation
* Regression: Given input $\bm{x}$ map it to a real number
  * Depth prediction, bounding box estimation

## Classification

### Softmax Output Layers

Softmax output layers are most often used as the output of a classifier, to represent the probability distribution over $K$ different classes

The Softmax output layer is comprised of:

* A linear transformation: $$z = WTh+b$$
* Followed by the Softmax function: $$\text{Softmax}(z_i)=\frac{e^{z_i}}{\sum_je^{z_j}}$$

### Cross-Entropy Loss Function

By considering the output of the softmax output layer as a probability distribution, the Cross Entropy Loss function is derived using maximum likelihood as:
$$L(\theta)=-\log(\text{Softmax}(z_i))=-z_i+\log\sum_je^{z_j}$$

## Regression

### Linear Output Layers

Linear Output Units are based only on an affine transformation with no non-linearity
$$z=W^Th+b$$
Linear Output Units are usually used with the Mean
Squared Error loss function to model the mean of a
probability distribution:$$L(\theta)=\sum_i(z_i-f^*(x_i))^2$$

## Additional material

Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep Learning (Vol. 1)](https://www.deeplearningbook.org/). Cambridge: MIT press. Read sections 6.2, 6.4.
