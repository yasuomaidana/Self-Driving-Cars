# Convolutional Neural Networks

Used for processing data defined on grid
ID time series data, 2D images, 3D videos
Two major type of layers:

* Convolution Layers
* Pooling Layers

## Fully Connected vs Convolutional Layers

Fully Connected: $$h_n=g(\bm{W}^Th_{n-1}+b)$$

Although counter-intuitive, convolutional layers use cross-correlation not convolutions ([see](#convolution-vs-cross-correlation)) for their linear operator instead of general matrix multiplication.
$$h_n=g(\bm{W}*Th_{n-1}+b)$$

## Cross Correlation

<img alt="Cross Correlation" src="./cross correlation.png" height="300" style="display: block; margin: 0 auto"/>

* Width (3): horizontal dimension of input volume
* Height (3): vertical dimension of input volume
* Depth(3): number of channels of input volume
* Padding size(1): essential to retain shape
* Stride: The number of pixels the filter is moved in the vertical and horizontal direction.

### Output Volume Shape

* Filters are size $w\times h$
* Number of filters = $K$
* Stride = $S$
* Padding = $P$

$$\bm{W}_{\text{out}}=\frac{\bm{W}_{\text{in}}-w+2\times P}{S}+1$$

$$\bm{H}_{\text{out}}=\frac{\bm{H}_{\text{in}}-h+2\times P}{S}+1$$

$$\bm{D}_{\text{out}}=K$$

## Convolution vs Cross-Correlation

Here's why convolutional neural networks (CNNs) use cross-correlation and not strict mathematical convolution:

1. Computational Efficiency: Flipping the kernel in convolution is computationally expensive. Implementing true convolution requires flipping the kernel (filter) before sliding it over an image. This flipping adds overhead without affecting the core results of the operation.
2. Pattern Learning: The goal of CNNs is to learn patterns, regardless of the kernel's orientation CNNs are designed to identify patterns within images. The filters (kernels) used in a CNN learn to detect these patterns whether the kernel is mathematically flipped or not. The learned weights in the filter would simply adjust to accommodate the lack of flipping.
3. Terminology: "Convolution" has become a standard term in deep learning. Even though mathematically, the operation is more accurately described as cross-correlation, the term "convolution" has stuck in the field of deep learning.

Here's a breakdown of convolution vs. cross-correlation:

* Convolution: Involves flipping the kernel (horizontally and vertically) before performing the sliding and multiplication operation. More formally used in signal processing and image processing.
* Cross-correlation: Simply slides the kernel across the input without flipping it. The operation commonly used in convolutional neural networks. Let me know if you'd like a practical example illustrating the difference between the two!

Here's a simple example to illustrate the difference between convolution and cross-correlation:

**Scenario**: Imagine we have a 1D signal `[1, 2, 3]` and a small filter (kernel) `[2, 1]`. We want to find the "overlap" between the signal and the filter at different positions.

**Convolution**:

1. Flip the kernel: The flipped version of `[2, 1]` is `[1, 2]`.
2. Slide and multiply:</br>
Position 1: `[1, 2, 3] * [1, 2] = [1 * 1 + 2 * 2 + 3 * 0] = 5`</br>
Position 2: `[1, 2, 3] * [0, 1] = [1 * 0 + 2 * 1 + 3 * 2] = 7`

**Cross-correlation**:

1. No flipping needed: We use the original kernel `[2, 1]`.
2. Slide and multiply:</br>
Position 1: `[1, 2, 3] * [2, 1] = [1 * 2 + 2 * 1 + 3 * 0] = 4`</br>
Position 2: `[1, 2, 3] * [0, 2] = [1 * 0 + 2 * 2 + 3 * 1] = 7`

As you can see, both convolution and cross-correlation produce the same "peak" at position 2, indicating the strongest overlap between the signal and the filter. However, their intermediate values at position 1 differ due to the flipped kernel in convolution.

In the context of CNNs:

* We are primarily interested in identifying the strength of the pattern (filter) within the input (image).
* Flipping the kernel doesn't affect the final detection of the pattern, hence cross-correlation is computationally more efficient and commonly used in CNNs.

## Additional Material

Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep learning (Vol. 1)](https://www.deeplearningbook.org/). Cambridge: MIT press. Read sections 9.1-9.3
