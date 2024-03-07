# ConvNets for Semantic Segmentation

![modelv2](./modelv2.jpg)
![modelv2](./modelv3.jpg)
![encoder decoder](./encoder%20decoder.jpg)

## Feature extractor

![vgg model](../2D%20Object%20Detection/VGG%20feature%20extractor.jpg)

## Upsampling Layer

![upsampling](./upsampling.jpg)

Upsampling Multiplier $\bm{S}$

$$\bm{W}_{\text{out}}=S\times \bm{W_{\text{in}}}$$
$$\bm{H}_{\text{out}}=S\times \bm{H_{\text{in}}}$$
$$\bm{D}_{\text{out}}=\bm{D}_{\text{in}}$$

## Learning Same Resolution Feature Maps

![Same Resolution Feature Maps](./Same%20Resolution%20Feature%20Maps.jpg)

## The Feature Decoder

![The Feature Decoder](./The%20Feature%20Decoder.jpg)

## Output Representation and Loss

![Output Representation](./Output%20Representation.jpg)

$$L_{\text{cls}} = \frac{1}{N_{\text{total}}}\sum_i\text{CrossEntropy}(s_i^*,s_i)$$

* $N_{\text{total}}$ is the number of pixels in all images of our minibatch
* $s_i$ is the output of the neural network
* $s_i^*$ is the ground truth classification

## Additional Resources

* Badrinarayanan, V., Kendall, A., & Cipolla, R. (2015). Segnet: A deep convolutional encoder-decoder architecture for image segmentation. arXiv preprint arXiv:1511.00561.

* Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017, July). Pyramid scene parsing network. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) (pp. 2881-2890). (State of the art)
