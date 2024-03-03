# The Object Detection Problem

## Challenges

* Extent of objects is not fully observed!
  * Occlusion: Background objects covered by foreground objects
  * Truncation: Objects are out of image boundaries
* Scale: Object size gets smaller as the object moves farther away
* Illumination Changes:
  * Too bright
  * Too dark

## Mathematical Problem Formulation

![Formulation](./Mathematic%20Formulation.png)

$$f(\bm{x};\bm{\theta})=[x_{\text{min}},y_{\text{min}},x_{\text{max}},y_{\text{max}},S_{\text{class}_1},\dots,S_{\text{class}_k}]$$

## Evaluation Metrics

* Intersection-Over-Union (IOU): area of intersection of predicted box with a ground truth box, divided by the area of their union
* True Positive (TP): Object class score > score threshold, and IOU > IOU threshold
* False Positive (FP): Object class score > score threshold, and IOU < IOU threshold
* False Negative (FN): Number of ground truth objects not detected by the algorithm
* Precision: $\frac{TP}{TP + FP}$
* Recall: $\frac{TP}{TP + FN}$
* Precision Recall Curve (PR-Curve): Use multiple object class
score thresholds to compute precision and recall. Plot the
values with precision on y-axis, and recall on x-axis
* Average Precision (AP): Area under PR-Curve for a single
class. Usually approximated using 11 recall points

![average](./Average%20Precision.jpg)
We then proceed to plot the precision-recall curve, using the precision values on the y-axis and the recall values on the x-axis. Note that we also add the precision recall points of one and zero as the first in the plot, and zero one as the final point in the plot. This allows us to approximate the average precision by calculating the area under the P-R curve using 11 recall points between zero and one, at 0.01 recall increments.

## Resource

[Implementation Resources](https://github.com/tensorflow/models/tree/master/research/object_detection) (Fully implemented models ready to be used, from Google team)
