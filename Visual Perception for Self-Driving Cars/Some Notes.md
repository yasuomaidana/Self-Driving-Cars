# Some notes

## Camera Sensor Coordinate System

![Camera Sensor Coordinate System](./Camera%20Sensor%20Coordinate%20System.jpg)

## Plane Estimation

* compute_plane($xyz$):
  * Custom function provided to compute a plane estimate using SVD
* numpy.linalg.lstsq(A, b):
  * Solves the equation Ax = b by minimizing the Euclidean 2-norm
  * Be careful of minimum number of points required when randomly choosing points for RANSAC
  * Might fail if chosen points result in a poorly conditioned system
* numpy.linalg.svd(A):
  * Better numerical stability, and reliability over normal equations
  * SVD does not provide the final solution, you will need to manipulate the results yourself

## Lane Clustering, Merging and Filtering

Filtering:

* Remove any line with slope in the range $[0, \eta]$
* Try $\eta$ in the range of $0.1\sim0.3$ for best results

Clustering:

* Choose cluster center at random from the output of the Hough transform line detector
* A line belongs to this cluster if it has a slope and intercept that are close to the those of the cluster center

* Slope and Intercept closeness is determined if the
difference with the cluster's center is less than a threshold
* Slope difference threshold: $[0.1 \sim 0.3]$
* Intercept difference threshold: $[20 \sim 50]$

Lane Merging:

* For each cluster, compute the average slope and intercept

## Filtering Uncertain Output Of Object Detectors

* Count number of pixels belonging to the car category from the output of semantic segmentation
* Normalize the computed count by area to remove dependency on bounding box size
* Equivalent to computing the area inside the bounding box occupied by pixels belonging to the required category

How to filter out uncertain output of object detectors using the output from the semantic segmentation?

The output of object detection is usually reliable. But sometimes, you are given a high recall low precision detector that detects all objects in the scene, but also provides some false positives.

You are required to use the output from semantic segmentation to eliminate these false positives before estimating the distance to the obstacles. The results should be bounding boxes that reliably contain obstacles.

To perform this filtering, you will need to use the semantic segmentation output to count the number of pixels in the bounding box that have the same category as the classification output from the 2D object detector.

The trick here, is that this number will depend on the size of the bounding box. You will need to normalize the pixel count by the area of the bounding box before attempting to filter out the detections with a threshold.

The final normalized count is equivalent to computing the area inside the bounding box occupied by pixels belonging to the correct category. At this point, you should be ready to tackle the final assessment confidently.
