# Image Features and Feature Detectors

## Feature detection

Features are points of interest in an image

**Points of interest** should have the following characteristics:

* Saliency: distinctive, identifiable, and different from its immediate neighborhood
* Repeatability: can be found in multiple images using same operations
* Locality: occupies a relatively small subset of image space
* Quantity: enough points represented In the image
* Efficiency: reasonable computation time

## Feature Extraction

* Repetitive texture less patches are challenging to
detect consistently

* Patches with large contrast changes (gradients) are
easier to detect (edges)

* Gradients in at least two (significantly) different
orientations are the easiest to detect (corners)

## Feature Detection: Algorithms

* Harris {corners}: Easy to compute, but not scale invariant. [Harris and Stephens, 1988]
* Harris-Laplace {corners}: Same procedure as Harris detector, addition of scale selection based on Laplacian. Scale invariance. [Mikolajczyk, 2001]
* Features from accelerated segment test (FAST) {corners}: Machine learning approach for fast corner detection. [Rosten and Drummond, 2006]
* Laplacian of Gaussian (LOG) detector (blobs): Uses the concept of scale space in a large neighborhood (blob). Somewhat scale invariant. [Lindeberg, 1998]
* Difference of Gaussian (DOG) detector {blobs}: Approximates LOG but is faster to compute. [Lowe, 2004]
