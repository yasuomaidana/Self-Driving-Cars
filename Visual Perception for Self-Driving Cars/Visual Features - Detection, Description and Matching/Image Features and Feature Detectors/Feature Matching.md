# Feature matching

## Brute Force Feature Matching

1. Define a distance function $d(f_i,f_j)$ that compares two descriptors
2. Define distance threshold $\delta$
3. For every feature $f_i$ in image 1:
    * Compute $d(f_i,f_j)$ for each feature, f_i, in image 1, with all features $f_j$ in image 2
    * Find the closest match $f_c$, the match that has the minimum distance
    * Keep this match only if $d(f_i,f_j)$ is below threshold $\delta$

### Distance function

Sum of Squared Differences (SSD) $$d(f_i,f_j) = \sum_{k=1}^D(f_{i,k}-f_{j,k})^2$$

Other tistances functions

* Sum of absolute differences (SAD) $$d(f_i,f_j) = \sum_{k=1}^D|f_{i,k}-f_{j,k}|$$
* Hamming Distance $$d(f_i,f_j) = \sum_{k=1}^DXOR(f_{i,k}-f_{j,k})$$

### Considerations

* Brute force feature matching might not be fast enough for extremely large amounts of features
* Use a multidimensional search tree, usually a k-d tree to speed the search by constraining it spatially
* Both of these matchers are implemented in OpenCV ([tutorial](https://docs.opencv.org/4.0.0/dc/dc3/tutorial_py_matcher.html)) as:
  * cv2.BFMatcher(): Brute force matcher
  * cv2.FlannBasedMatcher(): K-D tree based approximate nearest neighbor matcher

## Distance ratio

* Compute $d(f_i,f_j)$ for each feature, f_i, in image 1, with all features $f_j$ in image 2
* Find the closest match $f_c$
* Find the second closest match $f_s$
* Find how better the closest match is than the second closest match. This can be done through distance ratio: $$0 \leq \frac{d(f_i,f_c)}{d(f_i,f_s)}\leq1$$

## Brute Force Feature Updated

1. Define a distance function $d(f_i,f_j)$ that compares two descriptors
2. Define distance threshold $\rho$
3. For every feature $f_i$ in image 1:
    1. Compute $d(f_i,f_j)$ with all features $f_j$ in image 2
    2. Find the closest match $f_c$ and the second closest match $f_s$
    3. Compute the distancee distance ratio: $0 \leq \frac{d(f_i,f_c)}{d(f_i,f_s)}\leq1$
    4. Keep matches with distance ratio $<\rho$

## Additional material

Feature Matching + Homography to find Objects ([tutorial](https://docs.opencv.org/4.0.0/d1/de0/tutorial_py_feature_homography.html))
