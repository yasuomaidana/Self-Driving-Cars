# Semantic Segmentation for Road Scene Understanding

## 3D Drivable Surface Estimation

1. Generate semantic segmentation output
2. Associate 3D point coordinates with 2D image pixels
3. Choose 3D points belonging to the Drivable Surface category
4. Estimate 3D drivable surface model

Plane Model: $$ax+by+z=d$$
Least squares formulation:
$$\bm{p}=\begin{bmatrix}a,b,d\end{bmatrix}, \quad \argmin_A\left(\bm{A}\bm{p}-\bm{B}\right)\\
\bm{A}=\begin{bmatrix}x_1&y_1&-1\\
x_2&y_2&-1\\
\vdots&\vdots&\vdots
\\x_N&y_N&-1\end{bmatrix}, \quad
\bm{B}=\begin{bmatrix}-z_1\\z_2\\
\vdots\\ -z_N\end{bmatrix}$$

Solution:$$p=\left(\bm{A}^T\bm{A}\right)^{-1}\bm{A}^T\bm{B}$$

Minimum number of points to estimatel model: **3 non-collinear points**

Ransac Algorithm:

1. From your data, randomly select 3 points.
2. Compute model parameters $a$, $b$, and $d$ using least squares estimation.
3. Compute number of inliers, $N$.
4. If $N$ > threshold, terminate and return the computed plane parameters. Else, go back to step 1.
5. Recompute the model parameter using all the inliers in the inlier set.

## Semantic Lane Estimation

Estimate the lane, the area where the car can drive on the drivable surface

Estimate what is at the boundaries of the lane:
* Curb
* Road
* Car

1. Extract segmentation mask from pixels belonging to lane separators such as lane markings or curbs.
2. Extract edges from this segmentation mask using an edge detector.
3. Linear Lane Model: Use the Hough transform to detect lines in the output edge map.
4. Filter lines based on slope to remove horizontal lines.
5. Remove any line that does not belong to the drivable space.
6. Determine which classes occur at the boundary of the lane.

## Additional resources

* <https://www.mapillary.com/dataset/vistas>
* [Hough Transform Line Detection](https://docs.opencv.org/4.9.0/d9/db0/tutorial_hough_lines.html)
* [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)
* Forsyth, D.A. and J. Ponce (2003). Computer Vision: a modern approach (2nd edition). New Jersey: Pearson. Read section 8.1, 8.2, 8.3 (Edge Detection); 16.1, 16.2 (Hough transform)
