# Visual Odometry

Visual Odometry (VO): is the process of incrementally estimating the pose of the vehicle by examining the changes that motion induces on the images of its onboard cameras

Pros:

* Not affected by wheel slip in uneven terrain, rainy/snowy weather, or other adverse conditions.
* More accurate trajectory estimates compared to wheel odometry.

Cons:

* Usually need an external sensor to estimate absolute scale
* Camera is a passive sensor, might not be very robust against weather conditions and illumination changes
* Any form of odometry (incremental state estimation) drifts over time, as seen in Course 2

## Problem Formulation

Estimate the camera motion $T_k$ between consecutive
images $I_{k-1}$ and  $I_k$
$$T_k=\begin{bmatrix} R_{k,k-1} & t_{k,k-1}\\ 0 & 1\end{bmatrix}$$
Concatenating these single movements allows the
recovery of the full trajectory of the camera, given
frames $C_1,\dots,\dots, C_m$

![visual odometry problem](./visual%20odometry%20problem.jpg)

## Implementation

* Given: two consecutive image frames $I_{k-1}$ and $I_k$
* Extract and match features $f_{k-1}$ and $f_k$ between
two frames $I_{k-1}$ and $I_k$
* Estimate motion between frames to get $T_k$

### Motion Estimation

* Correspondence types:
  * 2D-2D: both $f_{k-1}$ and $f_k$ are defined in Image coordiantes.
  * 3D-3D: both $f_{k-1}$ and $f_k$ are specified in 3D
  * 3D-2D: $f_{k-1}$ is specified in 3D and $f_k$ are their corresponding projection on 2D

#### 3D-2D motion estimation


![3d-2d motion estimation](./3d-2d%20motion%20estimation.jpg)

Given:

* 3D world coordinates features in frame $k-1$
* 2D image coordinates in frame $k$

Camera Projection:
$$\begin{bmatrix} su\\ sv \\ s\end{bmatrix} = K[R|t]\begin{bmatrix} X \\ Y\\ Z\\ 1 \end{bmatrix}$$

Where $K$ is known from calibration
Estimate $[\bm{R}|\bm{t}]$

##### Perspective N point PNP

Camera projection:

$$\begin{bmatrix} su\\ sv \\ s\end{bmatrix} = K[R|t]\begin{bmatrix} X \\ Y\\ Z\\ 1 \end{bmatrix}\forall f_i$$

PnP:

* Solve for initial guess of $[\bm{R}|\bm{t}]$ using direct linear transform (DLT)
  * Forms a linear model and solves $[\bm{R}|\bm{t}]$, with methods such as singluar value decomposition (SVD)
* Improve solution using Levenberg-Marquardt algorithm (LM)
* Need at least 3 points to solve (P3P), 4 if we don't want
ambiguous solutions.

###### Implementations in OpenCV

* [`cv2.solvePnP()`](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d): Solves for camera position given 3D
points in frame $k-1$, their projection in frame k, and the
camera intrinsic calibration matrix
* [`cv2.solvePnPRansac()`](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html#ga50620f0e26e02caa2e9adc07b5fbf24e): Same as above, but uses
RANSAC to handle outliers
