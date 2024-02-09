# Camera Calibration

## Computing the projection

World to camera
$$o_{\text{image}}=PO=K[R|t]O_{\text{world}}$$
![world to camera](../The%20Camera%20Sensor/World%20to%20image%20(simplified%20camera).jpg)

World coordinates to Image coordinates
$$o_{\text{image}}=\begin{bmatrix}x\\ y\\ z\end{bmatrix}=K[R|t]\begin{bmatrix}X\\ Y\\ Z\\ 1\end{bmatrix}$$
Image coordinates to Pixel coordinates
$$\begin{bmatrix}x\\ y\\ z\end{bmatrix}\rightarrow \begin{bmatrix}u\\ v\\ 1\end{bmatrix} =\frac{1}{z} \begin{bmatrix}x\\ y\\ z\end{bmatrix}=\begin{bmatrix}su\\ sv\\ s\end{bmatrix}$$

## Problem formulation

World coordinates and Camera coordinates known
$$o=PO=K[R|t]O_{\text{world}} = \begin{bmatrix}su\\ sv\\ s\end{bmatrix} = \begin{bmatrix}?&?&?&?\\
?&?&?&?\\
?&?&?&?
\end{bmatrix}\begin{bmatrix}X\\ Y\\ Z\\ 1\end{bmatrix}$$

Use scenes with known geometry to:
* Correspond 2D image coordinates to 3D world coordinates
* Find the Least Squares Solution (or non-linear solution) of the parameters of $\bm{P}$
$$\begin{bmatrix}su\\ sv\\ s\end{bmatrix} = \begin{bmatrix}p_{11}&p_{12}&p_{13}&p_{14}\\
p_{21}&p_{22}&p_{23}&p_{24}\\
p_{31}&p_{32}&p_{33}&p_{34}
\end{bmatrix}\begin{bmatrix}X\\ Y\\ Z\\ 1\end{bmatrix}$$

### Solution
1. Expanding the equation results in:$$
su=p_{11}X+p_{12}Y+p_{13}Z+p_{14} \quad(1)\\
sv=p_{21}X+p_{22}Y+p_{23}Z+p_{24} \quad(2)\\
s=p_{31}X+p_{32}Y+p_{33}Z+p_{34} \quad(3)$$
2. Move to LHS (left hand side)$$
su-p_{11}X-p_{12}Y-p_{13}Z-p_{14}=0\\
sv-p_{21}X-p_{22}Y-p_{23}Z-p_{24}=0\\
s-p_{31}X-p_{32}Y-p_{33}Z-p_{34}=0$$
3. Replace Eq. (3) in Eq. (1) and Eq. (2) to get 2 equations per point:
$$p_{31}Xu+p_{32}Yu+p_{33}Zu+p_{34}u
-p_{11}X-p_{12}Y-p_{13}Z-p_{14}=0 \\
p_{31}Xu+p_{32}Yu+p_{33}Zu+p_{34}u
-p_{21}X-p_{22}Y-p_{23}Z-p_{24}=0$$

If we have N 3D points and their corresponding N 2D projections,
set up homogeneous linear system
Solved with Singular Value Decomposition (SVD)
$$\begin{bmatrix}
X_1&Y_1&Z_1&1&0&0&0&0&-u_1X_1&-u_1Y_1&-u_1Z_1&-u_1 \\
0&0&0&0&X_1&Y_1&Z_1&1&-v_1X_1&-v_1Y_1&-v_1Z_1&-v_1 \\
\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots&\vdots\\
X_N&Y_N&Z_N&1&0&0&0&0&-u_NX_N&-u_NY_N&-u_NZ_N&-u_N \\
0&0&0&0&X_N&Y_N&Z_N&1&-v_NX_N&-v_NY_N&-v_NZ_N&-v_N
\end{bmatrix}\begin{bmatrix}
p_{11}\\ p_{12} \\ p_{13} \\ p_{14} \\
p_{21}\\ p_{22} \\ p_{23} \\ p_{24} \\
p_{31}\\ p_{32} \\ p_{33} \\ p_{34}
\end{bmatrix}=0$$

### Advantages disadvantages
|Advantages|Disadvantages|
|-|-|
|Easy to formulate|Does not directly provide camera parameters|
|Closed form solution|Does not model radial distortion and other complex phenomena|
| | Does not allow for constraints such as known focal length tobe imposed|

## Factoring the P matrix

$$P = K[R|t]\\=K[R|-RC]\\=[KR|-KRC]$$

$C$ is the center of the camera such that $PC=0$

Let $M=KR$
$$P=[M|-MC]\\ M=\mathcal{R}Q = KR$$

$R$ and $Q$ are $[3\times3]$ matrices

$\mathcal{R}$ is not the rotation matrix $R$

Intrinsic Calibration Matrix: $K=\mathcal{R}$

Rotation Matrix: $R=Q$

Translation vector: $t=-K^{-1}P[:,4] = -K^{-1}MC$

## Additional material

* Forsyth, D. A. and J. Ponce. (2003). Computer vision: a modern approach (2nd edition). New Jersey: Pearson. Read sections 5.3.

* Szeliski, R. (2010). [Computer vision: algorithms and applications. Springer Science & Business Media.](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf) Read sections 6.1, 6.2. 6.3

* Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision. Cambridge university press. Read sections 7.1, 7.2, 7.4, 8.4, 8.5

* [Camera Calibration with OpenCV](https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html)
