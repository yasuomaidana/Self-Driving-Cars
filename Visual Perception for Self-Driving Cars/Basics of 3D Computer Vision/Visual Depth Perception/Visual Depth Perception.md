# Visual Depth Perception

## Stereo Camera Model

![stereo camera model](./stereo%20camera%20model.jpg)

Assumptions:

1. The two cameras used to construct the stereo sensors are identical.
2. We will assume that while manufacturing the stereo sensor, we tried as hard as possible to keep the two cameras optical axes aligned. (The two cameras have parallel optical axes)
3. Project to Bird's eye view for easier geometry

The focal length $\bm{f}$ is, once again, the distance between the camera center and the image plane

 The baseline $\bm{b}$ is defined as the distance along the shared x-axis between the left and right camera centers. By defining a baseline to represent the transformation between the two camera coordinate frames, we are assuming that the rotation $\bm{R}$ matrix is identity and there is only a non-zero x component in the translation vector $\bm{t}$.

> The $\bm{R}$ and $\bm{t}$ transformation therefore boils down to a single baseline parameter $\bm{b}$

![stereo model simplified](./stereo%20camera%20model%20simplified.jpg)

## Computing 3D Point Coordinates

![stereo solution](./stereo%20camera%20solution.jpg)

Disparity: $$d=x_L-x_R$$
Where:$$x_L=u_L-u_0 \\ x_R=u_R-u_0 \\ y_L=v_L-v_0$$
$$\frac{Z}{f}=\frac{X}{x_L} \rightarrow Zx_L=fX$$
$$\frac{Z}{f}=\frac{X-b}{x_R} \rightarrow Zx_R=fX-fb$$
$$Zx_R=Zx_L-fb$$
$$Z=\frac{fb}{d}$$
$$X=\frac{Zx_L}{f}$$
$$Y=\frac{Zy_L}{f}$$

We need to know $f$, $b$, $u_0$, $v_0$

* Use stereo camera calibration

We need to find corresponding $x_R$ for each $x_L$

* Use disparity computation algorithms

## Epipolar Constraint for Correspondence

![epipolar line](./epipolar%20line.jpg)

Let's move our 3D point along the line connecting it with the left cameras center. Its projection on the left camera image plane does not change.

However, what can you notice about the projection on the right camera plane, the projection moves along the horizontal line. This is called an epipolar line and follows directly from the fixed lateral offset and image plane alignment of the two cameras in a stereo pair.

We can constrain our correspondence search to be along the epipolar line, reducing the search from 2D to 1D.

### Non-ParaIIeI Optical Axes

![multiview](./multi-view%20geometry.jpg)

Horizontal epipolar lines only occur when the optical axes of the two cameras are parallel.

If this condition is not met, epipolar lines will be skewed (Known as multi-view geometry).

### Disparity computarion

We can use stereo rectification to warp images originating from two cameras with
non-parallel optical axes to force epipolar lines to be horizontal.

## Additional resources

* Forsyth, D. A. and J. Ponce. (2003). Computer vision: a modern approach (2nd edition). New Jersey: Pearson. Read sections 11.1, 12.1, 12.2.

* Szeliski, R. (2010). [Computer vision: algorithms and applications](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf). Springer Science & Business Media. Read sections 11.1

* Hartley, R., & Zisserman, A. (2003). Multiple view geometry in computer vision. Cambridge university press. Read sections 1. 9.1, 10.1, 11.12

* [Epipolar Geometry (OpenCV)](https://docs.opencv.org/3.4.3/da/de9/tutorial_py_epipolar_geometry.html)

* [Depth Map from Stereo Images (OpenCV)](https://docs.opencv.org/3.4.3/dd/d53/tutorial_py_depthmap.html)
