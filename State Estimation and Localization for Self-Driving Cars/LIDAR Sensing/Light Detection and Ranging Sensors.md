# Light Detection and Ranging Sensors

## Measuring Distance with Time-of-Flight

![Measuring Distance with Time-of-Flight](./Distance%20with%20Time%20of%20Flight.jpg)

## Measurement Models for 3D LIDAR Sensors

3D LIDAR sensors report range, azimuth angle and elevation angle (+ return intensity)

![3D LIDAR Measuring](./3D%20LIDAR%20Measuring.jpg)

Inverse Sensor Model
$$\begin{bmatrix}x\\y\\z\end{bmatrix}=\bm{h}^{-1}\left(r,\alpha,\epsilon \right)=\begin{bmatrix}r\cos\alpha\cos\epsilon \\
r\sin\alpha\cos\epsilon \\
r\sin\epsilon
\end{bmatrix}$$

Forwards Sensor Model

$$\begin{bmatrix}r\\ \alpha \\ \epsilon\end{bmatrix}=\bm{h}\left(x,y,z\right)=\begin{bmatrix} \sqrt{x^2+y^2+z^2} \\
\tan^{-1}\left(\frac{y}{x}\right) \\
\sin^{-1}\left(\frac{z}{\sqrt{x^2+y^2+z^2}}\right)
\end{bmatrix}$$

> For 2D Environments $z$ and $\epsilon$ are $0$

## Sources of Measurement Noise

* Uncertainty in determining the exact time of arrival of the reflected signal
* Uncertainty in measuring the exact orientation of the mirror
* Interaction with the target (surface absorption, specular reflection, etc.)
* Variation of propagation speed (e.g., through materials)

Forwards Sensor Model (with Noise)

$$\begin{bmatrix}r\\ \alpha \\ \epsilon\end{bmatrix}=\bm{h}\left(x,y,z,\bm{v}\right)=\begin{bmatrix} \sqrt{x^2+y^2+z^2} \\
\tan^{-1}\left(\frac{y}{x}\right) \\
\sin^{-1}\left(\frac{z}{\sqrt{x^2+y^2+z^2}}\right)
\end{bmatrix}+\bm{v} \quad\quad \bm{v}\sim\mathcal{N}\left( \bm{0},\bm{R}\right)$$

## Motion Distortion

* Typical scan rate for a 3D LIDAR is 5-20 Hz
* For a moving vehicle, each point in a scan is taken from a slightly different place
* Need to account for this if the vehicle is moving quickly, otherwise motion distortion becomes a problem
