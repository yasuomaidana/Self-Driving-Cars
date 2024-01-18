# The Inertial Measurement Unit (IMU)

An IMU is typically composed of

* **gyroscopes** (measure angular rotation rates about three separate axes)
* **accelerometers** (measure accelerations along three orthogonal axes)

## The accelerometer

Accelerometers measure acceleration relative to *free-fall* - this is also called the proper acceleration or specific force:
$$\bm{a}_{meas}=\bm{f}=\frac{\bm{F}_{non-gravity}}{m}$$
> Sitting still at your desk, your proper acceleration is g upwards! (think of the 'normal' force holding you up)

In localization, we typically require the acceleration relative to a fixed reference frame

* 'coordinate' acceleration
* computed using fundamental equation for accelerometers in a gravity field: $$\bm{f}+\bm{g}=\ddot{\bm{r}}_i$$

### The accelerometer - Examples

* An accelerometer in a stationary car measures:
$$\bm{f}=\ddot{\bm{r}}_i-\bm{g}\approx \bm{0}-\bm{g} \approx -\bm{g}$$

* An accelerometer on the International Space Station
measures:
$$\bm{f}=\ddot{\bm{r}}_i-\bm{g}\approx \bm{g}-\bm{g} \approx 0$$

## Measurement Model: Gyroscope

$$\bm{\omega}(t)=\bm{\omega}_s(t)+\bm{b}_{gyro}(t)+\bm{n}_{gyro}(t)$$

Where:

* $\bm{\omega}_s(t)$ angular velocity of the sensor
expressed in the sensor frame.
* $\bm{b}_{gyro}(t)$ slowly evolving bias
* $\bm{n}_{gyro}(t)$ noise term

## Measurement Model: Accelerometer

$$\bm{a}(t)=\bm{C}_{sn}(t)\left(\ddot{\bm{r}}_{n}^{sn}(t)-\bm{g}_{n}\right)+\bm{b}_{accel}(t)+\bm{n}_{accel}(t)$$

Where:

* $\bm{C}_{sn}(t)$ orientation of the sensor (computed by integrating the rotational rates from the gyroscope)
* $\bm{b}_{accel}(t)$ bias term
* $\bm{n}_{accel}(t)$ noise term
* $\bm{g}_{n}$ gravity in the navigation frame

## Inertial Navigations: Important Notes

When using an IMU for localization, keep in mind:

* If we inaccurately keep track of $\bm{C}_{sn}(t)$, we incorporate components of $\bm{g}_{n}$ into $\ddot{\bm{r}}_{n}^{sn}(t)$. </br>
This will ultimately lead to terrible estimates of position ($\bm{r}_{n}^{sn}(t)$)
* Both measurement models ignore the effect of Earth's
* We only consider **strapdown** IMUS - where the individual sensors are rigidly attached to the vehicle and are not gimballed

## Additional resources

* Read the [page](https://en.wikipedia.org/wiki/Inertial_measurement_unit) about IMUs on Wikipedia.

* Follow along with the PDF slides from Gordon Wetzstein's [lecture on IMUs](http://stanford.edu/class/ee267/lectures/lecture9.pdf) at Stanford University.

* Examine a technical description of inertial navigation in Chapter 11, Section 1 of [Jay A. Farrell, Aided Navigation (2008)](https://books.google.ca/books/about/Aided_Navigation_GPS_with_High_Rate_Sens.html?id=yNujEvIMszYC&redir_esc=y).

* Reflect on 40 years of advances in inertial navigation, summarized in this article by Anthony King (1998)
 from the magazine GEC Review.
