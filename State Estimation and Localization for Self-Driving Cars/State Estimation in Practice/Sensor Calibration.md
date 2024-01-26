# Sensor Calibration

* Intrinsinc Calibration: Sensor specific paremters (itself)
    1. How can we determine the parameters of our sensor models?
        * Manufacturer specification
        * Measure by hand
        * [Estimate as part of the state](#calibration-by-estimation)
* [Extrinsic Calibration](#extrinsic-sensor-calibration): How are positioned and oriented respect to the system (vehicle)
    1. How do we determine the relative poses of all the sensors?
        * CAD model
        * Measure by hand
        * Estimate as part of the state
* Temporal Calibration: Time offset respect other elements
    1. How do we determine the relative time delays of all the sensors?
        * Assume zero
        * Hardware synchronization
        * Estimate as part of the state

## Calibration by Estimation

![motion model](./Motion%20Model.jpg)

## Extrinsic Sensor Calibration

![Sensors](./Sensors.jpg)
$$\{\bm{C}_{LI},\bm{r}_1^{LI} \}$$

## Additional material

* Read an [interesting article](https://www.rscal.com/all-you-need-to-know-about-sensor-calibration/) Mon why sensor calibration is necessary.
* Read a [blog_post](https://aimotive.com/blog/content/1227) from Almotive about the need for sensor spatial calibration and temporal synchronization.
* Explore the [cloud-based calibration](http://apollo.auto/platform/perception.html) service for self-driving cars provided by Baidu's Apollo initiative.
