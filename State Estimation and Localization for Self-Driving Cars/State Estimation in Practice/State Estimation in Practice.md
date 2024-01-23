# State Estimation in Practice

## State Estimation with Multiple Sensors

* Self-driving vehicles rely on data streams from many different sensors (cameras, IMUS, LIDAR, RADAR, GPS, etc.)
* How can we combine information from all these sources?

## Calibration

* What do we need to know about our sensors
and the vehicle to do sensor fusion?
  * Sensor models, which may depend on car-specific parameters (e.g., wheel radius)
  * Relative poses between each sensor pair, so we can combine information in a common reference frame
  * Time offsets between sensor polling times, so we combine only synchronized information

## Accuracy Requirements

* How accurate does the estimator need to be for safe self-driving?
* Typically less than a meter for highway lane keeping
  * Less for driving in dense traffic
* GPS accuracy is 1-5 meters in optimal
conditions
  * Need additional sensors!

## Speed Requirements

* How fast do we need to update the vehicle state to ensure safe driving?
* How much computation power does the vehicle have on-board?
* How much power can our computing resources consume?

## Localization Failures

* How can localization fail?
  * Sensors fail or provide bad data (e.g., GPS in a tunnel)
  * Estimation error (e.g., linearization error in the EKF)
  * Large state uncertainty (e.g., relying on IMU for too long)

## Our Dynamic World

* Many of the models we use in practice for sensors like LIDAR, RADAR, cameras, etc. assume that the world is static and unchanging
* In reality, the world is always moving and changing
* We need to account for this in our models, or find ways of ignoring objects that don't fit our assumptions
