# Feedforward Speed Control

## Feedback vs. Feedforward Control

![Feedback vs. Feedforward Control](./Feedback%20vs.%20Feedforward%20Control.jpg)

## Combined Feedforward and Feedback Control

![Combined Feedforward and Feedback Control](./Combined%20Feedforward%20and%20Feedback%20Control.jpg)

Feedforward and feedback are often used together:

* Feedforward controller provides predictive response, non-zero offset
* Feedback controller corrects the response, compensating for
disturbances and errors in the model

## Vehicle Speed Control

![Vehicle Speed Control](./Vehicle%20Speed%20Control.jpg)
Throttling & Braking:

* The output of the feedforward and feedback control blocks are the throttling or braking signals to accelerate or decelerate the vehicle (plant) to keep the vehicle velocity close to the reference velocity.

## Controller Actuators

![Controller Actuators](./Controller%20Actuators.jpg)

Actuators (throttle angle):

* The feedforward controller generates the actuator signal $u_{ff}$ based on the predefined table and the feedback controller generates the actuator signal $(u_{fb})$ based on the velocity error.

## Feedforward Table

![Feedforward Table](./Feedforward%20Table.jpg)

## Additional resources

Sailan, K., Kuhnert, K.D., ["Modeling and Design of Cruise Control System with Feedforward For All Terrain Vehicles"](https://airccj.org/CSCP/vol3/csit3828.pdf), Computer Science & Information Technology (CS & IT). 2013.
