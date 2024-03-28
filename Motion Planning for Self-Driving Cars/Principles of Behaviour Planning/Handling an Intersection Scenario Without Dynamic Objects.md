# Handling an Intersection Scenario Without Dynamic Objects

Scenario Evaluation

* 4 way Intersection
* TWO lane
* Stop Sign for every direction
* Be able to travel:
  * Through the intersection
  * Left at the intersection
  * Right at the intersection
* No other dynamic vehicles

## Discretizing the Intersection

![intersections](./Intersections.jpg)

* Approaching an intersection <span style="color:red">**&#9974;**</span>.
* At an intersection <span style="color:green">**&#9974;**</span>.
* On an intersection <span style="color:orange">**&#9974;**</span>.
* Determining the size of each zone:
  * Ego vehicle velocity
  * Size of the intersection

## State Machine States

![finite states](./Finite%20States.jpg)

* Track Speed — Follow the current speed limit
* Decelerate to Stop — Stop to a particular point
* Stop — Stay stopped at the current location

## Behavior Planning Testing

* Code based tests
* Simulation tests
* Private track tests
* Limited scoped close supervision road tests
