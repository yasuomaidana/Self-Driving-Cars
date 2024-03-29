# Behaviour Planning

A behavior planning system plans the set of high level driving actions, or maneuvers to safely achieve
the driving mission under various driving situations

Behavior planner considers:

* Rules of the road
* Static objects around the vehicle
* Dynamic objects around the vehicle

Planned path must be safe and efficient

## Driving maneuvers

* Track Speed - maintain current speed of the road
* Follow leader - match the speed of the leading vehicle and maintain a safe distance
* Decelerate to stop - begin decelerating and stop before a given space
* Stop - remain stopped in the current position
* Merge - join or switch onto a new drive lane

## Output of Behavior Planner

* Driving maneuver to be executed
* Set of constraints which must be obeyed by the planned trajectory of the self driving car which include:
  * Ideal path
  * Speed limit
  * Lane boundaries
  * Stop locations
  * Set of interest vehicles

## Input Requirements

* High definition road map
* Mission path
* Localization information

### Perception Information

* All observed dynamic objects
  * Prediction of future movement
  * Collision points and time to collision
* All observed static objects
  * Road signs
* Occupancy grid

## Finite State Machines

<img alt="finite state machine" src="./Finite%20State%20Machines.jpg" style="height:50vh;margin: 1em auto; display:block;"/>

* Each state is a driving maneuver
* Transitions define movement from one maneuver to another
* Transitions define the rule implementation that needs to be met before a transition can occur
* Entry action are modification to the constraints

### Advantages of Finite State Machines in Behaviour Planning

* Limiting number of rule checks
* Rule become more targeted and simple
* Implementation of the behavior plannerbecomes simpler
