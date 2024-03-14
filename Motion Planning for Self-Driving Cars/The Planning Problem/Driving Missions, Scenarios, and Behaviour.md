# Driving Missions, Scenarios, and Behaviour

## Autonomous Driving Mission

* Mission is to navigate from point A to point B on the map
* Mission planning is higher-level planning
* Low-level details are abstracted away
* Goal is to find most efficient path (in terms of time or distance)

## Road Structure Scenarios

* Road structure influences driving scenario through lane boundaries and regulatory elements
* Simplest case is driving straight, following the center of the lane
* Minimize deviation from centerline
* Attain reference speed for efficiency

### Lane change and Shape trajectory

* Lane changes are more complex
* Different shapes for different situations
* Shape depends on vehicle speed, acceleration limitations
* Time horizon of execution affects the aggressiveness of the lane change

### Left-Right Turn

* Left and right turn scenarios are common in intersections and drivelanes
* Shape of turn varies, similar to lane changes
* State of surrounding environment impacts the ability of the vehicle to make turns

#### U turn

* U-turns are useful for efficient direction changes
* Shape of U-turn will depend on car's speed and acceleration limits
* Not always possible at all intersections

## Obstacle Scenarios

* Static and dynamic obstacles also impact the driving scenario
* Static obstacles restrict which locations our path can occupy
* Most important dynamic obstacle is often the leading vehicle in front of the ego vehicle:
  * Need to maintain time gap for safety

### Turn and Lane Change

* Dynamic obstacles impact turns/lane changes as well
* Depending on locations and speed, different time windows of execution are available for the autonomous vehicle
* Need to use estimation and prediction to calculate these windows of opportunity

### Other obstacles (cyclist and people)

* Different dynamic obstacles in the scenario have
different characteristics and behaviours

## Behaviours

* Speed Tracking
* Decelerate to Stop
* Stay Stopped
* Yield
* Emergency Stop
  * Not an exhaustive list
* Lane change
* Direction change

## Challenges

* Only covered a small subset of scenarios
  * Focused on common cases that follow the rules of the road
* Edge cases make the driving task complex
  * e.g. lane splitting, jaywalking

## Hierarchical Planning Introduction

<img alt="Hierarchy" src="./Hierarchy.png" style="height:60vh;margin:auto;display:block"/>

Driving mission and scenarios are complex problems.
Break them into a hierarchy of optimization problems.

Each optimization problem tailored to the correct scope and level of abstraction

Higher in the hierarchy means more abstraction.
Each optimization problem will have constraints and objective functions
