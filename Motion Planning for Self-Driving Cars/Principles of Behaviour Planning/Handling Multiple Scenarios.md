# Handling Multiple Scenarios

Scenario Evaluation

<img alt="intersections" src="./Intersections.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

* 4 way intersection
* Every direction has a stop sign
* Be able to travel:
  * Through the intersection
  * Left at the intersection
  * Right at the intersection
* **Only vehicles** as dynamic objects
  * 1, 2, 3 or 4 other vehicles

## Single State Machine

<img alt="single state machine" src="./State Machine_v2.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

* Single state machine method
  * Add transitions
  * Add additional transition conditions
* **Issues** with single state machine method:
  * Rule explosion (number of rules required to add additional scenarios becomes unimaginably large)
  * Increase in computational time
  * Complicated to create and maintain

## Hierarchical State Machine

<img alt="Hierarchical State Machine" src="./Hierarchical State Machine.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

### Entry and Exit Transitions - Intersection

How do we switch between scenarios when processing the sub-state maneuvers?

One method is to introduce transitions to key maneuver sub-states out of the current scenario super-state.

**Example**:

<img alt="Entry Exit" src="./Entry Exit.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

First, let's establish which state will be the key exit state out of the scenario.

The only way we're able to exit an intersection is while in the track speed or follow leader states after having passed the intersection.

So let's put the exit transitions there. The condition on the transitions would be to confirm that this scenario has indeed been completed.

In this example, we can use the distance to the next stop sign intersection. If this is larger than a given threshold, we can exit the intersection scenario. With this method of transition, we were able to also maintain maneuvers between scenario switches.

In this case, track speed in the intersection super-state will connect to the track speed of the next scenario that we enter.

### Multi-lane Scenario

<img alt="Multi lane scenario" src="./Multilane Scenario.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

Adding an additional multi-lane scenario to this hierarchical state machine is done by creating a new super-state scenario node, with a sub-state state machine capable of handling that scenario.

### Advantages and Disadvantages

<img alt="Advantages Disadvantages" src="./Advantages Disadvantages Hierarchical.jpg" style="height:40vh;margin: 1em auto; display:block;"/>

* Advantages:
  * Decrease in computational time
  * Simpler to create and maintain
* Disadvantages:
  * Rule Explosion
  * Repetition of many rules in the low level state machines.
