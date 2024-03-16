# Occupancy Grids

* Discretized fine grain grid map
  * Which can be 2D or 3D
* Occupancy by a static object
  * Trees and buildings
  * Curbs and other non drivable surfaces
* Each cell is a binary value $$m^i\in \left\{0,1 \right\}$$

## Assumption of Occupancy Grid

* Static environment
* Independence of each cell
* Known vehicle state at each time step

## Occupancy Grid - Sensor (LIDAR)

* LIDAR

### LIDAR Data Filtering

The first step is to **filter all LIDAR points which comprise the *ground plane***. In this case, the ground plane is the road surface which the autonomous car can safely drive on.

Next, **all points which appear above the highest point of the vehicle are also filtered out (*objects above car height*)**. This set of LIDAR points can be ignored as they will not impede the progression of the autonomous vehicle.

Finally, **all non-static objects (*dynamic objects*)** which had been captured by the LIDAR need to be removed. This includes:

* all vehicles,
* pedestrians,
* bicycles,
* and animals.

Once all filtering of the LIDAR data is complete, the 3D LIDAR data will need to be projected down to a 2D plane to be used to construct our occupancy grid. **Projection onto a 2D plane**

### Range Sensor

* 2D range sensor measuring distance to static objects

### LIDAR Data Noise

* Sensor Noise
* Map Uncertainties

### Probabilistic Occupancy Grid

* Probability of occupancy will be stored $$m^i\in \left\{0,1 \right\}$$
* A belief map is built $$bel_t(m^i) =p(m^i|(x,y))$$

Where:

* $m^i$: Current map cell
* $(x,y)$: Sensor measurement for given cell

* Threshold of certainty will be used to establish occupancy

#### Bayesian Update of the Occupancy Grid

To improve robustness multiple timesteps are used to produce the current map
$$bel_t(m^i) =p(m^i|(x,y)_{1:t})$$

In fact, we can update beliefs in a recursive manner so that at each time step $t$, we use all prior information from $t=1$ onwards to define our belief.

The belief at time $t$ over the map cell $m^i$ is defined as the probability that $m^i$ is occupied given all measurements and the vehicle position from $t=1$ to $t$.

To combine multiple measurements into a single belief map, Bayes theorem can be applied.

Bayes' theorem is applied for at each update step for each
cell.

$$bel_t(m^i) =\eta p(y_t|m^i)bel_{t-1}(m^i)$$

Where:

* $p(y_t|m^i)$: Current measurement. The distribution $p(y_t|m^i)$[^1] , is the probability of getting a particular measurement given a cell $m^i$ is occupied. This is known as the **measurement model**
* $bel_{t-1}(m^i)$: **Previous belief map**. The belief at time $t-1$ over $m^i$ corresponds to the prior belief stored in our occupancy grid from the previous time step.
* $\eta$ corresponds to a **normalizing constant** applied to the belief map. This is needed to scale the results to make sure it remains a probability distribution. Lets see an occupancy grid in action.

We rely on the Markov assumption, that all necessary information for estimating cell occupancy is captured in the belief map at each time step. So no earlier history needs to be considered in the cell update equation.

[^1]: $p$ of $y_t$ given $m^i$
