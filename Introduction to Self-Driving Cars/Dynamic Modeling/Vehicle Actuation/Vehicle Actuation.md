# Vehicle Actuation

## Coupled Lateral & Longitudinal

Main Control Task:
To keep the vehicle on the defined path at the desired velocity
![input output model](./media/Input%20output%20Model.jpg)

### Steering

![steering model](./media/Steering.jpg)

* Simple Steering (Usually used for simulations)
![simple  steering model](./media/Simple%20Steering.jpg)
* Actual Steering System
![actual steering model](./media/Actual%20Steering.jpg)

### Powertrain System (Driveline)

Throttle and brake commands affect torque balance

![power train model](./media/Power%20train%20.jpg)

#### Throttling (Accelerating)

![throttiling model](./media/Throttling.jpg)

* Accelerating model ![accelerating model](./media/accelerating%20model.jpg)
* Accelerating characteristics plots ![characteristics plots](./media/accelerating%20characteristics%20models.jpg)

#### Typical Torque Curves for Gasoline Engines

![Typical Torques Curves for Gasolines Engines](./media/Typical%20Torques%20Curves%20for%20Gasolines%20Engines.jpg)

$$T_{e_{max}}\left(\omega_e\right) = A_0 + A_1\omega_e + A_2\omega_e^2$$

$$T_{e_{max}}\left(\omega_e,x_\theta\right) \approx x_\theta \left(A_0 + A_1\omega_e + A_2\omega_e^2 \right)$$

Where $x_\theta$ is the throttle position (persentage)

#### Braking (decelerating)

![decelerating](./media/Braking.jpg)

* Model
![braking model](./media/braking%20model.jpg)
$$T_{brake} = k \Delta P$$

## Additional Resources

Read more about vehicle steering system in the Journal article below:

Reimann G., Brenner P., Büring H. (2015) "Steering Actuator Systems". In: Winner H., Hakuli S., Lotz F., Singer C. (eds) Handbook of Driver Assistance Systems. Springer, Cham

Read more about vehicle driveline (throttling and braking system) in the textbook below:

Mashadi, B., Crolla, D, Vehicle Powertrain Systems. Wiley (2012)
