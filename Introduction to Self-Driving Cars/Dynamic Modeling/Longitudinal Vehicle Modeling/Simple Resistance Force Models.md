# Simple Resistance Force Models

* Total resistance load : $$F_{load}=F_{aero} + R_x + mg\alpha$$
* The aerodynamic force can depend on air density $\rho$, frontal area $A$ which can be represented as aerodynamic coeficient $c_a$, frontal area, on the sepeed of the vehicle: $$F_{aero} = \frac{1}{2}C_a\rho A\dot{x}^2 = c_a\dot{x}^2$$
* The rolling resistance can depend on the tire normal force, tire presures and vehicle speed which can be simplified by using rolling resistance coefficient $c_{r,1}$: $$R_x = N(\hat{c}_{r,0}+ \hat{c}_{r,1}|\dot{x}|+ \hat{c}_{r,2}\dot{x}^2\approx c_{r,1}|\dot{x}|)$$
