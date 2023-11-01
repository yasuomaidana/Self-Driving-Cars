# Bicycle Kinematic Model

## Instantanous center of rotation (ICR)

![Bicycle ICR model](./Bicycle%20Kinematic%20Model%20Instantaneous%20Center%20of%20Rotation.jpg)

$$\dot{\theta} = \omega = \frac{v}{R} = \frac{v\tan{\delta}}{L}$$
$$\tan{\delta} = \frac{L}{R}$$

## Rear Axle Model

![Rear model](./Bicycle%20Kinematic%20Model%20Rear%20Axle.jpg)
$$\dot{x}_r=v\cos{\theta}$$
$$\dot{y}_r=v\sin{\theta}$$
$$\dot{\theta}=\frac{v\tan{\delta}}{L}$$

## Front Axle model

![Front model](./Bicycle%20Kinematic%20Model%20Front%20Axle.jpg)
$$\dot{x}_f=v\cos{(\theta+\delta)}$$
$$\dot{y}_f=v\sin{(\theta+\delta)}$$
$$\dot{\theta}=\frac{v\sin{\delta}}{L}$$

## Center of gravity model

![Center of gravity model](./Bicycle%20Kinematic%20Model%20Center%20of%20the%20Gravity.jpg)

$$\dot{x}_f=v\cos{(\theta+\beta)}$$
$$\dot{y}_f=v\sin{(\theta+\beta)}$$
$$\dot{\theta}=\frac{v\cos{\beta}\tan{\delta}}{L}$$
$$\beta = \tan^{-1}(\frac{l_r\tan\delta}{L})$$

## State-Space Representation

State: $[x,y,\theta,\delta]^T$
Inputs: $[v,\psi]^T$

$$\dot{x}_f=v\cos{(\theta+\beta)}$$
$$\dot{y}_f=v\sin{(\theta+\beta)}$$
$$\dot{\theta}=\frac{v\cos{\beta}\tan{\delta}}{L}$$
$$\dot{\delta} = \psi$$

Where $\psi$ is a modified input: rate of change of steering angle.

## Additional resources

Read more about the Kinematic Bicycle Model (pages 15-26) in the PDF below:

["Chapter 2, Lateral Vehicle Dynamics"](./Lateral%20Vehicle%20Dynamics-e.pdf), R. Rajamani, Vehicle Dynamics and Control, Mechanical Engineering Series. (2012)
