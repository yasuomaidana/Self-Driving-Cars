# CARLA Simulation Integration & Benchmarking

## 1. CARLA Autonomous Driving Architecture

In CARLA, the planning system interfaces with the Python API:
- `carla.World`: Provides access to the simulation map (`world.get_map()`), actor lists, and weather conditions.
- `carla.Vehicle`: Ego actor capable of receiving `carla.VehicleControl(throttle=..., steer=..., brake=...)`.
- Sensors: Transform point clouds from `sensor.lidar.ray_cast` and image frames from `sensor.camera.rgb`.

---

## 2. Benchmark Metrics for Motion Planners

Autonomous driving planners are evaluated across 4 quantitative dimensions:

1. **Safety:** Zero collisions, zero near-miss incidents ($\text{TTC} > 1.5s$), minimum clearance $> 1.0m$.
2. **Comfort:** Longitudinal acceleration $|a_{lon}| \le 2.5 m/s^2$, lateral acceleration $|a_{lat}| \le 2.0 m/s^2$, jerk $|j| \le 2.5 m/s^3$.
3. **Efficiency:** Total trip completion time, average speed relative to speed limit.
4. **Traffic Rule Adherence:** Full stops at stop lines ($v < 0.1 m/s$), correct right-of-way yielding.
