"""
Session 09 Solution: Path Optimization & Curvature-Constrained Velocity Profiling
"""

import numpy as np
import polars as pl
from exercise_path_optimizer_velocity import generate_curved_path_waypoints, fit_smooth_spline_path, compute_velocity_profile

if __name__ == "__main__":
    x_coarse, y_coarse = generate_curved_path_waypoints()
    s, x_smooth, y_smooth, curvature = fit_smooth_spline_path(x_coarse, y_coarse)
    
    a_lat_max = 2.5
    v_prof = compute_velocity_profile(s, curvature, v_cruise=12.0, a_lat_max=a_lat_max, a_accel_max=1.5, a_decel_max=2.0)
    
    # Verify final stop
    assert abs(v_prof[-1]) < 1e-3, f"Final velocity must be 0.0 (got {v_prof[-1]})"
    # Verify lateral acceleration comfort limit
    a_lat = (v_prof ** 2) * np.abs(curvature)
    max_lat = np.max(a_lat)
    assert max_lat <= a_lat_max + 1e-2, f"Lateral acceleration exceeded limit: {max_lat} > {a_lat_max}"
    
    print(f"Velocity Profiling Solution Verified! Smooth profile generated. Max a_lat: {max_lat:.2f} m/s^2 <= {a_lat_max} m/s^2, Final speed = {v_prof[-1]:.2f} m/s")
