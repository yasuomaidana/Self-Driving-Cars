"""
Session 10 Solution: Full-Stack Integrated Autonomous Motion Planning Capstone
"""

import polars as pl
from integrated_planner_capstone import run_full_simulation

if __name__ == "__main__":
    df_capstone = run_full_simulation(total_time=18.0, dt=0.05)
    
    # 1. Verify all states were reached
    states = df_capstone["state"].unique().to_list()
    assert "FOLLOW_LANE" in states
    assert "APPROACH_STOP" in states
    assert "STAY_STOPPED" in states
    assert "CROSS_INTERSECTION" in states
    
    # 2. Verify complete stop occurred near stop sign
    stopped_rows = df_capstone.filter(pl.col("state") == "STAY_STOPPED")
    assert len(stopped_rows) > 0, "Vehicle never came to a complete stop at stop sign"
    min_speed = stopped_rows["velocity_mps"].min()
    assert min_speed < 0.1, f"Expected near zero speed when stopped, got {min_speed}"
    
    # 3. Verify vehicle successfully bypassed parked obstacle at x=25m without colliding
    obstacle_x = 25.0
    near_obs = df_capstone.filter((pl.col("x_m") >= 20.0) & (pl.col("x_m") <= 30.0))
    max_lateral_dev = near_obs["y_m"].abs().max()
    assert max_lateral_dev > 0.8, f"Vehicle should swerve around obstacle, max dev was {max_lateral_dev}m"
    
    print("=" * 60)
    print("🎉 FULL-STACK CAPSTONE VERIFIED SUCCESSFULLY! 🎉")
    print(f"- Traversed States: {states}")
    print(f"- Total Mission Distance: {df_capstone['x_m'].max():.2f} meters")
    print(f"- Max Lateral Deviation for Obstacle Avoidance: {max_lateral_dev:.2f} meters")
    print(f"- Stopped Time Duration: {len(stopped_rows) * 0.05:.2f} seconds")
    print("=" * 60)
