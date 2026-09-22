"""
Session 10 Practice: Full-Stack Integrated Autonomous Motion Planning Capstone
-----------------------------------------------------------------------------
In this capstone, you will integrate all components into a closed-loop simulation:
1. HSM Behaviour Planner (Cruising, Stop-Signs, Collision-Avoidance)
2. Local Reactive Trajectory Rollouts & Circle Collision Checking
3. Curvature-Constrained Velocity Profiling
4. Kinematic Bicycle Closed-Loop Tracking Simulator

Tools used:
- Polars: Full-trip vehicle telemetry logging and KPI metric calculation
- Plotly: Interactive end-to-end mission dashboard (2D map + speed/acceleration plots)
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FullStackAutonomousPlanner:
    def __init__(self, stop_sign_s: float = 65.0, v_cruise: float = 10.0):
        self.stop_sign_s = stop_sign_s
        self.v_cruise = v_cruise
        
        # State machine
        self.state = "FOLLOW_LANE"
        self.stop_timer = 0.0
        
        # Vehicle params
        self.L = 2.8
        self.max_lat_accel = 2.5
        self.collision_radius = 1.25
        self.circle_offsets = [-1.0, 0.5, 2.0]

    def plan_step(self, ego_s: float, ego_x: float, ego_y: float,
                  ego_theta: float, ego_v: float,
                  static_obstacles: list[tuple], dt: float) -> tuple[float, float, str]:
        """
        Executes one full planning cycle:
        1. Behaviour decision (Target speed and stop sign state)
        2. Local rollout generation around static obstacles
        3. Steering angle command and longitudinal acceleration command
        """
        dist_to_stop = self.stop_sign_s - ego_s
        
        # --- 1. Behaviour Layer ---
        if self.state == "FOLLOW_LANE":
            target_v = self.v_cruise
            if 0 < dist_to_stop <= 25.0:
                self.state = "APPROACH_STOP"
                
        elif self.state == "APPROACH_STOP":
            target_v = max(0.0, min(self.v_cruise, (dist_to_stop / 25.0) * self.v_cruise))
            if dist_to_stop <= 1.2 and ego_v < 0.2:
                self.state = "STAY_STOPPED"
                self.stop_timer = 0.0
                
        elif self.state == "STAY_STOPPED":
            target_v = 0.0
            self.stop_timer += dt
            if self.stop_timer >= 3.0: # 3 second wait
                self.state = "CROSS_INTERSECTION"
                
        elif self.state == "CROSS_INTERSECTION":
            target_v = 7.0
            if ego_s > self.stop_sign_s + 15.0:
                self.state = "FOLLOW_LANE"
                
        # --- 2. Local Rollout Selection (Avoid Obstacles) ---
        steer_candidates = np.linspace(-0.35, 0.35, 17)
        best_delta = 0.0
        best_cost = float('inf')
        
        for delta in steer_candidates:
            # Propagate 2.0s forward
            collides = False
            cur_x, cur_y, cur_th = ego_x, ego_y, ego_theta
            
            for _ in range(10): # 10 steps of 0.2s
                cur_x += ego_v * math.cos(cur_th) * 0.2
                cur_y += ego_v * math.sin(cur_th) * 0.2
                cur_th += (ego_v / self.L) * math.tan(delta) * 0.2
                
                # Check circle collision against static obstacles
                for off in self.circle_offsets:
                    cx = cur_x + off * math.cos(cur_th)
                    cy = cur_y + off * math.sin(cur_th)
                    for ox, oy, robs in static_obstacles:
                        if math.hypot(cx - ox, cy - oy) < (self.collision_radius + robs):
                            collides = True
                            break
                    if collides:
                        break
                if collides:
                    break
                    
            if not collides:
                # Cost: Track road centerline (y=0) + minimize steering effort
                cost = abs(cur_y - 0.0) + 0.8 * abs(delta)
                if cost < best_cost:
                    best_cost = cost
                    best_delta = delta
                    
        # --- 3. Longitudinal Control Command ---
        accel_cmd = max(-3.0, min(1.5, (target_v - ego_v) * 1.2))
        
        return best_delta, accel_cmd, self.state


def run_full_simulation(total_time: float = 20.0, dt: float = 0.05) -> pl.DataFrame:
    """Runs closed-loop simulation of the full-stack autonomous system."""
    planner = FullStackAutonomousPlanner(stop_sign_s=55.0, v_cruise=9.0)
    
    # Parked car obstacle on right lane at x=25m, y=0m
    obstacles = [(25.0, 0.0, 1.4)]
    
    # Initial state
    ego_s = 0.0
    ego_x = 0.0
    ego_y = 0.0
    ego_theta = 0.0
    ego_v = 4.0
    
    telemetry = []
    num_steps = int(total_time / dt)
    
    for step in range(num_steps):
        t = step * dt
        
        # Planning cycle
        steer_cmd, accel_cmd, state_str = planner.plan_step(
            ego_s, ego_x, ego_y, ego_theta, ego_v, obstacles, dt
        )
        
        # Vehicle Kinematics Update
        ego_x += ego_v * math.cos(ego_theta) * dt
        ego_y += ego_v * math.sin(ego_theta) * dt
        ego_theta += (ego_v / planner.L) * math.tan(steer_cmd) * dt
        ego_v = max(0.0, ego_v + accel_cmd * dt)
        ego_s += ego_v * dt
        
        kappa = math.tan(steer_cmd) / planner.L
        a_lat = (ego_v ** 2) * kappa
        
        telemetry.append({
            "time_s": round(t, 2),
            "ego_s_m": round(ego_s, 2),
            "x_m": round(ego_x, 2),
            "y_m": round(ego_y, 2),
            "theta_rad": round(ego_theta, 3),
            "velocity_mps": round(ego_v, 2),
            "steer_rad": round(steer_cmd, 3),
            "accel_mps2": round(accel_cmd, 2),
            "lat_accel_mps2": round(a_lat, 2),
            "state": state_str
        })
        
    return pl.DataFrame(telemetry)


def generate_capstone_dashboard(df: pl.DataFrame, obstacles: list[tuple],
                                output_html: str = "capstone_dashboard.html"):
    """Creates a comprehensive interactive Plotly dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("2D Vehicle Trajectory & Obstacle Avoidance", "Velocity Profile v(t)",
                        "Steering Angle δ(t)", "Lateral Acceleration vs Comfort Limit")
    )
    
    # 1. 2D Path
    for ox, oy, robs in obstacles:
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=ox - robs, y0=oy - robs, x1=ox + robs, y1=oy + robs,
                      fillcolor="red", line_color="darkred", row=1, col=1)
    fig.add_trace(go.Scatter(x=df["x_m"], y=df["y_m"], mode="lines+markers",
                             name="Vehicle Trajectory", line=dict(color="blue", width=3)), row=1, col=1)
    fig.add_vline(x=55.0, line_dash="dash", line_color="orange", annotation_text="Stop Sign", row=1, col=1)
    
    # 2. Velocity
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["velocity_mps"], mode="lines",
                             name="Velocity (m/s)", line=dict(color="green", width=2)), row=1, col=2)
    
    # 3. Steering
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["steer_rad"], mode="lines",
                             name="Steering (rad)", line=dict(color="darkorange", width=2)), row=2, col=1)
    
    # 4. Lateral Acceleration
    fig.add_trace(go.Scatter(x=df["time_s"], y=df["lat_accel_mps2"], mode="lines",
                             name="a_lat (m/s^2)", line=dict(color="purple", width=2)), row=2, col=2)
    fig.add_hline(y=2.5, line_dash="dash", line_color="red", row=2, col=2)
    fig.add_hline(y=-2.5, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_layout(title="Full-Stack Autonomous Driving Planning Capstone Dashboard", height=750)
    fig.write_html(output_html)
    print(f"Capstone interactive dashboard saved to {output_html}")


if __name__ == "__main__":
    print("Running Full-Stack Motion Planning Capstone simulation...")
    df_capstone = run_full_simulation(total_time=18.0, dt=0.05)
    
    print("\n--- Capstone Telemetry Summary (Polars DataFrame) ---")
    print(df_capstone.head(10))
    
    print("\n--- State Distribution ---")
    print(df_capstone.group_by("state").agg(pl.count().alias("steps")))
    
    generate_capstone_dashboard(df_capstone, [(25.0, 0.0, 1.4)], "capstone_dashboard_plotly.html")
