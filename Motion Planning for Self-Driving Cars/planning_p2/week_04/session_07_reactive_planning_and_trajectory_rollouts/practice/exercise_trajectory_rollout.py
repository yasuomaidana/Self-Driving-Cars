"""
Session 07 Practice: Reactive Trajectory Rollouts & Collision Evaluator
----------------------------------------------------------------------
In this lab, you will generate a bundle of candidate trajectory rollouts using the
kinematic bicycle model, filter out colliding trajectories using multi-circle checking,
and select the optimal collision-free path.

Tools used:
- Polars: Rollout candidate logging and scoring
- Plotly: Interactive rollout bundle visualization with obstacles
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go


class CircleCollisionChecker:
    def __init__(self, circle_offsets: list[float] = [-1.0, 0.5, 2.0], circle_radius: float = 1.25):
        self.offsets = circle_offsets
        self.radius = circle_radius

    def is_collision(self, traj_points: list[tuple], obstacles: list[tuple]) -> bool:
        """
        Checks if any point along traj (x, y, theta) collides with any obstacle (ox, oy, r_obs).
        """
        for x, y, theta in traj_points:
            for d in self.offsets:
                cx = x + d * math.cos(theta)
                cy = y + d * math.sin(theta)
                for ox, oy, r_obs in obstacles:
                    dist = math.hypot(cx - ox, cy - oy)
                    if dist < (self.radius + r_obs):
                        return True
        return False


def generate_rollouts(x0: float, y0: float, theta0: float, v0: float,
                      num_paths: int = 15, max_steer: float = 0.4,
                      horizon: float = 3.0, dt: float = 0.1, L: float = 2.8) -> list[dict]:
    """
    Generates a bundle of candidate trajectories parameterized by fixed steering angles.
    """
    rollouts = []
    steer_angles = np.linspace(-max_steer, max_steer, num_paths)
    num_steps = int(horizon / dt)
    
    for idx, delta in enumerate(steer_angles):
        path = []
        x, y, theta, v = x0, y0, theta0, v0
        for _ in range(num_steps):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += (v / L) * math.tan(delta) * dt
            path.append((x, y, theta))
            
        rollouts.append({
            "rollout_id": idx,
            "steer_angle_rad": delta,
            "path": path,
            "end_x": path[-1][0],
            "end_y": path[-1][1],
        })
    return rollouts


def score_and_select_best(rollouts: list[dict], obstacles: list[tuple],
                          goal_y: float = 0.0) -> tuple[dict, pl.DataFrame]:
    """
    Evaluates rollouts, filters colliding trajectories, and selects best path.
    
    TODO (Student):
    1. For each rollout, check collision with obstacles
    2. If safe, compute cost: J = abs(end_y - goal_y) + 0.5 * abs(steer_angle)
    3. If colliding, set cost = float('inf')
    4. Select rollout with minimum cost
    """
    checker = CircleCollisionChecker()
    scored = []
    best_rollout = None
    min_cost = float('inf')
    
    for r in rollouts:
        colliding = checker.is_collision(r["path"], obstacles)
        if colliding:
            cost = float('inf')
        else:
            # Objective: Follow goal_y (e.g. lane center) with minimal steering effort
            cost = abs(r["end_y"] - goal_y) + 0.5 * abs(r["steer_angle_rad"])
            if cost < min_cost:
                min_cost = cost
                best_rollout = r
                
        scored.append({
            "rollout_id": r["rollout_id"],
            "steer_angle_rad": round(r["steer_angle_rad"], 3),
            "is_colliding": colliding,
            "cost": round(cost, 3) if cost != float('inf') else 999.0,
            "end_x": round(r["end_x"], 2),
            "end_y": round(r["end_y"], 2)
        })
        
    df_scored = pl.DataFrame(scored)
    return best_rollout, df_scored


def plot_rollout_bundle(rollouts: list[dict], best_rollout: dict, obstacles: list[tuple],
                        df_scored: pl.DataFrame, output_html: str = "trajectory_rollouts.html"):
    """Visualizes the candidate rollout bundle, obstacles, and selected path with Plotly."""
    fig = go.Figure()
    
    # 1. Obstacles
    for ox, oy, r_obs in obstacles:
        fig.add_shape(type="circle",
                      xref="x", yref="y",
                      x0=ox - r_obs, y0=oy - r_obs,
                      x1=ox + r_obs, y1=oy + r_obs,
                      fillcolor="crimson", line_color="darkred", opacity=0.8)
        
    # 2. All Rollouts
    for r in rollouts:
        r_id = r["rollout_id"]
        is_col = df_scored.filter(pl.col("rollout_id") == r_id)["is_colliding"][0]
        color = "rgba(255, 0, 0, 0.3)" if is_col else "rgba(100, 150, 255, 0.5)"
        xs = [p[0] for p in r["path"]]
        ys = [p[1] for p in r["path"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=color, width=1.5),
                                 showlegend=False, hoverinfo="text",
                                 text=f"Rollout {r_id} (Col: {is_col})"))
        
    # 3. Best Selected Rollout
    if best_rollout:
        b_xs = [p[0] for p in best_rollout["path"]]
        b_ys = [p[1] for p in best_rollout["path"]]
        fig.add_trace(go.Scatter(x=b_xs, y=b_ys, mode="lines+markers",
                                 line=dict(color="green", width=4),
                                 marker=dict(size=4),
                                 name=f"Selected Best Rollout #{best_rollout['rollout_id']}"))
        
    fig.update_layout(
        title="Reactive Trajectory Rollout Bundle & Collision Checking",
        xaxis_title="Global X (m)",
        yaxis_title="Global Y (m)",
        width=850,
        height=600
    )
    fig.write_html(output_html)
    print(f"Interactive rollout plot saved to {output_html}")


if __name__ == "__main__":
    # Parked car obstacle directly in front at (15m, 0m) with radius 1.5m
    obstacles = [(15.0, 0.0, 1.5)]
    
    # Generate rollouts
    rollouts = generate_rollouts(x0=0.0, y0=0.0, theta0=0.0, v0=8.0, num_paths=21, max_steer=0.35)
    best_rollout, df_scored = score_and_select_best(rollouts, obstacles, goal_y=0.0)
    
    print("--- Rollout Candidates Evaluation (Polars DataFrame) ---")
    print(df_scored)
    print(f"\nBest Selected Rollout ID: {best_rollout['rollout_id']} (Steer: {best_rollout['steer_angle_rad']:.3f} rad)")
    
    plot_rollout_bundle(rollouts, best_rollout, obstacles, df_scored, "trajectory_rollouts_plotly.html")
