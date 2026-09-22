"""
Session 05 Practice: Hierarchical State Machine (HSM) Behaviour Planner
------------------------------------------------------------------------
In this lab, you will implement a rule-based HSM Behaviour Planner that handles
cruising, approaching stop lines, waiting at stop signs, and crossing intersections.

Tools used:
- Polars: For event tracking and state transition telemetry logging
- Plotly: For timeline visualization of behaviour states
"""

from enum import Enum
import polars as pl
import plotly.express as px


class State(Enum):
    FOLLOW_LANE = "FOLLOW_LANE"
    DECELERATE_TO_STOP = "DECELERATE_TO_STOP"
    STAY_STOPPED = "STAY_STOPPED"
    CROSS_INTERSECTION = "CROSS_INTERSECTION"


class HSMBehaviourPlanner:
    def __init__(self, stop_wait_time: float = 3.0, stop_dist_threshold: float = 25.0):
        self.state = State.FOLLOW_LANE
        self.stop_wait_time = stop_wait_time
        self.stop_dist_threshold = stop_dist_threshold
        self.stop_timer = 0.0
        self.target_speed = 12.0 # default cruising speed m/s (~43 km/h)

    def update(self, ego_s: float, ego_v: float, stop_sign_s: float,
               is_intersection_clear: bool, dt: float) -> tuple[State, float]:
        """
        Executes one behaviour planning decision cycle.
        
        Parameters:
            ego_s: Current longitudinal distance along route (m)
            ego_v: Current vehicle speed (m/s)
            stop_sign_s: Location of upcoming stop sign (or None / inf if none)
            is_intersection_clear: True if perception confirms no crossing traffic
            dt: Time step (s)
            
        Returns:
            (current_state, commanded_target_speed)
            
        TODO (Student):
        Implement transitions between FOLLOW_LANE, DECELERATE_TO_STOP, STAY_STOPPED, CROSS_INTERSECTION:
        - FOLLOW_LANE -> DECELERATE_TO_STOP when dist_to_stop <= stop_dist_threshold and ego_s < stop_sign_s
        - DECELERATE_TO_STOP -> STAY_STOPPED when ego_v < 0.2 and dist_to_stop <= 1.5m
        - STAY_STOPPED: Increment self.stop_timer. When timer >= stop_wait_time and is_intersection_clear -> CROSS_INTERSECTION
        - CROSS_INTERSECTION -> FOLLOW_LANE when ego_s > stop_sign_s + 15.0m
        """
        dist_to_stop = stop_sign_s - ego_s if stop_sign_s is not None else float('inf')
        
        if self.state == State.FOLLOW_LANE:
            self.target_speed = 12.0
            if 0 < dist_to_stop <= self.stop_dist_threshold:
                self.state = State.DECELERATE_TO_STOP

        elif self.state == State.DECELERATE_TO_STOP:
            # Command gradual deceleration to stop at line
            self.target_speed = max(0.0, min(12.0, (dist_to_stop / self.stop_dist_threshold) * 12.0))
            if dist_to_stop <= 1.5 and ego_v < 0.2:
                self.state = State.STAY_STOPPED
                self.stop_timer = 0.0

        elif self.state == State.STAY_STOPPED:
            self.target_speed = 0.0
            self.stop_timer += dt
            if self.stop_timer >= self.stop_wait_time and is_intersection_clear:
                self.state = State.CROSS_INTERSECTION

        elif self.state == State.CROSS_INTERSECTION:
            self.target_speed = 8.0 # moderate speed through turn/intersection
            if ego_s > stop_sign_s + 15.0: # passed intersection
                self.state = State.FOLLOW_LANE
                self.stop_timer = 0.0

        return self.state, self.target_speed


def simulate_scenario(dt: float = 0.1, total_time: float = 15.0) -> pl.DataFrame:
    """Simulates an autonomous vehicle encountering and negotiating a stop sign."""
    planner = HSMBehaviourPlanner(stop_wait_time=3.0, stop_dist_threshold=25.0)
    
    stop_sign_s = 40.0
    ego_s = 0.0
    ego_v = 12.0
    
    logs = []
    num_steps = int(total_time / dt)
    
    for step in range(num_steps):
        t = step * dt
        
        # Intersection is clear after t = 7.0s
        is_clear = (t > 7.0)
        
        state, target_v = planner.update(ego_s, ego_v, stop_sign_s, is_clear, dt)
        
        # Simple longitudinal dynamics simulation towards target_v
        accel = (target_v - ego_v) * 1.5
        accel = max(-3.0, min(2.0, accel)) # Clamp accel
        ego_v = max(0.0, ego_v + accel * dt)
        ego_s += ego_v * dt
        
        logs.append({
            "time_s": round(t, 2),
            "ego_s_m": round(ego_s, 2),
            "ego_v_mps": round(ego_v, 2),
            "target_v_mps": round(target_v, 2),
            "state": state.value,
            "stop_timer_s": round(planner.stop_timer, 2),
            "dist_to_stop_m": round(max(0.0, stop_sign_s - ego_s), 2)
        })
        
    return pl.DataFrame(logs)


def plot_behaviour_timeline(df: pl.DataFrame, output_html: str = "behaviour_fsm.html"):
    """Visualizes state machine transitions and vehicle kinematics using Plotly."""
    fig = px.line(
        df, x="time_s", y=["ego_v_mps", "target_v_mps", "dist_to_stop_m"],
        color_discrete_map={"ego_v_mps": "blue", "target_v_mps": "green", "dist_to_stop_m": "orange"},
        title="HSM Behaviour Planner Stop-Sign Execution Profile"
    )
    
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Metric Values (m / m/s)",
        width=850,
        height=500
    )
    fig.write_html(output_html)
    print(f"Interactive behaviour plot saved to {output_html}")


if __name__ == "__main__":
    df_sim = simulate_scenario(dt=0.1, total_time=16.0)
    print("--- Behaviour State Transitions (Polars DataFrame) ---")
    print(df_sim.filter(pl.col("state").is_not_null()).head(10))
    
    # State summary count
    print("\n--- Time Spent in Each State ---")
    print(df_sim.group_by("state").agg(pl.count().alias("time_steps_0.1s")))
    
    plot_behaviour_timeline(df_sim, "behaviour_timeline_plotly.html")
