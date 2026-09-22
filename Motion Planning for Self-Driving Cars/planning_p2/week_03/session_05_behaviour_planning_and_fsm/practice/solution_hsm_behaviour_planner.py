"""
Session 05 Solution: Hierarchical State Machine (HSM) Behaviour Planner
"""

import polars as pl
from exercise_hsm_behaviour_planner import State, HSMBehaviourPlanner, simulate_scenario

if __name__ == "__main__":
    df_sim = simulate_scenario(dt=0.1, total_time=16.0)
    
    unique_states = df_sim["state"].unique().to_list()
    expected_states = [s.value for s in State]
    
    for exp in expected_states:
        assert exp in unique_states, f"State {exp} was not reached during simulation"
        
    print(f"HSM Behavior Planner Solution Verified! All states traversed: {unique_states}")
