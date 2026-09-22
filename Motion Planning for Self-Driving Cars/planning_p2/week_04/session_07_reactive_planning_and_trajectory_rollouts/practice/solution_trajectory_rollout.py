"""
Session 07 Solution: Reactive Trajectory Rollouts & Collision Evaluator
"""

from exercise_trajectory_rollout import generate_rollouts, score_and_select_best

if __name__ == "__main__":
    obstacles = [(15.0, 0.0, 1.5)]
    rollouts = generate_rollouts(x0=0.0, y0=0.0, theta0=0.0, v0=8.0, num_paths=21, max_steer=0.35)
    best_rollout, df_scored = score_and_select_best(rollouts, obstacles, goal_y=0.0)
    
    assert best_rollout is not None, "A valid collision-free rollout should be found"
    num_colliding = len(df_scored.filter(df_scored["is_colliding"]))
    assert num_colliding > 0, "At least some rollouts should collide with the obstacle"
    
    print(f"Trajectory Rollouts Solution Verified! Found safe rollout #{best_rollout['rollout_id']} among {len(rollouts)} candidates ({num_colliding} collided)")
