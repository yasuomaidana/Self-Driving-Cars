"""
Session 06 Solution: Dynamic Motion Prediction & Time-to-Collision (TTC) Evaluator
"""

from exercise_dynamic_prediction_ttc import run_intersection_risk_assessment

if __name__ == "__main__":
    df_risk, ttc = run_intersection_risk_assessment()
    assert ttc > 0 and ttc < float('inf'), "Collision should be detected in test scenario"
    print(f"Dynamic Prediction Solution Verified! Correctly detected collision at TTC = {ttc:.2f}s")
