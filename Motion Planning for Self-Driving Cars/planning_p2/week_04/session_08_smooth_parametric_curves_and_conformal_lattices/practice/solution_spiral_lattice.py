"""
Session 08 Solution: Conformal Lattice Planning in Frenet Coordinates
"""

from exercise_spiral_lattice import generate_conformal_lattice

if __name__ == "__main__":
    paths, df_meta = generate_conformal_lattice(s0=0.0, d0=0.0)
    assert len(paths) == 6, "Expected 6 lattice candidate paths (2 horizons x 3 offsets)"
    print(f"Conformal Lattice Solution Verified! Generated {len(paths)} smooth polynomial paths.")
