"""
Session 08 Practice: Conformal Lattice Planning in Frenet Coordinates
---------------------------------------------------------------------
In this lab, you will generate a Conformal Lattice of smooth Quintic Polynomial paths
in Frenet space (s, d) and transform them back into global Cartesian space (x, y).

Tools used:
- Polars: For lattice endpoint and candidate trajectory path storage
- Plotly: For interactive visualization of the road-aligned conformal lattice
"""

import math
import numpy as np
import polars as pl
import plotly.graph_objects as go


class QuinticPolynomial1D:
    def __init__(self, xs: float, vxs: float, axs: float,
                 xe: float, vxe: float, axe: float, T: float):
        """
        Fits a 5th-degree polynomial: x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        satisfying boundary conditions at t=0 and t=T.
        """
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0
        
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([
            xe - self.a0 - self.a1*T - self.a2*T**2,
            vxe - self.a1 - 2*self.a2*T,
            axe - 2*self.a2
        ])
        
        coeffs = np.linalg.solve(A, b)
        self.a3 = coeffs[0]
        self.a4 = coeffs[1]
        self.a5 = coeffs[2]

    def calc_point(self, t: float) -> float:
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_first_derivative(self, t: float) -> float:
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def calc_second_derivative(self, t: float) -> float:
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3


def frenet_to_cartesian(s_vals: np.ndarray, d_vals: np.ndarray,
                        ref_line_fn) -> list[tuple[float, float]]:
    """
    Transforms Frenet points (s, d) to global Cartesian (x, y) coordinates.
    ref_line_fn(s) returns (xr, yr, thetar).
    """
    cartesian_pts = []
    for s, d in zip(s_vals, d_vals):
        xr, yr, thetar = ref_line_fn(s)
        # Normal vector: (-sin(theta), cos(theta))
        x = xr - d * math.sin(thetar)
        y = yr + d * math.cos(thetar)
        cartesian_pts.append((x, y))
    return cartesian_pts


def generate_conformal_lattice(s0: float, d0: float,
                               longitudinal_horizons: list[float] = [30.0, 40.0],
                               lateral_offsets: list[float] = [-3.5, 0.0, 3.5],
                               ds: float = 0.5) -> tuple[list[dict], pl.DataFrame]:
    """
    Generates a bundle of quintic polynomial paths across (s_f, d_f) lattice targets.
    """
    lattice_paths = []
    records = []
    
    # Simple reference path: A gentle curved road
    def curved_reference(s):
        # Radius of curvature R = 200m
        R = 200.0
        angle = s / R
        xr = R * math.sin(angle)
        yr = R * (1.0 - math.cos(angle))
        thetar = angle
        return xr, yr, thetar
        
    path_id = 0
    for s_target in longitudinal_horizons:
        for d_target in lateral_offsets:
            # Quintic polynomial for lateral offset d as function of s in [0, s_target]
            poly = QuinticPolynomial1D(xs=d0, vxs=0.0, axs=0.0,
                                       xe=d_target, vxe=0.0, axe=0.0,
                                       T=s_target)
            
            s_samples = np.arange(0.0, s_target + ds, ds)
            d_samples = np.array([poly.calc_point(s) for s in s_samples])
            d_prime = np.array([poly.calc_first_derivative(s) for s in s_samples])
            d_double_prime = np.array([poly.calc_second_derivative(s) for s in s_samples])
            
            # Rough lateral curvature approximation
            max_lat_curvature = float(np.max(np.abs(d_double_prime)))
            
            pts = frenet_to_cartesian(s0 + s_samples, d_samples, curved_reference)
            
            lattice_paths.append({
                "path_id": path_id,
                "s_target": s_target,
                "d_target": d_target,
                "points": pts,
                "max_lat_curv": max_lat_curvature
            })
            
            records.append({
                "path_id": path_id,
                "s_target_m": s_target,
                "d_target_m": d_target,
                "num_waypoints": len(pts),
                "max_lateral_curvature": round(max_lat_curvature, 4)
            })
            path_id += 1
            
    return lattice_paths, pl.DataFrame(records)


def visualize_lattice(lattice_paths: list[dict], output_html: str = "conformal_lattice.html"):
    """Visualizes the road-aligned conformal lattice with Plotly."""
    fig = go.Figure()
    
    # Reference center line
    s_ref = np.linspace(0, 50, 100)
    R = 200.0
    xr_ref = R * np.sin(s_ref / R)
    yr_ref = R * (1.0 - np.cos(s_ref / R))
    fig.add_trace(go.Scatter(x=xr_ref, y=yr_ref, mode="lines",
                             line=dict(color="black", dash="dash", width=2),
                             name="Road Centerline Reference"))
    
    # Lattice candidate paths
    for p in lattice_paths:
        xs = [pt[0] for pt in p["points"]]
        ys = [pt[1] for pt in p["points"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(width=2),
            name=f"Lattice #{p['path_id']} (s={p['s_target']}, d={p['d_target']})"
        ))
        
    fig.update_layout(
        title="Conformal Spatiotemporal Lattice Planning in Frenet Frame",
        xaxis_title="Global X (m)",
        yaxis_title="Global Y (m)",
        width=850,
        height=650
    )
    fig.write_html(output_html)
    print(f"Interactive lattice plot saved to {output_html}")


if __name__ == "__main__":
    paths, df_meta = generate_conformal_lattice(s0=0.0, d0=0.0)
    print("--- Conformal Lattice Candidates (Polars DataFrame) ---")
    print(df_meta)
    
    visualize_lattice(paths, "conformal_lattice_plotly.html")
