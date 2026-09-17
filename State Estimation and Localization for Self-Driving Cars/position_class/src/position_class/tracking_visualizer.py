"""Image Output and Visualization Module for Visual Tracking.

Renders bounding boxes, polygon segmentations, center of mass, Kalman predictions,
trajectory history, error vectors to image center, and on-screen HUD telemetry.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np


class TrackingVisualizer:
    """Renders comprehensive tracking overlays and diagnostics on video/image frames."""

    def __init__(
        self,
        max_trail_length: int = 30,
        show_center_crosshair: bool = True,
        show_error_vector: bool = True,
        show_trajectories: bool = True,
        show_hud: bool = True
    ):
        self.max_trail_length = max_trail_length
        self.show_center_crosshair = show_center_crosshair
        self.show_error_vector = show_error_vector
        self.show_trajectories = show_trajectories
        self.show_hud = show_hud

        # History trails for visualization
        self.meas_trail: Deque[Tuple[int, int]] = deque(maxlen=max_trail_length)
        self.est_trail: Deque[Tuple[int, int]] = deque(maxlen=max_trail_length)
        self.pred_trail: Deque[Tuple[int, int]] = deque(maxlen=max_trail_length)

    def reset_trails(self) -> None:
        """Clear historical trajectory trails."""
        self.meas_trail.clear()
        self.est_trail.clear()
        self.pred_trail.clear()

    def draw_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        polygon: Optional[List[Tuple[float, float]]] = None,
        meas_point: Optional[Tuple[float, float]] = None,
        est_point: Optional[Tuple[float, float]] = None,
        pred_point: Optional[Tuple[float, float]] = None,
        velocity: Optional[Tuple[float, float]] = None,
        error_distance: Optional[float] = None,
        status_text: str = "TRACKING",
        label: str = "Target",
        fps: Optional[float] = None,
        is_live: bool = False,
        model_name: Optional[str] = None,
        selection_alpha: float = 0.30,
        fill_color: Tuple[int, int, int] = (0, 220, 255),  # Vibrant Amber/Cyan
        fast_points: Optional[List[Tuple[float, float]]] = None
    ) -> np.ndarray:
        """Draw complete visual tracking overlay on a copy of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx_img = w // 2
        cy_img = h // 2

        # 1. Draw Image Center Crosshair
        if self.show_center_crosshair:
            cv2.drawMarker(vis, (cx_img, cy_img), (200, 200, 200), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            cv2.circle(vis, (cx_img, cy_img), 4, (255, 255, 255), -1)
            cv2.putText(vis, "IMG CENTER", (cx_img + 8, cy_img - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        # 2. Draw Polygon Segmentation or Bounding Box with Alpha Transparency Color Fill
        is_dead_reckoning = (meas_point is None and (est_point is not None or pred_point is not None))
        active_fill_color = (180, 0, 220) if is_dead_reckoning else fill_color
        active_border_color = (255, 0, 255) if is_dead_reckoning else (0, 255, 120)

        box_coords = bbox
        if box_coords is None and polygon is not None and len(polygon) >= 3:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            box_coords = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

        if polygon is not None and len(polygon) >= 3 and not is_dead_reckoning:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            # Alpha color fill
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color=active_fill_color)
            cv2.addWeighted(overlay, selection_alpha, vis, 1.0 - selection_alpha, 0, vis)
            # Solid perimeter
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        elif box_coords is not None:
            bx, by, bw, bh = box_coords
            # Alpha color fill for bounding box
            overlay = vis.copy()
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), active_fill_color, -1)
            cv2.addWeighted(overlay, 0.22 if is_dead_reckoning else selection_alpha, vis, 0.78 if is_dead_reckoning else (1.0 - selection_alpha), 0, vis)
            
            # Border
            border_thick = 1 if is_dead_reckoning else 2
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), active_border_color, border_thick, cv2.LINE_AA)
            
            # Corner accents
            c_len = min(16, bw // 4, bh // 4)
            if c_len > 0 and not is_dead_reckoning:
                cv2.line(vis, (bx, by), (bx + c_len, by), (0, 255, 255), 3)
                cv2.line(vis, (bx, by), (bx, by + c_len), (0, 255, 255), 3)
                cv2.line(vis, (bx + bw, by), (bx + bw - c_len, by), (0, 255, 255), 3)
                cv2.line(vis, (bx + bw, by), (bx + bw, by + c_len), (0, 255, 255), 3)
                cv2.line(vis, (bx, by + bh), (bx + c_len, by + bh), (0, 255, 255), 3)
                cv2.line(vis, (bx, by + bh), (bx, by + bh - c_len), (0, 255, 255), 3)
                cv2.line(vis, (bx + bw, by + bh), (bx + bw - c_len, by + bh), (0, 255, 255), 3)
                cv2.line(vis, (bx + bw, by + bh), (bx + bw, by + bh - c_len), (0, 255, 255), 3)

        # Object Name Badge / Pill Label (Always rendered whenever label or box is available)
        if box_coords is not None and label:
            bx, by, bw, bh = box_coords
            prefix = "[KF PRED] " if is_dead_reckoning else ""
            label_text = f" {prefix}{label} "
            (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            lbl_y1 = max(4, by - lbl_h - 8)
            lbl_y2 = lbl_y1 + lbl_h + 6
            lbl_x2 = min(w - 2, bx + lbl_w + 8)
            
            # Pill background
            pill_overlay = vis.copy()
            cv2.rectangle(pill_overlay, (bx, lbl_y1), (lbl_x2, lbl_y2), (20, 20, 20), -1)
            cv2.addWeighted(pill_overlay, 0.85, vis, 0.15, 0, vis)
            cv2.rectangle(vis, (bx, lbl_y1), (lbl_x2, lbl_y2), active_border_color, 1)
            cv2.putText(vis, label_text, (bx + 2, lbl_y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 100, 255) if is_dead_reckoning else (255, 255, 255), 1, cv2.LINE_AA)

        # 2b. Draw FAST 3-Point Refreshed Feature Markers
        if fast_points is not None and len(fast_points) > 0 and not is_dead_reckoning:
            for f_idx, f_pt in enumerate(fast_points[:3]):
                fx, fy = int(f_pt[0]), int(f_pt[1])
                # Small reticle marker
                cv2.drawMarker(vis, (fx, fy), (0, 255, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                cv2.circle(vis, (fx, fy), 3, (0, 240, 255), 1, cv2.LINE_AA)
                f_label = f"F{f_idx+1}"
                cv2.putText(vis, f_label, (fx + 5, fy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Update trajectory history
        if meas_point is not None:
            self.meas_trail.append((int(meas_point[0]), int(meas_point[1])))
        if est_point is not None:
            self.est_trail.append((int(est_point[0]), int(est_point[1])))
        if pred_point is not None:
            self.pred_trail.append((int(pred_point[0]), int(pred_point[1])))

        # 4. Draw Trajectory Trails
        if self.show_trajectories:
            for i in range(1, len(self.est_trail)):
                alpha = i / len(self.est_trail)
                thickness = max(1, int(3 * alpha))
                cv2.line(vis, self.est_trail[i - 1], self.est_trail[i], (0, 255, 0), thickness, cv2.LINE_AA)

            for i in range(1, len(self.pred_trail)):
                cv2.line(vis, self.pred_trail[i - 1], self.pred_trail[i], (255, 0, 255), 1, cv2.LINE_AA)

        # 5. Draw Target Measurement Center of Mass (CoM)
        if meas_point is not None:
            mx, my = int(meas_point[0]), int(meas_point[1])
            cv2.circle(vis, (mx, my), 6, (0, 255, 255), -1, cv2.LINE_AA) # Yellow dot
            cv2.circle(vis, (mx, my), 10, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, "CoM", (mx + 12, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # 6. Draw Kalman Prediction Point
        if pred_point is not None:
            px, py = int(pred_point[0]), int(pred_point[1])
            cv2.drawMarker(vis, (px, py), (255, 0, 255), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
            cv2.putText(vis, "KF Pred", (px + 10, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)

        # 7. Draw Kalman Estimated Filtered State
        active_target_pt = None
        if est_point is not None:
            ex, ey = int(est_point[0]), int(est_point[1])
            cv2.circle(vis, (ex, ey), 5, (0, 200, 0), -1, cv2.LINE_AA)
            active_target_pt = (ex, ey)
        elif meas_point is not None:
            active_target_pt = (int(meas_point[0]), int(meas_point[1]))

        # 8. Draw Error Vector (Line from Image Center to Tracked Object)
        if self.show_error_vector and active_target_pt is not None:
            tx, ty = active_target_pt
            # Draw connecting error vector
            cv2.line(vis, (cx_img, cy_img), (tx, ty), (0, 140, 255), 2, cv2.LINE_AA)
            
            # Midpoint text for error distance
            mid_x = (cx_img + tx) // 2
            mid_y = (cy_img + ty) // 2
            dist_val = error_distance if error_distance is not None else float(np.hypot(tx - cx_img, ty - cy_img))
            cv2.putText(vis, f"e = {dist_val:.1f}px", (mid_x + 5, mid_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

        # 9. Draw Telemetry HUD
        if self.show_hud:
            self._draw_hud(vis, frame_idx, status_text, error_distance, velocity, active_target_pt, (cx_img, cy_img), fps, is_live, model_name, label)

        return vis

    def _draw_hud(
        self,
        img: np.ndarray,
        frame_idx: int,
        status_text: str,
        error_distance: Optional[float],
        velocity: Optional[Tuple[float, float]],
        target_pt: Optional[Tuple[int, int]],
        img_center: Tuple[int, int],
        fps: Optional[float] = None,
        is_live: bool = False,
        model_name: Optional[str] = None,
        target_label: Optional[str] = None
    ) -> None:
        """Draw top-left telemetry overlay bar."""
        overlay = img.copy()
        hud_w = 370
        hud_h = 160 if (model_name is not None or target_label is not None) else 135
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)
        cv2.rectangle(img, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 80), 1)

        # Status indicator color
        status_color = (0, 255, 0) if "TRACK" in status_text.upper() else (0, 165, 255)
        if "IDLE" in status_text.upper() or "PAUSED" in status_text.upper():
            status_color = (240, 220, 0) if "PAUSED" in status_text.upper() else (200, 200, 200)
        elif "LOST" in status_text.upper() or "OCCLUDED" in status_text.upper():
            status_color = (0, 0, 255)

        cv2.putText(img, f"STATE: {status_text}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

        # Live Camera Badge
        if is_live:
            cv2.circle(img, (290, 26), 6, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "LIVE", (302, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        # Frame and FPS
        fps_str = f" | FPS: {fps:.1f}" if fps is not None else ""
        cv2.putText(img, f"Frame: #{frame_idx}{fps_str}", (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        y_offset = 66
        if target_label and "IDLE" not in status_text.upper():
            cv2.putText(img, f"Target: {target_label}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1, cv2.LINE_AA)
            y_offset += 18

        if error_distance is not None:
            cv2.putText(img, f"Center Error: {error_distance:6.1f} px", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 180, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Center Error: N/A", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)
        y_offset += 18

        if velocity is not None:
            vx, vy = velocity
            speed = float(np.hypot(vx, vy))
            cv2.putText(img, f"Est. Vel: ({vx:+.1f}, {vy:+.1f}) | {speed:.1f} px/s", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Est. Vel: (0.0, 0.0) px/s", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)
        y_offset += 18

        if model_name is not None:
            cv2.putText(img, f"AI: {model_name[:38]}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 220, 255), 1, cv2.LINE_AA)
            y_offset += 16

        # Controls & Legend
        cv2.putText(img, "[Yellow=CoM  Magenta=Pred  Green=Est]", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, "[Space=Pause & Click | 'r'=Reset | 'q'=Quit]", (20, y_offset + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (140, 200, 255), 1, cv2.LINE_AA)


