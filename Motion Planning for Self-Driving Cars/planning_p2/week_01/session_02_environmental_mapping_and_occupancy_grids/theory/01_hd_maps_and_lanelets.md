# HD Maps & Lanelet Representations

## 1. High-Definition (HD) Vector Maps in Autonomous Driving

Unlike standard consumer GPS navigation maps (e.g. Google Maps or OpenStreetMap), autonomous vehicles require centimeter-accurate **High-Definition (HD) Maps** containing topological, geometric, and semantic traffic data:

```
HD Map Layers:
1. Geometric Layer: 3D point cloud / meshes of physical road boundaries, curbs, barriers.
2. Lane-Level Topological Layer: Left/right lane boundaries, centerline reference paths, lane widths, connectivity.
3. Semantic / Regulatory Layer: Traffic light spatial locations & signal associations, stop signs, crosswalks, speed zones.
```

---

## 2. The Lanelet Representation

A **Lanelet** is an atomic drivable road segment bounded by an interconnected left polyline and right polyline:

- **Left Boundary:** Polyline $[p_0^L, p_1^L, \dots, p_n^L]$
- **Right Boundary:** Polyline $[p_0^R, p_1^R, \dots, p_n^R]$
- **Centerline:** Interpolated geometric path between left and right boundaries $p_i^C = \frac{p_i^L + p_i^R}{2}$
- **Regulatory Elements (RegEls):** Explicit links to applicable traffic rules (e.g., speed limit 50 km/h, stop sign at end node, lane change allowed/prohibited).

### Topological Graph Connections
Lanelets connect through graph adjacency:
- **Predecessor & Successor Lanelets:** For continuous longitudinal travel across road segments.
- **Neighboring Lanelets (Left/Right):** For lane-changing maneuvers.
- **Conflicting Lanelets:** Crossing paths inside intersections to detect potential collision paths with other vehicles.
