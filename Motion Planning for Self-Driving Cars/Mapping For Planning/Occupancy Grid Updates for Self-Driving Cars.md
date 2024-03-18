# Occupancy Grid Updates for Self-Driving Cars

Downsampling is the process of reducing the number of Lidar points to be considered by removing or ignoring redundant Lidar points.

* Removal of redundant points
* Improves computation

## Removal of overhanging objects

* Removing all Lidar points that are above a given threshold of the height limit of the car

## Removal of ground plane

* Difficult to estimate due to several complications
* Differing road geometries
* Curbs, lane boundaries
* Don't want to miss small objects

## Ground plane Classification

* Utilize segmentation to remove points of road elements
* Keep points from no drivable surfaces

## Removal of Dynamic Objects

This can be done once again with the reliance on the perception stack which must detect and track all dynamic objects in the scene.

The 3D bounding box of the detected dynamic object is used to remove all the points in the affected area. A small threshold is also added to the size of the bounding box used to account for any small mistakes in the perception algorithm object location estimate increasing robustness of the point remove filter.

### Removal of Dynamic Objects Improvement

* Not all vehicles are dynamic, so they should be included
* History of dynamic object location can be used to identify parked vehicle
* The dynamic objects are identified from the previous
LIDAR frame
* Predicted future location improvement

## Projection of LIDAR Onto a 2D Plane

Simple solution:

* Collapse all points by Zeroing the Z coordinate
* Sum up the number of LIDAR points in each grid location
  * More points indicated greater chance of occupation of that grid cell
