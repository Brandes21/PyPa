# Introduction
This package includes 3 PSL tracing functions that can trace a layout of principal stress lines with custom features inside the GH/Rhino environment. In total there are three functions inside the package ranging from manual seeds point selection, to automated selction and finally to automated force-based selection all tracing a principal stress line field (vector field).

## Theory
All tracers are based on the same tracing logic. They are based on biderctional tracing where the RK4 method is utilized to approximate the next point. The interpolation between points in the discrete structured mesh is done using k-nearest neighbours IDW. Additional to the logic all tracers incorporate custom features such as: Edge detection, loop detection, distance thresholds, and optional merging with snapping. This ensures a tool that can be informed by user defined constraints generating a custom layout adhering to geometric rules. It works for 3D shapes and uses projection to stay on the surface.


## Tracer A
Tracer A takes a manual selection of seed points as an input and traces those bidirectionally. The order of the seed point list defines the order in which the seeds are traced. 

Inputs:
1. h = step length
2. num_steps = maximum iterations (steps)
3. k = k nearest neighbours for interpolation
4. collision_threshold = distance threshold
5. snap_radius = radius in which snapping occurs for merging 
6. n_back = if merging is initiated how many steps back the connection is (to avoid 90° connections)
7. sample_count = sampling of boundaries for edge connection
8. k_edge = how many edge samples to check for when building the edge connection
9. seed_points = manually selected seed points
10. principal_vectors = vector field
11. points = points of the structural mesh
12. domain_surface = surface or brep
13. boundary_curves = edge curves of the surface (OBS: needs to be a list)


def main(h,num_steps,k,collision_threshold,snap_radius,n_back,sample_count,k_edge,seed_points,principal_vectors,points,domain_surface,boundary_curves):

Outputs:
1. main_lines = traced PSL's
2. Edge and merge connections




## Tracer B
Tracer B takes one initial seed and generates a equally distributed PSL network automatically. 

Inputs:
1. h = step length
2. num_steps = maximum iterations (steps)
3. k = k nearest neighbours for interpolation
4. collision_threshold = distance threshold
5. snap_radius = radius in which snapping occurs for merging 
6. n_back = if merging is initiated how many steps back the connection is (to avoid 90° connections)
7. offset_distance = distance for the offsetting of sample points/seeds
8. sample_interval = sampling of PSL
9. sample_count = sampling of boundaries for edge connection 
10. k_edge = how many edge samples to check for when building the edge connection
11. ratio_minseed_coll = ratio for minimum seed distance based on collision threshold
12. seed_point = manually selected seed point
13. principal_vectors = vector field
14. points = points of the structural mesh
15. domain_surface = surface or brep
16. boundary_curves = edge curves of the surface (OBS: needs to be a list)


def main(h,num_steps,k,collision_threshold,snap_radius,n_back,offset_distance,sample_interval,sample_count,k_edge,ratio_minseed_coll,seed_point,principal_vectors,points,domain_surface,boundary_curves):


Outputs:
1. Traced PSL's
2. Edge and merge connections
3. Valid seeds



## Tracer C
Tracer C takes one initial seed and generates a equally distributed PSL network automatically based on the forces in the design domain.

Inputs:
1. h = step length
2. num_steps = maximum iterations (steps)
3. k = k nearest neighbours for interpolation
4. collision_threshold = distance threshold
5. snap_radius = radius in which snapping occurs for merging 
6. n_back = if merging is initiated how many steps back the connection is (to avoid 90° connections)
7. offset_distance = distance for the offsetting of sample points/seeds
8. sample_interval = sampling of PSL
9. sample_count = sampling of boundaries for edge connection 
10. k_edge = how many edge samples to check for when building the edge connection
11. ratio_minseed_coll = ratio for minimum seed distance based on collision threshold
12. seed_point = manually selected seed point
13. principal_vectors = vector field
14. points = points of the structural mesh
15. domain_surface = surface or brep
16. boundary_curves = edge curves of the surface (OBS: needs to be a list)


Outputs:
1. Traced PSL's
2. Edge and merge connections
3. Valid seeds

















