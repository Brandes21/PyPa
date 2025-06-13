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


a, b = Tracer_A.main(h,num_steps,k,collision_threshold,snap_radius,n_back,sample_count,k_edge,seed_points,principal_vectors,points,domain_surface,boundary_curves):

Outputs:
1. main_lines = traced PSL's (a)
2. Edge and merge connections (b)


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


a, b, c = Tracer_B.main(h,num_steps,k,collision_threshold,snap_radius,n_back,offset_distance,sample_interval,sample_count,k_edge,ratio_minseed_coll,seed_point,principal_vectors,points,domain_surface,boundary_curves):


Outputs:
1. Traced PSL's (a)
2. Edge and merge connections (b)
3. Valid seeds (c)



## Tracer C
Tracer C takes one initial seed and generates a equally distributed PSL network automatically based on the forces in the design domain.

Inputs:
1. ghenv = global GH variable "ghenv"
2. h = step length
3. num_steps = maximum iterations (steps)
4. k = k nearest neighbours for interpolation
5. collision_threshold = distance threshold
6. snap_radius = radius in which snapping occurs for merging 
7. n_back = if merging is initiated how many steps back the connection is (to avoid 90° connections)
8. max_offset_distance = max distance for the offsetting of sample points/seeds
9. sample_interval = sampling of PSL
10. sample_count = sampling of boundaries for edge connection 
11. k_edge = how many edge samples to check for when building the edge connection
12. ratio_minseed_coll = ratio for minimum seed distance based on collision threshold
13. seed_point = manually selected seed point
14. principal_vectors = vector field
15. points = points of the structural mesh
16. domain_surface = surface or brep
17. boundary_curves = edge curves of the surface (OBS: needs to be a list)
18. strength = strength value for inlay
19. stress_values = stress values from fx Karamba3D (for each point of the points input)


a, b ,c = Tracer_C.main(ghenv,h,num_steps,k,collision_threshold,snap_radius,n_back,max_offset_distance,sample_interval,sample_count,ratio_minseed_coll,seed_point,principal_vectors,points,domain_surface,boundary_curves,strength,stress_values)

Outputs:
1. Traced PSL's (a)
2. Edge and merge connections (b)
3. Valid seeds (c)




## Example files
A GH example file can be found on: https://github.com/Brandes21/PyPa/tree/main/Example_files 



For any questions or in case of bugs feel free to contact me on: niclasbrandt97@gmail.com












