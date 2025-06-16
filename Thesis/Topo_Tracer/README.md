# Introduction
This package includes 2 python functions that together are able to extract the topological skeleton of the stress tensor field in the GH/Rhino environment. 


# Theory
At first degenerate points are identified by comparing the tensors two eigenvalues for equality with a user defined tolerance. Due to teh discrete mesh multiple points around a degeneracy may be identified as degenrate which is why clustering is applied and the mean point of each cluster is chosen. After localizing degenerate points they are classified by delta and the their tensor index. Based on the classification and the theory from Delmarcelle and Hesselink (94) the separating directions are extracted as vectors. Thsi information gets passed to the second function that traces the separating streamlines. Togetehr that forms the topological skeleton. 



## Topo_Tracer_1
This function locates degenerate points and calculates the separating direction for each of them.


import package in GH component: from Topo_Tracer import Topo_Tracer_1


Inputs:
1. neighbors  = for partial derivative
2. tol_deg = Tolerance for degeneracy
3. offset_eps = For partial derivative 
4. cluster_threshold = distance threshold for clustering
5. seed_amplitude = distance of the new seed points from the degenerate points
6. index_radius = radius of the jordan curve used for index calculation
7. n_sep_merged = the number of separating streamlines for higher order degeneracies (saddles)
8. domain_surface = surface or brep
9. points = points from the structural mesh in karamba (points where stress values are defined)
10. stress_values_1 = stress values from first principal direction
11. stress_values_2 = stress values from secondary principal direction
12. vectors_1 = first principal stress trajectories as vectors
12. vectors_2 = first principal stress trajectories as vectors


Outputs:
1. singularities = degenerate points
2. seed_vectors_field_1 = separating directions for first principal stress field given as 3D vectors
3. seed_vectors_field_2 = separating directions for secondary principal stress field given as 3D vectors
4. new_seeds_1 = new seed for each separating direction slighlty offsetted from the degenerate point for first principal stress field 
5. new_seeds_2 = new seed for each separating direction slighlty offsetted from the degenerate point for first principal stress field 
6. dege_index = tensor index of the degenerate points measured by eigenvector rotations
7. dege_type = classification/type of degenerate point (wedge, trisector or merged singualrity)

Call function in GH:
singularities, seed_vectors_field_1, seed_vectors_field_2, new_seeds_1, new_seeds_2, dege_index, dege_type = Topo_Tracer_1.main(domain_surface, points, stress_values_1, vectors_1, stress_values_2, vectors_2,tol_deg, offset_eps, neighbors, cluster_threshold, seed_amplitude, index_radius, n_sep_merged)



## Topo_Tracer_2
This function takes the calculated separating directions and the new seed points and traces the separatrices.

import package in GH component: from Topo_Tracer import Topo_Tracer_2


Inputs:
1. points = points from the structural mesh in karamba (points where stress values are defined)
2. principal_vectors = principal stress trajectories/vectors for chosen field
3. seed_points = the new seed points from Topo_Tracer_1
4. seed_vectors = the new seed vectors / separating direction from Topo_Tracer_1
5. boundary_curves = the curves that form the boundary of the design
6. domain_surface = surface or brep
7. k = k nearest neighbours for interpolation
8. h = step length
9. num_steps = max iterations
10. step_sign = 1
11. boundary_tolerance = from what distance boundaries are detected
12. collision_threshold = threshold for when a PSL merges into anotehr psl
13. n_back = how many traced points to go back when merging
14. seed_dist = distance at what snapping to other seed points occurs.


Outputs:
1. separatrices = traced separatrices
2. connecting_points = lists with start and end point of connecting lines


Call function in GH:
separatrices, connecting_points = Topo_Tracer_2.main(points, principal_vectors, seed_points, seed_vectors,boundary_curves, domain_surface, k, h, num_steps, step_sign , boundary_tolerance, collision_threshold, n_back, seed_dist)



## Example files
A GH example file can be found on: https://github.com/Brandes21/PyPa/tree/main/Example_files 



For any questions or in case of bugs feel free to contact me on: niclasbrandt97@gmail.com