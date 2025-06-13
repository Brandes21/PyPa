# Introduction
This python package tailored to the GH/Rhino environment allows for distance based removal of curves. When a bunch of curves are too close to adjacent curves the python package identifies the curves that have to be removed in order to fulfill the distance constraints but also keep as many curves as possible.



## Theory
A group of curves can be representet as a graph where each curve is a vertex (curve = V) and each edge represents a distance violation between two curves (E(u,v) if dist(u,v)< dist_limit). To find the maximum set of curves that do not violate the distance constraint one has to identify the maximum independent set of vertices that do not have an edge (thus no distance violation). This is done utilizing a greedy MIS (maximum independent set) algorithm on the graph based. The heuristic removes the vertices with the most conflicts at first offering a approximation to the "perfect" solution as it is an NP-hard problem.


## Input
The python package takes at most three inputs:

1. A list of curves                                 (Required)
2. A distance threshold                             (Required)
3. User-defined curves that should not be removed.  (Optional)

The python package can just be called inside of a python3 grasshopper component. 

## Example files
A GH example file can be found on: 



For any questions or in case of bugs feel free to contact me on: niclasbrandt97@gmail.com

