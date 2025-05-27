# PSL Tracer package
This package includes 3 PSL tracing functions that can trace a layout of principal stress lines with custom features explained in the following.


## Dependencies
The package depends on the rhino/grasshopper environment thus one can use it only in the python component in grasshopper.


## Tracing logic
The tracing builds up on the logic shown in the following image:


![Image 1]](/assets/images/Asset 28IDW.jpg "IDW")



Encompassing IDW interpolation method, RK4 method, and bi-directional tracing. (for 3d shapes each new step is projected back on the surface/brep)



# Features
1. Distance threshold
2. Loop detection
3. Merging





For any questions feel free to contact me!