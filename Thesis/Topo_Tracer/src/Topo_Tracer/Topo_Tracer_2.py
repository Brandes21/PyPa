def main(seed_xyz, principal_vectors, points_3d, domain_surface, boundary_curves, start_dir, h, num_steps, k, collision_threshold, seed_dist, n_back):   



    import math
    import rhinoscriptsyntax as rs
    import Rhino
    import Rhino.Geometry as rg
    import scriptcontext as sc
    import numpy as np
    from scipy.spatial import KDTree
    from ghpythonlib import treehelpers as tr





    # ------------------------------------------------------------------------
    # 3D UTILITY FUNCTIONS
    # ------------------------------------------------------------------------

    def to_xyz(pt_or_vec):
        """Converts Rhino.Geometry.Point3d or Vector3d (or numeric [x,y,z]) to a Python list [x, y, z]."""
        if hasattr(pt_or_vec, "X"):
            return [pt_or_vec.X, pt_or_vec.Y, pt_or_vec.Z]
        else:
            return [pt_or_vec[0], pt_or_vec[1], pt_or_vec[2]]

    def normalize_3d(vec):
        """Normalize a 3D vector [x, y, z] to unit length. Returns [0,0,0] if near zero length."""
        mag = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
        if mag < 1e-12:
            return [0.0, 0.0, 0.0]
        return [vec[0]/mag, vec[1]/mag, vec[2]/mag]

    def find_closest_neighbors_kd_3d(point, kd_tree, k):
        """
        Given a 3D point [x, y, z] and a KDTree, returns a list of indices of the k closest points.
        """
        distances, indices = kd_tree.query(point, k=k)
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def interpolate_vector_3d_consistent(pt, points_3d, vectors_3d, neighbors, ref_dir):
        """
        Distance-weighted interpolation of neighbor vectors in 3D,
        flipping each neighbor's vector if dot < 0 with respect to 'ref_dir'.

        - pt: [x, y, z], the point where we want the interpolated vector
        - points_3d: Nx3 array of point coords
        - vectors_3d: Nx3 array of principal vectors
        - neighbors: list of indices from the KDTree
        - ref_dir: [dx, dy, dz], the direction from the previous step (or None if first step)
        """
        weights = []
        weighted_vec = [0.0, 0.0, 0.0]

        for i in neighbors:
            vx, vy, vz = vectors_3d[i]  # copy so we can flip locally
            if ref_dir is not None:
                dotp = vx*ref_dir[0] + vy*ref_dir[1] + vz*ref_dir[2]
                if dotp < 0:
                    vx, vy, vz = -vx, -vy, -vz
            
            npt = points_3d[i]
            dx = pt[0] - npt[0]
            dy = pt[1] - npt[1]
            dz = pt[2] - npt[2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            w = 1.0 / (dist + 1e-6)
            weights.append(w)
            
            weighted_vec[0] += vx*w
            weighted_vec[1] += vy*w
            weighted_vec[2] += vz*w

        total_w = sum(weights)
        if total_w > 1e-12:
            weighted_vec[0] /= total_w
            weighted_vec[1] /= total_w
            weighted_vec[2] /= total_w

        mag = (weighted_vec[0]**2 + weighted_vec[1]**2 + weighted_vec[2]**2)**0.5
        if mag < 1e-12:
            return [0.0, 0.0, 0.0]
        
        return [weighted_vec[0]/mag, weighted_vec[1]/mag, weighted_vec[2]/mag]

    def adjust_step_size_3d(current_pt, neighbors, points_3d, step_sign):

        if not neighbors:
            return 0.1 * step_sign
        
        distances = []
        for i in neighbors:
            npt = points_3d[i]
            dx = current_pt[0] - npt[0]
            dy = current_pt[1] - npt[1]
            dz = current_pt[2] - npt[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            distances.append(dist)
        
        avg_dist = sum(distances) / len(distances) if len(distances) else 0.1
        base_step = min(0.1, avg_dist / 2.0)
        return base_step * step_sign

    # ------------------------------------------------------------------------#
    # SURFACE / BREP PROJECTION UTILS
    # ------------------------------------------------------------------------#

    def project_onto_surface(surface, pt3d):
        """
        Project a 3D point onto a Rhino surface or the first face of a Brep.
        Returns [x, y, z] on the surface, or None if projection fails.
        """
        pt_rh = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        
        # If 'surface' is actually a Brep, get the face or use ClosestPoint on the brep
        if isinstance(surface, rg.Brep):
            # For a single-face brep (like an untrimmed surface), we can do:
            #   face = surface.Faces[0]
            #   rc, u, v = face.ClosestPoint(pt_rh)
            #   ...
            # Otherwise, you can do surface.ClosestPoint(pt_rh) to get a face index, etc.
            face = surface.Faces[0] # <--- if you know it has only one face
            rc, u, v = face.ClosestPoint(pt_rh)
            if rc:
                new_pt = face.PointAt(u, v)
                return [new_pt.X, new_pt.Y, new_pt.Z]
            else:
                return None
        
        elif isinstance(surface, rg.Surface):
            rc, u, v = surface.ClosestPoint(pt_rh)
            if rc:
                new_pt = surface.PointAt(u, v)
                return [new_pt.X, new_pt.Y, new_pt.Z]
            else:
                return None
        
        else:
            # If it's something else, or None
            return None

    def is_on_surface(surface, pt3d, tol=0.01):
        """
        Returns True if pt3d is within 'tol' of the surface. 
        """
        if not surface: 
            return True  # no surface provided
        pproj = project_onto_surface(surface, pt3d)
        if pproj is None:
            return False
        dx = pt3d[0] - pproj[0]
        dy = pt3d[1] - pproj[1]
        dz = pt3d[2] - pproj[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return (dist < tol)


    def distance_to_brep_edges(pt3d, boundary_curves):
        """
        Given a 3D point (x,y,z) and a list of boundary curves,
        returns the minimum distance from the point to any of those curves.
        """
        test_pt = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        min_dist = float('inf')
        for crv in boundary_curves:
            
            # ClosestPoint returns (bool, parameter)
            rc, t = crv.ClosestPoint(test_pt)
            if rc:
                cp = crv.PointAt(t)
                dist = cp.DistanceTo(test_pt)
                if dist < min_dist:
                    min_dist = dist
                    close_pt = cp
        return min_dist, close_pt

    def runge_kutta_step_3d(
        current_point, 
        current_dir,
        h, 
        k, 
        principal_vectors, 
        points_3d, 
        step_sign, 
        kd_tree, 
        boundary_curves=None, 
        boundary_tolerance=0.01,prev_dir=None):
        """
        Perform one RK4 step in 3D. Stop if next point is within 'boundary_tolerance'
        of any boundary curve (edges).
        Returns the next 3D point or None if out-of-bound / near boundary.
        """
        
        # 1) Neighbors & dynamic step size
        neighbors = find_closest_neighbors_kd_3d(current_point, kd_tree, k)
        h = adjust_step_size_3d(current_point, neighbors, points_3d, step_sign)
        
        # k1
        k1_dir = interpolate_vector_3d_consistent(current_point, points_3d, principal_vectors, neighbors, current_dir)
        mid1 = [
            current_point[0] + 0.5*h*k1_dir[0],
            current_point[1] + 0.5*h*k1_dir[1],
            current_point[2] + 0.5*h*k1_dir[2]
        ]
        
        # k2
        neigh_mid1 = find_closest_neighbors_kd_3d(mid1, kd_tree, k)
        k2_dir = interpolate_vector_3d_consistent(mid1, points_3d, principal_vectors, neigh_mid1, k1_dir)
        mid2 = [
            current_point[0] + 0.5*h*k2_dir[0],
            current_point[1] + 0.5*h*k2_dir[1],
            current_point[2] + 0.5*h*k2_dir[2]
        ]
        
        # k3
        neigh_mid2 = find_closest_neighbors_kd_3d(mid2, kd_tree, k)
        k3_dir = interpolate_vector_3d_consistent(mid2, points_3d, principal_vectors, neigh_mid2, k2_dir)
        end_pt = [
            current_point[0] + h*k3_dir[0],
            current_point[1] + h*k3_dir[1],
            current_point[2] + h*k3_dir[2]
        ]
        
        # k4
        neigh_end = find_closest_neighbors_kd_3d(end_pt, kd_tree, k)
        k4_dir = interpolate_vector_3d_consistent(end_pt, points_3d, principal_vectors, neigh_end, k3_dir)
        
        # Summation
        dx = h*(k1_dir[0] + 2*k2_dir[0] + 2*k3_dir[0] + k4_dir[0]) / 6.0
        dy = h*(k1_dir[1] + 2*k2_dir[1] + 2*k3_dir[1] + k4_dir[1]) / 6.0
        dz = h*(k1_dir[2] + 2*k2_dir[2] + 2*k3_dir[2] + k4_dir[2]) / 6.0
        
        next_point = [
            current_point[0] + dx,
            current_point[1] + dy,
            current_point[2] + dz
        ]
        next_dir = k4_dir

        return next_point, next_dir

    def build_polyline_curve_3d(poly_pts):
        pts3d = [rg.Point3d(pt[0], pt[1], pt[2]) for pt in poly_pts]
        poly = rg.Polyline(pts3d)
        return rg.PolylineCurve(poly)
    # ------------------------------------------------------------------------
    # COLLISION / MERGING CHECKS IN 3D
    # ------------------------------------------------------------------------


    def closest_point_on_polyline_3d(pt3d, poly_curve):
        """
        Return (closest_point, distance) from a 3D point to a polyline curve.
        """
        test_pt = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        rc, t = poly_curve.ClosestPoint(test_pt)
        if rc:
            cp = poly_curve.PointAt(t)
            dist = cp.DistanceTo(test_pt)
            return [cp.X, cp.Y, cp.Z], dist
        else:
            return None, float('inf')

    def find_closest_existing_line_3d(next_point, existing_trajectories, threshold):
        """
        Among all previously traced lines (in 3D), find if 'next_point' is within 'threshold' of any line.
        Returns (closest_line_index, closest_point_on_line, distance).
        """
        min_dist = float('inf')
        closest_line_idx = None
        closest_pt = None
        
        for i, (polyline_pts, poly_curve) in enumerate(existing_trajectories):
            cp, dist = closest_point_on_polyline_3d(next_point, poly_curve)
            if dist < min_dist:
                min_dist = dist
                closest_line_idx = i
                closest_pt = cp
        
        if min_dist < threshold:
            return (closest_line_idx, closest_pt, min_dist)
        else:
            return (None, None, float('inf'))


    def distance_to_seedpt(pt):
        min_dist = float('inf')
        closest_seed = None
        for seed in seed_points:
            dist_to_s = seed.DistanceTo(pt)
            if dist_to_s < min_dist:
                min_dist = dist_to_s
                closest_seed = seed
        return min_dist, closest_seed

    # ------------------------------------------------------------------------
    # MAIN PSL TRACING FUNCTION (3D)
    # ------------------------------------------------------------------------

    def trace_principal_stress_line_3d(
        p_start, 
        h, 
        num_steps, 
        k, 
        principal_vectors, 
        points_3d, 
        boundary_curves=None, 
        boundary_tolerance= None,
        existing_trajectories=None,
        collision_threshold=None,
        step_sign=1,
        start_dir=None):
        """
        Trace a principal stress line starting from p_start (3D) using RK4 in 3D,
        following the provided starting direction 'start_dir'.
        """
        if existing_trajectories is None:
            existing_trajectories = []
        
        trajectory = [rg.Point3d(p_start[0], p_start[1], p_start[2])]
        bridging_line = None
        
        current_point_3d = p_start[:]
        # Use the provided starting direction
        current_dir = start_dir
        nn = 0
        for _ in range(num_steps):
            next_result = runge_kutta_step_3d(
                current_point_3d, 
                current_dir,
                h, 
                k, 
                principal_vectors,
                points_3d,
                step_sign,
                kd_tree,
                boundary_curves,
                boundary_tolerance,
            )
            nn = nn+1
            if not next_result or next_result[0] is None:
                print("PSL stopped: near or off boundary.")
                break
            next_pt, next_dir = next_result
            
            # Collision check with existing lines
            if existing_trajectories:
                line_idx, close_pt, dist = find_closest_existing_line_3d(next_pt, existing_trajectories, collision_threshold)
                if line_idx is not None and close_pt is not None:
                    steps_back = n_back
                    if len(trajectory) > steps_back:
                        bridging_line = [
                            trajectory[-steps_back],
                            rg.Point3d(close_pt[0], close_pt[1], close_pt[2])
                        ]
                        del trajectory[-(steps_back-1):]
                    else:
                        bridging_line = [
                            trajectory[-1],
                            rg.Point3d(close_pt[0], close_pt[1], close_pt[2])
                        ]
                    print("PSL stopped: close to line %d at distance %.3f - merging initiated" % (line_idx, dist))
                    break
        
            if seed_points:
 
                    dist_to_seed, seed_pt = distance_to_seedpt(rg.Point3d(next_pt[0],next_pt[1],next_pt[2]))
    
                    if dist_to_seed < seed_dist:
       
                        # create connection to close seed_point
                        bridging_line = [
                        trajectory[-1],
                        rg.Point3d(seed_pt[0], seed_pt[1], seed_pt[2])
                        ]
                        break

            if boundary_curves:
                dist_to_edge,cp = distance_to_brep_edges(next_pt, boundary_curves)
                
                if dist_to_edge < boundary_tolerance:
                    # We consider that "off" or "too close" => stop
                
                    bridging_line = [cp,trajectory[-1]]
                    
                    break
    
            trajectory.append(rg.Point3d(next_pt[0], next_pt[1], next_pt[2]))
            current_point_3d = next_pt
            current_dir = next_dir
        
        return trajectory, bridging_line

    # ------------------------------------------------------------------------
    # "MAIN" LOGIC IN GRASSHOPPER
    # ------------------------------------------------------------------------
    # Below is just an example of how you'd use these functions in GH:
    #   1) Build the 3D KDTree from your data.
    #   2) Loop over seed points.
    #   3) Keep a global 'existing_trajectories' for collisions.
    # ------------------------------------------------------------------------

    # Example usage in a GH Python component:

    # 1) Convert your input points/vectors to lists of [x,y,z].
    points_array_3d = [to_xyz(pt) for pt in points]

    boundary_curves = boundary_curves #get_brep_edge_curves(domain_surface)


    # 1.1) Make vector field globally consistent (local consistency is handled on the go)
    def get_knn_adjacency(points_3d, k=6):
        """
        Build adjacency by connecting each point to its 'k' nearest neighbors.
        points_3d: Nx3 array (or list) of 3D coords
        Returns: adjacency (list of lists),
                where adjacency[i] is the list of neighbor indices for point i.
        """
        points_np = np.array(points_3d)
        tree = KDTree(points_np)
        adjacency = []

        n = len(points_3d)
        for i in range(n):
            # Query K+1 because the nearest neighbor list includes the point itself at distance 0
            dists, indices = tree.query(points_np[i], k=k+1)
            # Drop 'i' itself
            neighbors = [idx for idx in indices if idx != i]
            adjacency.append(neighbors)
        
        return adjacency

    def unify_vector_field(points_3d, vectors_3d, adjacency):
        """
        points_3d: Nx3 array of point positions
        vectors_3d: Nx3 array of directions
        adjacency: a list of lists, adjacency[i] = indices of neighbors of i
        """
        visited = [False]*len(points_3d)
        queue = [0]  # start BFS from index 0, or any random index
        visited[0] = True
        
        while queue:
            i = queue.pop(0)
            v_i = vectors_3d[i]
            
            for nbr in adjacency[i]:
                if not visited[nbr]:
                    # Compare dot product
                    dotp = (v_i[0]*vectors_3d[nbr][0] + 
                            v_i[1]*vectors_3d[nbr][1] + 
                            v_i[2]*vectors_3d[nbr][2])
                    if dotp < 0:
                        # Flip neighbor's vector
                        vectors_3d[nbr][0] = -vectors_3d[nbr][0]
                        vectors_3d[nbr][1] = -vectors_3d[nbr][1]
                        vectors_3d[nbr][2] = -vectors_3d[nbr][2]
                    visited[nbr] = True
                    queue.append(nbr)
        
        # Now vectors_3d should be locally consistent
        return vectors_3d

    adjacency = get_knn_adjacency(points_array_3d,k=1)
    constant_vectors = unify_vector_field(points,principal_vectors,adjacency)
    vectors_array_3d = [to_xyz(vec) for vec in constant_vectors]



    w = tr.list_to_tree(vectors_array_3d)





    # 2) Build a 3D KD-tree
    points_np = np.array(points_array_3d)  # shape (N, 3)
    kd_tree = KDTree(points_np)

    # 3) We'll store PSL results in 'existing_trajectories' as: ( [pt3d_list], polyline_curve )
    existing_trajectories = []
    bridging_lines_out = []

    # If 'domain_surface' is a Rhino Surface or a single-face Brep, you can store it here:
    surface_id = None
    if domain_surface:
        surface_id = domain_surface  # rename for clarity

    # 4) Loop over seed points (3D)
    # Assume seed_points and seed_vectors are lists of seed points and their associated starting vectors
    for seed, svec in zip(seed_points, seed_vectors):
        seed_xyz = to_xyz(seed)
        start_direction = to_xyz(svec)  # convert starting vector if necessary
        
        psl, bridge = trace_principal_stress_line_3d(
            seed_xyz,
            h,
            num_steps,
            k,
            principal_vectors=vectors_array_3d,
            points_3d=points_array_3d,
            boundary_curves=boundary_curves,
            boundary_tolerance=collision_threshold,
            existing_trajectories=existing_trajectories,
            collision_threshold = collision_threshold,
            start_dir=start_direction  # use the provided starting vector
        )
        
        # Build the cached polyline for collisions
        psl_curve_3d = build_polyline_curve_3d([[pt.X, pt.Y, pt.Z] for pt in psl])
        existing_trajectories.append((psl, psl_curve_3d))
        if bridge:
            bridging_lines_out.append(bridge)


    # 5) Output
    a = tr.list_to_tree(existing_trajectories)  # Each branch: (pts, polylineCurve)
    b = tr.list_to_tree(bridging_lines_out)
    return a, b
