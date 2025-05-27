def main(h,num_steps,k,collision_threshold,merge_radius,n_back,sample_count,k_edge,seed_points,principal_vectors,points,domain_surface,boundary_curves):


    import math
    import rhinoscriptsyntax as rs
    import Rhino
    import Rhino.Geometry as rg
    import scriptcontext as sc
    import numpy as np
    from scipy.spatial import KDTree
    from ghpythonlib import treehelpers as tr
    import cProfile, pstats, io


    # ------------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------------





    #set values
    tolerance_proj = 2000        #Max distance for projection
    boundary_tolerance= h
    closing_threshold=h


    # ------------------------------------------------------------------------
    # 3D UTILITY FUNCTIONS
    # ------------------------------------------------------------------------





    def distance_3d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

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



    # ------------------------------------------------------------------------
    # TRACING FUNCTIONS
    # ------------------------------------------------------------------------


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
        boundary_tolerance=boundary_tolerance,prev_dir=None):
        """
        Perform one RK4 step in 3D. Stop if next point is within 'boundary_tolerance'
        of any boundary curve (edges).
        Returns the next 3D point or None if out-of-bound / near boundary.
        """
        
        # 1) Neighbors & dynamic step size
        neighbors = find_closest_neighbors_kd_3d(current_point, kd_tree, k)
        h = h * step_sign#adjust_step_size_3d(current_point, neighbors, points_3d, step_sign)
        
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

        if domain_surface is not None:
            projected = project_onto_surface(domain_surface, next_point, tolerance_proj)
            #print(projected)
            if projected is None:
                # Means we're off domain or near an open edge you consider invalid
                return None
            next_point = projected  # accept the projected coordinate


        # Otherwise return the next valid point
        return next_point, next_dir


    # ------------------------------------------------------------------------#
    # SURFACE / BREP PROJECTION UTILS
    # ------------------------------------------------------------------------#

    def project_onto_surface(surface, pt3d, tolerance_proj):
        pt_rh = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        
        if isinstance(surface, rg.Brep):
            # Note the 6 returned values:
            rc, closest_pt, cindex, u, v, normal = surface.ClosestPoint(pt_rh, tolerance_proj)
            if rc:
                # Check if cindex is a face
                if cindex.ComponentIndexType == rg.ComponentIndexType.BrepFace:
                    face_id = cindex.Index
                    face = surface.Faces[face_id]
                    
                    # 'closest_pt' is the 3D point on that face
                    # Optionally confirm it's within the face domain or do face.IsPointOnFace(u,v)
                    
                    return [closest_pt.X, closest_pt.Y, closest_pt.Z]
                else:
                    # The closest component might be an edge or vertex
                    return None
            else:
                return None
        
        elif isinstance(surface, rg.Surface):
            rc, u, v = surface.ClosestPoint(pt_rh)
            if rc:
                pt_srf = surface.PointAt(u, v)
                return [pt_srf.X, pt_srf.Y, pt_srf.Z]
            else:
                return None
        else:
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



    # ------------------------------------------------------------------------
    # COLLISION/DISTANCE CHECKS
    # ------------------------------------------------------------------------


    def build_polyline_curve_3d(poly_pts):
        pts3d = [rg.Point3d(pt[0], pt[1], pt[2]) for pt in poly_pts]
        poly = rg.Polyline(pts3d)
        return rg.PolylineCurve(poly)

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

    def is_point_close_to_any(point_list, single_point, threshold):
        flat = [item for sublist in point_list for item in sublist]
        #print(f"single = {single_point}")
        single_point_1 = rg.Point3d(single_point[0],single_point[1],single_point[2])
        
        
        for pt in flat:
            #print(f"pt = {pt}")
            if single_point_1.DistanceTo(pt) <= threshold:
                return pt
        return single_point

    def find_most_parallel_boundary_point(
        current_pt,
        last_step_vec,
        boundary_points,
        k=10):
        """
        Given a KD-tree of boundary sample points (boundary_kdtree)
        and the original array (boundary_points),
        find the boundary sample whose direction from current_pt 
        is most opposite (or "most parallel" – depends on how you measure) 
        to last_step_vec. Return (best_point, min_dist).

        boundary_points should be a Nx3 np.array or a list of [x,y,z].
        """
        
        if boundary_kdtree is None:
            return None, float('inf')
        
        cur_np = np.array(current_pt, dtype=float)
        step_np = np.array(last_step_vec, dtype=float)
        
        # if zero length, can't define direction
        step_len = np.linalg.norm(step_np)
        if step_len < 1e-12:
            return None, float('inf')
        step_np /= step_len
        
        # 1) Quickly find the 'k' closest boundary samples to current_pt
        #    using the KD-tree:
        #    If k=1, you'll get a single distance & index as floats
        #    If k>1, arrays
        distances, indices = boundary_kdtree.query(cur_np, k=k)

        # If k=1, make them arrays for consistency
        if k == 1:
            distances = [distances]
            indices = [indices]

        # 2) Among these k points, pick the direction which has 
        #    the smallest dot or largest negative dot, etc.
        best_dot = 9999
        best_pt = None
        min_dist = float('inf')

        # boundary_points is Nx3

        for dist, idx in zip(distances, indices):
            cand_np = boundary_points_array[idx]  # shape (3,)
            dir_vec = cand_np - cur_np
            dir_len = np.linalg.norm(dir_vec)
            if dir_len < 1e-12:
                continue
            dir_unit = dir_vec / dir_len
            dot_val = np.dot(dir_unit, step_np)
            
            # Suppose you want the direction that is "most parallel but reversed"
            # => you'd pick the smallest dot_val. If you want "most parallel, same direction,"
            # => you'd pick largest dot_val. 
            if dot_val < best_dot:
                best_dot = dot_val
                best_pt = cand_np
                # optional: track the boundary distance as well
                if dist < min_dist:
                    min_dist = dist

        # Convert best_pt to Rhino point
        if best_pt is not None:
            return rg.Point3d(best_pt[0], best_pt[1], best_pt[2]), min_dist
        else:
            return None, float('inf')


    # ------------------------------------------------------------------------
    # TRACE PSL IN BOTH DIRECTIONS (3D)
    # ------------------------------------------------------------------------               


    def trace_psl_both_directions_in_one_loop_3d(
        seed_point_3d,
        h, 
        num_steps,
        k,
        principal_vectors, 
        points_3d,
        kd_tree,
        boundary_curves=None,
        boundary_tolerance=boundary_tolerance,
        existing_trajectories=None,
        collision_threshold=collision_threshold,
        closing_threshold=closing_threshold,
        existing_merge_pts = None):
        """
        Traces a PSL forward (+1) and backward (-1) in ONE loop, letting each direction
        stop or collide independently. If forward is done, backward can keep going (and vice versa).
        If both are done, or tips meet, we stop entirely.
        """
        if existing_trajectories is None:
            existing_trajectories = []
        
        import Rhino.Geometry as rg
        from math import sqrt

        # Initialize lines
        forward_line = [rg.Point3d(*seed_point_3d)]
        backward_line = [rg.Point3d(*seed_point_3d)]

        # Current states
        f_current_pt = list(seed_point_3d)
        f_current_dir = None
        b_current_pt = list(seed_point_3d)
        b_current_dir = None

        bridging_lines = []
        merge_points = []

        # NEW: Flags for forward/backward
        forward_active = True
        backward_active = True

        for step_i in range(num_steps):
            # --------------------------------------------------
            # FORWARD STEP (if still active)
            # --------------------------------------------------
            if forward_active:
                f_next = runge_kutta_step_3d(
                    f_current_pt,
                    f_current_dir,
                    h,
                    k,
                    principal_vectors,
                    points_3d,
                    step_sign=+1,
                    kd_tree=kd_tree,
                    boundary_curves=boundary_curves,
                    boundary_tolerance=boundary_tolerance
                )
                if not f_next or (f_next[0] is None):
                    print(f"PSL forward direction stopped at iteration {step_i}.")
                    # Do NOT break from the entire loop
                    forward_active = False
                else:
                    # Accept the new point
                    f_next_pt, f_next_dir = f_next
                    
                    # Optional collision check
                    if existing_trajectories:
                        line_idx, close_pt, dist_cl = find_closest_existing_line_3d(
                            f_next_pt, existing_trajectories, collision_threshold
                        )
                        if len(existing_merge_pts) > 1 and close_pt is not None:
                            
                            close_pt = is_point_close_to_any(existing_merge_pts, close_pt, merge_radius)

                        if line_idx is not None and close_pt is not None:
                            # Instead of break, we do bridging & disable forward
                            steps_back = n_back
                            if len(forward_line) > steps_back:
                                bridging_line = rg.Line(
                                    forward_line[-steps_back],
                                    rg.Point3d(*close_pt)
                                )
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))
                                del forward_line[-(steps_back-1):]
                            if len(forward_line)==0:
                                print("too short for bridge")
                            else:
                                bridging_line = rg.Line(
                                    forward_line[-1],
                                    rg.Point3d(*close_pt)
                                )
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))
                            
                            print(f"PSL forward collided with line {line_idx} at dist {dist_cl:.3f}.")
                            forward_active = False

                        # 2) Check distance to boundary edges, if we have them

                    if boundary_curves:
                        dir_vec = [-1*x for x in f_next_dir]
                        cp,dist_to_edge = find_most_parallel_boundary_point(f_next_pt,dir_vec,boundary_curves,k_edge)
                        if dist_to_edge < boundary_tolerance:
                            # We consider that "off" or "too close" => stop
                            print(f"PSL forward reached boundary.")
                            forward_active = False
                            bridging_line = rg.Line(
                                    forward_line[-1],
                                    rg.Point3d(cp)
                            )
                            bridging_lines.append(bridging_line)
                            
                    # If still active after collision checks, append the step
                    if forward_active:
                        forward_line.append(rg.Point3d(*f_next_pt))
                        f_current_pt = f_next_pt
                        f_current_dir = f_next_dir

            # --------------------------------------------------
            # BACKWARD STEP (if still active)
            # --------------------------------------------------
            if backward_active:
                b_next = runge_kutta_step_3d(
                    b_current_pt,
                    b_current_dir,
                    h,
                    k,
                    principal_vectors,
                    points_3d,
                    step_sign=-1,
                    kd_tree=kd_tree,
                    boundary_curves=boundary_curves,
                    boundary_tolerance=boundary_tolerance
                )
                if not b_next or (b_next[0] is None):
                    print(f"PSL backward direction stopped at iteration {step_i}.")
                    backward_active = False
                else:
                    b_next_pt, b_next_dir = b_next
                    
                    if existing_trajectories:
                        line_idx, close_pt, dist_cl = find_closest_existing_line_3d(
                            b_next_pt, existing_trajectories, collision_threshold
                        )
                        if len(existing_merge_pts) > 1 and close_pt is not None:
                            close_pt = is_point_close_to_any(existing_merge_pts, close_pt, merge_radius)

                        if line_idx is not None and close_pt is not None:
                            steps_back = n_back
                        
                            if len(backward_line) > steps_back:
                                bridging_line = rg.Line(
                                    backward_line[-steps_back],
                                    rg.Point3d(*close_pt)
                                )
                                del backward_line[-(steps_back-1):]
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))

                            if len(backward_line)==0:
                                print("too short for bridge")
                            else:
                                bridging_line = rg.Line(
                                    backward_line[-1],
                                    rg.Point3d(*close_pt)
                                )
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))
                            print(f"PSL backward collided with line {line_idx} at dist {dist_cl:.3f}.")
                            backward_active = False




                    #print(f"direction = {b_current_dir}")
                    if boundary_curves:
                        cp,dist_to_edge = find_most_parallel_boundary_point(b_next_pt,b_next_dir,boundary_curves,k_edge)
                        if dist_to_edge < boundary_tolerance:
                            # We consider that "off" or "too close" => stop
                            print(f"PSL forward reached boundary.")
                            backward_active = False
                            bridging_line = rg.Line(
                                    backward_line[-1],
                                    rg.Point3d(cp)
                            )
                            bridging_lines.append(bridging_line)





                    if backward_active:
                        backward_line.append(rg.Point3d(*b_next_pt))
                        b_current_pt = b_next_pt
                        b_current_dir = b_next_dir

            # --------------------------------------------------
            # If BOTH directions are inactive => stop
            # --------------------------------------------------
            if not forward_active and not backward_active:
                print("Both directions inactive => done.")
                break

            # --------------------------------------------------
            # TIP DISTANCE CHECK
            # (only if both are active, or up to you)
            # --------------------------------------------------
            if forward_active and backward_active and step_i >= 10:
                f_tip = [forward_line[-1].X, forward_line[-1].Y, forward_line[-1].Z]
                b_tip = [backward_line[-1].X, backward_line[-1].Y, backward_line[-1].Z]
                dist_fb = sqrt(
                    (f_tip[0] - b_tip[0])**2 +
                    (f_tip[1] - b_tip[1])**2 +
                    (f_tip[2] - b_tip[2])**2
                )
                if dist_fb < closing_threshold:
                    # bridging line if you like
                    bridging_line = [forward_line[-1], backward_line[-1]]
                    bridging_lines.append(bridging_line)

                    print(
                        f"PSL tips meet at iteration {step_i}, dist= {dist_fb:.3f} < {closing_threshold}."
                    )
                    forward_active = False
                    backward_active = False
                    break

        # MERGE
        backward_line.reverse()
        if len(backward_line) > 1:
            backward_line.pop()
        merged_pts = backward_line + forward_line

        return merged_pts, bridging_lines, merge_points


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
    vectors_array_3d = [to_xyz(vec) for vec in principal_vectors]




    sampled_boundary_pts = []
    for crv in boundary_curves:
        params = crv.DivideByCount(sample_count, True)
        for t in params:
            p = crv.PointAt(t)
            sampled_boundary_pts.append([p.X, p.Y, p.Z])

    # Build KDTree for boundary points
    if sampled_boundary_pts:
        boundary_points_array = np.array(sampled_boundary_pts, dtype=float)
        #print(boundary_points_array)
        boundary_kdtree = KDTree(boundary_points_array)
        #print(boundary_kdtree)
    else:
        boundary_kdtree = None
        



    # 2) Build a 3D KD-tree
    points_np = np.array(points_array_3d)  # shape (N, 3)
    #print(points_np)
    kd_tree = KDTree(points_np)
    #print(f"old_kdTree = {kd_tree}")

    # 3) We'll store PSL results in 'existing_trajectories' as: ( [pt3d_list], polyline_curve )
    existing_trajectories = []
    bridging_lines_out = []
    existing_merge_pts = []
    psl_out = []



    # 4) Loop over seed points (3D)
    for seed in seed_points:
        seed_xyz = to_xyz(seed)
        
        merged_psl,  new_bridges, merge_points = trace_psl_both_directions_in_one_loop_3d(
            seed_point_3d=seed_xyz,
            h=h,
            num_steps=num_steps,
            k=k,
            principal_vectors=vectors_array_3d,
            points_3d=points_array_3d,
            kd_tree=kd_tree,
            boundary_curves=boundary_curves,
            boundary_tolerance=boundary_tolerance,
            existing_trajectories=existing_trajectories,
            collision_threshold=collision_threshold,
            closing_threshold=closing_threshold,  
            existing_merge_pts = existing_merge_pts
        )
        
        # Build the cached polyline for collisions
        psl_curve_3d = build_polyline_curve_3d([[pt.X, pt.Y, pt.Z] for pt in merged_psl])
        existing_trajectories.append((merged_psl, psl_curve_3d))
        existing_merge_pts.append(merge_points)

        psl_out.append(psl_curve_3d)
        bridging_lines_out.extend(new_bridges)
        

    # 5) Output
    main_lines = tr.list_to_tree(psl_out)  
    connections = tr.list_to_tree(bridging_lines_out)

    return main_lines, connections