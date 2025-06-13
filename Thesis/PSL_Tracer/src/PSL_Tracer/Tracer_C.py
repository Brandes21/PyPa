def main(ghenv,h,num_steps,k,collision_threshold,snap_radius,n_back,max_offset_distance,sample_interval,sample_count,ratio_minseed_coll,seed_point,principal_vectors,points,domain_surface,boundary_curves,strength, stress_values):

    import math
    import numpy as np
    import rhinoscriptsyntax as rs
    import Rhino.Geometry as rg
    import scriptcontext as sc
    from scipy.spatial import KDTree
    import heapq
    from ghpythonlib import treehelpers as tr


    # ------------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------------

    boundary_tolerance = h
    min_seed_distance = collision_threshold * ratio_minseed_coll
    seed_boundary_dist = min_seed_distance
    closing_threshold=h*5



    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------
    def to_xyz(pt_or_vec):

        if hasattr(pt_or_vec, "X"):
            return [pt_or_vec.X, pt_or_vec.Y, pt_or_vec.Z]
        else:
            # Already [x,y,z]
            return [pt_or_vec[0], pt_or_vec[1], pt_or_vec[2]]

    def distance_3d(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def normalize(vector):
        magnitude = math.sqrt(vector[0]**2 + vector[1]**2+vector[2]**2)
        return [vector[0] / magnitude, vector[1] / magnitude,vector[2] / magnitude] if magnitude != 0 else [0, 0, 0]

    def to_numeric(vec):

        try:
            return [vec.X, vec.Y, vec.Z]
        except AttributeError:
            return vec

    def find_closest_neighbors_kd(point, kd_tree, k):
        #use kd tree for neighbour lookup

        distances, indices = kd_tree.query(point,k=k)
        if isinstance(indices, int):
            return [indices]
        return list(indices)

    def build_polyline_curve_3d(poly_pts):
        pts3d = [rg.Point3d(pt[0], pt[1], pt[2]) for pt in poly_pts]
        poly = rg.Polyline(pts3d)
        return rg.PolylineCurve(poly)

    def project_onto_surface(surface, pt3d, tolerance=200):
        pt_rh = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        
        if isinstance(surface, rg.Brep):
            
            rc, closest_pt, cindex, u, v, normal = surface.ClosestPoint(pt_rh, tolerance)
            if rc:
                # Check if cindex is a face
                if cindex.ComponentIndexType == rg.ComponentIndexType.BrepFace:
                    face_id = cindex.Index
                    face = surface.Faces[face_id]
                    
                    # 'closest_pt' is the 3D point on that face
                    
                    
                    return [closest_pt.X, closest_pt.Y, closest_pt.Z]
                else:
                    
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

    def distance_to_brep_edges(pt3d, boundary_curves):
        test_pt = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        min_dist = float('inf')
        best_cp = None  
        
        for crv in boundary_curves:
            rc, t = crv.ClosestPoint(test_pt)
            if rc:
                cp = crv.PointAt(t)
                dist = cp.DistanceTo(test_pt)
                if dist < min_dist:
                    min_dist = dist
                    best_cp = cp
        
        return min_dist, best_cp



    def is_on_surface(surface, pt3d, tol=0.01):

        if not surface: 
            return True  # no surface provided
        pproj = project_onto_surface(surface, pt3d)
        if pproj is None:
            return False
        dx = pt3d[0] - pproj[0]
        dy = pt3d[1] - pproj[1]
        dz = pt3d[2] - pproj[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        return (dist < tol) # returns boolean



    # ------------------------------------------------------------------------
    # Tracing functions
    # ------------------------------------------------------------------------

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
        boundary_tolerance=1, prev_dir=None):

        # Perform one RK4 step in 3D. Stop if next point is within 'boundary_tolerance'of any boundary curve (edges).

        
        # 1) Neighbors & step size
        neighbors = find_closest_neighbors_kd(current_point, kd_tree, k)
        h = h * step_sign
        
        # k1
        k1_dir = interpolate_vector(current_point, points_3d, principal_vectors, neighbors, current_dir)
        mid1 = [
            current_point[0] + 0.5*h*k1_dir[0],
            current_point[1] + 0.5*h*k1_dir[1],
            current_point[2] + 0.5*h*k1_dir[2]
        ]
        
        # k2
        neigh_mid1 = find_closest_neighbors_kd(mid1, kd_tree, k)
        k2_dir = interpolate_vector(mid1, points_3d, principal_vectors, neigh_mid1, k1_dir)
        mid2 = [
            current_point[0] + 0.5*h*k2_dir[0],
            current_point[1] + 0.5*h*k2_dir[1],
            current_point[2] + 0.5*h*k2_dir[2]
        ]
        
        # k3
        neigh_mid2 = find_closest_neighbors_kd(mid2, kd_tree, k)
        k3_dir = interpolate_vector(mid2, points_3d, principal_vectors, neigh_mid2, k2_dir)
        end_pt = [
            current_point[0] + h*k3_dir[0],
            current_point[1] + h*k3_dir[1],
            current_point[2] + h*k3_dir[2]
        ]
        
        # k4
        neigh_end = find_closest_neighbors_kd(end_pt, kd_tree, k)
        k4_dir = interpolate_vector(end_pt, points_3d, principal_vectors, neigh_end, k3_dir)
        
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
            projected = project_onto_surface(domain_surface, next_point)
            #print(projected)
            if projected is None:
                # Means we're off domain or near an open edge you consider invalid
                return None
            next_point = projected  # accept the projected coordinate

        # Otherwise return the next valid point
        return next_point, next_dir

    def interpolate_vector(point, points, vectors, neighbors, ref_dir):

        # Distance-weighted interpolation of neighbor vectors in 3D, flipping each neighbor's vector if dot < 0 with respect to 'ref_dir'.

        weights = []
        weighted_vec = [0.0, 0.0, 0.0]

        for i in neighbors:
            vx, vy, vz = vectors[i]  # copy so we can flip locally
            if ref_dir is not None:
                dotp = vx*ref_dir[0] + vy*ref_dir[1] + vz*ref_dir[2]
                if dotp < 0:
                    vx, vy, vz = -vx, -vy, -vz
            
            npt = points[i]
            dx = point[0] - npt[0]
            dy = point[1] - npt[1]
            dz = point[2] - npt[2]
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

    # ------------------------------------------------------------------------
    # Collision checks
    # ------------------------------------------------------------------------


    def closest_point_on_polyline_3d(pt3d, poly_curve):

        test_pt = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
        rc, t = poly_curve.ClosestPoint(test_pt)
        if rc:
            cp = poly_curve.PointAt(t)
            dist = cp.DistanceTo(test_pt)
            #print(dist)
            return [cp.X, cp.Y, cp.Z], dist
        else:
            return None, float('inf')

    def find_closest_existing_line_3d(next_point, existing_trajectories, threshold):
 
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

    def find_most_parallel_boundary_point(
        current_pt,
        last_step_vec,
        boundary_points,
        k=10):

        # Given a KD-tree of boundary sample points (boundary_kdtree)
        # and the original array (boundary_points),
        # find the boundary sample whose direction from current_pt 
        # is most opposite (or "most parallel" – depends on how you measure) 
        # to last_step_vec. Return (best_point, min_dist).
        
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
  
        distances, indices = boundary_kdtree.query(cur_np, k=k)

        # If k=1, make them arrays for consistency
        if k == 1:
            distances = [distances]
            indices = [indices]

        # 2) Among these k points, pick the direction which has the smallest dot or largest negative dot, etc.
        best_dot = 9999
        best_pt = None
        min_dist = float('inf')



        for dist, idx in zip(distances, indices):
            cand_np = boundary_points_array[idx]  
            dir_vec = cand_np - cur_np
            dir_len = np.linalg.norm(dir_vec)
            if dir_len < 1e-12:
                continue
            dir_unit = dir_vec / dir_len
            dot_val = np.dot(dir_unit, step_np)
            
            # most parallel, same direction => pick largest dot_val. 
            if dot_val < best_dot:
                best_dot = dot_val
                best_pt = cand_np

                if dist < min_dist:
                    min_dist = dist

        # Convert best_pt to Rhino point
        if best_pt is not None:
            return rg.Point3d(best_pt[0], best_pt[1], best_pt[2]), min_dist
        else:
            return None, float('inf')

    # ------------------------------------------------------------------------
    # Sampling and offset functions
    # ------------------------------------------------------------------------


    def surface_normal_at_point(surface, pt3d, tolerance=2):

        pt_rh = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])

        if isinstance(surface, rg.Brep):
            stuff = surface.ClosestPoint(pt_rh, tolerance)
            if not stuff or len(stuff) < 6:
                return None
            b = stuff[0]
            rc = stuff[1]

            comp_index = stuff[2]
            u = stuff[3]
            v = stuff[4]

            if not rc:
                return None
            
            # If it's a face, we evaluate the face normal at (u, v)
            if comp_index.ComponentIndexType == rg.ComponentIndexType.BrepFace:
                face_id = comp_index.Index
                face = surface.Faces[face_id]
                exact_normal = face.NormalAt(u, v)
                if exact_normal:
                    return [exact_normal.X, exact_normal.Y, exact_normal.Z]
                else:
                    return None
            else:
   
                return None
        
        elif isinstance(surface, rg.Surface):
            rc, u, v = surface.ClosestPoint(pt_rh)
            if rc:
                exact_normal = surface.NormalAt(u, v)
                return [exact_normal.X, exact_normal.Y, exact_normal.Z]
            else:
                return None

        # Not a Brep or Surface
        return None

    def offset_seed(psl_pts_3d, index, offset_distance, surface):
        """
        Offsets the PSL at psl_pts_3d[index] in the local tangent plane of 'surface'.
        Steps:
        1) Compute PSL tangent at index
        2) Get surface normal at that point
        3) binormal = cross(surface_normal, psl_tangent)
        4) candidate1 = point + offset_distance*binormal
            candidate2 = point - offset_distance*binormal
        5) (optional) project candidates back onto the surface
        Returns: (candidate1, candidate2) as [x, y, z], or (None, None) on error.
        """
        n = len(psl_pts_3d)
        if n < 2:
            return None, None
        
        # 0) Current point
        px, py, pz = psl_pts_3d[index]
        
        # 1) PSL TANGENT
        if index == 0:
            # Use next point - current point
            nx, ny, nz = psl_pts_3d[1]
            tx = nx - px
            ty = ny - py
            tz = nz - pz
        elif index == n - 1:
            # Use current point - prev point
            px_prev, py_prev, pz_prev = psl_pts_3d[n - 2]
            tx = px - px_prev
            ty = py - py_prev
            tz = pz - pz_prev
        else:
            # Middle: use psl_pts_3d[index+1] - psl_pts_3d[index-1]
            px_prev, py_prev, pz_prev = psl_pts_3d[index - 1]
            px_next, py_next, pz_next = psl_pts_3d[index + 1]
            tx = px_next - px_prev
            ty = py_next - py_prev
            tz = pz_next - pz_prev

        # Normalize tangent
        t_len = math.sqrt(tx*tx + ty*ty + tz*tz)
        if t_len < 1e-12:
            return None, None
        tx /= t_len
        ty /= t_len
        tz /= t_len
        
        # 2) SURFACE NORMAL
        normal = surface_normal_at_point(surface, [px, py, pz])
        if not normal:
            print("no normal")
            return None, None
        nx, ny, nz = normal
        n_len = math.sqrt(nx*nx + ny*ny + nz*nz)
        if n_len < 1e-12:
            return None, None
        nx /= n_len
        ny /= n_len
        nz /= n_len
        
        # 3) BINORMAL = cross(normal, tangent)
        # This is guaranteed to be in the tangent plane if PSL is truly tangent to the surface
        # If the PSL isn't exactly tangent, you'll get some tilt. 
        bx = ny*tz - nz*ty
        by = nz*tx - nx*tz
        bz = nx*ty - ny*tx
        b_len = math.sqrt(bx*bx + by*by + bz*bz)
        if b_len < 1e-12:
            # Means normal and tangent are parallel or something degenerate
            return None, None
        bx /= b_len
        by /= b_len
        bz /= b_len
        
        # 4) OFFSET POINTS
        candidate1 = [px + offset_distance*bx,
                    py + offset_distance*by,
                    pz + offset_distance*bz]
        candidate2 = [px - offset_distance*bx,
                    py - offset_distance*by,
                    pz - offset_distance*bz]
        
        
        candidate1_proj = project_onto_surface(surface, candidate1)
        #print(f"this point is seed {candidate1_proj}")
        candidate2_proj = project_onto_surface(surface, candidate2)


        return candidate1_proj, candidate2_proj

    def sample_psl_3d(psl_pts, sample_interval):
        """
        Given a PSL (list of Rhino Point3d or numeric [x,y,z] points),
        sample it every 'sample_interval' points.
        Returns a list of tuples: ( [x,y,z], original_index )
        """
        samples = []
        for i in range(0, len(psl_pts), sample_interval):
            pt3d = psl_pts[i]
  
            if hasattr(pt3d, "X"):
                samples.append(([pt3d.X, pt3d.Y, pt3d.Z], i))
            else:
                samples.append((pt3d, i))
        return samples

    def is_valid_seed(candidate, existing_trajectories, min_distance, boundary_curves):
        """
        Check if candidate seed is valid:
        - It is not too close to any existing PSL (using min_distance).
        - It is not too close to any boundary curves (if boundary_curves is provided and
            candidate's distance < boundary_tolerance).
        """
        # Check candidate against existing PSLs
        idx, cp, dist = find_closest_existing_line_3d(candidate, existing_trajectories, min_distance)
        if idx is not None:
            return False

        # Check candidate against boundary curves if provided
        if boundary_curves:
            dist_edge, cp_edge = distance_to_brep_edges(candidate, boundary_curves)
            if dist_edge < seed_boundary_dist:
                return False

        return True

    def get_stress_value(sample_pt, kd_tree, stress_values):
        """
        Query the stress_kdtree to find the nearest stress sample.
        Return the corresponding stress value from stress_values.
        """
        # sample_pt is [x, y] in 2D
        dist, idx = kd_tree.query(sample_pt)  # single neighbor
        # stress_values[idx] is the stress at the nearest stress point
        return stress_values[idx]

    def is_point_close_to_any(point_list, single_point, threshold):
        flat = [item for sublist in point_list for item in sublist]
        #print(f"single = {single_point}")
        single_point_1 = rg.Point3d(single_point[0],single_point[1],single_point[2])
        
        
        for pt in flat:
            #print(f"pt = {pt}")
            if single_point_1.DistanceTo(pt) <= threshold:
                return pt
        return single_point




    # ------------------------------------------------------------------------
    # Tracing logic
    # ------------------------------------------------------------------------

    def trace_psl_both_directions(
        seed_point_3d, 
        h, 
        num_steps, 
        k, 
        principal_vectors, 
        points_3d,
        boundary_curves=None,
        boundary_tolerance=boundary_tolerance,
        existing_trajectories=None,
        collision_threshold=collision_threshold,
        closing_threshold=closing_threshold, 
        kd_tree=None,
        existing_merge_pts = None):
        """
        Single-loop approach where forward/backward can stop independently.
        If forward hits a collision/boundary, it stops, but backward can continue, and vice versa.
        We only stop the entire PSL when BOTH directions are inactive, or when tips meet.
        """
        if existing_trajectories is None:
            existing_trajectories = []
        
        
        bridging_lines = []
        edge_lines = []
        merge_points = []

        # We'll keep two lines: forward and backward
        forward_line = [rg.Point3d(*seed_point_3d)]
        backward_line = [rg.Point3d(*seed_point_3d)]
        
        # Current forward/backward points + directions
        f_current_pt = list(seed_point_3d)
        f_current_dir = None
        b_current_pt = list(seed_point_3d)
        b_current_dir = None

        # NEW: Boolean flags to track if forward/backward are still active
        forward_active = True
        backward_active = True

        for step_i in range(num_steps):
            
            # ---------------------------
            # FORWARD STEP (if still active)
            # ---------------------------
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
                if not f_next or f_next[0] is None:
                    print(f"PSL forward direction stopped at iteration {step_i}.")
                    forward_active = False
                else:
                    f_next_pt, f_next_dir = f_next
                    # Collision check with existing PSLs
                    if existing_trajectories:
                        line_idx, close_pt, dist_cl = find_closest_existing_line_3d(
                            f_next_pt, existing_trajectories, collision_threshold
                        )

                        if len(existing_merge_pts) > 1 and close_pt is not None:
                            print(f"merged = {merge_points}")
                            close_pt = is_point_close_to_any(existing_merge_pts, close_pt, snap_radius)

                        if line_idx is not None and close_pt is not None:
                            # Instead of break, we do bridging & disable forward
    
                            if len(forward_line) > n_back:
                                bridging_line = rg.Line(
                                    forward_line[-n_back],
                                    rg.Point3d(*close_pt)
                                )
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))
                                del forward_line[-(n_back-1):]

                            if len(forward_line)==0 or len(backward_line)==1:
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
                        cp,dist_to_edge = find_most_parallel_boundary_point(f_next_pt,dir_vec,boundary_curves,10)
                        if dist_to_edge < boundary_tolerance:
                            # We consider that "off" or "too close" => stop
                            print(f"PSL forward reached boundary.")
                            forward_active = False
                            bridging_line = rg.Line(
                                    forward_line[-1],
                                    rg.Point3d(cp)
                            )
                            bridging_lines.append(bridging_line)

                    # If still active, accept the new forward step
                    if forward_active:
                        forward_line.append(rg.Point3d(*f_next_pt))
                        f_current_pt = f_next_pt
                        f_current_dir = f_next_dir

            # ---------------------------
            # BACKWARD STEP (if still active)
            # ---------------------------
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
                if not b_next or b_next[0] is None:
                    print(f"PSL backward direction stopped at iteration {step_i}.")
                    backward_active = False
                else:
                    b_next_pt, b_next_dir = b_next
                    # Collision check with existing PSLs
                    if existing_trajectories:
                        line_idx, close_pt, dist_cl = find_closest_existing_line_3d(
                            b_next_pt, existing_trajectories, collision_threshold
                        )

                        if len(existing_merge_pts) > 1 and close_pt is not None:
                            print(f"merged = {merge_points}")
                            close_pt = is_point_close_to_any(existing_merge_pts, close_pt, snap_radius)


                        if line_idx is not None and close_pt is not None:
                        
                            if len(backward_line) > n_back:
                                bridging_line = rg.Line(
                                    backward_line[-n_back],
                                    rg.Point3d(*close_pt)
                                )
                                bridging_lines.append(bridging_line)
                                merge_points.append(rg.Point3d(*close_pt))
                                del backward_line[-(n_back-1):]

                            if len(backward_line)==0 or len(backward_line)==1:
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


                    if boundary_curves:
                        cp,dist_to_edge = find_most_parallel_boundary_point(b_next_pt,b_next_dir,boundary_curves,10)
                        if dist_to_edge < boundary_tolerance:
                            # We consider that "off" or "too close" => stop
                            print(f"PSL forward reached boundary.")
                            backward_active = False
                            bridging_line = rg.Line(
                                    backward_line[-1],
                                    rg.Point3d(cp)
                            )
                            bridging_lines.append(bridging_line)

                    # If still active, accept the new backward step
                    if backward_active:
                        backward_line.append(rg.Point3d(*b_next_pt))
                        b_current_pt = b_next_pt
                        b_current_dir = b_next_dir
            
            # ---------------------------
            # Check if BOTH directions are done
            # ---------------------------
            if not forward_active and not backward_active:
                print("Both forward & backward directions inactive => stopping PSL.")
                break

            # ---------------------------
            # TIP DISTANCE CHECK
            # ---------------------------
            # Only if both are still active
            if forward_active and backward_active and step_i >= 10:
                f_xyz = [forward_line[-1].X, forward_line[-1].Y, forward_line[-1].Z]
                b_xyz = [backward_line[-1].X, backward_line[-1].Y, backward_line[-1].Z]
                dist_fb = distance_3d(f_xyz, b_xyz)
                if dist_fb < closing_threshold:
                    bridging_line = rg.Line(
                        forward_line[-1],
                        backward_line[-1]
                    )
                    bridging_lines.append(bridging_line)

                    print(
                        f"PSL tips meet at iteration {step_i}, dist= {dist_fb:.3f} < {closing_threshold}."
                    )
                    # Disable both directions and break
                    forward_active = False
                    backward_active = False
                    break
        
        # Merge lines
        backward_line.reverse()
        if len(backward_line) > 1:
            backward_line.pop()
        merged_pts = backward_line + forward_line
        
        return merged_pts, bridging_lines, edge_lines, merge_points

    def generate_all_psls_3d(
        initial_seed_3d,
        h, 
        num_steps, 
        k, 
        principal_vectors, 
        points_3d,
        kd_tree,
        boundary_curves,
        collision_threshold,
        sample_interval,
        min_seed_distance,
        # NEW: for stress-based offset
        stress_values):
        """
        Generate PSLs using an iterative seeding strategy (3D version)
        with a GLOBAL PRIORITY QUEUE for candidate seeds based on stress.

        1) Trace an initial PSL from initial_seed_3d.
        2) Sample it, pushing candidate seeds into a priority queue (max stress first).
        3) Pop the highest-stress seed, validate it, trace a PSL, sample it, push new seeds...
        4) Stop when the priority queue is empty.

        Returns:
        - all_psls: list of (psl_points_3d, polyline_curve_3d)
        - bridging_lines: bridging segments
        - positive_candidates: seeds that were actually used
        """
        existing_merge_pts = []

        # 1) Trace the initial PSL
        initial_psl, initial_bridges, initial_edges, merge_points = trace_psl_both_directions(
            seed_point_3d=initial_seed_3d,
            h=h,
            num_steps=num_steps,
            k=k,
            principal_vectors=principal_vectors,
            points_3d=points_3d,
            boundary_curves=boundary_curves,
            boundary_tolerance=boundary_tolerance,
            existing_trajectories=None,
            collision_threshold=collision_threshold,
            kd_tree=kd_tree,
            existing_merge_pts = existing_merge_pts
        )
        # Convert to curve for collision checks
        initial_curve = build_polyline_curve_3d(initial_psl)

        all_psls = [(initial_psl, initial_curve)]
        bridging_lines = []
        positive_candidates = []
        edges = []
        offsets = []


        if initial_bridges:
            bridging_lines.extend(initial_bridges)

        if initial_edges:
            edges.extend(initial_edges)

        # Helper: sample a PSL, push seeds to a global priority queue
        def add_candidates_from_psl(psl_points, sample_interval):
            samples = sample_psl_3d(psl_points, sample_interval)  # => [ ([x,y,z], index), ... ]
            print(len(samples))
            psl_numeric = [[pt.X, pt.Y, pt.Z] for pt in psl_points]

            for (sample_pt_3d, orig_index) in samples:
                # 1) Query stress
                stress_val = get_stress_value(sample_pt_3d,kd_tree, stress_values)
                # 2) Compute dynamic offset e.g. offset = bar_capacity / stress_val, clamped
                if stress_val < 1e-6:
                    continue
                raw_offset = strength / stress_val
                #print(raw_offset)
                

                dyn_offset = min(raw_offset, max_offset_distance) 
                #print(f"dyn_offset: {dyn_offset}")
                offsets.append(dyn_offset)
                if dyn_offset < collision_threshold: # issue warning here
                    from Grasshopper.Kernel import GH_RuntimeMessageLevel as RML
                    ghenv.Component.AddRuntimeMessage(RML.Warning, "Warning: Needed offset is smaller than collision_threshold -> Choose higher rebar diameter")
                    print("Warning: Needed offset is smaller than collision_threshold -> Choose higher rebar diameter")
                    warning = "Warning: Needed offset is smaller than collision_threshold -> Choose higher rebar diameter"

                # 3) Generate offset candidates
                cand1, cand2 = offset_seed(psl_numeric, orig_index, dyn_offset, domain_surface)
                for c in [cand1, cand2]:
                    if c is not None:
                        # Use negative stress as priority => max stress first
                        heapq.heappush(candidate_queue, (-stress_val, c))
                        
        # 2) Initialize a global priority queue for seeds
        candidate_queue = []
        # Sample the initial PSL, push seeds
        add_candidates_from_psl(initial_psl, sample_interval)

        # 3) Process the global priority queue
        while candidate_queue:
            neg_stress, candidate_3d = heapq.heappop(candidate_queue)
            s_val = -neg_stress
            
            # Check if candidate is valid: 
            #   1) not too close to existing PSLs
            #   2) not too close to domain boundary if you do so in is_valid_seed
            #   3) optionally: on surface, etc.
            if not is_valid_seed(candidate_3d, all_psls, min_seed_distance, boundary_curves):
                continue
            positive_candidates.append(candidate_3d)
            # Now trace a PSL from this candidate
            new_psl, new_bridges, new_edges, merge_points = trace_psl_both_directions(
                seed_point_3d=candidate_3d,
                h=h,
                num_steps=num_steps,
                k=k,
                principal_vectors=principal_vectors,
                points_3d=points_3d,
                boundary_curves=boundary_curves,
                boundary_tolerance=boundary_tolerance,
                existing_trajectories=all_psls,
                collision_threshold=collision_threshold,
                kd_tree=kd_tree,
                existing_merge_pts = existing_merge_pts
            )
            new_curve = build_polyline_curve_3d(new_psl)
            all_psls.append((new_psl, new_curve))
            if new_bridges:
                bridging_lines.extend(new_bridges)

            if new_edges:
                edges.extend(new_edges)
            
            existing_merge_pts.append(merge_points)

            print(f"checkkkkkk = {existing_merge_pts}")

            # Immediately sample the new PSL => add new seeds
            add_candidates_from_psl(new_psl, sample_interval)

        return all_psls, bridging_lines, positive_candidates, edges, offsets


    # ------------------------------------------------------------------------
    # Build KD Trees
    # ------------------------------------------------------------------------

    #1) mesh points kdtree
    points_array_3d = [to_xyz(pt) for pt in points]
    points_np = np.array(points_array_3d)  # shape (N, 3)
    kd_tree = KDTree(points_np)

    #2) boundary sample points kdtree
    sampled_boundary_pts = []
    for crv in boundary_curves:
        params = crv.DivideByCount(sample_count, True)
        for t in params:
            p = crv.PointAt(t)
            sampled_boundary_pts.append([p.X, p.Y, p.Z])

    # Build KDTree for boundary points
    if sampled_boundary_pts:
        boundary_points_array = np.array(sampled_boundary_pts, dtype=float)
        boundary_kdtree = KDTree(boundary_points_array)
    else:
        boundary_kdtree = None







    initial_seed_3d = to_xyz(seed_point[0])
    principal_vectors_3d = [to_xyz(vec) for vec in principal_vectors]








    # ----------------------------------------------------------------------
    # CALL YOUR 3D PSL GENERATION FUNCTION
    # ----------------------------------------------------------------------




    all_psls, bridging_lines, positive_candidates,edges,offsets = generate_all_psls_3d(
        initial_seed_3d,
        h,
        num_steps,
        k,
        principal_vectors_3d,   # 3D vectors
        points_array_3d,        # 3D points
        kd_tree,
        boundary_curves,
        collision_threshold,
        sample_interval,
        min_seed_distance,
        stress_values

    )


    # ----------------------------------------------------------------------
    # PREPARE OUTPUTS FOR GH
    # ----------------------------------------------------------------------
    # 1) PSL polylines as GH curves
    psl_curves = [psl_curve for (psl_points, psl_curve) in all_psls]



    # 3) Convert positive_candidates to Point3d
    generated_seed_points = [rg.Point3d(pc[0], pc[1], pc[2]) for pc in positive_candidates]
    
    # 4) Finally set GH outputs (A, B, C, e.g.)
    a = tr.list_to_tree(psl_curves)
    b = tr.list_to_tree(bridging_lines)
    w = tr.list_to_tree(generated_seed_points)
    #c = tr.list_to_tree(offsets)

    return a, b, w