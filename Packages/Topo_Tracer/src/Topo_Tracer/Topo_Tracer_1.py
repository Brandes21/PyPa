def main(surface, points, sigma1_list, p1_list, sigma2_list, p2_list,tol_deg, offset_eps,neighbors, cluster_threshold, seed_amplitude,index_radius, n_sep_merged):

    # -- Prerequisites ------------------------------------------------------------
    import math
    from typing import List, Optional, Tuple
    import numpy as np
    from scipy.spatial import KDTree
    import Rhino.Geometry as rg
    from ghpythonlib import treehelpers as tr
    import ghpythonlib.components as ghcomp

    def measure_unoriented_angles(
        x0,
        y0,
        slope,
        eps,
        pts,
        Tlist,
        kdtree,
        tol=0.15
    ):
        """
        1) Offsets from (x0,y0) by eps in direction (1, slope).
        2) Gets local p1, p2 from nearest non-degenerate neighbor.
        3) Computes the *unoriented* angles angle1, angle2 of (1,slope)
        vs. p1 and p2, respectively.

        Returns (label, angle1, angle2):
        label: '1' if angle1 <= angle2, '2' otherwise
        angle1, angle2: the actual angles in [0, π/2].

        If something fails (degenerate region, zero dir, etc.), returns (None, None, None).
        """
        dir_vec = np.array([1.0, slope], dtype=float)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-14:
            return None, None, None
        dir_vec_n = dir_vec / norm

        x_eps = x0 + eps * dir_vec_n[0]
        y_eps = y0 + eps * dir_vec_n[1]

        # Re-use existing local-eigens code
        p1, p2 = get_local_eigenvectors(x_eps, y_eps, pts, Tlist, kdtree, sigma1_list, sigma2_list, tol=tol)
        if p1 is None or p2 is None:
            return None, None, None





        p1_n = p1 / np.linalg.norm(p1)
        p2_n = p2 / np.linalg.norm(p2)

        # Compute angles via arccos(|dot|)
        dot1 = abs(np.dot(dir_vec_n, p1_n))
        dot2 = abs(np.dot(dir_vec_n, p2_n))

        angle1 = math.acos(np.clip(dot1, 0, 1))
        angle2 = math.acos(np.clip(dot2, 0, 1))

        # Decide ownership
        if angle1 <= angle2:
            return '1', angle1, angle2
        else:
            return '2', angle1, angle2
    # -- Basic 2D Stress Tensor ---------------------------------------------------

    def stress_tensor_2d(
        sigma1: float,
        p1: Tuple[float, float, float],
        sigma2: float,
        p2: Tuple[float, float, float]) -> np.ndarray:
        """
        Construct the 2D stress tensor:
            T = sigma1 * (p1 p1^T) + sigma2 * (p2 p2^T)
        Only the x and y components of p1, p2 are used.
        """
        x1, y1, _ = p1
        x2, y2, _ = p2
        T1 = np.array([[x1*x1, x1*y1], [y1*x1, y1*y1]])
        T2 = np.array([[x2*x2, x2*y2], [y2*x2, y2*y2]])
        return sigma1 * T1 + sigma2 * T2



    def sample_directions_around_merged_point(
        x0, y0,
        pts,
        Tlist,
        kdtree,
        n_samples=16,
        offset_eps=0.02,
        deg_tol=0.01
    ):
        """
        1) Sample n_samples directions around the circle.
        2) For each, measure unoriented angles to both fields.
        3) Collect candidates in two groups (label '1' or '2').
        4) Sort each by fit angle and keep best four.
        Returns p1_vecs, p2_vecs, p1_candidates.
        """
        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        p1_candidates, p2_candidates = [], []

        for theta in angles:
            dx, dy = math.cos(theta), math.sin(theta)
            # slope fallback
            slope = dy/dx if abs(dx)>1e-14 else np.sign(dy)*1e14
            label, angle1, angle2 = measure_unoriented_angles(
                x0, y0, slope, offset_eps, pts, Tlist, kdtree, tol=deg_tol
            )
            if label is None:
                continue
            vec = np.array([dx, dy], float)
            if np.linalg.norm(vec)<1e-14:
                continue
            vec /= np.linalg.norm(vec)
            fit = angle1 if label=='1' else angle2
            (p1_candidates if label=='1' else p2_candidates).append((vec, fit))

        # keep best 4 of each
        p1_candidates.sort(key=lambda x: x[1]); p2_candidates.sort(key=lambda x: x[1])
        p1_vecs = [v for v,_ in p1_candidates[:n_sep_merged]]
        p2_vecs = [v for v,_ in p2_candidates[:n_sep_merged]]
        return p1_vecs, p2_vecs, p1_candidates




    # -- Cubic Solver for Separatrices --------------------------------------------

    def separatrix_slopes(
        a: float,
        b: float,
        c: float,
        d: float,
        imag_tol: float = 1e-7) -> List[float]:
        """
        Solve cubic: d x^3 + (c+2b) x^2 + (2a-d) x - c = 0
        Return all real roots (slope values).
        """
        coeffs = [d, c + 2*b, 2*a - d, -c]
        roots = np.roots(coeffs)
        return [r.real for r in roots if abs(r.imag) < imag_tol]

    # -- Classification of Degenerate Type ---------------------------------------

    def classify_degenerate(
        x0: float,
        y0: float,
        points: list[rg.Point3d],
        tensors: list[np.ndarray],
        kdtree: KDTree,
        radius: float,
        a: float,
        b: float,
        c: float,
        d: float,
        tol: float = 1e-15,
        n_samples: int = 50,
        which_field: str = 'major'
    ) -> str:
        """
        Compute delta = a*d - b*c and Delmarcelle index, then choose the index-based
        classification if it differs from delta-based.

        Delta-based:
        delta>tol  -> 'wedge'
        delta<-tol -> 'trisector'
        else       -> 'merged'

        Index-based (from compute_index_delmarcelle):
        index ≈ +0.5 -> 'wedge'
        index ≈ -0.5 -> 'trisector'
        index ≈ -1.0 -> 'saddle'
        else         -> 'merged'
        """
        # delta classification
        delta = a*d - b*c
        if delta > 1e-15:
            delta_class = 'wedge'
        elif delta < -1e-15:
            delta_class = 'trisector'
        else:
            delta_class = 'merged'

        # index classification always computed
        idx_value, _ = compute_index_delmarcelle(
            x0, y0, points, tensors, kdtree,
            radius, n_samples, which_field
        )

        print(f"index new = {idx_value}")
        # map index to type
        if abs(idx_value - 0.5) <= 0.1:
            index_class = 'wedge'
        elif abs(idx_value + 0.5) <= 0.1:
            index_class = 'trisector'
        elif abs(idx_value + 1.0) <= 0.1:
            index_class = 'saddle'
        else:
            index_class = 'merged'

        # if they differ, trust index
        if index_class != delta_class:
            return index_class
        return delta_class

    # -- Approximate Partial Derivatives via Local Regression --------------------

    def approximate_partials(
        center_idx: int,
        points: List[rg.Point3d],
        tensors: List[np.ndarray],
        kdtree: KDTree,
        neighbors) -> Optional[Tuple[float, float, float, float]]:
        """
        Fit local plane to f=T11-T22 and g=T12 via least squares over nearest neighbors.
        For idx_center, take up to `neighbors` nearest points (excluding center).
        Returns (a, b, c, d) where:
        a=0.5*df/dx, b=0.5*df/dy, c=dg/dx, d=dg/dy.
        """
        # center coordinates
        x0, y0 = points[center_idx].X, points[center_idx].Y
        #print(x0)
        # query neighbors+1 (including self)
        dists, idxs = kdtree.query((x0, y0), k=neighbors+1)
        #print(idxs)
        # flatten and exclude center
        all_idxs = list(np.atleast_1d(idxs))
        neighbor_idxs = [i for i in all_idxs if i != center_idx][:neighbors]
        print(neighbor_idxs)
        
        if not neighbor_idxs:
            return None

        # build design matrix X = [1, x, y]
        X = []
        f_vals = []
        g_vals = []
        for j in neighbor_idxs:
            xj, yj = points[j].X, points[j].Y
            X.append([1.0, xj, yj])
            T = tensors[j]
            f_vals.append(T[0, 0] - T[1, 1])
            g_vals.append(T[0, 1])
        X = np.array(X)
        f_vals = np.array(f_vals)
        g_vals = np.array(g_vals)

        # solve least squares: f ~ A0 + A1 x + A2 y
        sol_f, *_ = np.linalg.lstsq(X, f_vals, rcond=None)
        sol_g, *_ = np.linalg.lstsq(X, g_vals, rcond=None)

        # partials
        df_dx, df_dy = sol_f[1], sol_f[2]
        dg_dx, dg_dy = sol_g[1], sol_g[2]
        return 0.5 * df_dx, 0.5 * df_dy, dg_dx, dg_dy

    # -- Local Eigenvector Retrieval ---------------------------------------------

    def get_local_eigenvectors(
        x: float,
        y: float,
        points: List[rg.Point3d],
        tensors: List[np.ndarray],
        kdtree: KDTree,
        sigma1_vals: List[float],
        sigma2_vals: List[float],
        tol: float = 1e-2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find nearest non-degenerate point, return its eigenvectors (p1,p2) sorted by eigenvalue descending.
        """
        dists, idxs = kdtree.query((x, y), k=5)
        for dist, idx in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            if abs(sigma1_vals[idx] - sigma2_vals[idx]) > tol:
                eigvals, eigvecs = np.linalg.eig(tensors[idx])
                order = np.argsort(eigvals)[::-1]
                return eigvecs[:,order[0]], eigvecs[:,order[1]]
        return None, None

    # -- Assign Field Ownership by Slope -----------------------------------------

    def which_field_owns(
        x0: float,
        y0: float,
        slope: float,
        eps: float,
        points: List[rg.Point3d],
        tensors: List[np.ndarray],
        kdtree: KDTree,
        sigma1_vals: List[float],
        sigma2_vals: List[float],
        tol: float = 1e-2) -> Optional[str]:
        """
        Offset from (x0,y0) by eps along slope, get local eigenvectors,
        return '1' if closer to major, else '2'.
        """
        vec = np.array([1.0, slope])
        vec /= np.linalg.norm(vec)
        x_eps, y_eps = x0 + eps*vec[0], y0 + eps*vec[1]
        p1, p2 = get_local_eigenvectors(x_eps, y_eps, points, tensors, kdtree, sigma1_vals, sigma2_vals, tol)
        if p1 is None or p2 is None:
            return None
        ang1 = math.acos(np.clip(abs(np.dot(vec, p1/np.linalg.norm(p1))), 0, 1))
        ang2 = math.acos(np.clip(abs(np.dot(vec, p2/np.linalg.norm(p2))), 0, 1))
        return '1' if ang1 <= ang2 else '2'

    # -- Compute Delmarcelle-Hesselink Index -------------------------------------
    def compute_index_delmarcelle(
        x0: float,
        y0: float,
        points: List[rg.Point3d],
        tensors: List[np.ndarray],
        kdtree: KDTree,
        radius: float,
        n_samples: int = 50,
        which_field: str = 'major') -> Tuple[float, List[Tuple[float,float]]]:
        """
        Sample the chosen eigenvector field around a circle and compute its net rotation index.
        Returns (index_value, circle_samples).
        """
        thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        alpha = np.zeros(n_samples)
        pts_circle: List[Tuple[float,float]] = []
        v_prev = None
        lam_tol = 1e-10
        nominal = 0 if which_field.lower().startswith('major') else 1

        for i, th in enumerate(thetas):
            x_s = x0 + radius*math.cos(th)
            y_s = y0 + radius*math.sin(th)
            pts_circle.append((x_s, y_s))
            _, idx = kdtree.query((x_s, y_s), k=1)
            vals, vecs = np.linalg.eig(tensors[idx])
            vals, vecs = np.real(vals), np.real(vecs)
            order = np.argsort(vals)[::-1]
            vec = vecs[:, order[nominal]]
            # branch tracking
            if abs(vals[0] - vals[1]) < lam_tol and v_prev is not None:
                d0 = abs(np.dot(v_prev, vecs[:,order[0]]))
                d1 = abs(np.dot(v_prev, vecs[:,order[1]]))
                vec = vecs[:, order[1 if d1>d0 else 0]]
            if v_prev is not None and np.dot(vec, v_prev) < 0:
                vec = -vec
            v_prev = vec
            alpha[i] = math.atan2(vec[1], vec[0])

        # unwrap
        alpha_u = alpha.copy()
        for i in range(1, n_samples):
            diff = alpha[i] - alpha_u[i-1]
            if diff > math.pi:
                diff -= 2*math.pi
            elif diff < -math.pi:
                diff += 2*math.pi
            alpha_u[i] = alpha_u[i-1] + diff

        total = alpha_u[-1] - alpha_u[0]
        return total/(2*math.pi), pts_circle


    # -- Main Analysis Function --------------------------------------------------
    def find_degenerate_points(
        points: list[rg.Point3d],
        sigma1_vals: list[float],
        p1_dirs: list[tuple[float,float,float]],
        sigma2_vals: list[float],
        p2_dirs: list[tuple[float,float,float]],
        tol_deg,
        offset_eps,
        slope_tol: float = 0.3,
        index_radius: float = 0.1
    ) -> tuple[
        list[rg.Point2d],
        list[str],
        list[list[str]],
        list[list[tuple[float,float]]],
        list[float]
    ]:
        """
        Detect degeneracies, classify, extract separatrices, and compute index.
        Returns pts2d, types, field_labels, directions, indices
        """
        assert len(points)==len(sigma1_vals)==len(sigma2_vals)==len(p1_dirs)==len(p2_dirs)
        # build tensors and tree
        tensors = [stress_tensor_2d(s1,p1,s2,p2)
                for s1,p1,s2,p2 in zip(sigma1_vals,p1_dirs,sigma2_vals,p2_dirs)]
        coords = np.array([(pt.X,pt.Y) for pt in points])
        kdtree = KDTree(coords)

        pts2d, types, fields_sublists, dirs_sublists, indices = [], [], [], [], []
        for i, pt in enumerate(points):
            # skip non-degenerate by eigenvalue gap
            if abs(sigma1_vals[i]-sigma2_vals[i])>tol_deg:
                continue
            derivs = approximate_partials(i, points, tensors, kdtree, neighbors)

            if derivs is None:
                pts2d.append(rg.Point2d(pt.X, pt.Y)); types.append('uncertain')
                fields_sublists.append([]); dirs_sublists.append([]); indices.append(0.0)
                continue
            a,b,c,d = derivs
            kind = classify_degenerate(
                pt.X, pt.Y, points, tensors, kdtree,
                index_radius, a, b, c, d
            )
            # separatrix slopes or sampling
            slopes = []
            local_fields, local_dirs = [], []
            print(kind)
            if kind in ['wedge','trisector']:
                slopes = separatrix_slopes(a,b,c,d, imag_tol=slope_tol)
                for m in slopes:
                    L = math.hypot(1,m)
                    for sgn in (1,-1):
                        sl = sgn*m
                        label = which_field_owns(
                            pt.X,pt.Y,sl,offset_eps,
                            points,tensors,kdtree,
                            sigma1_vals,sigma2_vals,tol_deg
                        )
                        if label:
                            vec = (sgn/L, sgn*m/L)
                            local_fields.append(label); local_dirs.append(vec)
            else:
                # merged or saddle: fallback sampling
                p1_vecs, p2_vecs, _ = sample_directions_around_merged_point(
                    pt.X, pt.Y, points, tensors, kdtree,
                    n_samples=16, offset_eps=offset_eps, deg_tol=tol_deg
                )
                for v in p1_vecs:
                    local_fields.append('1'); local_dirs.append((v[0],v[1]))
                for v in p2_vecs:
                    local_fields.append('2'); local_dirs.append((v[0],v[1]))

            # compute index always
            idx_val, _ = compute_index_delmarcelle(
                pt.X, pt.Y, points, tensors, kdtree,
                radius=index_radius, n_samples=50, which_field='major'
            )
            pts2d.append(rg.Point2d(pt.X, pt.Y))
            types.append(kind)
            fields_sublists.append(local_fields)
            dirs_sublists.append(local_dirs)
            indices.append(idx_val)

        return pts2d, types, fields_sublists, dirs_sublists, indices


    def cluster_representatives(
        pts2d: List[rg.Point2d],
        types: List[str],
        fields: List[List[str]],
        dirs: List[List[Tuple[float,float]]],
        indices: List[float],
        threshold: float) -> Tuple[
        List[rg.Point2d], List[str], List[List[str]], List[List[Tuple[float,float]]], List[float]]:
        """
        Group degenerate points within `threshold` distance, then for each cluster
        pick the point closest to the cluster mean as representative.
        """
        coords = np.array([(p.X,p.Y) for p in pts2d])
        kdtree = KDTree(coords)
        visited = set()
        reps_pts, reps_types, reps_fields, reps_dirs, reps_idxs = [], [], [], [], []

        for i in range(len(coords)):
            if i in visited:
                continue
            # build cluster via BFS of neighbors within threshold
            cluster, queue = [], [i]
            visited.add(i)
            while queue:
                j = queue.pop()
                cluster.append(j)
                nbrs = kdtree.query_ball_point(coords[j], r=threshold)
                for nb in nbrs:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            # find centroid and closest index
            pts_arr = coords[cluster]
            centroid = pts_arr.mean(axis=0)
            dists = np.linalg.norm(pts_arr - centroid, axis=1)
            sel = cluster[int(np.argmin(dists))]
            reps_pts.append(pts2d[sel])
            reps_types.append(types[sel])
            reps_fields.append(fields[sel])
            reps_dirs.append(dirs[sel])
            reps_idxs.append(indices[sel])

        return reps_pts, reps_types, reps_fields, reps_dirs, reps_idxs




    # 1) Run script
    pts2d, typs, fl, dl, idxs = find_degenerate_points(
        points, sigma1_list, p1_list, sigma2_list, p2_list,tol_deg, offset_eps
    )
    print(fl)


    # 2) cluster degenerate points
    reps_pts, reps_typs, reps_fl, reps_dl, reps_idx = cluster_representatives(
        pts2d, typs, fl, dl, idxs, cluster_threshold
    )


    # 3) Sort out vectors for mean degenerate point

    field1_vecs = [
        [rg.Vector3d(dx, dy, 0) for lbl, (dx, dy) in zip(fl_sub, dl_sub) if lbl == '1']
        for fl_sub, dl_sub in zip(reps_fl, reps_dl)
    ]
    field2_vecs = [
        [rg.Vector3d(dx, dy, 0) for lbl, (dx, dy) in zip(fl_sub, dl_sub) if lbl == '2']
        for fl_sub, dl_sub in zip(reps_fl, reps_dl)
    ]



    # 3) Set seed amplitude
    seed_points_field_1 = []
    for base_pt, vec_list in zip(reps_pts, field1_vecs):
        sub_list_1 = []
        for vec in vec_list:
            if isinstance(base_pt, rg.Point2d):
                base_pt = rg.Point3d(base_pt.X, base_pt.Y, 0)
            # convert tuple→Vector3d if needed
            if isinstance(vec, tuple):
                v = rg.Vector3d(vec[0], vec[1], 0)
            else:
                v = vec
            # normalize & scale
            if v.Length > 1e-9:
                v.Unitize()
                v *= seed_amplitude
            # build line

            sub_list_1.append(base_pt + v)
        seed_points_field_1.append(sub_list_1)
    print(seed_points_field_1)


    seed_points_field_2 = []
    for base_pt, vec_list in zip(reps_pts, field2_vecs):
        sub_list_2 = []
        for vec in vec_list:
            if isinstance(base_pt, rg.Point2d):
                base_pt = rg.Point3d(base_pt.X, base_pt.Y, 0)
            # convert tuple→Vector3d if needed
            if isinstance(vec, tuple):
                v = rg.Vector3d(vec[0], vec[1], 0)
            else:
                v = vec
            # normalize & scale
            if v.Length > 1e-9:
                v.Unitize()
                v *= seed_amplitude
            # build line

            sub_list_2.append(base_pt + v)
        seed_points_field_2.append(sub_list_2)
    print(seed_points_field_2)


    def project_onto_surface(surface, pt3d):
            """
            Projects a point vertically onto a Brep or Surface using Grasshopper's Project component.
            Assumes a vertical direction of +Z.
            """
            # convert inputs
            gh_pt = rg.Point3d(pt3d[0], pt3d[1], pt3d[2])
            direction = rg.Vector3d(0, 0, 1)
            # use GH Project component
            projected = ghcomp.ProjectPoint(gh_pt, direction, surface)
        
            # Project returns a list of projected points; take first
            if projected and len(projected)>0:
                return projected[0]
            return None

    
    # Map all points on the surface and reconstruct guiding vectors
    reps_pts = [project_onto_surface(surface,rg.Point3d(p.X, p.Y, -0.01)) for p in reps_pts]

    seed_points_field_1 = [[project_onto_surface(surface,rg.Point3d(p.X, p.Y, -0.01)) for p in sub] for sub in seed_points_field_1]
    seed_points_field_2 = [[project_onto_surface(surface,rg.Point3d(p.X, p.Y, -0.01)) for p in sub] for sub in seed_points_field_2]
  
    field1_vecs = [
        [
            rg.Vector3d(
                seed.X - base_pt.X,
                seed.Y - base_pt.Y,
                seed.Z - base_pt.Z
            )
            for seed in seed_list
        ]
        for base_pt, seed_list in zip(reps_pts, seed_points_field_1)
    ]

    field2_vecs = [
        [
            rg.Vector3d(
                seed.X - base_pt.X,
                seed.Y - base_pt.Y,
                seed.Z - base_pt.Z
            )
            for seed in seed_list
        ]
        for base_pt, seed_list in zip(reps_pts, seed_points_field_2)
    ]







    # 3) Output to Grasshopper trees:
    singularities = reps_pts                                # final degenerate points 
    field_1 = tr.list_to_tree(field1_vecs)                  # direction vectors for field 1
    field_2 = tr.list_to_tree(field2_vecs)                  # direction vectord for field 2
    new_seeds_1 = tr.list_to_tree(seed_points_field_1)      # Seeds for field 1
    new_seeds_2 = tr.list_to_tree(seed_points_field_2)      # Seeds for field 2
    dege_index = reps_idx                                   # Delmarcelle index
    dege_type = tr.list_to_tree(reps_typs)                  # Degenerate type

    
    return singularities, field_1, field_2, new_seeds_1, new_seeds_2, dege_index, dege_type