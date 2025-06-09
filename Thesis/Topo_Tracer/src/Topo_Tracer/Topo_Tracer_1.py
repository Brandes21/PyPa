def main(points, sigma1_list, p1_list, sigma2_list, p2_list,tol_deg, offset_eps):

    # -- Prerequisites ------------------------------------------------------------
    import math
    from typing import List, Optional, Tuple
    import numpy as np
    from scipy.spatial import KDTree
    import Rhino.Geometry as rg
    from ghpythonlib import treehelpers as tr


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
        a: float,
        b: float,
        c: float,
        d: float,
        tol: float = 1e-4) -> str:
        """
        Compute delta = a*d - b*c.
        Return 'wedge' if delta>tol, 'trisector' if delta<-tol,
        else 'merged'.
        """
        delta = a*d - b*c
        #print(delta)
        if delta > tol:
            return "wedge"
        if delta < -tol:
            return "trisector"
        return "merged"

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
        points: List[rg.Point3d],
        sigma1_vals: List[float],
        p1_dirs: List[Tuple[float,float,float]],
        sigma2_vals: List[float],
        p2_dirs: List[Tuple[float,float,float]],
        tol_deg,
        offset_eps,
        slope_tol: float = 0.3) -> Tuple[
        List[rg.Point2d],
        List[str],
        List[List[str]],
        List[List[Tuple[float,float]]],
        List[float]]:
        """
        Detect degeneracies, classify, extract separatrices, and compute index.

        Returns:
        pts2d, types, field_labels, directions, indices
        """
        assert len(points)==len(sigma1_vals)==len(sigma2_vals)==len(p1_dirs)==len(p2_dirs)
        tensors = [stress_tensor_2d(s1,p1,s2,p2)
                for s1,p1,s2,p2 in zip(sigma1_vals,p1_dirs,sigma2_vals,p2_dirs)]
        coords = np.array([(pt.X,pt.Y) for pt in points])
        kdtree = KDTree(coords)
    

        pts2d, types, fields, dirs, indices = [], [], [], [], []
        for i, pt in enumerate(points):
            if abs(sigma1_vals[i]-sigma2_vals[i])>tol_deg:
                continue
            derivs = approximate_partials(i, points, tensors, kdtree, neighbors)
            if derivs is None:
                pts2d.append(rg.Point2d(pt.X,pt.Y)); types.append('uncertain')
                fields.append([]); dirs.append([]); indices.append(0.0)
                continue
            a,b,c,d = derivs
            #print(derivs)
            kind = classify_degenerate(a,b,c,d)
            slopes = separatrix_slopes(a,b,c,d, imag_tol=slope_tol) if kind!='merged' else []

            flist, dlist = [], []
            if kind!='merged':
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
                            flist.append(label); dlist.append(vec)
            # index
            idx_val, _ = compute_index_delmarcelle(
                pt.X,pt.Y,points,tensors,kdtree, radius=offset_eps,
                n_samples=50, which_field='major'
            )
            pts2d.append(rg.Point2d(pt.X,pt.Y)); types.append(kind)
            fields.append(flist); dirs.append(dlist); indices.append(idx_val)

        return pts2d, types, fields, dirs, indices


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