def main(points, principal_vectors, seed_points, seed_vectors, boundary_curves, domain_surface,
    k, num_steps, step_sign,boundary_tolerance,collision_threshold, n_back, seed_dist):
    """
    Wraps full PSL tracing workflow into one function.
    Inputs:
      points             - list of Point3d or [x,y,z]
      principal_vectors  - list of Vector3d or [x,y,z]
      seed_points        - list of Point3d or [x,y,z]
      seed_vectors       - list of Vector3d or [x,y,z]
      k                  - neighbors for interpolation
      num_steps          - max steps per seed
      step_sign          - +1 or -1 integration direction
      boundary_curves    - list of domain edge Curves
      domain_surface     - Brep or Surface to derive edges if curves not provided
      boundary_tolerance - stop if closer than this to boundary
      collision_threshold- merge if within this of existing line
    Outputs:
      a - GH tree of (trajectory_pts, polylineCurve) per seed
      b - GH tree of bridging segment curves
    """

    import math
    import rhinoscriptsyntax as rs
    import Rhino
    import Rhino.Geometry as rg
    import scriptcontext as sc
    import numpy as np
    from scipy.spatial import KDTree
    from ghpythonlib import treehelpers as tr


    # --------------------
    # Utility functions
    # --------------------
    def to_xyz(o):
        if hasattr(o, "X"): return [o.X, o.Y, o.Z]
        return list(o)

    def normalize_3d(v):
        m = math.sqrt(v[0]**2+v[1]**2+v[2]**2)
        if m<1e-12: return [0.0,0.0,0.0]
        return [v[i]/m for i in range(3)]

    def find_closest_neighbors_kd_3d(pt, tree, k):
        d, idxs = tree.query(pt, k=k)
        return [idxs] if isinstance(idxs,int) else list(idxs)

    def interpolate_vector_3d_consistent(pt, pts3, vecs3, nbrs, ref_dir):
        wsum=[0,0,0]; wts=[]
        for i in nbrs:
            vx,vy,vz=vecs3[i]
            if ref_dir is not None and (vx*ref_dir[0]+vy*ref_dir[1]+vz*ref_dir[2])<0:
                vx,vy,vz=-vx,-vy,-vz
            dx=pt[0]-pts3[i][0]; dy=pt[1]-pts3[i][1]; dz=pt[2]-pts3[i][2]
            d=math.sqrt(dx*dx+dy*dy+dz*dz)
            w=1.0/(d+1e-6); wts.append(w)
            wsum[0]+=vx*w; wsum[1]+=vy*w; wsum[2]+=vz*w
        sw=sum(wts)
        if sw<1e-12: return [0,0,0]
        wsum=[c/sw for c in wsum]
        nm=math.sqrt(wsum[0]**2+wsum[1]**2+wsum[2]**2)
        return [wsum[i]/nm if nm>1e-12 else 0 for i in range(3)]

    def adjust_step_size_3d(pt, nbrs, pts3, sign):
        if not nbrs: return 0.1*sign
        ds=[math.dist(pt,pts3[i]) for i in nbrs]
        avg=sum(ds)/len(ds)
        return min(0.1,avg/2.0)*sign

    def project_onto_surface(surface, pt3d):
        P=rg.Point3d(*pt3d)
        if isinstance(surface, rg.Brep): face=surface.Faces[0]
        elif isinstance(surface, rg.Surface): face=surface
        else: return None
        rc,u,v=face.ClosestPoint(P)
        return [pt.X for pt in [face.PointAt(u,v)]] if rc else None

    def is_on_surface(surf,pt3d,tol):
        p=project_onto_surface(surf,pt3d)
        if p is None: return False
        return math.dist(p,pt3d)<tol

    def get_brep_edge_curves(surf):
        if isinstance(surf, rg.Surface):
            surf=rg.Brep.CreateFromSurface(surf)
            if not surf: return []
        return [e.ToNurbsCurve() for e in surf.Edges]

    def distance_to_brep_edges(pt3d,curves):
        P=rg.Point3d(*pt3d); md=float('inf'); cp=None
        for c in curves:
            rc,t=c.ClosestPoint(P)
            if rc:
                Q=c.PointAt(t); d=Q.DistanceTo(P)
                if d<md: md,cp=d,Q
        return md,cp

    def runge_kutta_step_3d(curr_pt,curr_dir,h,k,vecs3,pts3,sign,tree,curves,tol):
        nbrs=find_closest_neighbors_kd_3d(curr_pt,tree,k)
        h=adjust_step_size_3d(curr_pt,nbrs,pts3,sign)
        k1=interpolate_vector_3d_consistent(curr_pt,pts3,vecs3,nbrs,curr_dir)
        m1=[curr_pt[i]+0.5*h*k1[i] for i in range(3)]
        k2=interpolate_vector_3d_consistent(m1,pts3,vecs3,find_closest_neighbors_kd_3d(m1,tree,k),k1)
        m2=[curr_pt[i]+0.5*h*k2[i] for i in range(3)]
        k3=interpolate_vector_3d_consistent(m2,pts3,vecs3,find_closest_neighbors_kd_3d(m2,tree,k),k2)
        end=[curr_pt[i]+h*k3[i] for i in range(3)]
        k4=interpolate_vector_3d_consistent(end,pts3,vecs3,find_closest_neighbors_kd_3d(end,tree,k),k3)
        np=[curr_pt[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6.0 for i in range(3)]
        return np,k4

    def build_polyline_curve_3d(pts):
        return rg.PolylineCurve([rg.Point3d(*p) for p in pts])

    def closest_point_on_polyline_3d(pt,curve):
        rc,t=curve.ClosestPoint(rg.Point3d(*pt))
        if not rc: return None,float('inf')
        Q=curve.PointAt(t); return [Q.X,Q.Y,Q.Z],Q.DistanceTo(rg.Point3d(*pt))

    def find_closest_existing_line_3d(pt,existing,thresh):
        md=float('inf'); idx=None; cp=None
        for i,(pts,crv) in enumerate(existing):
            cpt,d=closest_point_on_polyline_3d(pt,crv)
            if d<md: md,idx,cp=d,i,cpt
        return (idx,cp,md) if md<thresh else (None,None,float('inf'))

    def distance_to_seedpt(pt,seed_pts):
        md=float('inf'); sp=None
        P=rg.Point3d(*pt)
        for s in seed_pts:
            d=P.DistanceTo(s)
            if d<md: md,sp=d,s
        return md,sp

    def get_knn_adjacency(pts3,kAdj=6):
        arr=np.array(pts3);T=KDTree(arr);adj=[]
        for i in range(len(pts3)):
            d,inds=T.query(arr[i],kAdj+1)
            adj.append([j for j in inds if j!=i])
        return adj

    def unify_vector_field(pts3,vecs3,adj):
        vis=[False]*len(pts3);queue=[0];vis[0]=True
        while queue:
            i=queue.pop(0)
            for j in adj[i]:
                if not vis[j]:
                    if sum(vecs3[i][c]*vecs3[j][c] for c in range(3))<0:
                        vecs3[j]=[-c for c in vecs3[j]]
                    vis[j]=True;queue.append(j)
        return vecs3

    # --------------------
    # Prepare inputs
    # --------------------
    pts3=[to_xyz(p) for p in points]
    vecs3=[to_xyz(v) for v in principal_vectors]
    # global consistency
    adj=get_knn_adjacency(pts3,k)
    vecs3=unify_vector_field(pts3,vecs3,adj)
    seeds=[to_xyz(s) for s in seed_points]
    sdirs=[to_xyz(sv) for sv in seed_vectors]
    # boundary curves from surface
    if boundary_curves is None and domain_surface:
        boundary_curves=get_brep_edge_curves(domain_surface)
    # KD-tree
    tree=KDTree(np.array(pts3))

    existing_trajectories=[]
    bridging_lines_out=[]

    # --------------------
    # Trace seeds
    # --------------------
    for s_pt,s_dir in zip(seeds,sdirs):
        cur_pt=list(s_pt); cur_dir=normalize_3d(s_dir)
        traj=[rg.Point3d(*cur_pt)]; bridge=None;nn=0
        for _ in range(num_steps):
            nn+=1
            nxt,ndir=runge_kutta_step_3d(cur_pt,cur_dir,step_sign,k,vecs3,pts3,step_sign,tree,boundary_curves,boundary_tolerance)
            if boundary_curves:
                d_e,cp=distance_to_brep_edges(nxt,boundary_curves)
                if d_e<boundary_tolerance:
                    bridge=[traj[-1],cp]; break
            idx,cp,d=find_closest_existing_line_3d(nxt,existing_trajectories,collision_threshold)
            if idx is not None:
                # merge back 15 steps
                back=n_back if len(traj)>n_back else len(traj)-1
                bridge=[traj[-back],rg.Point3d(*cp)]; traj=traj[:-back+1]; break
            if nn>20:
                d_s,sp=distance_to_seedpt(nxt,[rg.Point3d(*pt) for pt in seeds])
                if d_s<seed_dist:
                    bridge=[traj[-1],sp]; break
            traj.append(rg.Point3d(*nxt)); cur_pt=nxt; cur_dir=ndir
        existing_trajectories.append((traj, build_polyline_curve_3d([[pt.X,pt.Y,pt.Z] for pt in traj])))
        if bridge: bridging_lines_out.append(bridge)

    # --------------------
    # Output
    # --------------------
    a=tr.list_to_tree(existing_trajectories)
    b=tr.list_to_tree(bridging_lines_out)
    return a,b