def main(curve_ids, min_dist, forced_keep_curves):
    import Rhino
    import Grasshopper.Kernel as gh
    from ghpythonlib import treehelpers as tr


    # === GHPython Run ===
    # inputs:
    #   curve_ids          : List[Curves]
    #   min_dist           : float
    #   forced_keep_curves : List[Curves]         !Optional input!
    # outputs:
    #   kept curves
    #   removed curves

    def curve_min_distance(crvA, crvB):
        success, ptA, ptB = crvA.ClosestPoints(crvB)
        return ptA.DistanceTo(ptB) if success else float('inf')


    def curves_equal_by_control_points(crvA, crvB, tol=1e-6):
        """True if A and B have identical NURBS control nets (within tol)."""
        ncA = crvA.ToNurbsCurve()
        ncB = crvB.ToNurbsCurve()

        # quick rejects 
        if ncA.Degree     != ncB.Degree:     return False
        if ncA.Points.Count != ncB.Points.Count: return False
        if ncA.Knots.Count  != ncB.Knots.Count:  return False

        # points & weights
        for i in range(ncA.Points.Count):
            pA = ncA.Points[i];  pB = ncB.Points[i]
            if pA.Location.DistanceTo(pB.Location) > tol: return False
            if abs(pA.Weight - pB.Weight)      > tol: return False

        # knots
        for i in range(ncA.Knots.Count):
            if abs(ncA.Knots[i] - ncB.Knots[i]) > tol: return False

        return True

    def find_forced_indices(curves, forced_keep, tol=1e-6):
        """
        Return set of indices in `curves` whose geometry matches any curve
        in `forced_keep`, by control-point equality.
        """
        forced_idx = set()
        for fk in forced_keep:
            # if user passed GH_Curve wrapper, unwrap it
            raw_fk = fk.Value if hasattr(fk, "Value") else fk
            matched = False
            for i, c in enumerate(curves):
                if curves_equal_by_control_points(c, raw_fk, tol):
                    forced_idx.add(i)
                    matched = True
                    break
            if not matched:
                ghenv.Component.AddRuntimeMessage(
                    gh.GH_RuntimeMessageLevel.Warning,
                    "Forced-keep curve #{} didn’t match any input curve.".format(
                        forced_keep.index(fk)
                    )
                )
        return forced_idx

    def remove_curves_too_close(curves, min_dist, forced_keep):

        n = len(curves)
        # 0) find which indices are forced
        if forced_keep:
            forced_idx = find_forced_indices(curves, forced_keep, tol=1e-6)
        else:
            forced_idx = []

        # 1) Build conflict adjacency list
        conflicts = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i+1, n):
                if curve_min_distance(curves[i], curves[j]) < min_dist:
                    conflicts[i].append(j)
                    conflicts[j].append(i)

        # 2) Warn if two forced curves conflict
        for i in forced_idx:
            for j in conflicts[i]:
                if j in forced_idx and i < j:
                    ghenv.Component.AddRuntimeMessage(
                        gh.GH_RuntimeMessageLevel.Warning,
                        "Forced-keep curves #{} and #{} are closer than {}."
                        .format(i, j, min_dist)
                    )

        # 3) Start with everything, immediately drop non-forced that conflict with forced
        keep = set(range(n))
        for i in forced_idx:
            for j in conflicts[i]:
                if j not in forced_idx and j in keep:
                    keep.remove(j)

        # 4) Greedy MIS on the rest (never drop forced)
        while True:
            # find any remaining conflict
            conflict_found = False
            for i in keep:
                for j in conflicts[i]:
                    if j in keep and i < j:
                        conflict_found = True
                        break
                if conflict_found:
                    break
            if not conflict_found:
                break

            # pick worst non-forced
            worst, max_deg = None, -1
            for i in keep:
                if i in forced_idx:
                    continue
                deg = sum(1 for x in conflicts[i] if x in keep)
                if deg > max_deg:
                    worst, max_deg = i, deg

            if worst is None:
                # only forced-vs-forced remain; we already warned
                break

            keep.remove(worst)

        kept   = [curves[i] for i in sorted(keep)]
        removed = [curves[i] for i in range(n) if i not in keep]
        return kept, removed


  
    #Output#
    kept, removed = remove_curves_too_close(curve_ids, min_dist, forced_keep_curves)

    kept = tr.list_to_tree(kept)
    removed = tr.list_to_tree(removed)
    return kept, removed


