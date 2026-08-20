# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import math
import time

from sage.all import GF, ZZ, matrix


def orbit_representatives(N):
    """
    Returns all triples 0 <= a <= b <= c with a^2 + b^2 + c^2 == N.
    """
    reps = []
    a = 0
    while 3 * a * a <= N:
        b = a
        while a * a + 2 * b * b <= N:
            rest = N - a * a - b * b
            c = math.isqrt(rest)
            if c * c == rest:
                reps.append((a, b, c))
            b += 1
        a += 1
    return reps


def orbit_of(v):
    """
    The orbit of v under the 48 signed coordinate permutations.
    """
    orbit = set()
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            orbit.add((s[0] * v[p[0]], s[1] * v[p[1]], s[2] * v[p[2]]))
    return sorted(orbit)


def orbits(N):
    """All orbits on the sphere |v|^2 == N, one per representative."""
    return [orbit_of(rep) for rep in orbit_representatives(N)]


def sphere_vectors(N):
    """All integer (x, y, z) with x^2 + y^2 + z^2 == N, listed orbit by
    orbit (built from orbit_representatives)."""
    return [v for orbit in orbits(N) for v in orbit]


def relation_matrix(S):
    """The Heuberger-type homotopy-relation matrix of S: rows are the points, columns are the
    homotopy relations -- one backtrack e_v + e_{-v} per antipodal pair, plus one
    parallelogram e_x + e_y - e_z - e_t per extra representation of a sum
    """
    n = len(S)
    index = {v: i for i, v in enumerate(S)}
    columns = []

    seen = set()
    for v in S:
        if v in seen:
            continue
        w = (-v[0], -v[1], -v[2])
        seen.add(v)
        seen.add(w)
        col = {}
        for u in (v, w):
            col[index[u]] = col.get(index[u], 0) + 1
        columns.append(col)

    by_sum = {}
    for i in range(n):
        xi, yi, zi = S[i]
        for j in range(i, n):
            xj, yj, zj = S[j]
            by_sum.setdefault((xi + xj, yi + yj, zi + zj), []).append((i, j))

    for pairs in by_sum.values():
        if len(pairs) < 2:
            continue
        i0, j0 = pairs[0]
        for i1, j1 in pairs[1:]:
            col = {}
            for k in (i0, j0):
                col[k] = col.get(k, 0) + 1
            for k in (i1, j1):
                col[k] = col.get(k, 0) - 1
            columns.append(col)

    entries = {(row, c): value
               for c, col in enumerate(columns)
               for row, value in col.items()}
    return matrix(ZZ, n, len(columns), entries)


def has_odd_torsion_vector(M, verbose=False):
    """The four-step check on a Heuberger matrix M (rows = points).

    Returns True if colspan_Q(M) ∩ Z^S contains a vector of odd
    augmentation, False otherwise."""
    t0 = time.time()
    n = M.nrows()

    A = M.left_kernel().basis_matrix()         
    A2 = A.change_ring(GF(2))                   
    
    ones = matrix(GF(2), 1, n, [1] * n)         
    rank_A = A2.rank()
    rank_stacked = A2.stack(ones).rank()
    exists = rank_stacked == rank_A + 1

    if verbose:
        print(f"  ker_Z(M^T): {A.nrows()} basis vector(s) of length {n}; "
              f"rank_F2(A)={rank_A}, rank_F2([A;1])={rank_stacked} -> "
              f"{'odd vector EXISTS' if exists else 'all augmentations even'} "
              f"({time.time()-t0:.2f}s)")
    return exists


def has_odd_torsion_on(S, verbose=False):
    """Build the Heuberger matrix on the points S and run the check."""
    return has_odd_torsion_vector(relation_matrix(S), verbose=verbose)


def check(d, D, verbose=True, full_sphere=False):
    """With full_sphere=True the check runs once on the whole sphere at once.
    Otherwise the orbits are added one at a time (in the order given by
    orbit_representatives) and the check is run after each, reporting how
    much of the sphere is needed before an odd-augmentation vector
    appears.

    Returns a dict: d, D, N, size_S, exists, orbits_needed, points_needed,
    seconds. orbits_needed/points_needed are None when nothing was found,
    and orbits_needed is None in full-sphere mode (not tracked there).
    """
    d = int(d)
    D = int(D)
    N = d * D * D
    t0 = time.time()
    parts = orbits(N)
    S = [v for orbit in parts for v in orbit]
    if verbose:
        print(f"d={d}, D={D}: N=d*D^2={N}, |S|={len(S)}, "
              f"{len(parts)} orbit(s)")

    result = {"d": d, "D": D, "N": N, "size_S": len(S), "exists": False,
              "orbits_needed": None, "points_needed": None, "seconds": 0.0}
    if not S:
        if verbose:
            print("S is empty.")
        result["seconds"] = time.time() - t0
        return result

    if full_sphere:
        result["exists"] = has_odd_torsion_on(S, verbose=verbose)
        if result["exists"]:
            result["points_needed"] = len(S)
    else:
        active = []
        for i, orbit in enumerate(parts):
            active = active + orbit
            exists = has_odd_torsion_on(active)
            if verbose:
                print(f"  orbit {i+1}/{len(parts)} (+{len(orbit)}, active "
                      f"{len(active)}/{len(S)}): "
                      f"{'EXISTS' if exists else 'not yet'}")
            if exists:
                result.update(exists=True, orbits_needed=i + 1,
                              points_needed=len(active))
                break

    result["seconds"] = time.time() - t0
    if verbose:
        if result["exists"]:
            where = (f"using {result['orbits_needed']} orbit(s), "
                     f"{result['points_needed']}/{len(S)} points"
                     if result["orbits_needed"] else "on the full sphere")
            print(f"  => EXISTS {where} ({result['seconds']:.2f}s)")
        else:
            print(f"  => does not exist on the full sphere "
                  f"({result['seconds']:.2f}s)")
    return result


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage:")
        print("  sage -python odd_cycle_check.py d D [--by-orbit]")
        sys.exit(1)

    res = check(int(args[0]), int(args[1]),
                full_sphere="--by-orbit" not in args)
    sys.exit(0 if res["exists"] else 1)
