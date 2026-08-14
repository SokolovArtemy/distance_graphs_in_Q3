# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import math
import os
import shutil
import time

from sage.all import GF, ZZ, matrix, vector


# --------------------------------------------------------------- step 1

def sphere_vectors_z3(N):
    """All integer (x,y,z) with x^2+y^2+z^2 == N.

    Empty if and only if N has the form 4^a*(8b+7)
    (Legendre's three-square theorem).
    """
    N = int(N)

    if N < 0:
        return []
    if N == 0:
        return [(0, 0, 0)]
    pts = []
    R = math.isqrt(N)
    for x in range(-R, R + 1):
        rem1 = N - x * x
        if rem1 < 0:
            continue
        Ry = math.isqrt(rem1)
        for y in range(-Ry, Ry + 1):
            rem2 = rem1 - y * y
            z = math.isqrt(rem2)
            if z * z == rem2:
                pts.append((x, y, z))
                if z != 0:
                    pts.append((x, y, -z))
    return pts


_OCT_PERMS = list(itertools.permutations(range(3)))
_OCT_SIGNS = list(itertools.product((1, -1), repeat=3))


def orbits_z3(S):
    """Groups S into orbits (48 signed
    permutations of the coordinates) -- automorphisms of the form
    x^2+y^2+z^2, so each orbit lies entirely in S. Returns a list of
    orbits (each a list of vectors from S) in the order their first
    representative appears in S.
    """
    Sset = set(S)
    seen = set()
    orbits = []
    for v in S:
        if v in seen:
            continue
        orb = set()
        for p in _OCT_PERMS:
            pv = (v[p[0]], v[p[1]], v[p[2]])
            for sgn in _OCT_SIGNS:
                w = (sgn[0] * pv[0], sgn[1] * pv[1], sgn[2] * pv[2])
                if w in Sset:
                    orb.add(w)
        orbits.append(sorted(orb))
        seen |= orb
    return orbits


# --------------------------------------------------------------- step 2

def _group_pairs_by_sum(S):
    """Groups unordered pairs {i,j} (i<=j, indices into S) by sum
    S[i]+S[j] using a plain dict keyed by the sum vector.

    Returns a list of groups (each a list of index pairs (i,j)), only for
    sums with >= 2 representations.
    """
    n = len(S)
    by_sum = {}
    for i in range(n):
        xi, yi, zi = S[i]
        for j in range(i, n):
            xj, yj, zj = S[j]
            key = (xi + xj, yi + yj, zi + zj)
            by_sum.setdefault(key, []).append((i, j))
    return [pairs for pairs in by_sum.values() if len(pairs) >= 2]


def build_relation_matrix(S, verbose=False):
    """Builds the Heuberger matrix H for S: columns = backtracks
    + parallelograms.

    Returns (M, n_backtrack, n_parallelogram, columns), where M is a dense
    sage matrix over ZZ of size len(S) x (n_backtrack+n_parallelogram), and
    columns is the same set of columns as a list of dict{row: coeff}
    (needed for the human-readable relation dump in save_run, without
    re-parsing M).
    """
    n = len(S)
    index = {v: i for i, v in enumerate(S)}
    columns = []

    seen = set()
    for v in S:
        if v in seen:
            continue
        nv = (-v[0], -v[1], -v[2])
        seen.add(v)
        seen.add(nv)
        i, j = index[v], index[nv]
        if i == j:
            columns.append({i: 2})
        else:
            columns.append({i: 1, j: 1})
    n_backtrack = len(columns)

    groups = _group_pairs_by_sum(S)
    for grp in groups:
        i0, j0 = grp[0]
        for i1, j1 in grp[1:]:
            col = {}
            for idx in (i0, j0):
                col[idx] = col.get(idx, 0) + 1
            for idx in (i1, j1):
                col[idx] = col.get(idx, 0) - 1
            columns.append(col)
    n_parallelogram = len(columns) - n_backtrack

    if verbose:
        print(f"  |S|={n}, columns: {n_backtrack} backtracks + "
              f"{n_parallelogram} parallelograms = {len(columns)}")

    mat_dict = {(row, ci): coeff for ci, col in enumerate(columns)
                for row, coeff in col.items()}
    M = matrix(ZZ, n, len(columns), mat_dict, sparse=False)
    return M, n_backtrack, n_parallelogram, columns


# --------------------------------------------------------------- step 3

def _combine_to_gcd(nums):
    """nums: a list of ZZ. Returns (g, coeffs) with g=gcd(nums),
    sum(coeffs[i]*nums[i]) == g -- an extended gcd over the whole list,
    """
    if not nums:
        return ZZ(0), []
    M = matrix(ZZ, len(nums), 1, [ZZ(a) for a in nums])
    H, U = M.hermite_form(transformation=True)
    return H[0, 0], list(U.row(0))


def _odd_vector_in_lattice(B, verbose=False):
    """Looks for a vector of odd augmentation in the 2-saturation L^(2) of
    the lattice L spanned by the rows of the integer matrix B.

    Algorithm: while the rows of B are dependent over GF(2) (i.e. there is
    a 0/1 combination c with c*B = 0 mod 2): take y = (sum of the marked
    rows)/2 (integral by construction); if the augmentation of y is odd,
    we are done; otherwise replace one of the rows involved with y and
    repeat. Once there are no more dependencies over GF(2), the lattice is
    already 2-saturated; then either some integer combination of the basis
    has odd augmentation (reduces to the extended gcd of the basis
    augmentations), or there is no odd vector in L^(2) at all.

    Returns a sage vector y over ZZ (indexed like the columns of B), or
    None.
    """
    t0 = time.time()
    B = matrix(ZZ, B)  # own copy, rows get replaced below

    steps = 0
    while B.nrows() > 0:
        ker = B.change_ring(GF(2)).left_kernel()
        if ker.dimension() == 0:
            break
        c = next(iter(ker.basis()))
        on = [i for i in range(B.nrows()) if c[i] != 0]
        combo = sum((B.row(i) for i in on[1:]), B.row(on[0]))
        y = vector(ZZ, [ZZ(x) // 2 for x in combo])
        steps += 1
        if int(sum(y)) % 2 == 1:
            if verbose:
                print(f"  found by 2-saturation in {steps} step(s), "
                      f"{time.time()-t0:.2f}s")
            return y
        rows = list(B.rows())
        rows[on[0]] = y
        B = matrix(ZZ, rows)

    if B.nrows() == 0:
        return None

    eps = [sum(row) for row in B.rows()]
    g, coeffs = _combine_to_gcd(eps)
    if verbose:
        print(f"  2-saturated in {steps} halving step(s), rank={B.nrows()}, "
              f"gcd(augmentations)={g}, {time.time()-t0:.2f}s")

    if g == 0 or g % 2 == 0:
        return None

    y = sum((c * B.row(i) for i, c in enumerate(coeffs)),
            vector(ZZ, B.ncols()))
    return y


def find_odd_torsion_vector(S, M, verbose=False):
    """Looks for a vector of odd augmentation in the 2-saturation L^(2) of
    the column lattice L=colspan_Z(M) -- see _odd_vector_in_lattice.
    First reduces M to a compact integer basis via HNF (this alone
    already absorbs the redundant columns of M).

    Returns a sage vector y over ZZ (indexed like S), or None.
    """
    t0 = time.time()
    Mt = M.transpose()
    B = Mt.hermite_form(include_zero_rows=False)
    if verbose:
        print(f"  rank(H_F)={B.nrows()}, HNF in {time.time()-t0:.2f}s")
    return _odd_vector_in_lattice(B, verbose=verbose)


def order_orbits(orbits, order="natural", greedy_max_points=150,
                  greedy_prime=1000000007, verbose=False):
    """Reorders the list of orbits (before they are fed into
    find_odd_torsion -- the order determines which orbits end up in the
    first, cheapest combinations and at the start of the fallback).

    order:
      "natural"     -- as returned by orbits_z3: the order in which the
                      first representative appears in S, essentially an
                      artifact of the nested-loop order in
                      sphere_vectors_z3, not motivated by anything.
      "size"        -- by increasing orbit size. A purely resource-based
                      heuristic: a small orbit (24, instead of 48 -- one
                      coordinate zero or two equal) is cheaper in the
                      matrix. In practice this barely distinguishes
                      orbits -- the size is almost always 24 or 48, with
                      many ties.
      "greedy_rank" -- greedy: at each step, take the not-yet-chosen orbit
                      that gives the largest increase in the rank of the
                      Heuberger matrix of the already-chosen set (rank
                      computed over GF(greedy_prime) -- faster than an
                      exact ZZ HNF; for ordering purposes alone this
                      precision is enough, the probability of an "unlucky"
                      prime is negligible). This aims directly at what is
                      needed (reach full rank as fast as possible) -- and
                      indeed gives noticeably more compact odd-torsion
                      vectors on the analyzed examples. BUT each step
                      rebuilds the matrix for every remaining candidate
                      orbit on the CURRENT active set -- if bounded by the
                      number of orbits (as before), the later steps still
                      build a matrix on hundreds of points and become
                      sharply more expensive. So instead of bounding the
                      number of orbits we bound the size of the trial set:
                      orbits are greedily chosen while
                      len(active)+len(candidate) <= greedy_max_points; the
                      rest (by construction, later ones, once trials are
                      already expensive) are appended by size. This
                      actually bounds the cost of each step from above,
                      not just their count.
    """
    if order == "natural":
        return list(orbits)
    if order == "size":
        return sorted(orbits, key=len)
    if order != "greedy_rank":
        raise ValueError(f"unknown order={order!r}")

    Fp = GF(greedy_prime)
    remaining = list(range(len(orbits)))
    chosen = []
    active = []
    cur_rank = 0
    while remaining:
        candidates = [i for i in remaining
                      if len(active) + len(orbits[i]) <= greedy_max_points]
        if not candidates:
            break
        best_i, best_rank = None, cur_rank
        for i in candidates:
            trial = active + orbits[i]
            M, _nb, _npar, _cols = build_relation_matrix(trial, verbose=False)
            r = int(M.change_ring(Fp).rank())
            if r > best_rank:
                best_rank, best_i = r, i
        if best_i is None:  # no candidate increases the rank
            best_i = candidates[0]
            trial = active + orbits[best_i]
            M, _nb, _npar, _cols = build_relation_matrix(trial, verbose=False)
            best_rank = int(M.change_ring(Fp).rank())
        chosen.append(best_i)
        active = active + orbits[best_i]
        cur_rank = best_rank
        remaining.remove(best_i)
        if verbose:
            print(f"    greedy_rank: orbit {best_i} (rank(GF{greedy_prime}) "
                  f"-> {cur_rank})")

    rest = sorted(remaining, key=lambda i: len(orbits[i]))
    return [orbits[i] for i in chosen] + [orbits[i] for i in rest]


def find_odd_torsion(S, verbose=False,
                      orbit_order="natural", greedy_max_points=150):
    """Searches for odd torsion on unions of orbits of the octahedral
    group (orbits_z3), rather than on all of S at once.

    Returns (y_or_None, active_S, M, columns, n_backtrack).
    """
    orbits = order_orbits(orbits_z3(S), order=orbit_order,
                           greedy_max_points=greedy_max_points, verbose=verbose)
    k = len(orbits)

    if verbose:
        print("  appending orbits in order...")
    active = []
    M = columns = None
    n_backtrack = 0
    for i, orb in enumerate(orbits):
        active = active + orb
        M, n_backtrack, _npar, columns = build_relation_matrix(active, verbose=False)
        y = find_odd_torsion_vector(active, M, verbose=False)
        if verbose:
            found_str = "found" if y is not None else "not found"
            print(f"  orbit {i+1}/{k} (+{len(orb)}, "
                  f"active {len(active)}/{len(S)}): {found_str}")
        if y is not None:
            return y, active, M, columns, n_backtrack
    return None, active, M, columns, n_backtrack


def find_odd_torsion_by_size(S, max_orbits=6, orbit_order="size",
                              greedy_max_points=150, max_combos_per_size=5000,
                              verbose=False):
    """Alternative search strategy to find_odd_torsion.

    find_odd_torsion grows the active set by appending whole orbits
    cumulatively, in a fixed order, and stops as soon as the accumulated
    set contains torsion -- so the found witness typically ends up
    touching every orbit added along the way, not just the ones it
    actually needs. Looking at the already-solved cases (Q_3/runs/),
    the found witnesses draw from a small number of distinct orbits
    (1 to 6 out of dozens available, median ~3) -- most of the orbits
    appended by cumulative growth turn out to be irrelevant padding.

    This function searches directly for a small set of orbits instead:
    it tries ALL combinations of exactly 1 orbit, then of 2, then 3, and
    so on up to max_orbits, stopping at the first combination that
    contains torsion. A combination size is skipped if the number of
    combinations C(k,size) exceeds max_combos_per_size (avoids drowning
    in combinatorics when there are many orbits) -- so this is only
    practical when max_orbits and the number of orbits are both modest;
    see search_few_orbits for a convenience wrapper.

    Unlike the old (removed) combos machinery, this does NOT fall back
    to cumulative growth if nothing is found within max_orbits -- it is
    an exploratory alternative to find_odd_torsion (aimed at finding a
    smaller, more legible witness, possibly faster when few orbits
    suffice), not a drop-in replacement with the same completeness
    guarantee.

    Returns (y_or_None, active_S, M, columns, n_backtrack); active_S/M/
    columns/n_backtrack are None/[]/0 if nothing was found within
    max_orbits.
    """
    orbits = order_orbits(orbits_z3(S), order=orbit_order,
                           greedy_max_points=greedy_max_points, verbose=verbose)
    k_total = len(orbits)

    for size in range(1, min(max_orbits, k_total) + 1):
        total = math.comb(k_total, size)
        if total > max_combos_per_size:
            if verbose:
                print(f"  skipping size {size}: C({k_total},{size})={total} "
                      f"> max_combos_per_size={max_combos_per_size}")
            continue
        if verbose:
            print(f"  trying {total} combination(s) of {size} orbit(s)...")
        for idxs in itertools.combinations(range(k_total), size):
            active = [v for i in idxs for v in orbits[i]]
            M, n_backtrack, _npar, columns = build_relation_matrix(active, verbose=False)
            y = find_odd_torsion_vector(active, M, verbose=False)
            if y is not None:
                if verbose:
                    print(f"  found on orbits {idxs} ({len(active)} points)")
                return y, active, M, columns, n_backtrack

    if verbose:
        print(f"  no torsion found using at most {max_orbits} orbit(s) "
              f"(out of {k_total} available).")
    return None, [], None, [], 0


def verify_certificate(S, M, y):
    """Independent check of the odd-torsion vector:
      - y is integral;
      - the augmentation epsilon(y) is odd;
      - sum_F y_F * F == 0 (the boundary is zero -- y is closed);
      - y in colspan_Q(M) (the rank criterion (3)).
    """
    eps = sum(y)
    assert eps % 2 == 1, f"augmentation {eps} is even"

    boundary = [0, 0, 0]
    for coeff, v in zip(y, S):
        if coeff == 0:
            continue
        c = int(coeff)
        boundary[0] += c * v[0]
        boundary[1] += c * v[1]
        boundary[2] += c * v[2]
    assert boundary == [0, 0, 0], f"nonzero boundary {boundary}"

    r1 = M.rank()
    Maug = M.augment(matrix(ZZ, len(y), 1, list(y)))
    r2 = Maug.rank()
    assert r1 == r2, f"y is outside colspan_Q(M): rank {r1} != {r2}"

    return {"augmentation": int(eps), "boundary": boundary, "rank_M": int(r1)}


# --------------------------------------------------------------- top level

def search(d, D, verbose=True, max_cols=2_000_000_000,
           orbit_order="natural", greedy_max_points=150):
    """Searches for an odd-torsion witness for Cay(Z^3, {v: |v|^2=d*D^2}).

    On success this proves chi(Q^3, d) = 4. Always returns a dict (even
    when S is empty -- then also a dict, but with S=[], M=None):

        d, D, N, S, M, columns, n_backtrack, found (bool), y, support,
        certificate

    y/support/certificate are None/[]/None if found=False (then it is
    worth trying a larger D). M, columns, n_backtrack -- the built
    Heuberger matrix and its columns as a list of dict{row:coeff} (needed
    by the caller, e.g. save_run, even on failure).

    S in the result is not necessarily the full sphere: find_odd_torsion
    adds points in orbits of the octahedral group and stops as soon as
    torsion is found (see find_odd_torsion), so when found=True, S is
    usually much smaller than the full sphere. When found=False, S equals
    the full sphere (all orbits were tried without success).

    d, D are cast to plain python int (important when called from a
    notebook with the Sage preparser, where numeric literals become
    sage.Integer -- this breaks the later JSON serialization in save_run).
    """
    d = int(d)
    D = int(D)
    N = d * D * D
    S = sphere_vectors_z3(N)
    if verbose:
        print(f"d={d}, D={D}: N=d*D^2={N}, |S|={len(S)}")
    empty = {"d": d, "D": D, "N": N, "S": S, "M": None, "columns": None,
              "n_backtrack": 0, "found": False,
              "y": None, "support": [], "certificate": None}
    if not S:
        if verbose:
            print("  N is not representable as a sum of three squares "
                  "(form 4^a(8b+7)) -- S is empty.")
        return empty

    # rough estimate of the number of pairs in the WORST case (if all orbits are needed)
    approx_pairs = len(S) * (len(S) + 1) // 2
    if approx_pairs > max_cols:
        raise ValueError(
            f"|S|={len(S)} gives too many pairs ({approx_pairs} > "
            f"max_cols={max_cols}); decrease D or increase max_cols.")

    y, active, M, columns, nb = find_odd_torsion(
        S, verbose=verbose, orbit_order=orbit_order, greedy_max_points=greedy_max_points)
    if y is None:
        if verbose:
            print("  no odd torsion found for this D.")
        empty["S"] = active
        empty["M"] = M
        empty["columns"] = columns
        empty["n_backtrack"] = nb
        return empty

    cert = verify_certificate(active, M, y)
    support = [(active[i], int(y[i])) for i in range(len(active)) if y[i] != 0]
    if verbose:
        print(f"  FOUND: augmentation={cert['augmentation']}, "
              f"support of {len(support)} vectors, rank(H_F)={cert['rank_M']}, "
              f"used {len(active)}/{len(S)} points of S")

    return {"d": d, "D": D, "N": N, "S": active, "M": M, "columns": columns,
            "n_backtrack": nb, "found": True,
            "y": y, "support": support, "certificate": cert}


def search_few_orbits(d, D, verbose=True, max_cols=2_000_000_000,
                       max_orbits=6, orbit_order="size",
                       greedy_max_points=150, max_combos_per_size=5000):
    """Like search(), but uses find_odd_torsion_by_size instead of
    find_odd_torsion: searches directly for a witness spanning at most
    max_orbits orbits, rather than growing a cumulative set. See
    find_odd_torsion_by_size's docstring for the motivation and the
    (important) caveat that this does not fall back to a full search --
    if no combination of at most max_orbits orbits works, it reports
    found=False even though a witness might still exist using more
    orbits (try search() for that, or increase max_orbits).

    Returns the same shape of dict as search().
    """
    d = int(d)
    D = int(D)
    N = d * D * D
    S = sphere_vectors_z3(N)
    if verbose:
        print(f"d={d}, D={D}: N=d*D^2={N}, |S|={len(S)}")
    empty = {"d": d, "D": D, "N": N, "S": S, "M": None, "columns": None,
              "n_backtrack": 0, "found": False,
              "y": None, "support": [], "certificate": None}
    if not S:
        if verbose:
            print("  N is not representable as a sum of three squares "
                  "(form 4^a(8b+7)) -- S is empty.")
        return empty

    approx_pairs = len(S) * (len(S) + 1) // 2
    if approx_pairs > max_cols:
        raise ValueError(
            f"|S|={len(S)} gives too many pairs ({approx_pairs} > "
            f"max_cols={max_cols}); decrease D or increase max_cols.")

    y, active, M, columns, nb = find_odd_torsion_by_size(
        S, max_orbits=max_orbits, orbit_order=orbit_order,
        greedy_max_points=greedy_max_points,
        max_combos_per_size=max_combos_per_size, verbose=verbose)
    if y is None:
        if verbose:
            print(f"  no odd torsion found using at most {max_orbits} orbit(s).")
        return empty

    cert = verify_certificate(active, M, y)
    support = [(active[i], int(y[i])) for i in range(len(active)) if y[i] != 0]
    if verbose:
        print(f"  FOUND: augmentation={cert['augmentation']}, "
              f"support of {len(support)} vectors, rank(H_F)={cert['rank_M']}, "
              f"used {len(active)}/{len(S)} points of S")

    return {"d": d, "D": D, "N": N, "S": active, "M": M, "columns": columns,
            "n_backtrack": nb, "found": True,
            "y": y, "support": support, "certificate": cert}


# --------------------------------------------------------------- text dump
#
# Every vector in S gets its own index. 
# matrix.txt columns and odd_torsion_vector.txt both
# reference these same plain indices directly (they are already exactly
# the row indices used internally, see build_relation_matrix), so no
# translation table is needed between the three files.

def _signed_token(i, sign):
    return ("+" if sign > 0 else "-") + str(i)


def _write_vectors_txt(S, path):
    """'index: vector' -- one line per vector in S (v and -v get separate
    indices)."""
    with open(path, "w") as f:
        for i, v in enumerate(S):
            f.write(f"{i}: {v}\n")


def _write_matrix_txt(columns, n_backtrack, path):
    """Each line is one relation (a column of H_F) written as a sum of
    signed indices into S: '+i' / '-i' means a coefficient of +1 / -1 at
    S[i] (multiplicities >1 are expanded by repeating the token).
    Backtracks e_v+e_{-v} are '+i +j' (both coefficients are genuinely
    +1, v and -v being different indices); parallelograms
    e_F1+e_F2-e_F3-e_F4 are '+i +j -k -l'."""
    with open(path, "w") as f:
        f.write(f"# {len(columns)} relations: {n_backtrack} backtracks "
                f"('+i +j') + {len(columns) - n_backtrack} parallelograms "
                f"('+i +j -k -l'); indices are in vectors.txt\n")
        for col in columns:
            terms = []
            for row, coeff in col.items():
                terms.extend([_signed_token(row, 1 if coeff > 0 else -1)] * abs(coeff))
            f.write(" ".join(terms) + "\n")


def _write_odd_torsion_vector_txt(S, support, path):
    """Lines of the form "index" x "count" -- one per nonzero coordinate
    of the found vector y; index is the plain position in S (see
    vectors.txt) and count is the actual signed coefficient y[index]."""
    index = {v: i for i, v in enumerate(S)}
    with open(path, "w") as f:
        for v, c in support:
            f.write(f'{index[v]} x {c}\n')


# --------------------------------------------------------------- saving

def save_run(result, base_dir=None):
    """Saves the result of search() into its own folder next to the
    script (Q_3/runs/d{d}_D{D}/ by default). The folder is created and
    filled under the "bare" name, and only right after the write is
    finished, if torsion was found, is it renamed -- a '+' is prepended:
    Q_3/runs/+d{d}_D{D}/.

        vectors.txt           -- "index: vector", one per vector of S
                                  (see the text-dump section above);
        matrix.txt              -- the same matrix as relation lines in
                                  signed indices ('+i +j -k -l' / '+i +j');
        odd_torsion_vector.txt  -- the found odd-torsion vector, lines
                                  '"index" x "count"', only if found=True;

    Returns the path to the folder (already with '+' if found=True).
    """
    d, D = result["d"], result["D"]
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "runs", f"d{d}_D{D}")
    base_dir = base_dir.rstrip(os.sep)
    parent, name = os.path.split(base_dir)
    if name.startswith("+"):
        name = name[1:]
        base_dir = os.path.join(parent, name)
    plus_dir = os.path.join(parent, "+" + name)

    # clean start: don't leave files from a previous run of this (d,D)
    # under either name
    for stale in (base_dir, plus_dir):
        if os.path.isdir(stale):
            shutil.rmtree(stale)
    os.makedirs(base_dir, exist_ok=True)

    S = result["S"]

    _write_vectors_txt(S, os.path.join(base_dir, "vectors.txt"))

    meta = {
        "d": d, "D": D, "N": result["N"],
        "size_S": len(S),
        "found": result["found"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    M = result.get("M")
    columns = result.get("columns")
    if M is not None:
        meta["matrix_shape"] = [int(M.nrows()), int(M.ncols())]
        n_backtrack = result.get("n_backtrack", 0)
        _write_matrix_txt(columns, n_backtrack,
                           os.path.join(base_dir, "matrix.txt"))

    if result["found"]:
        cert = result["certificate"]
        _write_odd_torsion_vector_txt(S, result["support"],
                                       os.path.join(base_dir, "odd_torsion_vector.txt"))
        meta["augmentation"] = cert["augmentation"]
        meta["support_size"] = len(result["support"])

    if result["found"]:
        os.rename(base_dir, plus_dir)
        return plus_dir
    return base_dir


def search_and_save(d, D, base_dir=None, verbose=True,
                     orbit_order="natural", greedy_max_points=150):
    """search(d, D) + save_run() in one call. Returns (result, path).

    orbit_order/greedy_max_points are forwarded to search() (see its
    docstring and find_odd_torsion/order_orbits) -- they control the
    order in which orbits are tried during the cumulative search.
    """
    result = search(d, D, verbose=verbose,
                     orbit_order=orbit_order, greedy_max_points=greedy_max_points)
    path = save_run(result, base_dir=base_dir)
    if verbose:
        print(f"  saved to {path}")
    return result, path


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage:")
        print("  sage -python odd_torsion.py d D")
        sys.exit(1)

    d = int(args[0])
    res = search(d, int(args[1]))

    path = save_run(res)
    print(f"Saved to {path}")

    if not res["found"]:
        print("Result: no torsion found.")
    else:
        print(f"Result: G(Q^3, {res['d']}) is not 3-colorable (chi=4), "
              f"D={res['D']}.")
