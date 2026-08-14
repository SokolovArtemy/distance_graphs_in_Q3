# -*- coding: utf-8 -*-
"""Independent verification of a saved odd-torsion run.

Deliberately does NOT import anything from odd_torsion.py: it re-parses the
plain-text files a run folder contains (vectors.txt, matrix.txt,
odd_torsion_vector.txt) from scratch and re-derives the mathematical claim
on its own, so a bug shared between the search code and the checker cannot
silently validate a wrong answer.

Given a run folder (e.g. Q_3/runs/+d30_D9/), it:
  1. reads vectors.txt to recover the point set S (every vector, including
     both v and -v, has its own plain index -- see
     odd_torsion.py:_write_vectors_txt);
  2. reads matrix.txt to rebuild the Heuberger matrix H_F (one column per
     relation line; '+i'/'-i' tokens are the signed coefficient at S[i]);
  3. reads odd_torsion_vector.txt to rebuild the vector y;
  4. checks, independently, the three conditions of Proposition 3.1 of the
     paper (Sokolov, "Chromatic number of distance graphs in Q^3 for
     several families of distances", 2026):
       - y is an integer vector (guaranteed by the text format itself);
       - the augmentation eps(y) = sum(y) is odd;
       - the boundary sum_F y_F * F is the zero vector in Z^3 (y is closed);
     and, the specific check requested for this script:
       - appending y as an extra column to H_F does NOT increase its rank,
         i.e. rank(H_F) == rank([H_F | y]), the rank criterion (3) of
         Sec. 5 of the paper for y lying in colspan_Q(H_F).

Usage:
    sage -python verify_independent.py <run_folder>
"""

from __future__ import annotations

import os
import re
import sys

from sage.all import ZZ, matrix, vector


def _parse_vectors_txt(path):
    """'index: (x, y, z)' per line -> S, a list with S[index] = (x,y,z)."""
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, vec_str = line.split(":", 1)
            coords = tuple(int(c) for c in re.findall(r"-?\d+", vec_str))
            if len(coords) != 3:
                raise ValueError(f"bad vector line in {path!r}: {line!r}")
            entries[int(idx_str)] = coords
    n = len(entries)
    S = [None] * n
    for i, v in entries.items():
        S[i] = v
    if any(s is None for s in S):
        raise ValueError("vectors.txt indices are not a contiguous 0..n-1 range")
    return S


def _parse_signed_token(token):
    sign = 1 if token[0] == "+" else -1
    return int(token[1:]), sign


def _parse_matrix_txt(path):
    """Each non-comment, non-blank line is one relation (a column of
    H_F): a whitespace-separated list of signed tokens '+i'/'-i', where
    the sign is the actual coefficient sign at the plain index i (repeated
    tokens accumulate). Returns a list of columns as dict{row: coeff}."""
    columns = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            col = {}
            for tok in line.split():
                row, sign = _parse_signed_token(tok)
                col[row] = col.get(row, 0) + sign
            columns.append(col)
    return columns


def _parse_odd_torsion_vector_txt(path, n_rows):
    """Lines of the form '"5" x "-3"' -> a length-n_rows vector y with
    y[index] = count (index is a plain, unsigned position in S; count is
    the actual signed coefficient). Accepts both the current unquoted
    format ('5 x -3') and the older quoted one ('"5" x "-3"')."""
    y = [0] * n_rows
    pattern = re.compile(r'"?(\d+)"?\s*x\s*"?(-?\d+)"?')
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                raise ValueError(f"bad odd_torsion_vector line: {line!r}")
            idx, count = int(m.group(1)), int(m.group(2))
            y[idx] += count
    return y


def verify_folder(folder, verbose=True):
    """Independently re-checks a saved run folder. Returns a dict:
        {"rank_ok": bool, "augmentation_odd": bool, "boundary_zero": bool,
         "all_ok": bool, "rank_H": int, "rank_H_plus_y": int,
         "augmentation": int, "boundary": [int,int,int]}
    Raises FileNotFoundError if any of the three expected files are
    missing (odd_torsion_vector.txt is required -- this checker only
    makes sense for a run that claims found=True).
    """
    vectors_path = os.path.join(folder, "vectors.txt")
    matrix_path = os.path.join(folder, "matrix.txt")
    vec_path = os.path.join(folder, "odd_torsion_vector.txt")
    for p in (vectors_path, matrix_path, vec_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    S = _parse_vectors_txt(vectors_path)
    n_rows = len(S)

    columns = _parse_matrix_txt(matrix_path)
    n_cols = len(columns)
    mat_dict = {(row, ci): coeff for ci, col in enumerate(columns)
                for row, coeff in col.items()}
    H = matrix(ZZ, n_rows, n_cols, mat_dict, sparse=False)

    y = _parse_odd_torsion_vector_txt(vec_path, n_rows)
    yv = vector(ZZ, y)

    augmentation = int(sum(y))
    augmentation_odd = augmentation % 2 == 1

    boundary = [0, 0, 0]
    for coeff, v in zip(y, S):
        if coeff == 0:
            continue
        boundary[0] += coeff * v[0]
        boundary[1] += coeff * v[1]
        boundary[2] += coeff * v[2]
    boundary_zero = boundary == [0, 0, 0]

    rank_H = int(H.rank())
    H_plus_y = H.augment(matrix(ZZ, n_rows, 1, list(yv)))
    rank_H_plus_y = int(H_plus_y.rank())
    rank_ok = rank_H == rank_H_plus_y  # appending y does not increase the rank

    all_ok = rank_ok and augmentation_odd and boundary_zero

    if verbose:
        print(f"folder: {folder}")
        print(f"  |S| = {n_rows}, relations (columns of H_F) = {n_cols}")
        print(f"  augmentation(y) = {augmentation}  "
              f"({'odd -- OK' if augmentation_odd else 'EVEN -- FAIL'})")
        print(f"  boundary(y) = {tuple(boundary)}  "
              f"({'zero -- OK' if boundary_zero else 'NONZERO -- FAIL'})")
        print(f"  rank(H_F) = {rank_H}, rank([H_F | y]) = {rank_H_plus_y}  "
              f"({'unchanged -- OK, y in colspan_Q(H_F)' if rank_ok else 'INCREASED -- FAIL, y is NOT in colspan_Q(H_F)'})")
        print(f"  => {'PASS: valid odd-torsion witness' if all_ok else 'FAIL: NOT a valid witness'}")

    return {
        "rank_ok": rank_ok, "augmentation_odd": augmentation_odd,
        "boundary_zero": boundary_zero, "all_ok": all_ok,
        "rank_H": rank_H, "rank_H_plus_y": rank_H_plus_y,
        "augmentation": augmentation, "boundary": boundary,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sage -python verify_independent.py <run_folder>")
        sys.exit(1)
    result = verify_folder(sys.argv[1])
    sys.exit(0 if result["all_ok"] else 1)
