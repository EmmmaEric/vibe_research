"""Baseline: dense matrix-vector product.

Deliberately the straightforward implementation, written to be correct rather
than fast, and *not* crippled: it uses the same pure-Python idiom as the
candidate so the comparison is not rigged by language-level handicapping. A
comparison against an under-tuned baseline is the classic way to manufacture a
speedup, which is exactly what this experiment must avoid.
"""

from __future__ import annotations


def dense_matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """y = A @ x for a dense row-major matrix.

    Visits every entry, including the zeros -- which is the cost the sparse
    candidate is trying to avoid.
    """
    if not matrix:
        return []
    n_cols = len(matrix[0])
    if any(len(row) != n_cols for row in matrix):
        raise ValueError("all rows must have the same length")
    if len(vector) != n_cols:
        raise ValueError(
            f"dimension mismatch: matrix has {n_cols} columns, vector has {len(vector)}"
        )

    out = [0.0] * len(matrix)
    for i, row in enumerate(matrix):
        total = 0.0
        for j, a in enumerate(row):
            total += a * vector[j]
        out[i] = total
    return out
