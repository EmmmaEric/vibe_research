"""Candidate: CSR (compressed sparse row) matrix-vector product.

The mechanism under test: store only non-zero entries and visit only those, so
the cost scales with the number of non-zeros rather than with rows x columns.

Memory note, since the contract carries a fixed-memory premise: CSR stores
``nnz`` values, ``nnz`` column indices and ``n_rows + 1`` row pointers, so it is
smaller than the dense form only when density is below roughly 1/2. At high
density it is *larger*, which is a real limit of this candidate and is asserted
in the edge-case test rather than left as a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CSRMatrix:
    values: list[float]
    col_indices: list[int]
    row_pointers: list[int]
    n_rows: int
    n_cols: int

    @property
    def nnz(self) -> int:
        return len(self.values)

    @property
    def density(self) -> float:
        cells = self.n_rows * self.n_cols
        return self.nnz / cells if cells else 0.0

    def storage_units(self) -> int:
        """Stored scalars: values + indices + row pointers."""
        return self.nnz * 2 + self.n_rows + 1


def to_csr(matrix: list[list[float]], *, tolerance: float = 0.0) -> CSRMatrix:
    """Convert a dense matrix to CSR, dropping entries at or below ``tolerance``."""
    if not matrix:
        return CSRMatrix([], [], [0], 0, 0)
    n_cols = len(matrix[0])
    if any(len(row) != n_cols for row in matrix):
        raise ValueError("all rows must have the same length")

    values: list[float] = []
    cols: list[int] = []
    pointers: list[int] = [0]
    for row in matrix:
        for j, a in enumerate(row):
            if abs(a) > tolerance:
                values.append(a)
                cols.append(j)
        pointers.append(len(values))
    return CSRMatrix(values, cols, pointers, len(matrix), n_cols)


def csr_matvec(csr: CSRMatrix, vector: list[float]) -> list[float]:
    """y = A @ x, visiting only stored non-zeros."""
    if len(vector) != csr.n_cols:
        raise ValueError(
            f"dimension mismatch: matrix has {csr.n_cols} columns, vector has "
            f"{len(vector)}"
        )
    values = csr.values
    cols = csr.col_indices
    pointers = csr.row_pointers

    out = [0.0] * csr.n_rows
    for i in range(csr.n_rows):
        total = 0.0
        for k in range(pointers[i], pointers[i + 1]):
            total += values[k] * vector[cols[k]]
        out[i] = total
    return out
