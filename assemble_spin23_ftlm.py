"""Assemble a two-current matrix from three Planckian-tJ FTLM spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spin23_two_current import (
    assemble_by_polarization,
    open_circuit_schur,
    response_diagnostics,
)


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least two columns")
    return data[:, 0], data[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct C_NN, C_NE, C_EE from charge, proxy-energy, and sum auto-spectra."
    )
    parser.add_argument("charge", type=Path)
    parser.add_argument("energy_proxy", type=Path)
    parser.add_argument("sum_proxy", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    omega_n, auto_n = read_spectrum(args.charge)
    omega_e, auto_e = read_spectrum(args.energy_proxy)
    omega_s, auto_s = read_spectrum(args.sum_proxy)
    if not (np.array_equal(omega_n, omega_e) and np.array_equal(omega_n, omega_s)):
        raise ValueError("all spectra must use exactly the same frequency grid")

    matrix = assemble_by_polarization(auto_n, auto_e, auto_s)
    determinant = np.linalg.det(matrix)
    minimum_eigenvalue = np.linalg.eigvalsh(matrix)[:, 0]
    schur = np.full(omega_n.shape, np.nan)
    nonsingular = np.abs(matrix[:, 0, 0]) > 1e-14
    schur[nonsingular] = open_circuit_schur(matrix[nonsingular])
    table = np.column_stack(
        [
            omega_n,
            matrix[:, 0, 0],
            matrix[:, 0, 1],
            matrix[:, 1, 1],
            determinant,
            minimum_eigenvalue,
            schur,
        ]
    )
    np.savetxt(
        args.output,
        table,
        header="omega C_NN C_NE_proxy C_EE_proxy determinant minimum_eigenvalue open_circuit_schur",
    )
    print(response_diagnostics(matrix))


if __name__ == "__main__":
    main()

