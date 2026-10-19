"""Two-current response tools for the Planckian-tJ calculation.

The original program computes a scalar charge-current spectrum.  This module
collects the model-independent part of the extension to a symmetric 2 x 2
response.  The two operators may be any Hermitian currents.  For the intended
thermoelectric application they are the particle and energy currents.

Important: :func:`energy_weighted_current_proxy` is a diagnostic proxy, not the
interacting t-J energy current.  A physical energy current must be obtained
from a declared local-energy density and its continuity equation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SpectralResponse:
    """A frequency grid and its Hermitian two-current spectral matrices."""

    omega: Array
    matrix: Array

    def __post_init__(self) -> None:
        omega = np.asarray(self.omega)
        matrix = np.asarray(self.matrix)
        if omega.ndim != 1:
            raise ValueError("omega must be one-dimensional")
        if matrix.shape != (omega.size, 2, 2):
            raise ValueError("matrix must have shape (len(omega), 2, 2)")


def _as_hermitian(name: str, matrix: Array, atol: float = 1e-11) -> Array:
    value = np.asarray(matrix, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(value, value.conj().T, atol=atol, rtol=0.0):
        raise ValueError(f"{name} must be Hermitian")
    return value


def kubo_detailed_balance(beta: float, omega: Array | float) -> Array:
    """Return (1-exp(-beta*omega))/omega with its beta limit at zero."""

    if beta < 0:
        raise ValueError("beta must be non-negative")
    value = np.asarray(omega, dtype=float)
    scaled = beta * value
    result = np.empty_like(value)
    small = np.abs(scaled) < 1e-7
    # beta * (1 - x/2 + x^2/6 + ...) is stable at omega=0.
    result[small] = beta * (
        1.0 - scaled[small] / 2.0 + scaled[small] ** 2 / 6.0
    )
    result[~small] = -np.expm1(-scaled[~small]) / value[~small]
    return result


def gaussian_spectral_matrix(
    hamiltonian: Array,
    currents: Sequence[Array],
    beta: float,
    omega: Iterable[float],
    broadening: float,
    volume: float = 1.0,
) -> SpectralResponse:
    """Evaluate the finite-temperature two-current Lehmann spectrum.

    The returned matrix is

        C_ab(w) = pi/(V Z) sum_mn exp(-beta E_m)
                  <m|J_a|n><n|J_b|m> delta_eta(w-E_n+E_m).

    A positive Gaussian replaces the delta function, so the matrix is
    positive semidefinite at every sampled frequency up to roundoff.
    """

    h = _as_hermitian("hamiltonian", hamiltonian)
    if len(currents) != 2:
        raise ValueError("exactly two current operators are required")
    js = [_as_hermitian(f"currents[{i}]", op) for i, op in enumerate(currents)]
    if any(op.shape != h.shape for op in js):
        raise ValueError("currents and Hamiltonian must have the same shape")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if broadening <= 0:
        raise ValueError("broadening must be positive")
    if volume <= 0:
        raise ValueError("volume must be positive")

    grid = np.asarray(tuple(omega), dtype=float)
    energies, vectors = np.linalg.eigh(h)
    transformed = [vectors.conj().T @ op @ vectors for op in js]
    weights = np.exp(-beta * (energies - energies.min()))
    weights /= weights.sum()

    response = np.zeros((grid.size, 2, 2), dtype=complex)
    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * broadening)
    for m in range(energies.size):
        for n in range(energies.size):
            transition = energies[n] - energies[m]
            kernel = normalization * np.exp(
                -0.5 * ((grid - transition) / broadening) ** 2
            )
            amplitude = np.array(
                [transformed[a][m, n] for a in range(2)], dtype=complex
            )
            gram = np.outer(amplitude, amplitude.conj())
            response += (
                np.pi * weights[m] / volume
            ) * kernel[:, None, None] * gram[None, :, :]

    # Remove only floating-point anti-Hermitian noise.
    response = 0.5 * (response + response.conj().swapaxes(-1, -2))
    return SpectralResponse(grid, response)


def conductivity_from_spectrum(beta: float, response: SpectralResponse) -> SpectralResponse:
    """Apply the same detailed-balance factor used by the original code."""

    factor = kubo_detailed_balance(beta, response.omega)
    return SpectralResponse(response.omega, factor[:, None, None] * response.matrix)


def assemble_by_polarization(
    auto_first: Array, auto_second: Array, auto_sum: Array
) -> Array:
    """Build a real symmetric 2 x 2 spectrum from three auto-spectra.

    C_12^sym = (C_(1+2)(1+2) - C_11 - C_22)/2.

    This is useful with the existing scalar FTLM routine: run it for J1, J2,
    and J1+J2 instead of implementing a separate cross-Krylov estimator.
    """

    aa = np.asarray(auto_first, dtype=float)
    bb = np.asarray(auto_second, dtype=float)
    ss = np.asarray(auto_sum, dtype=float)
    if aa.shape != bb.shape or aa.shape != ss.shape:
        raise ValueError("the three auto-spectra must have identical shapes")
    cross = 0.5 * (ss - aa - bb)
    result = np.empty(aa.shape + (2, 2), dtype=float)
    result[..., 0, 0] = aa
    result[..., 1, 1] = bb
    result[..., 0, 1] = cross
    result[..., 1, 0] = cross
    return result


def energy_weighted_current_proxy(hamiltonian: Array, particle_current: Array) -> Array:
    """Return {H,J_N}/2 as an explicitly non-physical diagnostic proxy.

    This agrees with the familiar energy-weighted velocity form in simple
    one-body settings.  It is not generally the interacting t-J energy
    current and must not be reported as a thermal-transport prediction.
    """

    h = _as_hermitian("hamiltonian", hamiltonian)
    jn = _as_hermitian("particle_current", particle_current)
    if h.shape != jn.shape:
        raise ValueError("hamiltonian and particle_current must have the same shape")
    proxy = 0.5 * (h @ jn + jn @ h)
    return 0.5 * (proxy + proxy.conj().T)


def heat_current_transform(chemical_potential: float) -> Array:
    """Map the (J_N,J_E) basis to (J_N,J_Q), J_Q=J_E-mu J_N."""

    return np.array([[1.0, 0.0], [-float(chemical_potential), 1.0]])


def congruence(response: Array, transform: Array) -> Array:
    """Apply L -> S L S^T to one matrix or a stack of matrices."""

    matrix = np.asarray(response)
    s = np.asarray(transform)
    if matrix.shape[-2:] != (2, 2) or s.shape != (2, 2):
        raise ValueError("response and transform must end in 2 x 2 matrices")
    return np.einsum("ia,...ab,jb->...ij", s, matrix, s)


def open_circuit_schur(response: Array, atol: float = 1e-14) -> Array:
    """Return L_QQ-L_QN L_NN^{-1} L_NQ for a 2 x 2 response."""

    matrix = np.asarray(response)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("response must end in a 2 x 2 matrix")
    denominator = matrix[..., 0, 0]
    if np.any(np.abs(denominator) <= atol):
        raise ZeroDivisionError("charge response is singular")
    return matrix[..., 1, 1] - matrix[..., 1, 0] * matrix[..., 0, 1] / denominator


def jordan_coordinates(response: Array) -> tuple[Array, Array, Array]:
    """Return x0,x1,x2 for [[x0+x1,x2],[x2,x0-x1]]."""

    matrix = np.asarray(response)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("response must end in a 2 x 2 matrix")
    x0 = 0.5 * (matrix[..., 0, 0] + matrix[..., 1, 1])
    x1 = 0.5 * (matrix[..., 0, 0] - matrix[..., 1, 1])
    x2 = 0.5 * (matrix[..., 0, 1] + matrix[..., 1, 0])
    return x0, x1, x2


def response_diagnostics(response: Array, atol: float = 1e-10) -> dict[str, float | bool]:
    """Report symmetry, positivity, and Lorentz-determinant diagnostics."""

    matrix = np.asarray(response)
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("response must end in a 2 x 2 matrix")
    hermitian_error = float(
        np.max(np.abs(matrix - matrix.conj().swapaxes(-1, -2)))
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().swapaxes(-1, -2)))
    determinants = np.linalg.det(matrix)
    min_eigenvalue = float(np.min(eigenvalues.real))
    min_determinant = float(np.min(determinants.real))
    return {
        "hermitian_error": hermitian_error,
        "minimum_eigenvalue": min_eigenvalue,
        "minimum_determinant": min_determinant,
        "is_hermitian": hermitian_error <= atol,
        "is_positive_semidefinite": min_eigenvalue >= -atol,
        "is_in_lorentz_cone": min_determinant >= -atol,
    }

