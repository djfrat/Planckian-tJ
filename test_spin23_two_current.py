import unittest

import numpy as np

from spin23_two_current import (
    assemble_by_polarization,
    conductivity_from_spectrum,
    congruence,
    energy_weighted_current_proxy,
    gaussian_spectral_matrix,
    heat_current_transform,
    jordan_coordinates,
    kubo_detailed_balance,
    open_circuit_schur,
    response_diagnostics,
)


class TwoCurrentResponseTests(unittest.TestCase):
    def setUp(self):
        self.h = np.array(
            [[0.0, 0.2, 0.0], [0.2, 0.8, 0.15], [0.0, 0.15, 1.7]]
        )
        self.jn = np.array(
            [[0.0, 1.0j, 0.2j], [-1.0j, 0.0, 0.4j], [-0.2j, -0.4j, 0.0]]
        )
        self.je = energy_weighted_current_proxy(self.h, self.jn)

    def test_detailed_balance_zero_limit(self):
        values = kubo_detailed_balance(2.5, np.array([0.0, 1e-12, 0.4]))
        self.assertAlmostEqual(values[0], 2.5)
        self.assertAlmostEqual(values[1], 2.5, places=10)
        self.assertGreater(values[2], 0.0)

    def test_lehmann_matrix_is_positive(self):
        omega = np.linspace(-0.5, 2.5, 151)
        spectrum = gaussian_spectral_matrix(
            self.h, [self.jn, self.je], beta=1.7, omega=omega,
            broadening=0.08, volume=3.0
        )
        diagnostics = response_diagnostics(spectrum.matrix)
        self.assertTrue(diagnostics["is_hermitian"])
        self.assertTrue(diagnostics["is_positive_semidefinite"])
        self.assertTrue(diagnostics["is_in_lorentz_cone"])

        conductivity = conductivity_from_spectrum(1.7, spectrum)
        self.assertTrue(
            response_diagnostics(conductivity.matrix)["is_positive_semidefinite"]
        )

    def test_heat_basis_is_congruence(self):
        matrix = np.array([[3.0, 0.8], [0.8, 2.0]])
        transform = heat_current_transform(0.35)
        changed = congruence(matrix, transform)
        self.assertTrue(response_diagnostics(changed)["is_positive_semidefinite"])
        self.assertAlmostEqual(
            np.linalg.det(changed), np.linalg.det(transform) ** 2 * np.linalg.det(matrix)
        )

    def test_open_circuit_is_schur_complement(self):
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        self.assertAlmostEqual(open_circuit_schur(matrix), 2.75)

    def test_polarization_identity(self):
        aa = np.array([2.0, 3.0])
        bb = np.array([5.0, 7.0])
        cross = np.array([0.5, -0.25])
        total = aa + bb + 2.0 * cross
        matrix = assemble_by_polarization(aa, bb, total)
        np.testing.assert_allclose(matrix[:, 0, 1], cross)
        np.testing.assert_allclose(matrix[:, 1, 0], cross)

    def test_jordan_determinant_is_lorentz_norm(self):
        matrix = np.array([[4.0, 1.25], [1.25, 2.0]])
        x0, x1, x2 = jordan_coordinates(matrix)
        self.assertAlmostEqual(np.linalg.det(matrix), x0 * x0 - x1 * x1 - x2 * x2)


if __name__ == "__main__":
    unittest.main()

