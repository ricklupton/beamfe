
"""Check mass, modal frequency and mode shapes against known values
for an example blade.
"""

import unittest
import numpy as np
from numpy import dot
from numpy.testing import assert_allclose

from beamfe import BeamFE, ModalBeamFE


def assert_allclose_ignoring_sign(x, y, **kwargs):
    jmax = np.argmax(abs(x))
    if x[jmax] * y[jmax] < 0:
        x = -x
    assert_allclose(x, y, **kwargs)


class ExampleBlade_Test(unittest.TestCase):
    def setUp(self):
        data = np.loadtxt("tests/example_blade_data.txt")
        x, density, EA, EI_flap, EI_edge, twist = data.T
        # Twist has been saved in my convention (+ve X rotation), not Bladed
        self.fe = BeamFE(x, density, EA, EI_flap, EI_edge, twist=twist)
        self.fe.set_boundary_conditions('C', 'F')
        self.modal = ModalBeamFE(self.fe)
        self.answers = np.load("tests/example_blade_mode_results.npz")

    def test_mass(self):
        assert_allclose(self.fe.mass, 6547, atol=0.5)

    def test_moment_of_mass(self):
        m = np.dot(self.fe.S1, self.fe.q0)
        assert_allclose(m[0], 84219, atol=0.5)

    def test_moment_of_inertia(self):
        J = np.einsum('p, ijpq, q -> ij', self.fe.q0, self.fe.S2, self.fe.q0)
        assert_allclose(J[0, 0], 1784881, atol=0.5)

    def test_first_four_modal_frequencies_match_saved_values(self):
        assert_allclose(self.modal.w[:4], self.answers['freqs'], atol=1e-3)

    def test_first_four_mode_shapes_match_saved_values(self):
        v = self.modal.shapes
        answer = self.answers['shapes']
        for i in range(4):
            assert_allclose_ignoring_sign(v[:, i], answer[:, i], atol=1e-5)

    def test_mode_shapes_are_normalised(self):
        w = self.modal.w
        v = self.modal.shapes
        M = self.fe.M
        K = self.fe.K
        for i in range(v.shape[1]):
            assert_allclose(dot(v[:, i].T, dot(M, v[:, i])), 1.0)
            assert_allclose(dot(v[:, i].T, dot(K, v[:, i])), w[i]**2)

    def test_mode_are_sorted(self):
        w = list(self.modal.w)
        self.assertEqual(w, sorted(w))

    def test_correct_number_of_modes_are_returned(self):
        w_all, v_all = self.fe.normal_modes()
        w, v = self.fe.normal_modes(4)
        assert_allclose(w, w_all[:4])
        for i in range(4):
            assert_allclose_ignoring_sign(v[:, i], v_all[:, i])
