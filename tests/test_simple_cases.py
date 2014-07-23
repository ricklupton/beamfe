
"""Simple examples from Reddy1993
"""

import numpy as np
from numpy import array, dot, zeros_like, linspace
from numpy.testing import assert_allclose

from beamfe import BeamFE, interleave


class Example41_Test:
    def setup(self):
        x = array([0.0, 10.0, 22.0, 28.0])
        EI = array([[2e7, 2e7],
                    [1e7, 1e7],
                    [1e7, 1e7]])
        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        self.fe = BeamFE(x, density=0, EA=0, EIy=EI, EIz=0)
        self.fe.set_boundary_conditions('P', 'C')
        self.fe.set_dofs([False, False, True, False, True, False])

    def test_stiffness(self):
        # Reddy1993, pp 161-162
        expected = 1e7 * array([
            [0.024, -0.12, -0.024, -0.12,  0,        0,        0,       0],
            [0,     0.80,  0.12,   0.40,   0,        0,        0,       0],
            [0,     0,     0.0309, 0.0783, -0.00694, -0.04167, 0,       0],
            [0,     0,     0,      1.133,  0.0417,   0.167,    0,       0],
            [0,     0,     0,      0,      0.0625,   -0.125,   -0.0556, -0.167],
            [0,     0,     0,      0,      0,        1,        0.1667,  0.333],
            [0,     0,     0,      0,      0,        0,        0.0556,  0.1667],
            [0,     0,     0,      0,      0,        0,        0,       0.6667],
        ])
        expected += expected.T - np.diag(expected.diagonal())  # make symmetric
        expected_II = expected[1:6, 1:6]
        expected_IB = expected[1:6, [0, 6, 7]]
        assert_allclose(expected_II, self.fe.K_II, atol=1e-2 * 1e7)
        assert_allclose(expected_IB, self.fe.K_IB, atol=1e-3 * 1e7)

    def test_distributed_load(self):
        # Distributed force on first element:
        load = array([0, 0, -2400, 0, 0, 0, 0, 0, -2400, 0, 0, 0])
        Q = self.fe.distribute_load_on_element(0, load)
        Q = Q[[2, 4, 8, 10, 14, 16, 20, 22]]
        assert_allclose(Q / 1e3, [-12, 20, -12, -20, 0, 0, 0, 0])

    def test_solution(self):
        # Distributed force on first element:
        load = array([0, 0, -2400, 0, 0, 0, 0, 0, -2400, 0, 0, 0])
        QF = self.fe.distribute_load_on_element(0, load)

        # Nodal forces:
        Q = np.zeros(self.fe.K.shape[0])
        Q[14] = -10000     # z force at 3rd node

        # Solve static deflection
        deflections, reactions = self.fe.static_deflection(QF + Q)

        # Expected values from Reddy
        expected_deflections = zeros_like(deflections)
        expected_deflections[[4, 8, 10, 14, 16]] = [0.03856, -0.2808, 0.01214,
                                                    -0.1103, -0.02752]
        assert_allclose(deflections, expected_deflections, atol=1e-4)
        expected_reactions = zeros_like(reactions)
        expected_reactions[[2, 20, 22]] = [18565.54, 15434.46, 92164.83]
        assert_allclose(reactions, expected_reactions, atol=1e-2)

    def test_load_matrix_gives_same_result_as_method(self):
        # Distributed force on all elements
        load = np.zeros(4 * 6)
        load[2::6] = [350, 324, 654, 54]  # Z component

        Q1 = self.fe.distribute_load(load)
        Q2 = dot(self.fe.F, load)

        assert_allclose(Q1, Q2)


class Example42_Test:
    def setup(self):
        x = array([0.0, 4.0, 10.0])
        EI = 144.0
        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        self.fe = BeamFE(x, density=0, EA=0, EIy=EI, EIz=0)
        self.fe.set_boundary_conditions('C', 'F')
        self.fe.set_dofs([False, False, True, False, True, False])

    def test_stiffness(self):
        # Reddy1993, p 164
        expected = array([
            [27,  0,   0,     0,      0,  0],
            [-54, 144, 0,     0,      0,  0],
            [-27, 54,  27+8,  0,      0,  0],
            [-54, 72,  54-24, 144+96, 0,  0],
            [0,   0,   -8,    24,     8,  0],
            [0,   0,   -24,   48,     24, 96],
        ])
        expected += expected.T - np.diag(expected.diagonal())  # make symmetric
        expected_II = expected[2:6, 2:6]
        expected_IB = expected[2:6, 0:2]
        assert_allclose(self.fe.K_II, expected_II)
        assert_allclose(self.fe.K_IB, expected_IB)

    def test_distributed_load(self):
        # Distributed force on second element:
        load = array([0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0])
        Q = self.fe.distribute_load_on_element(1, load)
        Q = Q[[2, 4, 8, 10, 14, 16]]
        assert_allclose(Q, [0, 0, -90, 120, -210, -180])

    def test_solution_without_spring(self):
        # Distributed force on first element:
        load = array([0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0])
        QF = self.fe.distribute_load_on_element(1, load)

        # Solve static deflection
        deflections, reactions = self.fe.static_deflection(QF)

        # Expected values from Reddy
        expected_deflections = zeros_like(deflections)
        expected_deflections[[8, 10, 14, 16]] = array(
            [-1.6, 0.72, -7.108, 0.99]) * 1e4 / 143.0
        assert_allclose(deflections, expected_deflections, atol=1e1)


# Eigenvalue problem
class Example62_Test:
    def setup(self):
        x = linspace(0, 1, 16)
        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        self.fe = BeamFE(x, density=1, EA=0, EIy=1, EIz=0)
        self.fe.set_boundary_conditions('C', 'F')
        self.fe.set_dofs([False, False, True, False, True, False])

    def test_mode_frequencies(self):
        modal = self.fe.modal_matrices()
        assert_allclose(modal.w[:4],
                        [3.5160, 22.0345, 61.6972, 120.9019],
                        atol=1e-1)
