
"""Test applied force matrices
"""

import numpy as np
from numpy import array, dot, zeros, linspace
from numpy.testing import assert_allclose

from beamfe import BeamFE


class AppliedLoads_Tests:
    def setup(self):
        self.fe = BeamFE(linspace(0, 10, 11), density=0, EA=0, EIy=1, EIz=1)

    def test_uniform_load(self):
        f = array([3.4, 5.6, 7.2])
        load = zeros(6 * 11)
        for i in range(3): load[i::6] = f[i]

        # Force
        F = dot(self.fe.F1, load)
        assert_allclose(F, f * 10)

        # Moment
        q = self.fe.q0
        I = np.vstack((
            dot(q.T, (self.fe.F2[1, 2] - self.fe.F2[2, 1])),
            dot(q.T, (self.fe.F2[2, 0] - self.fe.F2[0, 2])),
            dot(q.T, (self.fe.F2[0, 1] - self.fe.F2[1, 0])),
        ))
        Q = dot(I, load)
        assert_allclose(Q, [0, -f[2]*10*5, f[1]*10*5])

    def test_linear_load(self):
        f = array([3.4, 5.6, 7.2])
        load = zeros(6 * 11)
        for i in range(3): load[i::6] = linspace(0, f[i], 11)

        # Force
        F = dot(self.fe.F1, load)
        assert_allclose(F, f * 10 / 2)

        # Moment
        q = self.fe.q0
        I = np.vstack((
            dot(q.T, (self.fe.F2[1, 2] - self.fe.F2[2, 1])),
            dot(q.T, (self.fe.F2[2, 0] - self.fe.F2[0, 2])),
            dot(q.T, (self.fe.F2[0, 1] - self.fe.F2[1, 0])),
        ))
        Q = dot(I, load)
        assert_allclose(Q, [0, -f[2]*10/2 * 10*2/3, f[1]*10/2 * 10*2/3])

        # Check stresses
        assert_allclose(dot(self.fe.F, load), self.fe.distribute_load(load))
