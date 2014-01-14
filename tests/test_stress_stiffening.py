
"""Check centrifugal stiffening
"""

import numpy as np
from numpy.testing import assert_allclose

from beamfe import BeamFE, ModalBeamFE


class StressStiffening_Tests:
    length = 5.4
    density = 54.3
    EI = 494.2

    def test_uniform_tension_increases_stiffness(self):
        x = np.linspace(0, self.length, 10)
        fe1 = BeamFE(x, self.density, 0, self.EI, self.EI)
        fe2 = BeamFE(x, self.density, 0, self.EI, self.EI,
                     axial_force=20 * np.ones(10))
        assert np.all(np.diag(fe2.Ks) >= np.diag(fe1.Ks))
