import unittest
import numpy as np
from numpy import array
from beamfe import BeamFE


class TestBeamFEAttachmentModes(unittest.TestCase):
    def setUp(self):
        x = array([0.0, 10.0, 22.0, 28.0])
        EI = array([[2e7, 2e7],
                    [1e7, 1e7],
                    [1e7, 1e7]])
        # Using the z axis as the transverse direction gives the same
        # sign convention as Reddy uses in 2D, namely that rotations
        # are positive clockwise.
        self.fe = BeamFE(x, density=0, EA=0, EIy=EI, EIz=0)
        self.fe.set_boundary_conditions('C', 'C')
        self.fe.set_dofs([False, False, True, False, True, False])

    def test_simple_model(self):
        am = self.fe.attachment_modes()
        self.assertEqual(am.shape, (4*6, 2*2))
        # TODO: better test


if __name__ == '__main__':
    unittest.main()
