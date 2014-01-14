
"""Check mass, modal frequency and mode shapes against known values
for an example blade.
"""

import numpy as np
from numpy.testing import assert_allclose

from beamfe import BeamFE, ModalBeamFE


def transverse_load(N, magnitude, angle=0, only_tip=False):
    """Force in YZ plane at given angle from Y axis"""
    F = np.zeros(6 * N)
    if only_tip:
        assign_to = F[-6:]
    else:
        assign_to = F[:]
    assign_to[1::6] = magnitude * np.cos(angle)  # y force
    assign_to[2::6] = magnitude * np.sin(angle)  # z force
    return F


class UniformCantilever_Test:
    length = 3.2
    density = 54.3
    EI = 494.2

    def setup(self):
        x = np.linspace(0, self.length, 10)
        self.fe = BeamFE(x, self.density, 0, self.EI, self.EI)
        self.fe.set_boundary_conditions('C', 'F')
        self.modal = ModalBeamFE(self.fe)

    def test_mass(self):
        mass = self.length * self.density
        assert_allclose(self.fe.mass, mass, atol=0.5)

    def test_moment_of_mass(self):
        I = self.density * self.length ** 2 / 2
        m = np.dot(self.fe.S1, self.fe.q0)
        assert_allclose(m[0], I, atol=0.5)

    def test_moment_of_inertia(self):
        Iyy = self.density * self.length ** 3 / 3
        J = np.einsum('p, ijpq, q -> ij', self.fe.q0, self.fe.S2, self.fe.q0)
        assert_allclose(J[0, 0], Iyy, atol=0.5)
        assert_allclose(J[1:, 1:], 0)

    def test_deflection_with_tip_load(self):
        # Simple tip load should produce databook tip deflection
        W = 54.1  # N
        Q = transverse_load(10, W, only_tip=True)
        defl, reactions = self.fe.static_deflection(Q=Q)
        x, y, z = defl[0::6], defl[1::6], defl[2::6]

        assert_allclose(x, 0)
        assert_allclose(z, 0)
        assert_allclose(y[-1], W * self.length**3 / (3 * self.EI))

    def test_deflection_under_uniform_load(self):
        # Simple uniform load should produce databook tip deflection
        w = 34.2  # N/m
        Q = self.fe.distribute_load(transverse_load(10, magnitude=w))
        defl, reactions = self.fe.static_deflection(Q)
        x, y, z = defl[0::6], defl[1::6], defl[2::6]

        assert_allclose(x, 0)
        assert_allclose(z, 0)
        assert_allclose(y[-1], w * self.length**4 / (8 * self.EI))

    def test_deflection_is_parallel_to_uniform_loading(self):
        w = 67.4                  # distributed load (N/m)
        theta = np.radians(35.4)  # angle from y axis of load
        Q = self.fe.distribute_load(transverse_load(10, w, theta))
        defl, reactions = self.fe.static_deflection(Q)
        x, y, z = defl[0::6], defl[1::6], defl[2::6]

        assert_allclose(x, 0)
        load_angle = np.arctan2(z, y)
        assert_allclose(load_angle[1:], theta)

    def test_reaction_force(self):
        # Simple uniform load should produce databook tip deflection
        w = 34.2  # N/m
        F = transverse_load(10, w)
        # Reaction force is F1 * F
        R = np.dot(self.fe.F1, F)
        assert_allclose(R, [0, w * self.length, 0])


class UniformCantileverWithConstantTwist_Test:
    length = 3.2
    density = 54.3
    EIy = 494.2
    EIz = 654.2
    twist = np.radians(76.4)

    def setup(self):
        x = np.linspace(0, self.length, 10)
        self.fe = BeamFE(x, self.density, 0, self.EIy, self.EIz,
                         twist=self.twist)
        self.fe.set_boundary_conditions('C', 'F')
        self.modal = ModalBeamFE(self.fe)

    def test_mass(self):
        mass = self.length * self.density
        assert_allclose(self.fe.mass, mass, atol=0.5)

    def test_moment_of_mass(self):
        I = self.density * self.length ** 2 / 2
        m = np.dot(self.fe.S1, self.fe.q0)
        assert_allclose(m[0], I, atol=0.5)

    def test_moment_of_inertia(self):
        Iyy = self.density * self.length ** 3 / 3
        J = np.einsum('p, ijpq, q -> ij', self.fe.q0, self.fe.S2, self.fe.q0)
        assert_allclose(J[0, 0], Iyy, atol=0.5)
        assert_allclose(J[1:, 1:], 0)

    def test_deflection_under_uniform_load(self):
        w = 34.2  # N/m
        Q = self.fe.distribute_load(transverse_load(10, w))
        defl, reactions = self.fe.static_deflection(Q)
        print(reactions)
        x, y, z = defl[0::6], defl[1::6], defl[2::6]

        # Resolve into local blade coordinates
        tw = self.twist
        wy = +w * np.cos(tw)
        wz = -w * np.sin(tw)
        local_y = wy * self.length**4 / (8 * self.EIz)  # NB y defl -> EIzz
        local_z = wz * self.length**4 / (8 * self.EIy)

        assert_allclose(x, 0)
        assert_allclose(y[-1], local_y * np.cos(tw) - local_z * np.sin(tw))
        assert_allclose(z[-1], local_z * np.cos(tw) + local_y * np.sin(tw))

        assert_allclose(reactions[1], -w * self.length)


class UniformCantileverWithLinearTwist_Test:
    """Values from R.H. MacNeal, R.L. Harder: "Proposed standard set of
    problems to test FE accuracy"
    """
    length = 12.0
    width = 1.1
    depth = 0.32
    E = 29.0e6
    tip_twist = np.pi / 2
    num_elements = 12

    def setup(self):
        x = np.linspace(0, self.length, self.num_elements)
        twist = x * self.tip_twist / self.length
        EIy = self.E * self.width * self.depth ** 3 / 12
        EIz = self.E * self.depth * self.width ** 3 / 12
        self.fe = BeamFE(x, 0, 0, EIy, EIz, twist=twist)
        self.fe.set_boundary_conditions('C', 'F')

    def test_deflection_under_tip_load_y(self):
        Q = transverse_load(self.num_elements, magnitude=1, only_tip=True)
        defl, reactions = self.fe.static_deflection(Q=Q)
        assert_allclose(defl[-6:][1], 0.001754, atol=1e-5)

    def test_deflection_under_tip_load_z(self):
        Q = transverse_load(self.num_elements, magnitude=1,
                            angle=(np.pi / 2), only_tip=True)
        defl, reactions = self.fe.static_deflection(Q=Q)
        assert_allclose(defl[-6:][2], 0.005424, atol=1e-5)


class ExampleBlade_Test:
    def setup(self):
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

    def test_first_four_modal_frequencies(self):
        assert_allclose(self.modal.w[:4], self.answers['freqs'], atol=1e-3)

    def test_first_four_modal_shapes(self):
        assert_allclose(self.modal.shapes[:, :4], self.answers['shapes'],
                        atol=1e-5)
