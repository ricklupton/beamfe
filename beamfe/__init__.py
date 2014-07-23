import numpy as np
from numpy import array, ix_, zeros, dot, eye, cos, sin
from numpy.lib.stride_tricks import as_strided
from scipy import linalg
from . import tapered_beam_element_integrals
from . import tapered_beam_element_integrals as integrals


def boundary_condition_vector(condition):
    types = {
        'C': [True, True, True, True, True, True],
        'P': [True, True, True, False, False, False],
        'F': [False, False, False, False, False, False],
    }
    try:
        return types[condition]
    except KeyError as e:
        raise ValueError("Unknown boundary condition: '{}'".format(condition))


def _prepare_inputs(n, input):
    # If only one set of density values is given, assume it's continuous.
    # Otherwise, values should be given at start and end of each section.
    input = np.asarray(input)
    if input.ndim == 0:
        return input * np.ones((n - 1, 2))
    elif input.ndim == 1:
        assert input.shape[0] == n
        yy = np.zeros((n - 1, 2))
        yy[:, 0] = input[:-1]
        yy[:, 1] = input[1:]
        return yy
    elif input.ndim == 2:
        assert input.shape[0] == n - 1
        return input
    else:
        raise ValueError('input should be Nx x 1 or Nx x 2')


def interleave(x, n=None):
    """Flattens x into a sequence of DOFs, where each location has `n`
    DOFs. If `n` is not given, assume it from the number of colums in
    `x`"""
    if x.ndim < 2:
        x = np.atleast_2d(x).T
    assert x.ndim == 2
    if n is None:
        n = x.shape[1]
    y = np.zeros(x.shape[0] * n, x.dtype)
    for i in range(x.shape[1]):
        y[i::n] = x[:, i]
    return y


class BeamFE(object):
    def __init__(self, x, density, EA, EIy, EIz, GJ=0, twist=0,
                 axial_force=None):
        assert x[0] == 0
        N_nodes = len(x)
        N_dof = N_nodes * 6

        density = _prepare_inputs(N_nodes, density)
        EA = _prepare_inputs(N_nodes, EA)
        GJ = _prepare_inputs(N_nodes, GJ)
        EIy = _prepare_inputs(N_nodes, EIy)
        EIz = _prepare_inputs(N_nodes, EIz)
        twist = _prepare_inputs(N_nodes, twist)

        if axial_force is None:
            axial_force = np.zeros(N_nodes)
        else:
            axial_force = np.asarray(axial_force)
            assert axial_force.shape == (N_nodes,)

        # Set undeformed nodal coordinates - along x axis
        self.q0 = np.zeros(N_dof)
        self.q0[0::6] = x

        # Assemble shape integrals and stiffness & force matrices
        self.mass = 0.0
        self.S1 = np.zeros((3, N_dof))
        self.S2 = np.zeros((3, 3, N_dof, N_dof))
        self.F1 = np.zeros((3, N_dof))
        self.F2 = np.zeros((3, 3, N_dof, N_dof))
        self.K = np.zeros((N_dof, N_dof))
        self.Ks = np.zeros((N_dof, N_dof))

        # Only average EA and GJ matters; assume average twist is ok
        avgEA = EA.mean(axis=1)
        avgGJ = GJ.mean(axis=1)
        avgTw = twist.mean(axis=1)

        for i_el in range(N_nodes - 1):
            elem_length = x[i_el+1] - x[i_el]
            ke = integrals.K(
                elem_length,
                avgEA[i_el], avgGJ[i_el],
                EIy[i_el, 0], EIy[i_el, 1],
                EIz[i_el, 0], EIz[i_el, 1]
            )
            ks = (axial_force[i_el] + axial_force[i_el + 1]) / 2 * integrals.Ks(elem_length)

            r1, r2 = density[i_el, :]
            m = integrals.mass(elem_length, r1, r2)
            S1 = integrals.S1(elem_length, r1, r2)
            S2 = integrals.S2(elem_length, r1, r2)
            F1 = integrals.F1(elem_length)
            F2 = integrals.F2(elem_length)

            # The angle is the rotation of the element coordinates
            # about the X axis. So CX transforms from local element
            # coordinates back to body coordinates, and is therefore
            # an X-rotation through the negative of the angle.
            w = avgTw[i_el]
            CX = array([[1, 0, 0], [0, cos(w), -sin(w)], [0, sin(w), cos(w)]])
            Cq = linalg.block_diag(CX, CX, CX, CX).T

            i1, i2 = i_el * 6, (i_el + 2) * 6
            self.K[i1:i2, i1:i2] += dot(Cq.T, dot(ke, Cq))
            self.Ks[i1:i2, i1:i2] += dot(Cq.T, dot(ks, Cq))

            self.mass += m
            self.S1[:, i1:i2] += dot(CX, dot(S1, Cq))
            self.F1[:, i1:i2] += F1
            for i in range(3):
                for j in range(3):
                    self.S2[i, j, i1:i2, i1:i2] += dot(Cq.T, dot(S2[i][j], Cq))
                    self.F2[i, j, i1:i2, i1:i2] += F2[i][j]

        self.M = np.trace(self.S2)
        self.F = np.trace(self.F2)

        # Boundary conditions: clamped-clamped by default
        self.Bbound = np.zeros(self.K.shape[0], dtype=bool)
        self.set_boundary_conditions('C', 'C')

        # Included DOFs
        self.Bdof = np.ones(self.K.shape[0], dtype=bool)
        self.set_dofs([False, True, True, False, True, True])

    def set_boundary_conditions(self, left=None, right=None):
        """left and right are one of F, C or P:
         - F = free
         - C = clamped
         - P = pinned
        """
        if left is not None:
            self.Bbound[:6] = boundary_condition_vector(left)
        if right is not None:
            self.Bbound[-6:] = boundary_condition_vector(right)

    def set_dofs(self, dofs):
        if len(dofs) != 6:
            raise ValueError("dofs should be 6-long list of bools")
        assert dofs[0] is False and dofs[3] is False        # XXX
        for i, dof_enabled in enumerate(dofs):
            self.Bdof[i::6] = dof_enabled

    @property
    def K_II(self):
        # Pick out interior (I) and boundary (B) coordinates
        I = self.Bdof & ~self.Bbound
        return (self.K + self.Ks)[I, :][:, I]

    @property
    def K_IB(self):
        # Pick out interior (I) and boundary (B) coordinates
        I = self.Bdof & ~self.Bbound
        B = self.Bdof & self.Bbound
        return (self.K + self.Ks)[I, :][:, B]

    @property
    def K_BB(self):
        # Pick out interior (I) and boundary (B) coordinates
        B = self.Bdof & self.Bbound
        return (self.K + self.Ks)[B, :][:, B]

    @property
    def S1B(self):
        """S B2, i.e. with boundary dofs removed"""
        I = self.Bdof & ~self.Bbound
        return self.S1[:, I]

    @property
    def S2B(self):
        """S2 B2, i.e. with boundary dofs removed"""
        I = self.Bdof & ~self.Bbound
        return self.S2[:, I]

    def normal_modes(self, n_modes=None):
        """
        Calculate the normal mode shapes and frequencies, limited to the first
        ``n_modes`` modes if required.
        """

        # Subset of modes to calculate
        if n_modes is None:
            eigvals = None
        elif n_modes == 0:
            return np.empty((0,)), np.empty((len(self.M), 0))
        else:
            assert n_modes >= 1
            eigvals = (0, n_modes - 1)

        # Select interior nodes and find eigenvalues; note eigh assumes symmetry
        I = self.Bdof & ~self.Bbound
        w, v = linalg.eigh(self.K_II, self.M[I, :][:, I], eigvals=eigvals)
        w = np.sqrt(w.real)

        # Reintroduce missing DOFs as zeros to keep shape consistent
        vall = np.zeros((len(I), v.shape[1]))
        vall[I, :] = v

        return w, vall

    def attachment_modes(self):
        """
        Calculate the mode shapes with unit deflection at the ends of the beam
        """
        # Pick out interior (I) and boundary (B) coordinates
        I = self.Bdof & ~self.Bbound
        B = self.Bdof & self.Bbound

        # Calculate attachment modes
        attach_modes = -dot(linalg.inv(self.K[I, :][:, I]), self.K[I, :][:, B])
        num_boundary_dofs = sum(self.Bbound)
        Xi = zeros((len(I), num_boundary_dofs))
        Xi[ix_(I, self.Bdof[:num_boundary_dofs])] = attach_modes
        Xi[ix_(B, self.Bdof[:num_boundary_dofs])] = eye(num_boundary_dofs)
        return Xi

    def distribute_load_on_element(self, ielem, load):
        # generalised forces corresponding to applied distributed force
        elem_length = self.q0[6*(ielem+1)] - self.q0[6*ielem]
        F2 = integrals.F2(elem_length)
        Fmat = np.trace(F2)
        QF = zeros(self.F.shape[0], dtype=load.dtype)
        QF[6*ielem:6*(ielem+2)] = dot(Fmat, load)
        return QF

    def distribute_load(self, load):
        """Return nodal forces for linearly varying distributed load"""
        Q = np.zeros_like(self.q0, dtype=load.dtype)
        for i in range(len(Q) // 6 - 1):
            fi = load[(6 * i):(6 * (i + 2))]
            Q += self.distribute_load_on_element(i, fi)
        return Q

    def static_deflection(self, Q=None):
        """Calculate static deflection under given distributed load `F` and
        nodal loads `Q`.
        """
        # reduced stiffness matrix, excluding clamped ends and axial/torsion
        I = self.Bdof & ~self.Bbound
        B = self.Bdof & self.Bbound

        # Applied nodal forces
        if Q is None:
            Q = zeros(self.F.shape[0])
        assert Q.shape == (self.F.shape[0], )

        # solve for deflection, using only the free DOFs
        # (assume prescribed DOFs are 0, so they don't appear here)
        x = zeros(len(self.K))
        x[I] = linalg.solve(self.K_II, Q[I])

        # solve for boundary forces
        P = zeros(len(self.K))
        P[B] = -Q[B] + dot(self.K_IB.T, x[I])
        return x, P

    def modal_matrices(self, n_modes=None):
        return ModalBeamFE(self, n_modes)

    @staticmethod
    def centrifugal_force_distribution(qn, density):
        """Calculates the distribution of axial forces at radii `r` when the
        beam is rotating about an axis perpendicular to the beam
        axis. Multiply by omega**2 to get the actual force.
        """
        N = len(qn) // 6
        density = _prepare_inputs(N, density)
        elem_lengths = np.diff(qn[0::6])
        qn_chunks = as_strided(qn, shape=(len(qn) // 6 - 1, 12),
                               strides=(6*qn.itemsize, qn.itemsize))
        F = array([
            dot(integrals.S1(l, rho[0], rho[1]), qn_el)[0]
            for l, rho, qn_el in zip(elem_lengths, density, qn_chunks)
        ])
        return np.r_[np.cumsum(F[::-1])[::-1], 0]


class ModalBeamFE:
    def __init__(self, fe, n_modes=None):

        """Calculate normal mode shapes, limited to the first ``n_modes``
        modes if required.

        """

        self.fe = fe
        w, shapes = fe.normal_modes(n_modes)
        self.w = w
        self.shapes = shapes
        self.damping = 0 * w

        # Calculate projected matrices
        self.M = dot(shapes.T, dot(fe.M, shapes))
        self.K = dot(shapes.T, dot(fe.K + fe.Ks, shapes))
        self.S1 = dot(fe.S1, shapes)
        self.F1 = fe.F1
        self.S2 = np.einsum('ap, ijab, bq -> ijpq', shapes, fe.S2, shapes)
        self.F2 = np.einsum('ap, ijab     -> ijpb', shapes, fe.F2)
