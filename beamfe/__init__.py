import numpy as np
from numpy import array, ix_, zeros, dot, eye, cos, sin
from scipy import linalg


class BeamFE(object):
    def __init__(self, x, density, EA, EIy, EIz, GJ=0, twist=0):
        assert x[0] == 0
        self.x = x
        self.density = self._prepare_inputs(len(x), density)
        self.EA = self._prepare_inputs(len(x), EA)
        self.GJ = self._prepare_inputs(len(x), GJ)
        self.EIy = self._prepare_inputs(len(x), EIy)
        self.EIz = self._prepare_inputs(len(x), EIz)
        self.twist = self._prepare_inputs(len(x), twist)

        # Assemble matrices
        N_nodes = len(x)
        N_dof = N_nodes * 6
        self.M = np.zeros((N_dof, N_dof))
        self.K = np.zeros((N_dof, N_dof))
        self.F = np.zeros((N_dof, N_dof))

        # Only average EA and GJ matters; assume average twist is ok
        avgEA = self.EA.mean(axis=1)
        avgGJ = self.GJ.mean(axis=1)
        avgTw = self.twist.mean(axis=1)

        for i_el in range(N_nodes - 1):
            elem_length = x[i_el+1] - x[i_el]
            ke = self.element_stiffness_matrix(
                avgEA[i_el], avgGJ[i_el],
                self.EIy[i_el, 0], self.EIy[i_el, 1],
                self.EIz[i_el, 0], self.EIz[i_el, 1],
                elem_length
            )
            me = self.element_mass_matrix(
                self.density[i_el, 0], self.density[i_el, 1],
                elem_length
            )
            fe = self.element_distributed_force_matrix(elem_length)

            # Transform from local element to global coordinates (twist)
            w = avgTw[i_el]
            rot = array([[1, 0, 0], [0, cos(w), sin(w)], [0, -sin(w), cos(w)]])
            T = linalg.block_diag(rot, rot, rot, rot)

            i1, i2 = i_el * 6, (i_el + 2) * 6
            self.M[i1:i2, i1:i2] += dot(T.T, dot(me, T))
            self.K[i1:i2, i1:i2] += dot(T.T, dot(ke, T))
            self.F[i1:i2, i1:i2] += dot(T.T, dot(fe, T))

    def _prepare_inputs(self, n, input):
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

    def element_mass_matrix(self, rho_1, rho_2, l):
        """
        Finite element mass matrix, assuming cubic shape functions and
        linear density variation from ``r1`` to ``r2``.
        """
        e = array([[l*(rho_1/4 + rho_2/12), 0, 0, 0, 0, 0, l*(rho_1/12 + rho_2/12), 0, 0, 0, 0, 0], [0, l*(2*rho_1/7 + 3*rho_2/35), 0, 0, 0, l**2*(15*rho_1 + 7*rho_2)/420, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, 0, 0, l**2*(-7*rho_1 - 6*rho_2)/420], [0, 0, l*(2*rho_1/7 + 3*rho_2/35), 0, l**2*(-15*rho_1 - 7*rho_2)/420, 0, 0, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, l**2*(7*rho_1 + 6*rho_2)/420, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, l**2*(-15*rho_1 - 7*rho_2)/420, 0, l**3*(rho_1/168 + rho_2/280), 0, 0, 0, l**2*(-6*rho_1 - 7*rho_2)/420, 0, l**3*(-rho_1 - rho_2)/280, 0], [0, l**2*(15*rho_1 + 7*rho_2)/420, 0, 0, 0, l**3*(rho_1/168 + rho_2/280), 0, l**2*(6*rho_1 + 7*rho_2)/420, 0, 0, 0, l**3*(-rho_1 - rho_2)/280], [l*(rho_1/12 + rho_2/12), 0, 0, 0, 0, 0, l*(rho_1/12 + rho_2/4), 0, 0, 0, 0, 0], [0, l*(9*rho_1/140 + 9*rho_2/140), 0, 0, 0, l**2*(6*rho_1 + 7*rho_2)/420, 0, l*(3*rho_1/35 + 2*rho_2/7), 0, 0, 0, l**2*(-7*rho_1 - 15*rho_2)/420], [0, 0, l*(9*rho_1/140 + 9*rho_2/140), 0, l**2*(-6*rho_1 - 7*rho_2)/420, 0, 0, 0, l*(3*rho_1/35 + 2*rho_2/7), 0, l**2*(7*rho_1 + 15*rho_2)/420, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, l**2*(7*rho_1 + 6*rho_2)/420, 0, l**3*(-rho_1 - rho_2)/280, 0, 0, 0, l**2*(7*rho_1 + 15*rho_2)/420, 0, l**3*(rho_1/280 + rho_2/168), 0], [0, l**2*(-7*rho_1 - 6*rho_2)/420, 0, 0, 0, l**3*(-rho_1 - rho_2)/280, 0, l**2*(-7*rho_1 - 15*rho_2)/420, 0, 0, 0, l**3*(rho_1/280 + rho_2/168)]])
        # e = array([
        #     [r1/4 + r2/12, 0, 0, 0, 0, 0,
        #      r1/12 + r2/12, 0, 0, 0, 0, 0],
        #     [0, 2*r1/7 + 3*r2/35, 0, 0, 0, l*(15*r1 + 7*r2)/420,
        #      0, 9*r1/140 + 9*r2/140, 0, 0, 0, l*(-7*r1 - 6*r2)/420],
        #     [0, 0, 2*r1/7 + 3*r2/35, 0, l*(-15*r1 - 7*r2)/420, 0,
        #      0, 0, 9*r1/140 + 9*r2/140, 0, l*(7*r1 + 6*r2)/420, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, l*(-15*r1 - 7*r2)/420, 0, l**2*(r1/168 + r2/280), 0,
        #      0, 0, l*(-6*r1 - 7*r2)/420, 0, l**2*(-r1 - r2)/280, 0],
        #     [0, l*(15*r1 + 7*r2)/420, 0, 0, 0, l**2*(r1/168 + r2/280),
        #      0, l*(6*r1 + 7*r2)/420, 0, 0, 0, l**2*(-r1 - r2)/280],
        #     [r1/12 + r2/12, 0, 0, 0, 0, 0,
        #      r1/12 + r2/4, 0, 0, 0, 0, 0],
        #     [0, 9*r1/140 + 9*r2/140, 0, 0, 0, l*(6*r1 + 7*r2)/420,
        #      0, 3*r1/35 + 2*r2/7, 0, 0, 0, l*(-7*r1 - 15*r2)/420],
        #     [0, 0, 9*r1/140 + 9*r2/140, 0, l*(-6*r1 - 7*r2)/420, 0,
        #      0, 0, 3*r1/35 + 2*r2/7, 0, l*(7*r1 + 15*r2)/420, 0],
        #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     [0, 0, l*(7*r1 + 6*r2)/420, 0, l**2*(-r1 - r2)/280, 0,
        #      0, 0, l*(7*r1 + 15*r2)/420, 0, l**2*(r1/280 + r2/168), 0],
        #     [0, l*(-7*r1 - 6*r2)/420, 0, 0, 0, l**2*(-r1 - r2)/280, 0,
        #      l*(-7*r1 - 15*r2)/420, 0, 0, 0, l**2*(r1/280 + r2/168)]])
        return e

    def element_stiffness_matrix(self, EA, GJ, EIy_1, EIy_2, EIz_1, EIz_2, l):
        """
        Finite element stiffness matrix, assuming cubic shape functions and
        linear stiffness variation.
        """
        e = array([
            [EA/l, 0, 0, 0, 0, 0, -EA/l, 0, 0, 0, 0, 0],
            [0, 6*(EIz_1 + EIz_2)/l**3, 0, 0, 0, 2*(2*EIz_1 + EIz_2)/l**2, 0, -6*(EIz_1 + EIz_2)/l**3, 0, 0, 0, 2*(EIz_1 + 2*EIz_2)/l**2],
            [0, 0, 6*(EIy_1 + EIy_2)/l**3, 0, -2*(2*EIy_1 + EIy_2)/l**2, 0, 0, 0, -6*(EIy_1 + EIy_2)/l**3, 0, -2*(EIy_1 + 2*EIy_2)/l**2, 0],
            [0, 0, 0, GJ/l, 0, 0, 0, 0, 0, -GJ/l, 0, 0],
            [0, 0, -2*(2*EIy_1 + EIy_2)/l**2, 0, (3*EIy_1 + EIy_2)/l, 0, 0, 0, 2*(2*EIy_1 + EIy_2)/l**2, 0, (EIy_1 + EIy_2)/l, 0],
            [0, 2*(2*EIz_1 + EIz_2)/l**2, 0, 0, 0, (3*EIz_1 + EIz_2)/l, 0, -2*(2*EIz_1 + EIz_2)/l**2, 0, 0, 0, (EIz_1 + EIz_2)/l],
            [-EA/l, 0, 0, 0, 0, 0, EA/l, 0, 0, 0, 0, 0],
            [0, -6*(EIz_1 + EIz_2)/l**3, 0, 0, 0, -2*(2*EIz_1 + EIz_2)/l**2, 0, 6*(EIz_1 + EIz_2)/l**3, 0, 0, 0, -2*(EIz_1 + 2*EIz_2)/l**2],
            [0, 0, -6*(EIy_1 + EIy_2)/l**3, 0, 2*(2*EIy_1 + EIy_2)/l**2, 0, 0, 0, 6*(EIy_1 + EIy_2)/l**3, 0, 2*(EIy_1 + 2*EIy_2)/l**2, 0],
            [0, 0, 0, -GJ/l, 0, 0, 0, 0, 0, GJ/l, 0, 0],
            [0, 0, -2*(EIy_1 + 2*EIy_2)/l**2, 0, (EIy_1 + EIy_2)/l, 0, 0, 0, 2*(EIy_1 + 2*EIy_2)/l**2, 0, (EIy_1 + 3*EIy_2)/l, 0],
            [0, 2*(EIz_1 + 2*EIz_2)/l**2, 0, 0, 0, (EIz_1 + EIz_2)/l, 0, -2*(EIz_1 + 2*EIz_2)/l**2, 0, 0, 0, (EIz_1 + 3*EIz_2)/l]
        ])
        return e

    def element_distributed_force_matrix(self, l):
        """
        Distributed force matrix F, so generalised forces are
        F*[fx1, fy1, fz1, fx2, fy2, fz2]
        """
        F = zeros((12, 12))
        F[:, 0:3] = [
            [20*l, 0, 0],
            [0, 21*l, 0],
            [0, 0, 21*l],
            [0, 0, 0],
            [0, 0, -3*l**2],
            [0, 3*l**2, 0],
            [10*l, 0, 0],
            [0, 9*l, 0],
            [0, 0, 9*l],
            [0, 0, 0],
            [0, 0, 2*l**2],
            [0, -2*l**2, 0]
        ]
        F[:, 6:9] = [
            [10*l, 0, 0],
            [0, 9*l, 0],
            [0, 0, 9*l],
            [0, 0, 0],
            [0, 0, -2*l**2],
            [0, 2*l**2, 0],
            [20*l, 0, 0],
            [0, 21*l, 0],
            [0, 0, 21*l],
            [0, 0, 0],
            [0, 0, 3*l**2],
            [0, -3*l**2, 0]
        ]
        return F / 60

    def normal_modes(self, n_modes=None, clamped_left=True, clamped_right=True,
                     exclude_axial=True, exclude_torsion=True):
        """
        Calculate the normal mode shapes and frequencies, limited to the first
        ``n_modes`` modes if required.
        """
        assert exclude_axial and exclude_torsion

        # remove axial/torsion
        idx = [i for i in range(len(self.K)) if (i % 6) not in (0, 3)]

        # take away ends if clamped
        if clamped_left:
            idx = idx[4:]
        if clamped_right:
            idx = idx[:-4]

        w, v = linalg.eig(self.K[ix_(idx, idx)], self.M[ix_(idx, idx)])
        order = np.argsort(w)
        w = np.sqrt(w[order].real)
        v = v[:, order]

        # put back axial/torsion as zeros
        shapes = zeros((len(self.K), v.shape[1]))
        shapes[idx, :] = v

        if n_modes is not None:
            w = w[:n_modes]
            shapes = shapes[:, :n_modes]
        return w, shapes

    def attachment_modes(self, exclude_axial=True, exclude_torsion=True):
        """
        Calculate the mode shapes with unit deflection at the ends of the beam
        """
        assert exclude_axial and exclude_torsion

        N = self.K.shape[0]

        # remove axial/torsion
        idx_ok = array([(i % 6) != 3 for i in range(N)])
        #idx_ok = np.ones(N, dtype=bool)
        idx_B = zeros(N, dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        K_II = self.K[idx_ok & ~idx_B, :][:, idx_ok & ~idx_B]
        K_IB = self.K[idx_ok & ~idx_B, :][:, idx_ok & idx_B]

        # Attachment modes
        Xi = zeros((N, 12))
        Xi[ix_(idx_ok & ~idx_B, idx_ok[:12])] = -dot(linalg.inv(K_II), K_IB)
        Xi[ix_(idx_ok & idx_B,  idx_ok[:12])] = eye(sum(idx_ok[:12]))
        return Xi

    @property
    def total_mass(self):
        # There must be a better way of doing this? XXX
        test = np.zeros(self.M.shape[0])
        test[0::6] = 1
        mass = dot(test.T, dot(self.M, test))
        return mass

    @property
    def K_BI(self):
        idx_B = zeros(self.K.shape[0], dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        return self.K[idx_B, :][:, ~idx_B]

    @property
    def M_BI(self):
        idx_B = zeros(self.K.shape[0], dtype=bool)
        idx_B[:6] = idx_B[-6:] = True
        return self.M[idx_B, :][:, ~idx_B]

    def static_deflection(self, f, clamped_left=True, clamped_right=True,
                          exclude_axial=True, exclude_torsion=True):
        """Calculate static deflection under given distributed load `f`."""
        assert exclude_axial and exclude_torsion

        # remove axial/torsion
        idx = [i for i in range(len(self.K)) if (i % 6) not in (0, 3)]

        # take away ends if clamped
        if clamped_left:
            idx = idx[4:]
        if clamped_right:
            idx = idx[:-4]

        # reduced stiffness matrix, excluding clamped ends and axial/torsion
        K = self.K[ix_(idx, idx)]

        # generalised forces corresponding to applied distributed force
        F = dot(self.F, f)

        # solve for deflection
        x = zeros(len(self.K))
        x[idx] = linalg.solve(K, F[idx])
        return x

    def modal_matrices(self, n_modes=None,
                       clamped_left=True, clamped_right=True,
                       exclude_axial=True, exclude_torsion=True):
        """
        Shortcut which calculates normal mode shapes, limited to the first
        ``n_modes`` modes if required, and returns modal mass and stiffness.
        """
        w, shapes = self.normal_modes(n_modes, clamped_left, clamped_right,
                                      exclude_axial, exclude_torsion)
        modalM = dot(shapes.T, dot(self.M, shapes))
        modalK = dot(shapes.T, dot(self.K, shapes))
        return modalM, modalK
