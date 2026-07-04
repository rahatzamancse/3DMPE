"""Linear projection family used to map the 3D embedding into each 2D view."""

import sys
import math
import itertools
import numpy as np
import scipy.stats

families = ['linear']
constraints = [None, 'orthogonal', 'similar']


class PROJ(object):
    """Collection of allowed projection functions and helpers to call them."""

    def __init__(self, d1=3, d2=2, family='linear', constraint='orthogonal',
                 **kwargs):
        """Set up the projection family.

        Parameters
        ----------
        d1 : int
            Dimension of the input (embedding) space.
        d2 : int
            Dimension of the output (image) space.
        family : str
            Projection family. Only ``'linear'`` is supported.
        constraint : None or str
            Constraint on the projection parameters: ``None``, ``'orthogonal'``,
            or ``'similar'``.
        """
        assert isinstance(d1, int) and d1 > 0
        self.d1 = d1
        assert isinstance(d2, int) and d2 > 0
        self.d2 = d2
        assert family in families
        self.family = family
        assert constraint in constraints
        self.constraint = constraint

        if self.family == 'linear':
            self.setup_linear()
        else:
            sys.exit('PROJ family unknown.')

    def setup_linear(self):
        """Set up functions for the linear projection family."""
        assert self.d1 >= self.d2

        self.shape = (self.d2, self.d1)
        self.p = lambda q, x: x @ q.T
        self.P = lambda q, X: X @ q.T
        self.dp = lambda q, x: q

        def special(number, method='identity'):
            if method == 'identity':
                q = np.identity(self.d1)
                q = q[0:self.d2]
                return [q] * number
            elif method == 'standard':
                Q = []
                for comb in itertools.combinations(range(self.d1), self.d2):
                    Q.append(np.identity(self.d1)[comb, :])
                assert len(Q) >= number
                return Q[0:number]
            elif method == 'cylinder':
                assert self.d1 == 3 and self.d2 == 2
                Q = []
                for k in range(number):
                    theta = math.pi / number * k
                    Q.append(np.array([[math.cos(theta), math.sin(theta), 0],
                                       [0, 0, 1]]))
                return Q
        self.special = special

        if self.constraint is None:
            self.c = lambda x: x
            self.random = lambda: np.random.randn(self.d2, self.d1)
        elif self.constraint == 'orthogonal':
            def c(P):
                """Return the nearest orthogonal matrix to ``P`` (Frobenius)."""
                U, s, Vh = np.linalg.svd(P, full_matrices=False)
                return U @ Vh
            self.c = c
            self.random = lambda: scipy.stats.ortho_group.rvs(self.d1)[0:self.d2, :]
        elif self.constraint == 'similar':
            def c(P):
                """Return the nearest scaled-orthogonal matrix to ``P``."""
                U, s, Vh = np.linalg.svd(P, full_matrices=False)
                s = np.sum(s) / len(s)
                return s * U @ Vh
            self.c = c

            def random(rmax=2, rmin=0.5):
                q = scipy.stats.ortho_group.rvs(self.d1)[0:self.d2, :]
                q *= np.random.rand() * (rmax - rmin) + rmin
                return q
            self.random = random

    def check(self, q=None, Q=None, X=None):
        """Validate the shape of projection parameters and/or a coordinate array."""
        if q is not None:
            assert isinstance(q, np.ndarray)
            assert q.shape == self.shape
        if Q is not None:
            assert isinstance(Q, list) or isinstance(Q, np.ndarray)
            for q in Q:
                assert isinstance(q, np.ndarray)
                assert q.shape == self.shape
        if X is not None:
            assert isinstance(X, np.ndarray)
            if X.ndim == 1:
                assert len(X) == self.d1
            else:
                assert X.ndim == 2
                assert X.shape[1] == self.d1

    def project(self, q, X):
        """Project ``X`` with one projection ``q`` or a list/array of projections."""
        if isinstance(q, np.ndarray) and q.shape == self.shape:
            Y = self.P(q, X)
        else:
            Y = []
            for qk in q:
                Y.append(self.P(qk, X))
            if isinstance(q, np.ndarray):
                Y = np.array(Y)
        return Y

    def gradient(self, q, x):
        """Return the projection Jacobian(s) for ``q`` (single or multiple)."""
        if isinstance(q, np.ndarray) and q.shape == self.shape:
            dpx = self.dp(q, x)
        else:
            dpx = []
            for qk in q:
                dpx.append(self.dp(q, x))
            if isinstance(q, np.ndarray):
                dpx = np.array(dpx)
        return dpx

    def restrict(self, q):
        """Project parameters back onto the constraint set (single or multiple)."""
        if isinstance(q, np.ndarray) and q.shape == self.shape:
            qq = self.c(q)
        else:
            qq = []
            for qk in q:
                qq.append(self.c(qk))
            qq = np.array(qq)
        return qq

    def generate(self, number=3, method='random', **kwargs):
        """Return a list of ``number`` projection parameter arrays.

        ``method`` is either ``'random'`` or one of the special methods
        (``'identity'``, ``'standard'``, ``'cylinder'``).
        """
        if not method:
            method = 'random'
        if method == 'random':
            Q = [self.random() for _ in range(number)]
        else:
            Q = self.special(number, method=method)
        return Q
