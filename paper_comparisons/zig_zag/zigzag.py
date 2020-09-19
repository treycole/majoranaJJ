#!/usr/bin/env python3

import cmath
import functools
import itertools
import math
import operator
import re
import warnings

import kwant
from kwant.builder import Site
from kwant.continuum import discretize
import numpy as np
import scipy.constants
import scipy.interpolate
import scipy.optimize
import scipy.sparse as sp
import scipy.sparse.linalg as sla


constants = dict(
    # effective mass in kg,
    m_eff=0.023 * scipy.constants.m_e / (scipy.constants.eV * 1e-3) / 1e18,
    hbar=scipy.constants.hbar / (scipy.constants.eV * 1e-3),
    mu_B=scipy.constants.physical_constants['Bohr magneton'][0] / (
        scipy.constants.eV * 1e-3),
    exp=cmath.exp,
    cos=cmath.cos,
    sin=cmath.sin)


def remove_phs(H):
    return re.sub(r'kron\((sigma_[xyz0]), sigma_[xzy0]\)', r'\1', H)


@functools.lru_cache()
def get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands=True,
        k_x_in_sc=False, with_k_z=False, no_phs=False,
        phs_breaking_potential=False):
    kinetic = "(hbar^2 / (2*m_eff) * (k_y^2 + k_x^2 + k_z^2) - mu {}) * kron(sigma_0, sigma_z)"
    if mu_from_bottom_of_spin_orbit_bands:
        ham_str = kinetic.format("+ m_eff*alpha_middle^2 / (2 * hbar^2)")
    else:
        ham_str = kinetic.format("")

    if not with_k_z:
        ham_str = ham_str.replace('k_z', '0')

    spin_orbit = "- alpha_{} * kron(sigma_y, sigma_z) * k_x"
    ham_normal = ham_str + spin_orbit.format('middle')
    ham_sc_top = ham_str + spin_orbit.format('left')
    ham_sc_bot = ham_str + spin_orbit.format('right')

    if transverse_soi:
        tr_spin_orbit = """+ alpha_{} * kron(sigma_x, sigma_z) * k_y"""
        ham_normal += tr_spin_orbit.format('middle')
        ham_sc_top += tr_spin_orbit.format('left')
        ham_sc_bot += tr_spin_orbit.format('right')

    superconductivity = """+ Delta_{0} * (cos({1}phase / 2) * kron(sigma_0, sigma_x)
                                          + sin({1}phase / 2) * kron(sigma_0, sigma_y))"""
    ham_sc_top += superconductivity.format('left', '-')
    ham_sc_bot += superconductivity.format('right', '+')

    zeeman = "+ g_factor_{} * mu_B * (B_x * kron(sigma_x, sigma_0)"
    zeeman += "+ B_y * kron(sigma_y, sigma_0)"
    zeeman += "+ B_z * kron(sigma_z, sigma_0))"
    ham_normal += zeeman.format('middle')
    ham_sc_top += zeeman.format('left')
    ham_sc_bot += zeeman.format('right')

    ham_barrier = ham_normal + "+ V * kron(sigma_0, sigma_z)"

    if not k_x_in_sc:
        ham_sc_bot = ham_sc_bot.replace('k_x', '0')
        ham_sc_top = ham_sc_top.replace('k_x', '0')

    if phs_breaking_potential:
        ham_normal += "+ V_breaking(x) * kron(sigma_0, sigma_0)"

    template_strings = {'barrier': ham_barrier,
                        'normal': ham_normal,
                        'sc_top': ham_sc_top,
                        'sc_bot': ham_sc_bot}

    if no_phs:
        template_strings = {k: remove_phs(v) for k, v in template_strings.items()}

    return template_strings


class Shape:
    """Creates callable object serving as a 'shape' function
    to be used with `kwant.Builder`.

    This class supports multiple set operations. Let `s1` and `s2` be
    instances of the `Shape` object, then:
    s3 = s1*s2 is the intersection of the two shapes, meaning that s3(x)
        returns True if and only if both s1(x) and s2(x) return True.
    s3 = s1+s2 is the union of the two shapes, meaning that s3(x) returns
        True if and only if s1(x) and/or s2(x) returns True.
    s3 = s1-s2 is the difference of the two shapes, meaning that s3(x) returns
        True if and only if s1(x) returns True and s2(x) returns False.

    The Shape class also contains some other useful methods to obtain the
    inverse, the edge, and interior of the shape.

    Using slice indexing one creates rectangular bounds, i.e.:
    if s = Shape()[:0, :],
    then s(x, y) returns True if and only if x<=0.

    Arguments
    ---------
    shape : callable, optional
        shape(kwant.Site) -> bool (default always returns True).
    """

    def __init__(self, shape=None):
        self.shape = shape if shape is not None else lambda site: True
        prod = itertools.product([-1, 0, 1], repeat=2)
        self._directions = [tup for tup in prod if tup != (0, 0)]

    def __call__(self, site):
        return self.shape(site)

    def __add__(self, other_shape):
        shapes = (self.shape, other_shape.shape)
        union = lambda site: any(shape(site) for shape in shapes)
        return Shape(union)

    def __sub__(self, other_shape):
        shape_A, shape_B = (self.shape, other_shape.shape)
        difference = lambda site: shape_A(site) and not shape_B(site)
        return Shape(difference)

    def __mul__(self, other_shape):
        shapes = (self.shape, other_shape.shape)
        intersection = lambda site: all(shape(site) for shape in shapes)
        return Shape(intersection)

    @staticmethod
    def slice_func(_slice):
        def _func(x):
            y = True
            if _slice.start is not None:
                y &= _slice.start <= x
            if _slice.stop is not None:
                y &= x < _slice.stop
            return y
        return _func

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            # Make sure that item is iterable
            item = (item,)
        shapes = [Shape(lambda s, i=i, x=x: self.slice_func(x)(s.pos[i]))
                  for i, x in enumerate(item)]
        return functools.reduce(operator.mul, [self] + shapes)

    def inverse(self):
        """Returns inverse of shape."""

        return Shape(lambda site: not self.shape(site))

    def edge(self, which='inner'):
        """Returns edge of a 2D shape.

        If which is 'inner' ('outer'), the inner (outer) edge of the shape
        is returned, meaning the familiy of sites which lie inside (outside)
        the shape, but have at least one neighbor outside (inside) of the shape.
        """
        def edge_shape(site):
            in_shape = lambda x: self.shape(Site(site.family, site.tag + x))
            sites = [in_shape(x) for x in self._directions]
            if which == 'inner':
                return self.shape(site) and not all(sites)
            elif which == 'outer':
                return not self.shape(site) and any(sites)
        return Shape(edge_shape)

    def interior(self):
        """Returns shape minus its edge."""
        return Shape(self - self.edge('inner'))

    @classmethod
    def below_curve(cls, curve):
        """Returns instance of Shape which returns True if a site (x, y)
        is such that y < curve(x)."""
        def _shape(site):
            x, y = site.pos
            return y < curve(x)
        return Shape(_shape)

    @classmethod
    def above_curve(cls, curve):
        """Returns instance of Shape which returns True if a
        site (x, y) is such that y >= curve(x)."""
        return Shape.below_curve(curve).inverse()

    @classmethod
    def left_of_curve(cls, curve):
        """Returns instance of Shape which returns True if a
        site (x, y) is such that x < curve(y)."""
        def _shape(site):
            x, y = site.pos
            return x < curve(y)
        return Shape(_shape)

    @classmethod
    def right_of_curve(cls, curve):
        """Returns instance of Shape which returns True if a site (x, y)
        is such that x >= curve(y)."""
        return Shape.left_of_curve(curve).inverse()


def create_parallel_sine(distance, z_x, z_y, rough_edge=None):
    def _parallel_sine(x, distance, z_x, z_y):
        def g(t):
            return z_y * math.sin(2 * np.pi / z_x * t)

        def g_prime(t):
            return z_y * 2 * np.pi / z_x * math.cos(2 * np.pi / z_x * t)

        def _x(t):
            return t - distance * g_prime(t) / np.sqrt(1 + g_prime(t)**2) - x

        def y(t):
            return g(t) + distance / np.sqrt(1 + g_prime(t)**2)

        t = scipy.optimize.fsolve(_x, x)
        return y(t)

    xs = np.linspace(0, z_x, 1000)
    ys = [_parallel_sine(x, distance, z_x, z_y) for x in xs]

    if rough_edge is not None:
        X, Y, salt = rough_edge

        # Calculate the unit vector to the sine
        tck = scipy.interpolate.splrep(xs, ys)
        dydx = scipy.interpolate.splev(xs, tck, der=1)
        unit_vectors = 1 / (1 + dydx**2) * np.array([-dydx, np.ones_like(dydx)])

        # Generate a disordered boundary parameterized by (X, Y, salt)
        rands = [X * kwant.digest.uniform(str(i), salt=str(salt))
                 for i in range(int(z_x / Y) - 2)]
        rands = [0, *rands, 0]  # make sure that it starts and ends with 0 (for periodicity)
        ys_disorder = scipy.interpolate.interp1d(
            np.linspace(0, z_x, len(rands)), rands, kind='quadratic')(xs)
        ys_disorder -= ys_disorder.mean()

        # Disorder in the direction normal to the sine
        dxys = ys_disorder * unit_vectors

        # Modify xs and ys to include the disorder
        xs[1:-1] += dxys[0, 1:-1]
        ys[1:-1] += dxys[1, 1:-1]

    parallel_sine = scipy.interpolate.interp1d(xs, ys)

    return lambda x: parallel_sine(x % z_x)


def get_shapes(shape, a, z_x, z_y, W, L_x, L_sc_down, L_sc_up, rough_edge=None):
    if shape == 'parallel_curve':
        curve = create_parallel_sine(0, z_x, z_y, rough_edge=None)

        _curve_top = create_parallel_sine(W // 2, z_x, z_y, rough_edge=rough_edge)
        _below_shape = Shape.below_curve(_curve_top)

        if rough_edge is not None:  # Change salt for the order boundary
            X, Y, salt = rough_edge
            rough_edge = (X, Y, -salt)

        _curve_bottom = create_parallel_sine(-W // 2, z_x, z_y, rough_edge=rough_edge)
        _above_shape = Shape.above_curve(_curve_bottom)

        _middle_shape = (_below_shape * _above_shape)[0:L_x, :]
        sc_top_initial_site = (z_x // 4, W // 2 + z_y + L_sc_up // 2)
        sc_top_shape = _middle_shape.inverse()[0:L_x, :L_sc_up + W // 2 + z_y]
        sc_bot_initial_site = (z_x // 4, -W // 2 - z_y - L_sc_down // 2)
        sc_bot_shape = _middle_shape.inverse()[0:L_x, -L_sc_down - W // 2 - z_y:]
    elif shape == 'sawtooth':
        def curve(x):
            x += z_x / 4
            if x % z_x < z_x / 2:
                return 4 * z_y / z_x * (x % (z_x / 2)) - z_y
            else:
                return -4 * z_y / z_x * (x % (z_x / 2)) + z_y

        y_offset = W / np.cos(np.arctan(4 * z_y / z_x)) if z_y != 0 else W

        _below_shape = Shape.below_curve(lambda x: curve(x) + y_offset // 2)
        _above_shape = Shape.above_curve(lambda x: curve(x) - y_offset // 2)

        _middle_shape = (_below_shape * _above_shape)[0:L_x, :]
        sc_top_initial_site = (0, y_offset // 2 + a)
        sc_top_shape = _middle_shape.inverse()[0:L_x, :L_sc_up + y_offset // 2 + z_y]
        sc_bot_initial_site = (0, -y_offset // 2 - a)
        sc_bot_shape = _middle_shape.inverse()[0:L_x, -L_sc_down - y_offset // 2 - z_y:]
    else:
        raise ValueError('Only "parallel_curve" and "sawtooth" are implemented.')

    edge_shape = (_middle_shape.edge('inner')
                  * sc_bot_shape.edge('outer')
                  * sc_top_shape.edge('outer'))

    interior_shape = _middle_shape - edge_shape
    interior_initial_site = (a, a)

    return {'sc_top': (sc_top_shape, sc_top_initial_site),
            'sc_bot': (sc_bot_shape, sc_bot_initial_site),
            'normal': (interior_shape, interior_initial_site),
            'edge': edge_shape}


@functools.lru_cache()
def system(
        W, L_x, L_sc_up, L_sc_down, z_x, z_y, a, shape, transverse_soi,
        mu_from_bottom_of_spin_orbit_bands, k_x_in_sc, wraparound, infinite,
        sc_leads=False, no_phs=False, rough_edge=None,
        phs_breaking_potential=False):
    """Create zigzag system

    Parameters
    ----------
    W : float
        Width of the semiconductor (or contact seperation of the junction.)
    L_x : float
        Length of the system (x-dimension).
    L_sc_up : float
        Minimum width of the top superconductor.
    L_sc_down : float
        Minimum width of the bottom superconductor.
    z_x : float
        Period of zigzag.
    z_y : float
        Amplitude of zigzag.
    a : float
        Lattice spacing.
    shape : string
        Can be either 'sawtooth' for zigzag shape, or 'parallel_curve'
        for a shape formed by curve parallel to a sine curve.
    transverse_soi : bool
        Toggle Rashba spin-orbit in the y-direction.
    mu_from_bottom_of_spin_orbit_bands : bool
        Toggle counting chemical potential from bottom of spin orbit band.
    k_x_in_sc : bool
        Toggle whether superconductor have hopping in the x-direction.
    wraparound : bool
        Toggle a wraparound system, such that the translational invariance
        is transformed into the momentum parameter k_x.
    infinite : bool
        Toggle whether the system contains a z_x periodic
        translational invariance.
    sc_leads : bool, optional
        Toggle superconducting leads in y-direction.
    no_phs : bool, optional
        Remove particle-hole symmetry by removing the electron-hole orbital.
    rough_edge : bool, optional
        Toggle roughened edges to shape.
    phs_breaking_potential : bool, optional
        Add particle-hole symmetry breaking potential to allow for a
        computationally cheaper way to calculate the Majorana decay length.

    Returns
    -------
    kwant.builder.FiniteSystem or kwant.builder.InfiniteSystem

    """
    if wraparound and not infinite:
        raise ValueError('If you want to use wraparound, infinite must be True.')
    if sc_leads and not infinite or sc_leads and not wraparound:
        raise ValueError('If you want to use sc_leads, infinite and wraparound must be True.')

    template_strings = get_template_strings(
        transverse_soi, mu_from_bottom_of_spin_orbit_bands,
        k_x_in_sc, False, no_phs, phs_breaking_potential)

    template = {k: discretize(v, coords=('x', 'y'), grid_spacing=a)
                for k, v in template_strings.items()}

    shapes = get_shapes(shape, a, z_x, z_y, W, L_x, L_sc_down, L_sc_up, rough_edge)

    syst = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0]) if infinite else None)

    for y in np.arange(-W - L_sc_down, W + L_sc_up, a):
        # We're unsure about the location of the barrier
        # so we loop over all possible sites.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            sites = syst.fill(template['barrier'], shapes['edge'], (0, y))

    syst.fill(template['normal'], *shapes['normal'])

    if L_sc_up > 0:
        syst.fill(template['sc_top'], *shapes['sc_top'])

    if L_sc_down > 0:
        syst.fill(template['sc_bot'], *shapes['sc_bot'])

    if infinite and wraparound:
        syst = kwant.wraparound.wraparound(syst)
        if sc_leads:
            lead_up = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0], [0, a]))
            lead_down = kwant.Builder(kwant.TranslationalSymmetry([L_x, 0], [0, -a]))
            lead_up = kwant.wraparound.wraparound(lead_up, keep=1)
            lead_down = kwant.wraparound.wraparound(lead_down, keep=1)
            lead_up.fill(template_sc_top, lambda s: 0 <= s.pos[0] < L_x, (0, 0))
            lead_down.fill(template_sc_bot, lambda s: 0 <= s.pos[0] < L_x, (0, 0))
            syst.attach_lead(lead_up)
            syst.attach_lead(lead_down)

    return syst.finalized()


def mumps_eigsh(matrix, k, sigma, **kwargs):
    """Call sla.eigsh with mumps support.

    Please see scipy.sparse.linalg.eigsh for documentation.
    """
    class LuInv(sla.LinearOperator):

        def __init__(self, matrix):
            instance = kwant.linalg.mumps.MUMPSContext()
            instance.analyze(matrix, ordering='pord')
            instance.factor(matrix)
            self.solve = instance.solve
            sla.LinearOperator.__init__(self, matrix.dtype, matrix.shape)

        def _matvec(self, x):
            return self.solve(x.astype(self.dtype))

    opinv = LuInv(matrix - sigma * sp.identity(matrix.shape[0]))
    return sla.eigsh(matrix, k, sigma=sigma, OPinv=opinv, **kwargs)


def spectrum(syst, params, k=20):
    """Compute the k smallest magnitude eigenvalues and
    corresponding eigenvectors of a system.

    Parameters
    ----------
    syst : `kwant.FiniteSystem` or `kwant.InfiniteSystem`
        System of which the spectrum will be computed.
    params : dict
        Dictionary containing the parameters.
    k : int
        Number of eigenvalues to calculate.

    Returns
    -------
    energies : numpy array
        Array containing smallest magnitude eigenvalues.
    wfs : numpy array
        Array containing eigenvectors corresponding to eigenvalues.
    """
    ham = syst.hamiltonian_submatrix(params=params, sparse=True)
    (energies, wfs) = mumps_eigsh(ham, k=k, sigma=0)
    return energies, wfs


def translation_ev(h, t, tol=1e6):
    """Compute the eigenvalues of the translation operator of a lead.

    Adapted from kwant.physics.leads.modes.

    Parameters
    ----------
    h : numpy array, real or complex, shape (N, N) The unit cell
        Hamiltonian of the lead unit cell.
    t : numpy array, real or complex, shape (N, M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.

    Returns
    -------
    ev : numpy array
        Eigenvalues of the translation operator in the form lambda=r*exp(i*k),
        for |r|=1 they are propagating modes.
    """
    a, b = kwant.physics.leads.setup_linsys(h, t, tol, None).eigenproblem
    ev = kwant.physics.leads.unified_eigenproblem(a, b, tol=tol)[0]
    return ev


def cell_mats(lead, params, bias=0):
    h = lead.cell_hamiltonian(params=params)
    h -= bias * np.identity(len(h))
    t = lead.inter_cell_hopping(params=params)
    return h, t


def gap_minimizer(lead, params, energy):
    """Function that minimizes a function to find the band gap.
    This objective function checks if there are progagating modes at a
    certain energy. Returns zero if there is a propagating mode.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    energy : float
        Energy at which this function checks for propagating modes.

    Returns
    -------
    minimized_scalar : float
        Value that is zero when there is a propagating mode.
    """
    h, t = cell_mats(lead, params, bias=energy)
    ev = translation_ev(h, t)
    norm = (ev * ev.conj()).real
    return np.min(np.abs(norm - 1))


def gap_from_modes(lead, params, tol=1e-6):
    """Finds the gapsize by peforming a binary search of the modes with a
    tolarance of tol.

    Parameters
    ----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dict that is used to store Hamiltonian parameters.
    tol : float
        The precision of the binary search.

    Returns
    -------
    gap : float
        Size of the gap.

    Notes
    -----
    For use with `lead = funcs.make_lead()`.
    """
    Es = kwant.physics.Bands(lead, params=params)(k=0)
    lim = [0, np.abs(Es).min()]
    if gap_minimizer(lead, params, energy=0) < 1e-15:
        # No band gap
        gap = 0
    else:
        while lim[1] - lim[0] > tol:
            energy = sum(lim) / 2
            par = gap_minimizer(lead, params, energy)
            if par < 1e-10:
                lim[1] = energy
            else:
                lim[0] = energy
        gap = sum(lim) / 2
    return gap


def gap_from_band_structure(lead, params, Ns=31, full_output=False):
    def energies(k_x):
        Es = spectrum(lead, dict(params, k_x=float(k_x)), k=4)[0]
        return np.abs(Es).min()

    return float(scipy.optimize.brute(energies, ranges=((0, np.pi),),
                 Ns=Ns, full_output=full_output))


def phase_bounds_operator(lead, params, k_x=0):
    h_k = lead.hamiltonian_submatrix(
        params=dict(params, mu=0, k_x=k_x), sparse=True)
    sigma_z = sp.csc_matrix(np.array([[1, 0], [0, -1]]))
    _operator = sp.kron(sp.eye(h_k.shape[0] // 2), sigma_z) @ h_k
    return _operator


def phase_bounds(lead, params, k_x=0, num_bands=20, sigma=0):
    """Find the phase boundaries.
    Solve an eigenproblem that finds values of chemical potential at which the
    gap closes at momentum k=0. We are looking for all real solutions of the
    form H*psi=0 so we solve sigma_0 * tau_z H * psi = mu * psi.

    Parameters
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.
    params : dict
        A dictionary that is used to store Hamiltonian parameters.
    k_x : float
        Momentum value, by default set to 0.

    Returns
    --------
    chemical_potential : numpy array
        Twenty values of chemical potential at which a bandgap closes at k=0.
    """
    chemical_potentials = phase_bounds_operator(lead, params, k_x)

    if num_bands is None:
        mus = np.linalg.eigvals(chemical_potentials.todense())
    else:
        # This is not Hermitian, so we can't use mumps.
        mus = sla.eigs(chemical_potentials, k=num_bands, sigma=sigma, which='LM')[0]

    real_solutions = abs(np.angle(mus)) < 1e-10

    mus[~real_solutions] = np.nan  # To ensure it returns the same shape vector
    return np.sort(mus.real)


def lat_from_syst(syst):
    lats = set(s.family for s in syst.sites)
    if len(lats) > 1:
        raise Exception('No unique lattice in the system.')
    return list(lats)[0]


def majorana_size_from_modes(lead, params):
    """Find the slowest decaying (evanescent) mode and returns the
    Majorana size.

    It uses an adapted version of the function kwant.physics.leads.modes,
    in such a way that it returns the eigenvalues of the translation operator
    (lamdba = |r|*e^ik). The imaginary part of the wavevector k, is the part
    that makes it decay. The inverse of this Im(k) is the size of a Majorana
    bound state. The norm of the eigenvalue that is closest to one is the
    slowest decaying mode. Also called decay length.

    Parameters:
    -----------
    lead : kwant.builder.InfiniteSystem object
        The finalized infinite system.

    Returns:
    --------
    majorana_length : float
        The length of the Majorana.
    """
    h, t = cell_mats(lead, params, bias=0)
    ev = translation_ev(h, t)
    norm = ev * ev.conj()
    idx = np.abs(norm - 1).argmin()
    a = lat_from_syst(lead).prim_vecs[0, 0]
    majorana_length = np.abs(a / np.log(ev[idx]).real)
    return majorana_length


def majorana_state(syst, params):
    energies, wfs = spectrum(syst, params, k=2)
    wfs = wfs[:, energies > 0]
    energies = energies[energies > 0]
    i_min = np.argmin(energies)
    rho = kwant.operator.Density(syst)
    wf = rho(wfs[:, i_min])
    return energies[i_min], wf


def majorana_size_from_fit(syst, params):
    wf = majorana_state(syst, params)[1]

    # Project onto the x-axis.
    xs, ys = np.array([s.pos for s in syst.sites]).T
    ncols = len(np.where(xs == 0)[0])
    wf_proj = wf.reshape((-1, ncols)).sum(axis=1)
    xs = xs.reshape((-1, ncols))[:, 0]

    # Only fit the middle part of the wf.
    N = len(xs) // 4
    xs_fit = xs[N:2*N]
    wf_proj_fit = wf_proj[N:2*N]

    # Do the fit.
    def flog(x, a, b, x0):
        return np.log(a) - b * x - x0

    popt, pcov = scipy.optimize.curve_fit(flog, xs_fit, np.log(wf_proj_fit))

    return 1 / popt[1]
