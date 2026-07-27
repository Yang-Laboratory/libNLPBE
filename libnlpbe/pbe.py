import os
import ctypes
import numpy
import scipy
import tempfile
import h5py
import libnlpbe

from pyscf import df
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent
from pyscf.data.nist import *
from pyscf.data.radii import VDW
from pyscf.tools import cubegen

PI = numpy.pi
KB2HARTREE = BOLTZMANN / HARTREE2J
M2HARTREE = AVOGADRO*BOHR**3*1.e-27

libamgcl = lib.load_library(os.path.join(libnlpbe.__path__[0], 'lib', 'libdinmh.so'))
libamgcl.amg_create.restype = ctypes.c_void_p

def pbe_for_scf(mf, solvent_obj=None, dm=None):
    """
    Attach PBE solvation model to a SCF object.

    Creates a self-consistent-field calculator that includes solvation effects
    via the non-linear Poisson-Boltzmann model.

    Parameters
    ----------
    mf : pyscf.scf.RHF/RKS or pyscf.scf.UHF/UKS
        PySCF SCF mean-field object.
    solvent_obj : PBE, optional
        PBE solvation object. If None, creates a default PBE(mf.mol).
    dm : ndarray, optional
        Initial density matrix for solvation. Default: None (computed from mf).

    Returns
    -------
    solmf : pyscf.solvent.PCMSolver.SCFWithPolarization
        SCF object with solvation effects included via PBE.

    Examples
    --------
    >>> from pyscf import gto, dft
    >>> from fcdft.solvent.pbe import PBE, pbe_for_scf
    >>> mol = gto.M(atom='O 0 0 0; H 0 1 0; H 0 0 1', basis='6-31g')
    >>> mf = dft.RKS(mol, xc='b3lyp')
    >>> cm = PBE(mol, cb=1.0, length=15, ngrids=41)
    >>> cm.eps = 78.3553  # Water
    >>> solmf = pbe_for_scf(mf, cm)
    >>> solmf.kernel()
    """
    if solvent_obj is None:
        solvent_obj = NLPBE(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)


def make_gradient_matrix(ngrids):
    central = numpy.array([3, -32, 168, -672, 672, -168, 32, -3], dtype=numpy.float64) / 840.0
    offsets = (-4, -3, -2, -1, 1, 2, 3, 4)
    one_sided = numpy.array([-25, 48, -36, 16, -3], dtype=numpy.float64) / 12.0

    rows = []
    cols = []
    vals = []
    for i in range(4, ngrids - 4):
        for offset, weight in zip(offsets, central):
            rows.append(i)
            cols.append(i + offset)
            vals.append(weight)
    for i in range(4):
        for k in range(5):
            rows.append(i)
            cols.append(i + k)
            vals.append(one_sided[k])
    for i in range(ngrids - 4, ngrids):
        for k in range(5):
            rows.append(i)
            cols.append(i - k)
            vals.append(-one_sided[k])

    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(ngrids, ngrids))

def gradient(solvent_obj, phi, ngrids, spacing):
    """8th-order finite-difference gradient of a scalar field on the cubic grid.

    Returns an ``(n_grid, 3)`` array. Used at convergence to build the
    polarization charge density ``rho_pol``. Matches the stencil used to
    assemble the linear operator in :func:`make_operator`.
    """
    grad = make_gradient_matrix(ngrids)
    I = scipy.sparse.identity(ngrids, format='csr')
    G = (scipy.sparse.kron(grad, scipy.sparse.kron(I, I)),
         scipy.sparse.kron(I, scipy.sparse.kron(grad, I)),
         scipy.sparse.kron(I, scipy.sparse.kron(I, grad)))
    dphi = numpy.empty((phi.size, 3), dtype=numpy.float64)
    for xi in range(3):
        dphi[:, xi] = G[xi].dot(phi) / spacing
    return dphi

def make_lambda(solvent_obj, mol, probe, stern_mol, coords, delta, atomic_radii):

    atom_coords = mol.atom_coords()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe - stern_mol) / delta
    erf_list = 0.5 * (1.0 + scipy.special.erf(x))
    lambda_r = numpy.prod(erf_list, axis=0)

    return lambda_r

def make_sas(solvent_obj, mol, probe, coords, delta, atomic_radii):
    """
    Construct solvent-accessible surface (SAS) function.

    Returns a smooth function that is 0 inside the SAS (within atomic radii + probe)
    and 1 in the bulk solvent, with smooth transition via error function.

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    mol : gto.Mole
        Molecule specification.
    probe : float
        Solvent probe radius (water: 1.4 Å = 2.64 a.u.) in a.u.
    coords : ndarray, shape (n_grid, 3)
        Grid point coordinates in Cartesian (a.u.).
    delta2 : float
        Broadening width for SAS erf in a.u.
    atomic_radii : ndarray, shape (n_atoms,)
        Atomic van der Waals radii in a.u.

    Returns
    -------
    sas : ndarray, shape (n_grid,)
        Solvent-accessible surface function (0=inside, 1=bulk).
    """
    atom_coords = mol.atom_coords()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta
    erf = scipy.special.erf(x)
    erf_list = 0.5 * (1.0 + erf)
    sas = numpy.prod(erf_list, axis=0)
    return sas

def make_grad_sas(solvent_obj, mol, probe, coords, delta, atomic_radii):
    ngrids = solvent_obj.grids.ngrids
    atom_coords = mol.atom_coords()
    natm = mol.natm

    r = atom_coords[:,None,:]
    rp = coords - r
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    x = (dist - atomic_radii[:,None] - probe) / delta
    erf = scipy.special.erf(x)
    erf_list = 0.5 * (1.0 + erf)
    er = rp / dist[:,:,None]
    gauss = numpy.exp(-x**2)
    grad_list = numpy.multiply(er, gauss[:,:,None], out=er) / (delta * numpy.sqrt(PI))
    grad_sas = numpy.zeros((ngrids**3, 3), dtype=numpy.float64)
    atmlst = range(natm)
    for i in atmlst:
        mask = [False if i == j else True for j in atmlst]
        erf_prod = numpy.prod(erf_list[mask], axis=0)
        grad_sas += grad_list[i] * erf_prod[:,None]

    return grad_sas

def make_eps(solvent_obj, eps, sas):
    eps_r = 1.0e0 + (eps - 1.0e0) * sas
    return eps_r

def make_grad_eps(solvent_obj, eps, grad_sas):
    grad_eps = (eps - 1.0) * grad_sas
    return grad_eps

def make_phi_sol(solvent_obj, dm=None, coords=None):
    t0 = (logger.process_clock(), logger.perf_counter())
    if dm is None: dm = solvent_obj._dm
    if coords is None: coords = solvent_obj.grids.coords
    _intermediates = solvent_obj._intermediates

    mol = solvent_obj.mol
    tot_ngrids = coords.shape[0]

    # Nuclear part
    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    dist[dist < 1.0e-100] = numpy.inf # Machine precision
    Vnuc = numpy.tensordot(1.0e0 / dist, Z, axes=([0], [0]))

    # Electronic part
    if dm.ndim == 3: # Spin-unrestricted
        dm = dm[0] + dm[1]

    auxmol = _intermediates['auxmol']
    erifile = _intermediates['erifile']
    nao = mol.nao
    naux = auxmol.nao

    dms = numpy.asarray(dm.real)
    dm_tril = lib.pack_tril(dms + dms.T)
    idx = numpy.arange(nao)
    idx = idx * (idx + 1) // 2 + idx
    dm_tril[idx] *= 0.5

    with h5py.File(erifile, 'r') as feri:
        int2c2e = numpy.asarray(feri['int2c2e'])
        int3c2e = numpy.asarray(feri['int3c2e'])

    # Cholesky solve the RI equation
    g = dm_tril.dot(int3c2e)
    cK = scipy.linalg.cho_solve(scipy.linalg.cho_factor(int2c2e), g)

    Vele = numpy.empty(tot_ngrids, order='C')
    max_memory = solvent_obj.max_memory - lib.current_memory()[0] - Vele.nbytes*1e-6
    blksize = int(max(max_memory*.9e6/8/naux, 400))
    for p0, p1 in lib.prange(0, tot_ngrids, blksize):
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = gto.intor_cross('int2c2e', fakemol, auxmol)
        Vele[p0:p1] = ints.dot(cK)
        del ints

    MEP = Vnuc - Vele
    t0 = logger.timer(solvent_obj, 'phi_sol', *t0)
    return lib.tag_array(MEP, Vnuc=Vnuc, Vele=-Vele)

def make_rho_sol(solvent_obj, phi_sol=None, ngrids=None, spacing=None):
    """
    Compute solute charge density from electrostatic potential via Poisson's equation.

    Uses the Laplacian of φ_sol to recover the charge density:

        ρ_sol = -∇²φ_sol / 4π

    Parameters
    ----------
    solvent_obj : PBE
        PBE solvation object.
    phi_sol : ndarray, shape (n_grid,), optional
        Solute potential. If None, uses solvent_obj.phi_sol.
    ngrids : int, optional
        Number of grid points along each axis (cubic grid).
        If None, uses solvent_obj.grids.ngrids.
    spacing : float, optional
        Grid spacing in a.u. If None, uses solvent_obj.grids.spacing.

    Returns
    -------
    rho_sol : ndarray, shape (n_grid,)
        Solute charge density at grid points.
    """
    if phi_sol is None: phi_sol = solvent_obj.phi_sol
    if spacing is None: spacing = solvent_obj.grids.spacing
    if ngrids is None: ngrids = solvent_obj.grids.ngrids

    L = solvent_obj.L
    rho_sol = L.dot(phi_sol) / 4.0 / PI / spacing**2

    # Zero-out boundary values
    rho_sol = rho_sol.reshape((ngrids,)*3)
    idx = numpy.arange(ngrids)
    idx = (idx < 4) | (idx >= ngrids-4)
    rho_sol[idx,:,:] = 0.0
    rho_sol[:,idx,:] = 0.0
    rho_sol[:,:,idx] = 0.0
    rho_sol = rho_sol.reshape(ngrids**3)
    logger.info(solvent_obj, 'charge by poisson equation = %s', rho_sol.sum() * spacing**3)

    return rho_sol

def rho_ions_one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74 / (4.0/3.0 * PI * (cation_rad**3 + anion_rad**3))
    if cb == 0.0:
        return numpy.zeros_like(phi_tot)

    x = phi_tot / KB2HARTREE / T
    mask = abs(x) < 691.4
    sinh = numpy.full_like(x, 1.0e300) * numpy.sign(x)
    cosh = numpy.full_like(x, 1.0e300)
    sinh[mask] = numpy.sinh(x[mask])
    cosh[mask] = numpy.cosh(x[mask])

    rho_ions = -2.0 * lambda_r * cb * sinh / (1.0 - cb/c12 + cb/c12 * lambda_r * cosh)

    return rho_ions

def drho_ions_one_to_one(solvent_obj, phi_tot=None, cb=None, lambda_r=None, T=None):
    """drho_ions / dphi"""
    if phi_tot is None: phi_tot = solvent_obj.phi_tot
    if cb is None: cb = solvent_obj.cb * M2HARTREE
    if lambda_r is None: lambda_r = solvent_obj._intermediates['lambda_r']
    if T is None: T = solvent_obj.T

    if cb == 0.0:
        return numpy.zeros_like(phi_tot)

    cation_rad = solvent_obj.cation_rad / BOHR
    anion_rad = solvent_obj.anion_rad / BOHR
    c12 = 0.74 / (4.0 / 3.0 * PI * (cation_rad**3 + anion_rad**3))

    x = phi_tot / KB2HARTREE / T
    mask = abs(x) < 691.4
    drho_ions = numpy.zeros_like(x)
    cosh = numpy.cosh(x[mask])
    a = cb / c12 * lambda_r[mask]
    b = 1.0 - cb / c12
    D = b + a * cosh
    # / D / D instead of D**2 to avoid unnecessary value overflow error msg
    drho_ions[mask] = (-2.0 * lambda_r[mask] * cb) * (b * cosh + a) / D / D
    drho_ions /= KB2HARTREE * T
    return drho_ions

def energy_osm_one_to_one(solvent_obj, phi_tot, cb, lambda_r, T, spacing):
    ngrids = solvent_obj.grids.ngrids
    lnlambda = numpy.full(ngrids**3, -numpy.inf)
    lnlambda = numpy.log(lambda_r, where=(lambda_r > 0), out=lnlambda)
    x = phi_tot / KB2HARTREE / T
    lnA = numpy.log(0.5) + lnlambda - x
    lnB = numpy.log(0.5) + lnlambda + x
    mask = numpy.isfinite(lnlambda)
    lnmax = numpy.maximum(lnA, lnB)
    expsum = numpy.zeros(ngrids**3)
    expsum[mask] = numpy.exp(lnA[mask] - lnmax[mask]) + numpy.exp(lnB[mask] - lnmax[mask])
    lnsum = numpy.full((ngrids**3,), -numpy.inf)
    lnsum[mask] = lnmax[mask] + numpy.log(expsum[mask])

    cation_rad, anion_rad = solvent_obj.cation_rad / BOHR, solvent_obj.anion_rad / BOHR
    c12 = 0.74 / (4.0/3.0 * PI * (cation_rad**3 + anion_rad**3))

    Gsolv_osm = -2.0*KB2HARTREE*T*c12*(numpy.log(1.0 + cb/c12*(numpy.exp(lnsum)-1.0))).sum() * spacing**3
    return Gsolv_osm

def make_operator(solvent_obj, grad_lneps=None):
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    if grad_lneps is None:
        _intermediates = solvent_obj._intermediates
        eps = _intermediates['eps']
        grad_eps = _intermediates['grad_eps']
        grad_lneps = grad_eps / eps[:,None]

    L = solvent_obj.L # nabla**2 = -L / spacing**2
    # A = nabla ln(eps) * nabla - L / spacing**2
    if solvent_obj.operator is None:
        grad = make_gradient_matrix(ngrids)
        I = scipy.sparse.identity(ngrids, format='csr')
        G = (scipy.sparse.kron(grad, scipy.sparse.kron(I, I)),
             scipy.sparse.kron(I, scipy.sparse.kron(grad, I)),
             scipy.sparse.kron(I, scipy.sparse.kron(I, grad)))
        A = -L / spacing**2
        for xi in range(3):
            A = A + scipy.sparse.diags(grad_lneps[:, xi]) @ (G[xi] / spacing)
        solvent_obj.operator = A

    A = solvent_obj.operator
    c_indptr = A.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_indices = A.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_data = A.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_tot_ngrids = ctypes.c_int(tot_ngrids)

    def apply_A(v, out):
        c_v = v.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libamgcl.csr_matvec(c_indptr, c_indices, c_data, c_v, c_out, c_tot_ngrids)
        return out
    
    return apply_A

def make_precond(solvent_obj, drho_ions_scr=None):
    """Build (or rebuild) the screened-Poisson AMG V-cycle preconditioner M^{-1}.

    Assembles the symmetric positive-definite screened operator

        A_screen = L / spacing**2 + diag(reaction),

    where ``L`` is the constant Poisson matrix and ``reaction`` is the
    (non-negative) reaction diagonal.  An amgcl smoothed-aggregation hierarchy is
    built on ``A_screen`` and cached on the instance.  Because ``A_screen`` always
    has the sparsity pattern of ``L`` (whose stencil already includes the
    diagonal), changing only ``reaction`` is handled by amgcl's cheap
    ``rebuild`` (reuse transfer operators, recompute coarse operators/smoothers).

    The returned callable applies one V-cycle, i.e.
    ``r -> A_screen^{-1} r``.  Because the Jacobian's matrix part is
    ``J ~ -A_screen``, the Newton update uses ``M r = J^{-1} r ~ -A_screen^{-1} r``
    (see :func:`make_phi`).

    Parameters
    ----------
    solvent_obj : NewtonPBE
        Solvent object with a built ``solver``.
    reaction : ndarray or None
        Reaction diagonal added to ``L / spacing**2`` (``None`` -> reaction = 0).

    Returns
    -------
    vcycle : callable
        Function ``(r, out) -> M^{-1} r`` (writes into ``out``).
    """
    spacing = solvent_obj.grids.spacing
    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = ngrids**3

    L = solvent_obj.L
    precond = L / spacing**2 + scipy.sparse.diags(drho_ions_scr)

    c_tot_ngrids = ctypes.c_int(tot_ngrids)
    c_indptr = precond.indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_indices = precond.indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    c_data = precond.data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    c_max_levels = ctypes.c_int(4) # 4 levels
    c_coarse_enough = ctypes.c_int(1000)
    c_relax = ctypes.c_int(0)
    c_npre = ctypes.c_int(1)
    c_npost = ctypes.c_int(1)
    c_ncycle = ctypes.c_int(1)
    c_pre_cycles = ctypes.c_int(1)

    hierarchy = libamgcl.amg_create(c_tot_ngrids, c_indptr, c_indices, c_data,
                                   c_max_levels, c_coarse_enough, c_relax,
                                   c_npre, c_npost, c_ncycle, c_pre_cycles)
    solvent_obj.hierarchy = hierarchy

    def vcycle(r, out):
        c_r = r.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        libamgcl.amg_vcycle(ctypes.c_void_p(hierarchy), c_r, c_out, c_tot_ngrids)
        return out

    return vcycle

def _release_caches(solvent_obj):
    """Release memory after make_phi"""
    if getattr(solvent_obj, 'hierarchy', None) is not None:
        hierarchy = solvent_obj.hierarchy
        libamgcl.amg_destroy(ctypes.c_void_p(hierarchy))
    solvent_obj.hierarchy = None

def make_phi(solvent_obj, phi_sol=None, rho_sol=None):
    if solvent_obj._intermediates is None: solvent_obj.build()
    _intermediates = solvent_obj._intermediates

    ngrids = solvent_obj.grids.ngrids
    tot_ngrids = solvent_obj.grids.get_ngrids()
    T = solvent_obj.T
    spacing = solvent_obj.grids.spacing
    cb = solvent_obj.cb * M2HARTREE

    eps = _intermediates['eps']
    lambda_r = _intermediates['lambda_r']
    grad_eps = _intermediates['grad_eps']

    max_cycle = solvent_obj.max_cycle # Newton cycle
    inner_max_cycle = solvent_obj.inner_max_cycle

    C_TST = 0.01
    p_TST = 1.0

    grad_lneps = grad_eps / eps[:, None]
    get_rho_ions = solvent_obj._gen_get_rho_ions()
    get_drho_ions = solvent_obj._gen_drho_ions()

    bc, const_src = solvent_obj._boundary_conditions(ngrids, spacing)

    inv_eps = 4.0 * PI / eps

    A = solvent_obj.make_operator(grad_lneps)

    if cb == 0.0:
        drho_ions_scr = numpy.zeros(tot_ngrids)
    else:
        drho_ions_scr = -inv_eps * get_drho_ions(solvent_obj, numpy.zeros(tot_ngrids), cb, lambda_r, T)

    precond = solvent_obj.make_precond(drho_ions_scr)

    def residual(phi_opt, out):
        phi_tot = phi_opt + bc
        rho_ions = get_rho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        if numpy.isnan(rho_ions).any():
            return None, rho_ions
        rho_tot = rho_sol + rho_ions
        A(phi_opt, out)
        out += inv_eps * rho_tot + const_src
        return out, rho_ions

    def finalize(phi_opt, rho_ions):
        rho_tot = rho_sol + rho_ions
        dphi = solvent_obj.gradient(phi_opt, ngrids, spacing)
        rho_iter = 0.25 / PI * (grad_lneps * dphi).sum(axis=1)
        rho_pol = (1.0 - eps) / eps * rho_tot + rho_iter
        phi_tot = phi_opt + bc
        return phi_tot, rho_ions, rho_pol

    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    logger.info(solvent_obj, ' |  Poisson-Boltzmann Solver with the Multigrid Scheme  |')
    logger.info(solvent_obj, ' -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')

    phi_opt = numpy.zeros(tot_ngrids, dtype=numpy.float64)
    res_old = numpy.empty(tot_ngrids, dtype=numpy.float64)
    res_new = numpy.empty(tot_ngrids, dtype=numpy.float64)
    res_outer, rho_ions = residual(phi_opt, res_new)

    if res_outer is None:
        logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
        return None, None, None

    fnorm = numpy.linalg.norm(res_outer)

    res_inner = numpy.empty(tot_ngrids, dtype=numpy.float64)
    dv = numpy.empty(tot_ngrids, dtype=numpy.float64)
    Av = numpy.empty(tot_ngrids, dtype=numpy.float64)
    b = numpy.empty(tot_ngrids, dtype=numpy.float64)

    t0 = (logger.process_clock(), logger.perf_counter())

    cycle = 0
    while cycle < max_cycle:
        phi_tot = phi_opt + bc
        drho_ions = get_drho_ions(solvent_obj, phi_tot, cb, lambda_r, T)
        jac_diag = inv_eps * drho_ions

        v = numpy.zeros(tot_ngrids, dtype=numpy.float64)
        b = numpy.negative(res_outer, out=b)

        res_inner[:] = b
        rnorm = fnorm
        inner_thresh = C_TST * fnorm**(1.0 + p_TST)
        inner = 0
        while inner < inner_max_cycle:
            if rnorm <= inner_thresh and rnorm < fnorm:
                break
            dv = precond(res_inner, dv)
            v -= dv
            Av = A(v, Av)
            dv = numpy.multiply(jac_diag, v, out=dv)
            res_inner = numpy.subtract(b, Av, out=res_inner)
            res_inner -= dv
            rnorm = numpy.linalg.norm(res_inner)
            inner += 1

        # Line search by backtracking.
        damping = 1.0
        accepted = False
        while damping > 1.0e-5:
            res_try, rho_ions_try = residual(phi_opt + damping * v, res_old)
            if res_try is not None:
                fnorm_try = numpy.linalg.norm(res_try)
                if fnorm_try < (1.0 - damping / 10**4) * fnorm:
                    accepted = True
                    break
            damping *= 0.5

        if not accepted:
            logger.warn(solvent_obj, 'Newton line search failed at cycle %d '
                        '(||F|| = %4.3e, inner = %d).', cycle + 1, fnorm, inner)
            res_try, rho_ions_try = residual(phi_opt + damping * v, res_old)
            if res_try is None:
                logger.info(solvent_obj, 'Skipping PBE due to infinite ion charge density.')
                return None, None, None
            fnorm_try = numpy.linalg.norm(res_try)

        phi_opt = phi_opt + damping * v
        res_outer, rho_ions = res_try, rho_ions_try
        fnorm = fnorm_try
        res_new, res_old = res_old, res_new   # accepted trial becomes current
        cycle += 1
        logger.info(solvent_obj, 'PBE Iteration %3d ||F|| = %4.3e inner = %2d damp = %4.3e',
                    cycle, fnorm, inner, damping)

        if fnorm < 1.0e-9:
            t0 = logger.timer(solvent_obj, 'phi_tot', *t0)
            return finalize(phi_opt, rho_ions)

    raise RuntimeError('PBE solver failed to converge.')


class NLPBE(ddcosmo.DDCOSMO):
    def __init__(self, mol, cb=0.0, cation_rad=4.3, anion_rad=4.3, T=298.15,
                 stern_mol=0.44, **kwargs):
        ddcosmo.DDCOSMO.__init__(self, mol)
        self.grids = Grids(mol, **kwargs)
        self.radii_table = VDW
        self.probe = 1.4
        self.stern_mol = stern_mol
        self.delta = 0.265
        self.cb = cb
        self.T = T
        self.cation_rad = cation_rad
        self.anion_rad = anion_rad

        self.phi_tot = None
        self.rho_ions = None

        self.inner_max_cycle = 200
        self.hierarchy = None
        self.operator = None
        self.precond = None
        self.kappa = None
        self.L = None

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'Probe radius = %.5f Angs', self.probe)
        logger.info(self, 'Dielectric constant of the solvent = %.5f', self.eps)
        logger.info(self, 'Broadening of the molecular Stern layer = %.5f Angs', self.delta)
        logger.info(self, 'Electrolyte concentration = %.5f mol/L', self.cb)
        logger.info(self, 'Temperature = %.5f Kelvin', self.T)
        logger.info(self, 'Box length = %.5f Angs', self.grids.length * BOHR)
        logger.info(self, 'Total grids = %d', self.grids.get_ngrids())

    def _get_vind(self, dm):
        if not self._intermediates or self.grids.coords is None:
            self.build()
        if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
            dm = dm[0] + dm[1]

        spacing = self.grids.spacing
        coords = self.grids.coords
        ngrids = self.grids.ngrids

        phi_sol = self.make_phi_sol(dm, coords)
        self.phi_sol = phi_sol
        rho_sol = self.make_rho_sol(phi_sol, ngrids, spacing)
        self.rho_sol = rho_sol
        phi_tot, rho_ions, rho_pol = self.make_phi(phi_sol, rho_sol)

        if phi_tot is None:
            return 0.0, numpy.zeros(dm.shape)

        self.phi_tot = phi_tot
        self.rho_ions = rho_ions

        phi_pol = phi_tot - phi_sol

        epbe = numpy.dot(rho_sol, phi_pol) * spacing**3

        # Dielectric contribution by Fisicaro
        epbe -= 0.5*(numpy.dot(rho_sol, phi_pol)
                     + numpy.dot(rho_ions, phi_tot)) * spacing**3

        # Osmotic pressure contribution
        cb = self.cb * M2HARTREE
        lambda_r = self._intermediates['lambda_r']
        T = self.T
        if self.cb == 0.0:
            pass
        else:
            epbe += self.energy_osm(phi_tot, cb, lambda_r, T, spacing)

        vmat = self._get_vmat(phi_pol)
        return epbe, vmat

    def _get_vmat(self, phi_pol):
        t0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        coords = self.grids.coords
        spacing = self.grids.spacing
        nao = mol.nao
        tot_ngrids = self.grids.get_ngrids()

        vmat = numpy.zeros([nao, nao], order='C')
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(min(max(max_memory*.9e6/8/nao/2, 400), tot_ngrids))
        buf = numpy.empty((blksize, nao), order='C')
        for p0, p1 in lib.prange(0, tot_ngrids, blksize):
            ao = mol.eval_gto('GTOval', coords[p0:p1])
            buf[:p1-p0] = ao * phi_pol[p0:p1, None]
            vmat -= numpy.dot(buf[:p1-p0].T, ao)
        vmat *= spacing**3
        t0 = logger.timer(self, 'v_diel', *t0)
        return vmat

    def energy_osm(self, phi_tot=None, cb=None, lambda_r=None, T=None, spacing=None):
        if phi_tot is None: phi_tot = self.phi_tot
        if cb is None: cb = self.cb * M2HARTREE
        if lambda_r is None: lambda_r = self._intermediates['lambda_r']
        if T is None: T = self.T
        if spacing is None: spacing = self.grids.spacing
        return energy_osm_one_to_one(self, phi_tot, cb, lambda_r, T, spacing)

    def build(self, auxbasis=None):
        if self.grids.coords is None:
            self.grids.build()

        mol = self.mol
        coords = self.grids.coords
        ngrids = self.grids.ngrids
        probe = self.probe / BOHR # angstrom to a.u.
        stern_mol = self.stern_mol / BOHR # angstrom to a.u.
        delta = self.delta / BOHR # angstrom to a.u.
        atomic_radii = self.get_atomic_radii()
        eps_bulk = self.eps
        cb = self.cb * M2HARTREE # mol/L to a.u.

        lambda_r = self.make_lambda(mol, probe, stern_mol, coords, delta, atomic_radii)
        sas = self.make_sas(mol, probe, coords, delta, atomic_radii)
        grad_sas = self.make_grad_sas(mol, probe, coords, delta, atomic_radii)
        eps = self.make_eps(eps_bulk, sas)
        grad_eps = self.make_grad_eps(eps_bulk, grad_sas)

        if self.L is None:
            import fcdft.solvent.calculus_helper as ch
            self.L = ch.poisson((ngrids,)*3, format='csr')

        self.kappa = numpy.sqrt(8.0e0 * PI * cb / self.eps / KB2HARTREE / self.T)
        if auxbasis is None: auxbasis = 'def2-universal-jkfit'
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        erifile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
        int2c2e = auxmol.intor('int2c2e')
        int3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s2kl')
        
        with h5py.File(erifile, 'w') as feri:
            feri.create_dataset('int2c2e', data=int2c2e)
            feri.create_dataset('int3c2e', data=int3c2e)

        self._intermediates = {
            'grids': self.grids.coords,
            'lambda_r': lambda_r,
            'eps': eps,
            'grad_eps': grad_eps,
            'sas': sas,
            'grad_sas': grad_sas,
            'auxmol': auxmol,
            'erifile': erifile,
        }

    def _boundary_conditions(self, ngrids, spacing):
        "A hooker for boundary conditions"
        return 0.0, 0.0

    def _gen_get_rho_ions(self):
        return rho_ions_one_to_one

    def _gen_drho_ions(self):
        return drho_ions_one_to_one

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None

        _release_caches(self)
        self.hierarchy = None
        self.operator = None
        self.precond = None
        return self

    gradient = gradient
    make_lambda = make_lambda
    make_sas = make_sas
    make_grad_sas = make_grad_sas
    make_eps = make_eps
    make_grad_eps = make_grad_eps
    make_phi_sol = make_phi_sol
    make_rho_sol = make_rho_sol
    make_phi = make_phi
    make_operator = make_operator
    make_precond = make_precond
    make_phi = make_phi

class Grids(cubegen.Cube):
    def __init__(self, mol, ngrids=97, length=20):
        self.mol = mol
        self.ngrids=ngrids
        self.alignment = 0
        self.length = length / BOHR
        self.spacing = None
        self.coords = None
        self.verbose = mol.verbose
        self.center = None
        super().__init__(mol, nx=ngrids, ny=ngrids, nz=ngrids, margin=self.length/2.0,
                         extent=[self.length, self.length, self.length])
        
    def get_coords(self):
        atom_coords = self.mol.atom_coords()
        self.center = (atom_coords.max(axis=0) + atom_coords.min(axis=0)) / 2.0
        xs, ys, zs = self.xs, self.ys, self.zs
        frac_coords = lib.cartesian_prod([xs, ys, zs])
        box_center = self.box.sum(axis=1) / 2.0
        return frac_coords @ self.box + (self.center - box_center)

    def dump_flags(self, verbose=None):
        logger.info(self, 'Grid spacing = %.5f Angstrom', self.grids.spacing * BOHR)

    def build(self, mol=None, *args, **kwargs):
        if mol is None: mol = self.mol
        self.coords = self.get_coords()
        self.boxorig = self.coords[0]
        self.spacing = self.length / (self.nx - 1)
        return self

if __name__=='__main__':
    from pyscf import gto
    mol = gto.M(
        atom='''
    C       -1.1367537947      0.1104289172      2.4844663896
    C       -1.1385831318      0.1723328088      3.8772156394
    C        0.0819843127      0.0788096973      1.7730802291
    H       -2.0846565855      0.1966185690      4.4236084687
    C        0.0806058727      0.2041086872      4.5921211233
    C        1.2993389981      0.1104289172      2.4844663896
    H        2.2526138470      0.0865980845      1.9483127672
    C        1.2994126658      0.1723829840      3.8783367991
    H        2.2453411518      0.1966879024      4.4251589385
    H       -2.0869454458      0.0863720324      1.9432143952
    C        0.0810980584      0.2676328718      6.0213144069
    N        0.0819851974      0.3199013851      7.1972568519
    S        0.0000000000      0.0000000000      0.0000000000
    H        1.3390319419     -0.0095801980     -0.2157234144''',
        charge=0, basis='6-31++g**', verbose=5)
    mf = mol.RKS(xc='pbe')
    cm = NLPBE(mol, cb=1.0, length=20, ngrids=97)
    solmf = pbe_for_scf(mf, cm)
    solmf.kernel()

