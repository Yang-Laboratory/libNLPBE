import numpy
import scipy
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.lib import logger

def make_phi_sol(solvent_obj, dm=None, coords=None):
    t0 = (logger.process_clock(), logger.perf_counter())
    if dm is None: dm = solvent_obj._dm
    if coords is None: coords = solvent_obj.grids.coords
    
    mol = solvent_obj.mol
    nao = mol.nao
    ngrids = solvent_obj.grids.ngrids

    atom_coords = mol.atom_coords()
    Z = mol.atom_charges()
    dist = scipy.spatial.distance.cdist(atom_coords, coords)
    dist[dist < 1.0e-100] = numpy.inf # Machine precision
    Vnuc = numpy.tensordot(1.0e0 / dist, Z, axes=([0], [0]))

    if dm.ndim == 3: # Spin-unrestricted
        dm = dm[0] + dm[1]

    dms = numpy.asarray(dm.real)
    Vele = numpy.empty(ngrids**3, order='C')
    auxmol = df.addons.make_auxmol(mol)
    PQ = auxmol.intor('int2c2e')
    nao = mol.nao
    naux = auxmol.nao

    dm_tril = lib.pack_tril(dms + dms.T)
    idx = numpy.arange(nao)
    idx = idx * (idx + 1) // 2 + idx
    dm_tril[idx] *= 0.5
    int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e', aosym='s2kl')
    g = dm_tril.dot(int3c)

    # Cholesky solve the RI equation
    PQ = scipy.linalg.cho_factor(PQ)
    cK = scipy.linalg.cho_solve(PQ, g)

    Vele = numpy.empty(ngrids**3, order='C')
    max_memory = solvent_obj.max_memory - lib.current_memory()[0] - Vele.nbytes*1e-6
    blksize = int(max(max_memory*.9e6/8/naux, 400))
    for p0, p1 in lib.prange(0, ngrids**3, blksize):
        fakemol = gto.fakemol_for_charges(coords[p0:p1])
        ints = gto.intor_cross('int2c2e', fakemol, auxmol)
        Vele[p0:p1] = ints.dot(cK)
        del ints

    MEP = Vnuc - Vele
    t0 = logger.timer(solvent_obj, 'phi_sol', *t0)
    return lib.tag_array(MEP, Vnuc=Vnuc, Vele=-Vele)

