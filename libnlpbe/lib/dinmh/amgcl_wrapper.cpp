// amgcl_wrapper.cpp
//
// Minimal C ABI over the header-only amgcl library, used as a multigrid
// V-cycle operator inside the Newton PBE solver (fcdft/solvent/newton.py).
//
// Design
// ------
//   * builtin (OpenMP) backend, smoothed-aggregation coarsening.
//   * Parameters are set through the plain AMG::params struct, NOT through
//     boost::property_tree; this avoids the boost runtime path that segfaults
//     when fcdft/pyamgcl is loaded.
//   * The hierarchy is built once and cached on the Python side.  When only the
//     reaction diagonal changes (same sparsity pattern) it can be refreshed
//     cheaply with pbe_amg_rebuild, which reuses the smoothed-aggregation
//     transfer operators and only recomputes the Galerkin coarse operators and
//     smoothers.
//   * Python owns the defect-correction (Richardson) loop and the Holst
//     Alg. 7 alternative-TST stopping test, so the *true* matrix-free Jacobian
//     governs convergence; this wrapper only applies V-cycles (Holst Alg. 3:
//     pre-smooth / coarse-correct / post-smooth, direct coarse solve).
//
// The smoother is chosen at run time among two compile-time instantiations so
// it can be tuned from Python without rebuilding:
//     relax = 0 -> SPAI(0)        (fully parallel, weak)
//     relax = 1 -> Gauss-Seidel   (multicolor-parallel, stronger)
// The two instantiations are hidden behind a small abstract base (Hierarchy) so
// the create/rebuild/vcycle/destroy entry points do not each branch on relax.


#include <vector>
#include <amgcl/util.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

#if defined(__GLIBC__)
#include <malloc.h>   // malloc_trim: hand freed heap back to the OS
#endif

// Force glibc's malloc to return free chunks to the OS so a destroyed
// AMG hierarchy actually shrinks process RSS. Without this, freed blocks
// remain in the per-thread arenas (lib.current_memory()[0] stays high)
// and downstream code that gates blksize on remaining max_memory collapses.
static inline void release_to_os()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

typedef amgcl::backend::builtin<double> Backend;

template <template <class> class Relax>
using AMG = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, Relax>;

typedef AMG<amgcl::relaxation::spai0>        AMG_spai0;
typedef AMG<amgcl::relaxation::gauss_seidel> AMG_gs;

struct Hierarchy {
    // virtual void rebuild(double* data) = 0;   // val has nnz entries
    virtual void vcycle(double* rhs, double* x) = 0;
    virtual ~Hierarchy() {}
};

template <class Amg>
struct HierarchyImpl : Hierarchy {
    int tot_ngrids;
    std::vector<int> indptr;
    std::vector<int> indices;
    Amg amg;

    HierarchyImpl(int tot_ngrids_, int* indptr_, int *indices_, std::vector<double> data,
                  const typename Amg::params& prm)
        : tot_ngrids(tot_ngrids_), indptr(indptr_, indptr_ + tot_ngrids_ + 1), indices(indices_, indices_ + indptr_[tot_ngrids_]),
          amg(std::tie(tot_ngrids, indptr, indices, data), prm)
    {
        // amgcl has internalized ptr/col into its own representation; we keep
        // them only as the public sparsity pattern for rebuild(). Drop any
        // reserved-but-unused capacity so this Hierarchy holds the minimum.
        indptr.shrink_to_fit();
        indices.shrink_to_fit();
    }

    // void rebuild(double* data) override {
    //     std::vector<double> v(data, data + indptr[tot_ngrids]);   // ptr[n] == nnz
    //     amg.rebuild(std::tie(tot_ngrids, indptr, indices, v));
    // }

    void vcycle(double *rhs, double *x) override {
        auto r  = amgcl::make_iterator_range(rhs, rhs + tot_ngrids);
        auto xx = amgcl::make_iterator_range(x,   x   + tot_ngrids);
        amg.apply(r, xx);
    }
};

template <class Amg>
static void set_params(typename Amg::params& prm, int max_levels,
                       int coarse_enough, int npre, int npost, int ncycle,
                       int pre_cycles) {
    prm.direct_coarse = true;                 // coarsest level: direct solve
    prm.allow_rebuild = true;                 // enable cheap rebuild()
    if (max_levels > 0)    prm.max_levels    = static_cast<unsigned>(max_levels);
    if (coarse_enough > 0) prm.coarse_enough = static_cast<unsigned>(coarse_enough);
    if (npre  >= 0)        prm.npre   = static_cast<unsigned>(npre);
    if (npost >= 0)        prm.npost  = static_cast<unsigned>(npost);
    if (ncycle > 0)        prm.ncycle = static_cast<unsigned>(ncycle);
    if (pre_cycles >= 0)   prm.pre_cycles = static_cast<unsigned>(pre_cycles);
}

template <class Amg>
static Hierarchy* build(int tot_ngrids, int *indptr, int* indices, double *data,
                        int max_levels, int coarse_enough, int npre, int npost,
                        int ncycle, int pre_cycles) {
    typename Amg::params prm;
    set_params<Amg>(prm, max_levels, coarse_enough, npre, npost, ncycle, pre_cycles);
    std::vector<double> v(data, data + indptr[tot_ngrids]);   // amgcl copies the values
    return new HierarchyImpl<Amg>(tot_ngrids, indptr, indices, std::move(v), prm);
}

extern "C" {
    // Create amg hirarchy
    void* amg_create(int tot_ngrids, int *indptr, int *indices, double *data,
                    int max_levels, int coarse_enough, int relax, 
                    int npre, int npost, int ncycle, int pre_cycles) {
        Hierarchy* h;
        if (relax == 1)
            h = build<AMG_gs>   (tot_ngrids, indptr, indices, data, max_levels, coarse_enough,
                                npre, npost, ncycle, pre_cycles);
        else
            h = build<AMG_spai0>(tot_ngrids, indptr, indices, data, max_levels, coarse_enough,
                                npre, npost, ncycle, pre_cycles);
        return static_cast<void*>(h);
    }

    // V-cycle
    void amg_vcycle(void* h, double *rhs, double *x, int tot_ngrids) {
        (void)tot_ngrids;
        static_cast<Hierarchy*>(h)->vcycle(rhs, x);
    }

    // Destroy hirarchy and retrive memory.
    void amg_destroy(void* h) {
        delete static_cast<Hierarchy*>(h);
        release_to_os();
    }

    // Av = out
    void csr_matvec(int *indptr, int *indices, double *data, double *v, double *out, int n) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double s = 0.0;
            int end = indptr[i + 1];
            for (int j = indptr[i]; j < end; j++) {
                s += data[j] * v[indices[j]];
            }
            out[i] = s;
        }
    }
}