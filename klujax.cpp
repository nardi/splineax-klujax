// version: 0.5.0
// Imports

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "klu.h"
#include "pybind11/pybind11.h"
#include "xla/ffi/api/ffi.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

ffi::Error validate_args(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions ds_Ax,
    const ffi::AnyBuffer::Dimensions ds_x) {
    int d_x = ds_x.size();
    if (d_x != 3) {
        return ffi::Error::InvalidArgument("x is not 3D.");
    }
    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];

    int d_Ax = ds_Ax.size();
    if (d_Ax != 2) {
        return ffi::Error::InvalidArgument("Ax is not 2D.");
    }
    int n_lhs_bis = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];

    if (n_lhs != n_lhs_bis) {
        return ffi::Error::InvalidArgument(
            "n_lhs mismatch: Ax.shape[0] != x.shape[0]: Got " + std::to_string(n_lhs_bis) + " != " + std::to_string(n_lhs));
    }

    auto ds_Ai = Ai.dimensions();
    int d_Ai = ds_Ai.size();
    if (d_Ai != 1) {
        return ffi::Error::InvalidArgument("Ai is not 1D.");
    }
    int n_nz_bis = (int)ds_Ai[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_nz mismatch: Ai.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    auto ds_Aj = Aj.dimensions();
    int d_Aj = ds_Aj.size();
    if (d_Aj != 1) {
        return ffi::Error::InvalidArgument("Aj is not 1D.");
    }
    n_nz_bis = (int)ds_Aj[0];
    if (n_nz != n_nz_bis) {
        return ffi::Error::InvalidArgument(
            "n_nz mismatch: Aj.shape[0] != Ax.shape[1]: Got " + std::to_string(n_nz_bis) + " != " + std::to_string(n_nz));
    }

    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int n = 0; n < n_nz; n++) {
        i = _Ai[n];
        if (i < 0) {
            return ffi::Error::InvalidArgument("Ai contains negative index");
        }
        if (i >= n_col) {
            return ffi::Error::InvalidArgument("Ai.max() >= n_col");
        }
        j = _Aj[n];
        if (j < 0) {
            return ffi::Error::InvalidArgument("Aj contains negative index");
        }
        if (j >= n_col) {
            return ffi::Error::InvalidArgument("Aj.max() >= n_col");
        }
    }
    return ffi::Error::Success();
}

void coo_to_csc_analyze(
    const int n_col,
    const int n_nz,
    const int* Ai,
    const int* Aj,
    int* Bi,
    int* Bp,
    int* Bk) {
    // compute number of non-zero entries per row of A
    for (int n = 0; n < n_nz; n++) {
        Bp[Aj[n]] += 1;
    }

    // cumsum the n_nz per row to get Bp
    int cumsum = 0;
    int temp = 0;
    for (int j = 0; j <= n_col; j++) {
        temp = Bp[j];
        Bp[j] = cumsum;
        cumsum += temp;
    }

    // write Ai, Aj into Bi, Bk
    int col = 0;
    int dest = 0;
    for (int n = 0; n < n_nz; n++) {
        col = Aj[n];
        dest = Bp[col];
        Bi[dest] = Ai[n];
        Bk[dest] = n;
        Bp[col] += 1;
    }

    int last = 0;
    for (int i = 0; i <= n_col; i++) {
        temp = Bp[i];
        Bp[i] = last;
        last = temp;
    }
}

using Complex = std::complex<double>;

template <typename T>
struct KluTraits;

template <>
struct KluTraits<double> {
    static klu_numeric* factor(int* Ap, int* Ai, double* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_factor(Ap, Ai, Ax, Symbolic, Common);
    }
    static int refactor(int* Ap, int* Ai, double* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_refactor(Ap, Ai, Ax, Symbolic, Numeric, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, double* B, klu_common* Common) {
        return klu_solve(Symbolic, Numeric, d, nrhs, B, Common);
    }
    static int tsolve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, double* B, klu_common* Common) {
        return klu_tsolve(Symbolic, Numeric, d, nrhs, B, Common);
    }
    static int rcond(klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_rcond(Symbolic, Numeric, Common);
    }
    static int condest(int* Ap, double* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_condest(Ap, Ax, Symbolic, Numeric, Common);
    }
};

template <>
struct KluTraits<Complex> {
    static klu_numeric* factor(int* Ap, int* Ai, Complex* Ax, klu_symbolic* Symbolic, klu_common* Common) {
        return klu_z_factor(Ap, Ai, reinterpret_cast<double*>(Ax), Symbolic, Common);
    }
    static int refactor(int* Ap, int* Ai, Complex* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_z_refactor(Ap, Ai, reinterpret_cast<double*>(Ax), Symbolic, Numeric, Common);
    }
    static int solve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, Complex* B, klu_common* Common) {
        return klu_z_solve(Symbolic, Numeric, d, nrhs, reinterpret_cast<double*>(B), Common);
    }
    // conj_solve=0 gives plain transpose A^T; use 1 for conjugate transpose A^H
    static int tsolve(klu_symbolic* Symbolic, klu_numeric* Numeric, int d, int nrhs, Complex* B, klu_common* Common) {
        return klu_z_tsolve(Symbolic, Numeric, d, nrhs, reinterpret_cast<double*>(B), 0, Common);
    }
    static int rcond(klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_z_rcond(Symbolic, Numeric, Common);
    }
    static int condest(int* Ap, Complex* Ax, klu_symbolic* Symbolic, klu_numeric* Numeric, klu_common* Common) {
        return klu_z_condest(Ap, reinterpret_cast<double*>(Ax), Symbolic, Numeric, Common);
    }
};

// Fills a batch element's output slice with NaN, used when KLU fails to factor/solve
// a singular matrix. This mirrors how LAPACK-backed solvers return NaN on singular
// systems instead of raising, letting callers detect failure from the output values.
template <typename T>
void fill_nan(T* dst, int n);

template <>
void fill_nan<double>(double* dst, int n) {
    std::fill(dst, dst + n, std::numeric_limits<double>::quiet_NaN());
}

template <>
void fill_nan<Complex>(Complex* dst, int n) {
    std::fill(dst, dst + n,
              Complex(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()));
}

// Handle cache ================================================================
//
// A handle is a uint64 id into this process-global cache, never a raw pointer,
// so a stale id can never dereference freed memory. Each entry owns its KLU
// objects and frees them when dropped. A symbolic entry owns a klu_symbolic.
// A numeric entry also owns a klu_numeric and shares (ref-counts) the symbolic
// it was factored against, so analysis is reused, not recomputed.
//
// The cache is bounded: the least recently used entry is evicted once it grows
// past KLUJAX_FACTOR_CACHE, so forgetting to free leaks only a bounded amount.
// A call arriving with an id no longer resident (evicted or freed) rebuilds the
// state from the Ai/Aj/Ax the caller carries, so use of a freed handle is safe.

struct SymbolicObj {
    klu_symbolic* S = nullptr;
    int n_col = 0;
    ~SymbolicObj() {
        if (S) {
            klu_common c;
            klu_defaults(&c);
            klu_free_symbolic(&S, &c);
        }
    }
};

struct CacheEntry {
    std::shared_ptr<SymbolicObj> symbolic;  // always set
    klu_numeric* numeric = nullptr;         // null for a symbolic-only entry
    ~CacheEntry() {
        if (numeric) {
            klu_common c;
            klu_defaults(&c);
            klu_free_numeric(&numeric, &c);
        }
    }
};

// The registry and its LRU order. Every public method takes `mu`, so callers
// look up under the lock but do the heavy KLU work outside it (see resolve_*).
struct CacheRegistry {
    std::mutex mu;
    std::map<uint64_t, std::shared_ptr<CacheEntry>> entries;
    std::list<uint64_t> lru;  // front = most recently used
    std::atomic<uint64_t> next_id{1};
    std::atomic<long> rebuilds{0};

    static CacheRegistry& instance() {
        static CacheRegistry r;
        return r;
    }

    // Cache size, from KLUJAX_FACTOR_CACHE, defaulting to 8 live handles.
    static size_t capacity() {
        const char* e = std::getenv("KLUJAX_FACTOR_CACHE");
        if (e) {
            char* end = nullptr;
            long v = std::strtol(e, &end, 10);
            if (end != e && v > 0) return static_cast<size_t>(v);
        }
        return 8;
    }

    // Strict mode turns a rebuild into an error, for finding lost handles.
    static bool strict() {
        const char* e = std::getenv("KLUJAX_STRICT_CACHE");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }

    // caller holds mu. 0 and live ids are skipped so an id stays unique.
    uint64_t fresh_id() {
        uint64_t id;
        do {
            id = next_id.fetch_add(1);
        } while (id == 0 || entries.count(id));
        return id;
    }
    void touch(uint64_t id) {
        lru.remove(id);
        lru.push_front(id);
    }
    void evict() {
        size_t cap = capacity();
        while (entries.size() > cap && !lru.empty()) {
            uint64_t victim = lru.back();
            lru.pop_back();
            entries.erase(victim);
        }
    }
    // Register entry under id (0 mints a fresh one), evicting overflow.
    uint64_t insert(std::shared_ptr<CacheEntry> e, uint64_t id = 0) {
        std::lock_guard<std::mutex> lk(mu);
        if (id == 0) id = fresh_id();
        entries[id] = std::move(e);
        touch(id);
        evict();
        return id;
    }
    std::shared_ptr<CacheEntry> lookup(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = entries.find(id);
        if (it == entries.end()) return nullptr;
        touch(id);
        return it->second;
    }
    void erase(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        entries.erase(id);
        lru.remove(id);
    }
};

// Build a fresh symbolic analysis from a COO pattern. Returns null on failure.
std::shared_ptr<SymbolicObj> build_symbolic(const int* _Ai, const int* _Aj, int n_nz, int n_col) {
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bk = std::make_unique<int[]>(n_nz);
    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());
    klu_common Common;
    klu_defaults(&Common);
    klu_symbolic* S = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);
    if (!S) return nullptr;
    auto obj = std::make_shared<SymbolicObj>();
    obj->S = S;
    obj->n_col = n_col;
    return obj;
}

// Factor one left-hand side against a symbolic. Returns null on failure.
template <typename T>
klu_numeric* build_numeric(SymbolicObj& sym, const int* _Ai, const int* _Aj, int n_nz, const T* _Ax_i) {
    int n_col = sym.n_col;
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bx = std::make_unique<T[]>(n_nz);
    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());
    for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
    klu_common Common;
    klu_defaults(&Common);
    klu_numeric* N = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), sym.S, &Common);
    if (N == nullptr || Common.status < KLU_OK) {
        if (N) klu_free_numeric(&N, &Common);
        return nullptr;
    }
    return N;
}

// Resolve a symbolic id, rebuilding from the caller's pattern on a miss. The
// heavy klu_analyze runs outside the registry lock. Sets `err` and returns null
// only on strict-mode miss or an analyze failure.
std::shared_ptr<SymbolicObj> resolve_symbolic(uint64_t id, const int* _Ai, const int* _Aj, int n_nz,
                                              int n_col, ffi::Error& err) {
    auto& r = CacheRegistry::instance();
    if (auto e = r.lookup(id)) return e->symbolic;
    if (CacheRegistry::strict()) {
        err = ffi::Error::Internal("klujax: symbolic handle " + std::to_string(id) +
                                   " was evicted or freed and strict cache mode is on");
        return nullptr;
    }
    auto sym = build_symbolic(_Ai, _Aj, n_nz, n_col);
    if (!sym) {
        err = ffi::Error::Internal("klujax: rebuild of symbolic handle failed (klu_analyze)");
        return nullptr;
    }
    r.rebuilds.fetch_add(1);
    auto e = std::make_shared<CacheEntry>();
    e->symbolic = sym;
    r.insert(e, id);  // heal under the same id so later calls hit
    return sym;
}

// Resolve one numeric id, rebuilding analysis and factorization from the
// caller's matrix on a miss. Returns the whole entry so the solver can use its
// self-consistent symbolic/numeric pair.
template <typename T>
std::shared_ptr<CacheEntry> resolve_numeric(uint64_t id, const int* _Ai, const int* _Aj, int n_nz,
                                            int n_col, const T* _Ax_i, ffi::Error& err,
                                            bool* rebuilt = nullptr) {
    if (rebuilt) *rebuilt = false;
    auto& r = CacheRegistry::instance();
    if (auto e = r.lookup(id)) {
        if (e->numeric) return e;
    }
    if (rebuilt) *rebuilt = true;
    if (CacheRegistry::strict()) {
        err = ffi::Error::Internal("klujax: numeric handle " + std::to_string(id) +
                                   " was evicted or freed and strict cache mode is on");
        return nullptr;
    }
    auto sym = build_symbolic(_Ai, _Aj, n_nz, n_col);
    if (!sym) {
        err = ffi::Error::Internal("klujax: rebuild of numeric handle failed (klu_analyze)");
        return nullptr;
    }
    klu_numeric* N = build_numeric<T>(*sym, _Ai, _Aj, n_nz, _Ax_i);
    if (!N) {
        err = ffi::Error::Internal("klujax: rebuild of numeric handle failed (klu_factor)");
        return nullptr;
    }
    auto e = std::make_shared<CacheEntry>();
    e->symbolic = sym;
    e->numeric = N;
    r.rebuilds.fetch_add(1);
    r.insert(e, id);
    return e;
}

template <typename T>
ffi::Error dot_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const T* _Ax,
    const T* _x,
    T* _b) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_x);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_x[0];
    int n_col = (int)ds_x[1];
    int n_rhs = (int)ds_x[2];
    int n_nz = (int)ds_Ax[1];

    // initialize empty result
    for (int i = 0; i < n_lhs * n_col * n_rhs; i++) {
        _b[i] = 0.0;
    }

    // fill result (all multi-dim arrays are row-major)
    // x_mik = A_mij × x_mjk (einsum)
    // sizes: m<n_lhs; i<n_col<--Ai; j<n_col<--Aj; k<n_rhs
    // Loop order: m (batch) outer for better cache locality on Ax
    int i;
    int j;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_nz; n++) {
            i = _Ai[n];
            j = _Aj[n];
            for (int k = 0; k < n_rhs; k++) {
                _b[m * n_col * n_rhs + i * n_rhs + k] += _Ax[m * n_nz + n] * _x[m * n_col * n_rhs + j * n_rhs + k];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error dot_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> x,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> b) {
    return dot_impl<double>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                            Ax.typed_data(), x.typed_data(), b->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_f64_handler, dot_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // b
);

ffi::Error dot_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> x,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> b) {
    return dot_impl<Complex>(Ai, Aj, Ax.dimensions(), x.dimensions(),
                             reinterpret_cast<const Complex*>(Ax.typed_data()),
                             reinterpret_cast<const Complex*>(x.typed_data()),
                             reinterpret_cast<Complex*>(b->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    dot_c128_handler, dot_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // x
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // b
);

template <typename T>
ffi::Error solve_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const T* _Ax,
    const T* _b,
    T* _x) {
    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    // _b itself won't be used anymore. KLU works on _x_temp in-place.
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    // initialize KLU for given sparsity pattern
    klu_symbolic* Symbolic;
    klu_numeric* Numeric;
    klu_common Common;
    klu_defaults(&Common);
    Symbolic = klu_analyze(n_col, _Bp.get(), _Bi.get(), &Common);

    // solve for all elements in batch:
    // NOTE: same sparsity pattern for each element in batch assumed
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU. A singular matrix (or otherwise failed factor/solve) fills this
        // batch element's output with NaN instead of aborting the whole call, mirroring how
        // LAPACK-backed solvers behave on a singular system.
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            if (Numeric != nullptr) {
                klu_free_numeric(&Numeric, &Common);
            }
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose the last two dimensions since JAX expects row-major layout
    // NOTE: it feels a bit weird to have to do all this copying and transposing here. This might actually be
    // pretty inefficient. Ideally I'd like to get rid of this transpose. Maybe just represent b/x in python
    // as n_lhs x n_rhs x n_col in stead of n_lhs x n_col x n_rhs?
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    klu_free_symbolic(&Symbolic, &Common);
    return ffi::Error::Success();
}

ffi::Error solve_f64(
    ffi::Buffer<ffi::DataType::S32> Ai,
    ffi::Buffer<ffi::DataType::S32> Aj,
    ffi::Buffer<ffi::DataType::F64> Ax,
    ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                              Ax.typed_data(), b.typed_data(), x->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_f64_handler, solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

ffi::Error solve_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(),
                               reinterpret_cast<const Complex*>(Ax.typed_data()),
                               reinterpret_cast<const Complex*>(b.typed_data()),
                               reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(  // b = A x
    solve_c128_handler, solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error solve_with_symbol_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_id = *symbolic.typed_data();

    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Resolve the analysis by id, rebuilding it from Ai/Aj if it was evicted.
    auto _sym = resolve_symbolic(sym_id, _Ai, _Aj, n_nz, n_col, err);
    if (!_sym) return err;
    klu_symbolic* Symbolic = _sym->S;

    // get COO -> CSC transformation information (using RAII for automatic cleanup)
    auto _Bk = std::make_unique<int[]>(n_nz);  // Ax -> Bx transformation indices
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // copy _b into _x_temp and transpose the last two dimensions since KLU expects col-major layout
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    klu_numeric* Numeric;
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        // convert COO Ax to CSC Bx
        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // solve using KLU with provided Symbolic handle. A singular matrix (or otherwise
        // failed factor/solve) fills this batch element's output with NaN instead of
        // aborting the whole call, mirroring how LAPACK-backed solvers behave.
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            if (Numeric != nullptr) {
                klu_free_numeric(&Numeric, &Common);
            }
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        KluTraits<T>::solve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        klu_free_numeric(&Numeric, &Common);
    }

    // copy _x_temp into _x and transpose
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error solve_with_symbol_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                          Ax.typed_data(), b.typed_data(), x->typed_data());
}

ffi::Error solve_with_symbol_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           reinterpret_cast<const Complex*>(Ax.typed_data()),
                                           reinterpret_cast<const Complex*>(b.typed_data()),
                                           reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_f64_handler, solve_with_symbol_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_c128_handler, solve_with_symbol_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

// tsolve_with_symbol: identical to solve_with_symbol_impl but uses KluTraits<T>::tsolve
// Solves A^T x = b (transpose solve) reusing a pre-computed symbolic factorization.
template <typename T>
ffi::Error tsolve_with_symbol_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    const T* _b,
    T* _x) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_id = *symbolic.typed_data();

    ffi::Error err = validate_args(Ai, Aj, ds_Ax, ds_b);
    if (err.failure()) {
        return err;
    }

    int n_lhs = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];
    int n_nz = (int)ds_Ax[1];
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Resolve the analysis by id, rebuilding it from Ai/Aj if it was evicted.
    auto _sym = resolve_symbolic(sym_id, _Ai, _Aj, n_nz, n_col, err);
    if (!_sym) return err;
    klu_symbolic* Symbolic = _sym->S;

    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    klu_numeric* Numeric;
    for (int i = 0; i < n_lhs; i++) {
        int m = i * n_nz;
        int n = i * n_rhs * n_col;

        for (int k = 0; k < n_nz; k++) {
            _Bx[k] = _Ax[m + _Bk[k]];
        }

        // A singular matrix (or otherwise failed factor/solve) fills this batch element's
        // output with NaN instead of aborting the whole call, mirroring how LAPACK-backed
        // solvers behave.
        Numeric = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), Symbolic, &Common);
        if (Numeric == nullptr || Common.status < KLU_OK) {
            if (Numeric != nullptr) {
                klu_free_numeric(&Numeric, &Common);
            }
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        // NOTE: tsolve instead of solve
        KluTraits<T>::tsolve(Symbolic, Numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            klu_free_numeric(&Numeric, &Common);
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }
        klu_free_numeric(&Numeric, &Common);
    }

    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error tsolve_with_symbol_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return tsolve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           Ax.typed_data(), b.typed_data(), x->typed_data());
}

ffi::Error tsolve_with_symbol_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return tsolve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                            reinterpret_cast<const Complex*>(Ax.typed_data()),
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_f64_handler, tsolve_with_symbol_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_c128_handler, tsolve_with_symbol_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

template <typename T>
ffi::Error factor_impl(
    int64_t n_col_attr,
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const T* _Ax,
    uint64_t* _numeric) {
    if (symbolic.element_count() != 1) return ffi::Error::InvalidArgument("symbolic must be scalar");
    uint64_t sym_id = *symbolic.typed_data();

    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = (int)n_col_attr;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Resolve the analysis (rebuilding it if it was evicted) and reuse it for
    // every left-hand side. Each numeric shares that symbolic, so the split
    // API keeps its "analyze once" benefit.
    ffi::Error err = ffi::Error::Success();
    auto sym = resolve_symbolic(sym_id, _Ai, _Aj, n_nz, n_col, err);
    if (!sym) return err;

    auto& registry = CacheRegistry::instance();
    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        klu_numeric* Numeric = build_numeric<T>(*sym, _Ai, _Aj, n_nz, _Ax_i);
        if (Numeric == nullptr) {
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        // A fresh id names one immutable factorization: this is the data edge
        // XLA orders solves against, and why no optimization_barrier is needed.
        auto entry = std::make_shared<CacheEntry>();
        entry->symbolic = sym;
        entry->numeric = Numeric;
        _numeric[i] = registry.insert(entry);
    }
    return ffi::Error::Success();
}

ffi::Error factor_f64(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, Ax.typed_data(), numeric->typed_data());
}

ffi::Error factor_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> numeric) {
    return factor_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic,
                                reinterpret_cast<const Complex*>(Ax.typed_data()),
                                numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_f64_handler, factor_f64,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::F64>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    factor_c128_handler, factor_c128,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()
        .Arg<ffi::Buffer<ffi::DataType::C128>>()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()
        .Ret<ffi::Buffer<ffi::DataType::U64>>());

// Shared by refactor and refactor_status. With `_out_status == nullptr` a failed refactor
// aborts the call with an FFI error (the original refactor behaviour). Otherwise the KLU
// status of every batch element is written out and the call succeeds, so a caller can
// branch on it under jit. Failures that are not a property of the matrix (null handle,
// shape mismatch) stay FFI errors either way.
template <typename T>
ffi::Error refactor_impl(
    int64_t n_col_attr,
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    uint64_t* _out_numeric,
    int32_t* _out_status = nullptr) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = (int)n_col_attr;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");
    if ((int)numeric.element_count() != n_lhs) return ffi::Error::InvalidArgument("numeric array size must match n_lhs");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    // Shared CSC pattern, used by the in-place refactor of resident entries.
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        ffi::Error err = ffi::Error::Success();
        bool rebuilt = false;
        auto entry = resolve_numeric<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Ax_i, err, &rebuilt);
        if (!entry) return err;

        // A rebuilt entry was just factored with these values, so it is already
        // the refactored state. Only a resident entry needs the refactor.
        if (!rebuilt) {
            for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
            Common.status = KLU_OK;
            int ok = KluTraits<T>::refactor(_Bp.get(), _Bi.get(), _Bx.get(),
                                            entry->symbolic->S, entry->numeric, &Common);
            if (!ok && Common.status == KLU_OK) {
                Common.status = KLU_SINGULAR;
            }
            if ((!ok || Common.status < KLU_OK) && _out_status == nullptr) {
                return ffi::Error::InvalidArgument("klu_refactor/z_refactor failed (singular matrix?)");
            }
            if (_out_status != nullptr) _out_status[i] = (int32_t)Common.status;
        } else if (_out_status != nullptr) {
            _out_status[i] = KLU_OK;
        }
        // Same id back out, so XLA sees the data edge refactor -> solve.
        _out_numeric[i] = _numeric[i];
    }
    return ffi::Error::Success();
}

ffi::Error refactor_f64(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(), out_numeric->typed_data());
}

ffi::Error refactor_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric,
                                  reinterpret_cast<const Complex*>(Ax.typed_data()),
                                  out_numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_f64_handler, refactor_f64,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_c128_handler, refactor_c128,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
);

// refactor_status: same as refactor, but reports the KLU status per batch element instead
// of failing the call. See refactor_impl for the contract.
ffi::Error refactor_status_f64(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status) {
    return refactor_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(),
                                 out_numeric->typed_data(), out_status->typed_data());
}

ffi::Error refactor_status_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status) {
    return refactor_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric,
                                  reinterpret_cast<const Complex*>(Ax.typed_data()),
                                  out_numeric->typed_data(), out_status->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_status_f64_handler, refactor_status_f64,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_status (KLU status per batch element)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_status_c128_handler, refactor_status_c128,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_status (KLU status per batch element)
);

// refactor_and_solve: fused in-place refactorization followed by triangular solve.
// Avoids allocating the CSC buffer twice and saves a JAX dispatch round-trip.
// Returns two outputs: x (the solution) and out_numeric (same pointers as input numeric,
// threaded through so XLA sees the dependency edge: refactor → solve → next-op).
template <typename T>
ffi::Error refactor_and_solve_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    const T* _b,
    T* _x,
    uint64_t* _out_numeric,
    int32_t* _out_status = nullptr) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_lhs_ax = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");
    if ((int)numeric.element_count() != n_lhs_ax) return ffi::Error::InvalidArgument("numeric array size must match n_lhs");

    // Validate b dimensions: must be 3D (n_lhs, n_col, n_rhs). n_col comes from
    // here, so no separate handle is needed to rebuild an evicted numeric.
    if (ds_b.size() != 3) return ffi::Error::InvalidArgument("b must be 3D (n_lhs, n_col, n_rhs)");
    int n_lhs_b = (int)ds_b[0];
    int n_col = (int)ds_b[1];
    int n_rhs = (int)ds_b[2];

    if (n_lhs_ax != n_lhs_b) return ffi::Error::InvalidArgument("n_lhs mismatch between Ax and b");

    int n_lhs = n_lhs_ax;

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    // Transpose b into col-major temp buffer for KLU
    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        int n = i * n_rhs * n_col;

        ffi::Error err = ffi::Error::Success();
        bool rebuilt = false;
        auto entry = resolve_numeric<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Ax_i, err, &rebuilt);
        if (!entry) return err;

        // A rebuilt entry is already factored with these values. A resident one
        // is refactored in place, reusing its pivots. A singular result fills
        // this element's output with NaN rather than aborting the whole call.
        Common.status = KLU_OK;
        int ok = 1;
        if (!rebuilt) {
            for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
            ok = KluTraits<T>::refactor(_Bp.get(), _Bi.get(), _Bx.get(),
                                        entry->symbolic->S, entry->numeric, &Common);
            if (!ok && Common.status == KLU_OK) Common.status = KLU_SINGULAR;
        }
        // Same id back out on every path, for the XLA edge refactor -> solve.
        _out_numeric[i] = _numeric[i];
        if (!ok || Common.status < KLU_OK) {
            fill_nan(&_x_temp[n], n_rhs * n_col);
            if (_out_status != nullptr) _out_status[i] = (int32_t)Common.status;
            continue;
        }

        KluTraits<T>::solve(entry->symbolic->S, entry->numeric, n_col, n_rhs, &_x_temp[n], &Common);
        if (Common.status < KLU_OK) {
            fill_nan(&_x_temp[n], n_rhs * n_col);
        }
        if (_out_status != nullptr) _out_status[i] = (int32_t)Common.status;
    }

    // Transpose result back to row-major for JAX
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }

    return ffi::Error::Success();
}

ffi::Error refactor_and_solve_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_and_solve_impl<double>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        Ax.typed_data(), b.typed_data(), x->typed_data(), out_numeric->typed_data());
}

ffi::Error refactor_and_solve_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric) {
    return refactor_and_solve_impl<Complex>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        reinterpret_cast<const Complex*>(Ax.typed_data()),
        reinterpret_cast<const Complex*>(b.typed_data()),
        reinterpret_cast<Complex*>(x->typed_data()),
        out_numeric->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_f64_handler, refactor_and_solve_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_c128_handler, refactor_and_solve_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
);

// refactor_and_solve_status: as refactor_and_solve (x is still NaN-filled on failure), with
// the KLU status reported per batch element so callers can branch on it under jit.
ffi::Error refactor_and_solve_status_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status) {
    return refactor_and_solve_impl<double>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        Ax.typed_data(), b.typed_data(), x->typed_data(), out_numeric->typed_data(),
        out_status->typed_data());
}

ffi::Error refactor_and_solve_status_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status) {
    return refactor_and_solve_impl<Complex>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        reinterpret_cast<const Complex*>(Ax.typed_data()),
        reinterpret_cast<const Complex*>(b.typed_data()),
        reinterpret_cast<Complex*>(x->typed_data()),
        out_numeric->typed_data(), out_status->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_status_f64_handler, refactor_and_solve_status_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (same handles, for XLA dep edge)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_status (KLU status per batch element)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    refactor_and_solve_status_c128_handler, refactor_and_solve_status_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric (input handles)
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x (solution)
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (same handles, for XLA dep edge)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_status (KLU status per batch element)
);

// rcond: reciprocal pivot growth estimate min|Uii| / max|Uii|, computed in O(n) from an
// existing numeric factorization. Detects pivot degradation after a refactor without the
// cost of a probe solve. A failed call means the factorization is singular, so 0 is written.
// rcond reads only a factorization, but to stay safe when its handle was
// evicted it now also takes Ai/Aj/Ax/n_col so a missing numeric can be rebuilt.
template <typename T>
ffi::Error rcond_impl(
    int64_t n_col_attr,
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    double* _out_rcond) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_lhs = (int)numeric.element_count();
    int n_nz = (int)ds_Ax[1];
    int n_col = (int)n_col_attr;
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        ffi::Error err = ffi::Error::Success();
        auto entry = resolve_numeric<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Ax_i, err);
        if (!entry) return err;

        Common.status = KLU_OK;
        int ok = KluTraits<T>::rcond(entry->symbolic->S, entry->numeric, &Common);
        _out_rcond[i] = (!ok || Common.status < KLU_OK) ? 0.0 : Common.rcond;
    }
    return ffi::Error::Success();
}

ffi::Error rcond_f64(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_rcond) {
    return rcond_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(),
                              out_rcond->typed_data());
}

ffi::Error rcond_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_rcond) {
    return rcond_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric,
                               reinterpret_cast<const Complex*>(Ax.typed_data()),
                               out_rcond->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rcond_f64_handler, rcond_f64,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // rcond per batch element
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    rcond_c128_handler, rcond_c128,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // rcond per batch element
);

// condest: 1-norm condition number estimate (Hager/Higham, as in MATLAB's condest). Needs
// the matrix values as well as the factorization. The usual follow-up when rcond is
// borderline. A failed call means the factorization is singular, so infinity is written.
template <typename T>
ffi::Error condest_impl(
    int64_t n_col_attr,
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    double* _out_condest) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_lhs = (int)ds_Ax[0];
    int n_nz = (int)ds_Ax[1];
    int n_col = (int)n_col_attr;

    if (Ai.dimensions().size() != 1 || Aj.dimensions().size() != 1) return ffi::Error::InvalidArgument("Ai/Aj must be 1D");
    if (Ai.dimensions()[0] != n_nz || Aj.dimensions()[0] != n_nz) return ffi::Error::InvalidArgument("Ai/Aj size mismatch with Ax");
    if ((int)numeric.element_count() != n_lhs) return ffi::Error::InvalidArgument("numeric array size must match n_lhs");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    const uint64_t* _numeric = numeric.typed_data();

    // Shared CSC pattern and values scratch for the condest call itself.
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        ffi::Error err = ffi::Error::Success();
        auto entry = resolve_numeric<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Ax_i, err);
        if (!entry) return err;

        for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];

        Common.status = KLU_OK;
        int ok = KluTraits<T>::condest(_Bp.get(), _Bx.get(), entry->symbolic->S, entry->numeric, &Common);
        _out_condest[i] = (!ok || Common.status < KLU_OK)
                              ? std::numeric_limits<double>::infinity()
                              : Common.condest;
    }
    return ffi::Error::Success();
}

ffi::Error condest_f64(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_condest) {
    return condest_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(),
                                out_condest->typed_data());
}

ffi::Error condest_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> out_condest) {
    return condest_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric,
                                 reinterpret_cast<const Complex*>(Ax.typed_data()),
                                 out_condest->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    condest_f64_handler, condest_f64,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // condest per batch element
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    condest_c128_handler, condest_c128,
    ffi::Ffi::Bind()
        .Attr<int64_t>("n_col")
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric
        .Ret<ffi::Buffer<ffi::DataType::F64>>()   // condest per batch element
);

template <typename T>
ffi::Error solve_with_numeric_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    const T* _b,
    T* _x) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_numeric = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    // Ax carries, by contract, the values that produced these factorizations,
    // so an evicted numeric can be rebuilt from it. It is (n_nz,) when the
    // numeric is a single (broadcast) handle, else (n_numeric, n_nz).
    int n_nz = (int)((ds_Ax.size() == 1) ? ds_Ax[0] : ds_Ax[1]);
    int n_ax = (int)((ds_Ax.size() == 1) ? 1 : ds_Ax[0]);

    // Parse b dimensions
    int d_b = ds_b.size();
    int n_lhs_b, n_col, n_rhs;

    if (d_b == 1) {
        // b is (n_col,) -> treat as (1, n_col, 1)
        n_lhs_b = 1;
        n_col = ds_b[0];
        n_rhs = 1;
    } else if (d_b == 2) {
        // b is (n_lhs, n_col) or (n_col, n_rhs)
        // Decide based on numeric batch size
        if (n_numeric > 1 && (int)ds_b[0] == n_numeric) {
            // b is (n_lhs, n_col) -> (n_lhs, n_col, 1)
            n_lhs_b = ds_b[0];
            n_col = ds_b[1];
            n_rhs = 1;
        } else {
            // b is (n_col, n_rhs) -> (1, n_col, n_rhs)
            n_lhs_b = 1;
            n_col = ds_b[0];
            n_rhs = ds_b[1];
        }
    } else if (d_b == 3) {
        // b is already (n_lhs, n_col, n_rhs)
        n_lhs_b = ds_b[0];
        n_col = ds_b[1];
        n_rhs = ds_b[2];
    } else {
        return ffi::Error::InvalidArgument("b must be 1D, 2D, or 3D");
    }

    // Determine output batch size
    bool broadcast_numeric = (n_numeric == 1);
    bool broadcast_b = (n_lhs_b == 1);

    int n_lhs;
    if (!broadcast_numeric && !broadcast_b) {
        if (n_numeric != n_lhs_b) {
            return ffi::Error::InvalidArgument("numeric and b batch size mismatch");
        }
        n_lhs = n_numeric;
    } else if (!broadcast_numeric) {
        n_lhs = n_numeric;
    } else if (!broadcast_b) {
        n_lhs = n_lhs_b;
    } else {
        n_lhs = 1;
    }

    // Validate output dimensions match expected
    if (ds_x.size() != d_b) {
        return ffi::Error::InvalidArgument("output dimensions don't match input");
    }

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);

    // Copy b to x_temp (transpose) with broadcasting
    for (int m = 0; m < n_lhs; m++) {
        int m_b = broadcast_b ? 0 : m;
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m_b * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int n = i * n_rhs * n_col;
        int num_i = broadcast_numeric ? 0 : i;
        int ax_i = (n_ax == 1) ? 0 : num_i;
        const T* _Ax_i = _Ax + (size_t)ax_i * n_nz;

        ffi::Error err = ffi::Error::Success();
        auto entry = resolve_numeric<T>(_numeric[num_i], _Ai, _Aj, n_nz, n_col, _Ax_i, err);
        if (!entry) return err;

        KluTraits<T>::solve(entry->symbolic->S, entry->numeric, n_col, n_rhs, &_x_temp[n], &Common);

        // A failed solve fills this element's output with NaN instead of aborting the
        // whole call, mirroring how LAPACK-backed solvers behave on a singular system.
        if (Common.status < KLU_OK) {
            fill_nan(&_x_temp[n], n_rhs * n_col);
        }
    }

    // Copy x_temp to x (transpose)
    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error solve_with_numeric_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return solve_with_numeric_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                           symbolic, numeric, Ax.typed_data(), b.typed_data(),
                                           x->typed_data());
}

ffi::Error solve_with_numeric_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return solve_with_numeric_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                            symbolic, numeric,
                                            reinterpret_cast<const Complex*>(Ax.typed_data()),
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_f64_handler, solve_with_numeric_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_c128_handler, solve_with_numeric_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>());

// tsolve_with_numeric: identical to solve_with_numeric_impl but uses KluTraits<T>::tsolve.
// Solves A^T x = b using a pre-computed numeric factorization.
template <typename T>
ffi::Error tsolve_with_numeric_impl(
    const ffi::Buffer<ffi::DataType::S32>& Ai,
    const ffi::Buffer<ffi::DataType::S32>& Aj,
    const ffi::AnyBuffer::Dimensions& ds_Ax,
    const ffi::AnyBuffer::Dimensions& ds_b,
    const ffi::AnyBuffer::Dimensions& ds_x,
    const ffi::Buffer<ffi::DataType::U64>& symbolic,
    const ffi::Buffer<ffi::DataType::U64>& numeric,
    const T* _Ax,
    const T* _b,
    T* _x) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_numeric = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();
    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();
    // Ax carries, by contract, the values that produced these factorizations,
    // so an evicted numeric can be rebuilt from it. It is (n_nz,) when the
    // numeric is a single (broadcast) handle, else (n_numeric, n_nz).
    int n_nz = (int)((ds_Ax.size() == 1) ? ds_Ax[0] : ds_Ax[1]);
    int n_ax = (int)((ds_Ax.size() == 1) ? 1 : ds_Ax[0]);

    int d_b = ds_b.size();
    int n_lhs_b, n_col, n_rhs;

    if (d_b == 1) {
        n_lhs_b = 1;
        n_col = ds_b[0];
        n_rhs = 1;
    } else if (d_b == 2) {
        if (n_numeric > 1 && (int)ds_b[0] == n_numeric) {
            n_lhs_b = ds_b[0];
            n_col = ds_b[1];
            n_rhs = 1;
        } else {
            n_lhs_b = 1;
            n_col = ds_b[0];
            n_rhs = ds_b[1];
        }
    } else if (d_b == 3) {
        n_lhs_b = ds_b[0];
        n_col = ds_b[1];
        n_rhs = ds_b[2];
    } else {
        return ffi::Error::InvalidArgument("b must be 1D, 2D, or 3D");
    }

    bool broadcast_numeric = (n_numeric == 1);
    bool broadcast_b = (n_lhs_b == 1);

    int n_lhs;
    if (!broadcast_numeric && !broadcast_b) {
        if (n_numeric != n_lhs_b) return ffi::Error::InvalidArgument("numeric and b batch size mismatch");
        n_lhs = n_numeric;
    } else if (!broadcast_numeric) {
        n_lhs = n_numeric;
    } else if (!broadcast_b) {
        n_lhs = n_lhs_b;
    } else {
        n_lhs = 1;
    }

    if (ds_x.size() != d_b) return ffi::Error::InvalidArgument("output dimensions don't match input");

    auto _x_temp = std::make_unique<T[]>(n_lhs * n_col * n_rhs);

    for (int m = 0; m < n_lhs; m++) {
        int m_b = broadcast_b ? 0 : m;
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x_temp[m * n_rhs * n_col + p * n_col + n] = _b[m_b * n_col * n_rhs + n * n_rhs + p];
            }
        }
    }

    klu_common Common;
    klu_defaults(&Common);

    for (int i = 0; i < n_lhs; i++) {
        int n = i * n_rhs * n_col;
        int num_i = broadcast_numeric ? 0 : i;
        int ax_i = (n_ax == 1) ? 0 : num_i;
        const T* _Ax_i = _Ax + (size_t)ax_i * n_nz;

        ffi::Error err = ffi::Error::Success();
        auto entry = resolve_numeric<T>(_numeric[num_i], _Ai, _Aj, n_nz, n_col, _Ax_i, err);
        if (!entry) return err;

        // NOTE: tsolve instead of solve
        KluTraits<T>::tsolve(entry->symbolic->S, entry->numeric, n_col, n_rhs, &_x_temp[n], &Common);

        // A failed solve fills this element's output with NaN instead of aborting the
        // whole call, mirroring how LAPACK-backed solvers behave on a singular system.
        if (Common.status < KLU_OK) {
            fill_nan(&_x_temp[n], n_rhs * n_col);
        }
    }

    for (int m = 0; m < n_lhs; m++) {
        for (int n = 0; n < n_col; n++) {
            for (int p = 0; p < n_rhs; p++) {
                _x[m * n_col * n_rhs + n * n_rhs + p] = _x_temp[m * n_rhs * n_col + p * n_col + n];
            }
        }
    }
    return ffi::Error::Success();
}

ffi::Error tsolve_with_numeric_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x) {
    return tsolve_with_numeric_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                            symbolic, numeric, Ax.typed_data(), b.typed_data(),
                                            x->typed_data());
}

ffi::Error tsolve_with_numeric_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x) {
    return tsolve_with_numeric_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                             symbolic, numeric,
                                             reinterpret_cast<const Complex*>(Ax.typed_data()),
                                             reinterpret_cast<const Complex*>(b.typed_data()),
                                             reinterpret_cast<Complex*>(x->typed_data()));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_f64_handler, tsolve_with_numeric_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Arg<ffi::Buffer<ffi::DataType::F64>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()  // x
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_c128_handler, tsolve_with_numeric_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()   // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()   // numeric
        .Arg<ffi::Buffer<ffi::DataType::C128>>()  // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()  // x
);

// Freeing now just drops the cache entry, which frees its KLU objects. It is
// never required: a later use of the id rebuilds it. An unknown id is a no-op.
// ordering is an unused operand that only gives XLA a data dependency, so a
// free inside a jit trace runs after the solves it must follow. See free_numeric
// in klujax.py.
ffi::Error free_numeric(
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::S32> ordering,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status) {
    (void)ordering;
    int n = numeric.element_count();
    const uint64_t* _numeric = numeric.typed_data();
    auto& r = CacheRegistry::instance();
    for (int i = 0; i < n; i++) {
        if (_numeric[i] != 0) r.erase(_numeric[i]);
    }
    // returning value so function can be traced
    *status->typed_data() = 1;
    return ffi::Error::Success();
}

ffi::Error free_symbolic(
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::S32> ordering,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> status) {
    (void)ordering;
    if (symbolic.element_count() != 1) {
        return ffi::Error::InvalidArgument("symbolic must be a scalar.");
    }
    uint64_t sym_id = *symbolic.typed_data();
    if (sym_id == 0) {
        *status->typed_data() = 0;  // 0 = nothing to free
        return ffi::Error::Success();
    }
    CacheRegistry::instance().erase(sym_id);
    // returning value so function can be traced
    *status->typed_data() = 1;
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_numeric_handler, free_numeric,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // numeric
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ordering (unused)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // status (instead of no return)
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    free_symbolic_handler, free_symbolic,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::U64>>()  // symbolic
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // ordering (unused)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // status
);

ffi::Error analyze(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::S32> n_col_buf,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> symbolic) {
    if (n_col_buf.element_count() != 1) {
        return ffi::Error::InvalidArgument("n_col must be a scalar.");
    }
    int n_col = *n_col_buf.typed_data();

    auto ds_Ai = Ai.dimensions();
    if (ds_Ai.size() != 1) return ffi::Error::InvalidArgument("Ai must be 1D.");
    int n_nz = ds_Ai[0];

    auto ds_Aj = Aj.dimensions();
    if (ds_Aj.size() != 1) return ffi::Error::InvalidArgument("Aj must be 1D.");
    if (ds_Aj[0] != n_nz) return ffi::Error::InvalidArgument("Aj size mismatch.");

    const int* _Ai = Ai.typed_data();
    const int* _Aj = Aj.typed_data();

    // Validate indices
    for (int n = 0; n < n_nz; n++) {
        if (_Ai[n] < 0 || _Ai[n] >= n_col) return ffi::Error::InvalidArgument("Ai index out of bounds.");
        if (_Aj[n] < 0 || _Aj[n] >= n_col) return ffi::Error::InvalidArgument("Aj index out of bounds.");
    }

    // Build the analysis and register it. The returned id, not a pointer, is
    // the handle: a stale id is a safe cache miss, never a bad dereference.
    auto sym = build_symbolic(_Ai, _Aj, n_nz, n_col);
    if (!sym) {
        return ffi::Error::Internal("klu_analyze failed.");
    }
    auto entry = std::make_shared<CacheEntry>();
    entry->symbolic = sym;
    *symbolic->typed_data() = CacheRegistry::instance().insert(entry);
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    analyze_handler, analyze,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // Aj
        .Arg<ffi::Buffer<ffi::DataType::S32>>()  // n_col
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // symbolic ptr
);

// Python wrappers
PYBIND11_MODULE(klujax_cpp, m) {
    m.def("dot_f64",
          []() { return py::capsule((void*)&dot_f64_handler); });
    m.def("dot_c128",
          []() { return py::capsule((void*)&dot_c128_handler); });
    m.def("solve_f64",
          []() { return py::capsule((void*)&solve_f64_handler); });
    m.def("solve_c128",
          []() { return py::capsule((void*)&solve_c128_handler); });
    m.def("solve_with_symbol_f64",
          []() { return py::capsule((void*)&solve_with_symbol_f64_handler); });
    m.def("solve_with_symbol_c128",
          []() { return py::capsule((void*)&solve_with_symbol_c128_handler); });
    m.def("tsolve_with_symbol_f64",
          []() { return py::capsule((void*)&tsolve_with_symbol_f64_handler); });
    m.def("tsolve_with_symbol_c128",
          []() { return py::capsule((void*)&tsolve_with_symbol_c128_handler); });
    m.def("factor_f64",
          []() { return py::capsule((void*)&factor_f64_handler); });
    m.def("factor_c128",
          []() { return py::capsule((void*)&factor_c128_handler); });
    m.def("solve_with_numeric_f64",
          []() { return py::capsule((void*)&solve_with_numeric_f64_handler); });
    m.def("solve_with_numeric_c128",
          []() { return py::capsule((void*)&solve_with_numeric_c128_handler); });
    m.def("tsolve_with_numeric_f64",
          []() { return py::capsule((void*)&tsolve_with_numeric_f64_handler); });
    m.def("tsolve_with_numeric_c128",
          []() { return py::capsule((void*)&tsolve_with_numeric_c128_handler); });
    m.def("free_numeric",
          []() { return py::capsule((void*)&free_numeric_handler); });
    m.def("analyze",
          []() { return py::capsule((void*)&analyze_handler); });
    m.def("free_symbolic",
          []() { return py::capsule((void*)&free_symbolic_handler); });
    m.def("refactor_f64",
          []() { return py::capsule((void*)&refactor_f64_handler); });
    m.def("refactor_c128",
          []() { return py::capsule((void*)&refactor_c128_handler); });
    m.def("refactor_and_solve_f64",
          []() { return py::capsule((void*)&refactor_and_solve_f64_handler); });
    m.def("refactor_and_solve_c128",
          []() { return py::capsule((void*)&refactor_and_solve_c128_handler); });
    m.def("refactor_status_f64",
          []() { return py::capsule((void*)&refactor_status_f64_handler); });
    m.def("refactor_status_c128",
          []() { return py::capsule((void*)&refactor_status_c128_handler); });
    m.def("refactor_and_solve_status_f64",
          []() { return py::capsule((void*)&refactor_and_solve_status_f64_handler); });
    m.def("refactor_and_solve_status_c128",
          []() { return py::capsule((void*)&refactor_and_solve_status_c128_handler); });
    m.def("rcond_f64",
          []() { return py::capsule((void*)&rcond_f64_handler); });
    m.def("rcond_c128",
          []() { return py::capsule((void*)&rcond_c128_handler); });
    m.def("condest_f64",
          []() { return py::capsule((void*)&condest_f64_handler); });
    m.def("condest_c128",
          []() { return py::capsule((void*)&condest_c128_handler); });
    // Diagnostics for the handle cache: how many factorizations had to be
    // rebuilt because their handle was evicted or freed, and a reset.
    m.def("rebuild_count",
          []() { return (long)CacheRegistry::instance().rebuilds.load(); });
    m.def("reset_rebuild_count",
          []() { CacheRegistry::instance().rebuilds.store(0); });
}