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
#include <type_traits>

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
// so a stale id can never dereference freed memory. The id is a *content hash*
// of the matrix the handle names (dtype + sparsity pattern + values), so a
// handle equals hash(content) and whatever the cache returns for a key was
// built from content whose hash is that key. This makes a handle name a matrix,
// not a mutable slot. A refactor re-keys to the new values instead of overwriting
// the old id, so a stale alias can never be handed the wrong matrix. At worst it
// rebuilds. Identical matrices hash equal, so the cache also deduplicates.
//
// Each entry owns its KLU objects and frees them when dropped. A symbolic entry
// owns a klu_symbolic. A numeric entry also owns a klu_numeric and shares
// (ref-counts) the symbolic it was factored against, so analysis is reused.
//
// The cache is bounded. The least recently used entry is evicted once it grows
// past KLUJAX_FACTOR_CACHE, so forgetting to free leaks only a bounded amount.
// A call arriving with an id no longer resident (evicted, freed, or superseded
// by a refactor) rebuilds the state from the Ai/Aj/Ax the caller carries, so
// use of a gone handle is safe. A bounded tombstone remembers why a retired key
// went away, so a rebuild can report EVICTED / FREED / SUPERSEDED.

// Why a call rebuilt a factorization instead of using a resident one. Mirrored
// by RebuildReason in klujax.py. Reported per left-hand side by the *_status ops.
//   NONE:       cache hit, the resident factorization was used.
//   EVICTED:    fell out of the bounded cache (raise KLUJAX_FACTOR_CACHE).
//   FREED:      free_numeric or free_symbolic had dropped it (expected).
//   SUPERSEDED: a refactor re-keyed this handle, so a stale alias was used.
//   DTYPE:      reserved. Entry present but factored with the other dtype.
//   UNKNOWN:    never resident, or its tombstone was itself evicted.
//   STALE:      reserved. Content-check mismatch (collision or forged token).
enum class RebuildReason : int32_t {
    NONE = 0,
    EVICTED = 1,
    FREED = 2,
    SUPERSEDED = 3,
    DTYPE = 4,
    UNKNOWN = 5,
    STALE = 6,
};
static constexpr int kNumRebuildReasons = 7;

// A 128-bit content hash split into a primary key (the cache key) and a second
// independent check lane. The check makes dedup collision-safe on the cold path
// (a colliding primary key is caught before an entry is reused) without widening
// the id that flows through JAX as data.
struct ContentKey {
    uint64_t key;
    uint64_t check;
};

// Two-lane FNV-1a-style mixing. Not cryptographic. It only needs to spread
// content across 128 bits well enough that distinct matrices collide with
// negligible probability.
inline void hash_fold(uint64_t& a, uint64_t& b, const void* data, size_t n) {
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n; i++) {
        a = (a ^ p[i]) * 0x100000001b3ULL;
        b = (b + p[i]) * 0x9E3779B97F4A7C15ULL;
        b ^= b >> 29;
    }
}

// Hash canonical CSC content. `domain` separates symbolic ('s') from numeric
// ('n') keyspaces. For a symbolic key pass Bx=nullptr / bx_bytes=0.
inline ContentKey hash_content(char domain, bool is_complex, int n_col, int n_nz,
                               const int* Bp, const int* Bi,
                               const void* Bx, size_t bx_bytes) {
    uint64_t a = 0xcbf29ce484222325ULL;
    uint64_t b = 0x27d4eb2f165667c5ULL;
    unsigned char tag[2] = {static_cast<unsigned char>(domain),
                            static_cast<unsigned char>(is_complex ? 1 : 0)};
    hash_fold(a, b, tag, sizeof(tag));
    hash_fold(a, b, &n_col, sizeof(n_col));
    hash_fold(a, b, &n_nz, sizeof(n_nz));
    hash_fold(a, b, Bp, sizeof(int) * static_cast<size_t>(n_col + 1));
    hash_fold(a, b, Bi, sizeof(int) * static_cast<size_t>(n_nz));
    hash_fold(a, b, Bx, bx_bytes);
    uint64_t key = a ^ (b << 1);
    uint64_t check = b ^ (a >> 1);
    if (key == 0) key = 1;  // 0 is the null-handle sentinel, never a real key
    return {key, check};
}

struct SymbolicObj {
    klu_symbolic* S = nullptr;
    int n_col = 0;
    uint64_t key = 0;    // content key of the sparsity pattern
    uint64_t check = 0;  // second hash lane, for collision-safe dedup
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
    bool is_complex = false;                // dtype the numeric was factored with
    uint64_t key = 0;                       // content key this entry is filed under
    uint64_t check = 0;                     // second hash lane
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
    // Tombstones remember why a retired key went away, so a later miss on it can
    // name the cause. Bounded like the live cache.
    std::map<uint64_t, RebuildReason> tombstones;
    std::list<uint64_t> tomb_lru;
    std::atomic<long> rebuilds{0};
    std::atomic<long> reason_counts[kNumRebuildReasons];

    CacheRegistry() {
        for (auto& c : reason_counts) c.store(0);
    }

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

    void touch(uint64_t id) {  // caller holds mu
        lru.remove(id);
        lru.push_front(id);
    }
    // caller holds mu. Remember a retired key and why, bounded by capacity().
    void record_tombstone(uint64_t id, RebuildReason why) {
        if (id == 0) return;
        if (tombstones.find(id) == tombstones.end()) {
            tomb_lru.push_front(id);
        } else {
            tomb_lru.remove(id);
            tomb_lru.push_front(id);
        }
        tombstones[id] = why;
        size_t cap = capacity();
        while (tombstones.size() > cap && !tomb_lru.empty()) {
            uint64_t v = tomb_lru.back();
            tomb_lru.pop_back();
            tombstones.erase(v);
        }
    }
    RebuildReason tombstone_reason(uint64_t id) {  // caller holds mu
        auto it = tombstones.find(id);
        return it == tombstones.end() ? RebuildReason::UNKNOWN : it->second;
    }
    // Public: why is `id` no longer resident? Locks internally.
    RebuildReason reason_for_missing(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        return tombstone_reason(id);
    }
    void evict() {  // caller holds mu
        size_t cap = capacity();
        while (entries.size() > cap && !lru.empty()) {
            uint64_t victim = lru.back();
            lru.pop_back();
            entries.erase(victim);
            record_tombstone(victim, RebuildReason::EVICTED);
        }
    }
    // Register entry under its content key, evicting overflow. A live key is no
    // longer a tombstone.
    uint64_t insert(std::shared_ptr<CacheEntry> e, uint64_t key) {
        std::lock_guard<std::mutex> lk(mu);
        e->key = key;
        entries[key] = std::move(e);
        auto it = tombstones.find(key);
        if (it != tombstones.end()) {
            tombstones.erase(it);
            tomb_lru.remove(key);
        }
        touch(key);
        evict();
        return key;
    }
    // Move an entry from an old key to a new one (refactor). The old key becomes
    // a SUPERSEDED tombstone so a stale alias of it rebuilds and can say why.
    void rekey(uint64_t oldk, uint64_t newk, std::shared_ptr<CacheEntry> e) {
        std::lock_guard<std::mutex> lk(mu);
        if (oldk != newk) {
            entries.erase(oldk);
            lru.remove(oldk);
            record_tombstone(oldk, RebuildReason::SUPERSEDED);
        }
        e->key = newk;
        entries[newk] = e;
        auto it = tombstones.find(newk);
        if (it != tombstones.end()) {
            tombstones.erase(it);
            tomb_lru.remove(newk);
        }
        touch(newk);
        evict();
    }
    std::shared_ptr<CacheEntry> lookup(uint64_t id) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = entries.find(id);
        if (it == entries.end()) return nullptr;
        touch(id);
        return it->second;
    }
    void erase(uint64_t id, RebuildReason why) {
        std::lock_guard<std::mutex> lk(mu);
        if (entries.erase(id)) {
            lru.remove(id);
            record_tombstone(id, why);
        }
    }
    // True when only the registry and the caller's local shared_ptr hold `e`, so
    // it is safe to mutate in place (no other in-flight call can be reading it).
    // Best-effort: a conservative false just takes the fresh-factor path.
    bool uniquely_held(const std::shared_ptr<CacheEntry>& e) {
        return e.use_count() <= 2;
    }
    void note_rebuild(RebuildReason why) {
        rebuilds.fetch_add(1);
        int i = static_cast<int>(why);
        if (i >= 0 && i < kNumRebuildReasons) reason_counts[i].fetch_add(1);
    }
};

// Build a fresh symbolic analysis from a COO pattern, keyed by the pattern hash.
// Returns null on failure.
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
    ContentKey ck = hash_content('s', false, n_col, n_nz, _Bp.get(), _Bi.get(), nullptr, 0);
    obj->key = ck.key;
    obj->check = ck.check;
    return obj;
}

// Factor one left-hand side against a symbolic, content-addressed. On a hit for
// the same content (dtype + pattern + values) the resident entry is reused
// (dedup, skipping klu_factor). Otherwise a fresh entry is built and inserted
// under its content key. `out_key` receives the content key, `deduped` whether
// an existing entry was reused. Returns null only on a factor failure (singular).
template <typename T>
std::shared_ptr<CacheEntry> make_or_get_numeric(std::shared_ptr<SymbolicObj> sym, const int* _Ai,
                                                const int* _Aj, int n_nz, const T* _Ax_i,
                                                uint64_t* out_key, bool* deduped) {
    int n_col = sym->n_col;
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bx = std::make_unique<T[]>(n_nz);
    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());
    for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
    bool is_complex = std::is_same<T, Complex>::value;
    ContentKey ck = hash_content('n', is_complex, n_col, n_nz, _Bp.get(), _Bi.get(),
                                 _Bx.get(), sizeof(T) * static_cast<size_t>(n_nz));
    auto& r = CacheRegistry::instance();
    if (auto e = r.lookup(ck.key)) {
        // Reuse only if the second hash lane also agrees, so a (vanishingly rare)
        // primary-key collision never hands back the wrong factorization.
        if (e->numeric && e->is_complex == is_complex && e->check == ck.check) {
            if (out_key) *out_key = ck.key;
            if (deduped) *deduped = true;
            return e;
        }
    }
    klu_common Common;
    klu_defaults(&Common);
    klu_numeric* N = KluTraits<T>::factor(_Bp.get(), _Bi.get(), _Bx.get(), sym->S, &Common);
    if (N == nullptr || Common.status < KLU_OK) {
        if (N) klu_free_numeric(&N, &Common);
        return nullptr;
    }
    auto e = std::make_shared<CacheEntry>();
    e->symbolic = sym;
    e->numeric = N;
    e->is_complex = is_complex;
    e->check = ck.check;
    r.insert(e, ck.key);
    if (out_key) *out_key = ck.key;
    if (deduped) *deduped = false;
    return e;
}

// Resolve a symbolic id, rebuilding from the caller's pattern on a miss. The
// heavy klu_analyze runs outside the registry lock. Sets `err` and returns null
// only on strict-mode miss or an analyze failure. `reason` (if given) reports
// whether a rebuild happened and why.
std::shared_ptr<SymbolicObj> resolve_symbolic(uint64_t id, const int* _Ai, const int* _Aj, int n_nz,
                                              int n_col, ffi::Error& err,
                                              RebuildReason* reason = nullptr) {
    auto& r = CacheRegistry::instance();
    if (auto e = r.lookup(id)) {
        if (reason) *reason = RebuildReason::NONE;
        return e->symbolic;
    }
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
    RebuildReason why = r.reason_for_missing(id);
    r.note_rebuild(why);
    auto e = std::make_shared<CacheEntry>();
    e->symbolic = sym;
    r.insert(e, sym->key);  // file under the true content key
    if (reason) *reason = why;
    return sym;
}

// Resolve one numeric id, rebuilding analysis and factorization from the
// caller's matrix on a miss. Returns the whole entry so the solver can use its
// self-consistent symbolic/numeric pair. A resident entry whose dtype disagrees
// with the requested solve is a hard error (never a silent wrong-dtype solve).
template <typename T>
std::shared_ptr<CacheEntry> resolve_numeric(uint64_t id, const int* _Ai, const int* _Aj, int n_nz,
                                            int n_col, const T* _Ax_i, ffi::Error& err,
                                            RebuildReason* reason = nullptr) {
    bool is_complex = std::is_same<T, Complex>::value;
    auto& r = CacheRegistry::instance();
    if (auto e = r.lookup(id)) {
        if (e->numeric) {
            if (e->is_complex != is_complex) {
                err = ffi::Error::InvalidArgument(
                    "klujax: numeric handle dtype mismatch (factored as " +
                    std::string(e->is_complex ? "complex" : "real") + ", used as " +
                    std::string(is_complex ? "complex" : "real") + ")");
                return nullptr;
            }
            if (reason) *reason = RebuildReason::NONE;
            return e;
        }
    }
    if (CacheRegistry::strict()) {
        err = ffi::Error::Internal("klujax: numeric handle " + std::to_string(id) +
                                   " was evicted or freed and strict cache mode is on");
        return nullptr;
    }
    // Capture why `id` is missing before rebuilding: make_or_get_numeric below
    // inserts a live entry, which clears any tombstone under the *same* key, so
    // reading id's tombstone has to happen before that insert can (rarely) hit it.
    RebuildReason why = r.reason_for_missing(id);
    auto sym = build_symbolic(_Ai, _Aj, n_nz, n_col);
    if (!sym) {
        err = ffi::Error::Internal("klujax: rebuild of numeric handle failed (klu_analyze)");
        return nullptr;
    }
    uint64_t key = 0;
    bool deduped = false;
    auto e = make_or_get_numeric<T>(sym, _Ai, _Aj, n_nz, _Ax_i, &key, &deduped);
    r.note_rebuild(why);
    if (reason) *reason = why;
    // A singular rebuild is not a hard error: the caller fills NaN for this
    // element and moves on, mirroring a direct singular solve (err stays Success).
    return e;
}

// Shared by refactor and refactor_and_solve: bring the numeric named by K0 up to
// the new values, content-addressed. Returns the up-to-date entry and, via
// out-params, the new content key, the KLU status of any in-place refactor, and
// the rebuild reason. Returns null with `*klu_status < KLU_OK` on a singular
// matrix, or null with `err` set on an analyze failure.
template <typename T>
std::shared_ptr<CacheEntry> refactor_to_entry(uint64_t K0, const int* _Ai, const int* _Aj, int n_nz,
                                              int n_col, const int* _Bp, const int* _Bi, T* _Bx,
                                              const T* _Ax_i, const ContentKey& ck,
                                              uint64_t* out_key, int32_t* klu_status,
                                              RebuildReason* reason, ffi::Error& err) {
    bool is_complex = std::is_same<T, Complex>::value;
    auto& r = CacheRegistry::instance();
    uint64_t K1 = ck.key;
    *out_key = K1;
    *klu_status = KLU_OK;
    *reason = RebuildReason::NONE;

    // Values unchanged, or the new content is already cached: reuse it (dedup).
    if (auto e1 = r.lookup(K1)) {
        if (e1->numeric && e1->is_complex == is_complex && e1->check == ck.check) {
            return e1;
        }
    }

    // The old numeric is resident and safe to mutate: klu_refactor in place
    // (reusing its pivots) then move it to the new key. This keeps refactor fast
    // while a stale alias of K0 rebuilds correctly (SUPERSEDED).
    auto e0 = r.lookup(K0);
    if (e0 && e0->numeric && e0->is_complex == is_complex && r.uniquely_held(e0)) {
        klu_common Common;
        klu_defaults(&Common);
        Common.status = KLU_OK;
        int ok = KluTraits<T>::refactor(const_cast<int*>(_Bp), const_cast<int*>(_Bi), _Bx,
                                        e0->symbolic->S, e0->numeric, &Common);
        if (!ok && Common.status == KLU_OK) Common.status = KLU_SINGULAR;
        *klu_status = (int32_t)Common.status;
        if (ok && Common.status >= KLU_OK) {
            e0->check = ck.check;
            r.rekey(K0, K1, e0);
            return e0;
        }
        // A failed in-place refactor leaves the numeric partially overwritten,
        // so drop it and let a later use rebuild cleanly.
        r.erase(K0, RebuildReason::SUPERSEDED);
        return nullptr;
    }

    // Fresh factor under the new key (dedup-aware). If K0 was resident we leave it
    // intact, so its aliases still hit. Otherwise the tombstone says why it went.
    // Read the tombstone before make_or_get_numeric's insert below, which would
    // clear it first if the new key happens to coincide with K0.
    RebuildReason missing_reason = e0 ? RebuildReason::NONE : r.reason_for_missing(K0);
    auto sym = e0 ? e0->symbolic : build_symbolic(_Ai, _Aj, n_nz, n_col);
    if (!sym) {
        err = ffi::Error::Internal("klujax: analyze failed on refactor rebuild");
        return nullptr;
    }
    uint64_t key = 0;
    bool deduped = false;
    auto e = make_or_get_numeric<T>(sym, _Ai, _Aj, n_nz, _Ax_i, &key, &deduped);
    if (!e0) r.note_rebuild(missing_reason);
    *reason = missing_reason;
    if (!e) {
        *klu_status = KLU_SINGULAR;  // singular matrix, not an internal error
        return nullptr;
    }
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
    T* _x,
    int32_t* out_rebuild = nullptr) {
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
    // Only one symbolic handle exists per call, unlike the numeric handles (one
    // per left-hand side), so its rebuild reason is a single scalar, not an array.
    RebuildReason sym_reason = RebuildReason::NONE;
    auto _sym = resolve_symbolic(sym_id, _Ai, _Aj, n_nz, n_col, err, &sym_reason);
    if (!_sym) return err;
    if (out_rebuild != nullptr) *out_rebuild = (int32_t)sym_reason;
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
    T* _x,
    int32_t* out_rebuild = nullptr) {
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
    // Only one symbolic handle exists per call, unlike the numeric handles (one
    // per left-hand side), so its rebuild reason is a single scalar, not an array.
    RebuildReason sym_reason = RebuildReason::NONE;
    auto _sym = resolve_symbolic(sym_id, _Ai, _Aj, n_nz, n_col, err, &sym_reason);
    if (!_sym) return err;
    if (out_rebuild != nullptr) *out_rebuild = (int32_t)sym_reason;
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

// solve_with_symbol_status / tsolve_with_symbol_status: as the plain variants,
// plus a scalar RebuildReason saying whether the symbolic analysis was reused or
// rebuilt from the carried Ai/Aj.
ffi::Error solve_with_symbol_status_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return solve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                          Ax.typed_data(), b.typed_data(), x->typed_data(),
                                          out_rebuild->typed_data());
}

ffi::Error solve_with_symbol_status_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return solve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           reinterpret_cast<const Complex*>(Ax.typed_data()),
                                           reinterpret_cast<const Complex*>(b.typed_data()),
                                           reinterpret_cast<Complex*>(x->typed_data()),
                                           out_rebuild->typed_data());
}

ffi::Error tsolve_with_symbol_status_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::F64> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return tsolve_with_symbol_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                           Ax.typed_data(), b.typed_data(), x->typed_data(),
                                           out_rebuild->typed_data());
}

ffi::Error tsolve_with_symbol_status_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::C128> b,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return tsolve_with_symbol_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic,
                                            reinterpret_cast<const Complex*>(Ax.typed_data()),
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()),
                                            out_rebuild->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_status_f64_handler, solve_with_symbol_status_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (scalar)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_symbol_status_c128_handler, solve_with_symbol_status_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()   // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (scalar)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_status_f64_handler, tsolve_with_symbol_status_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // Ax
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (scalar)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_symbol_status_c128_handler, tsolve_with_symbol_status_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // Ax
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // b
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Ret<ffi::Buffer<ffi::DataType::C128>>()   // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (scalar)

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

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        // The id is the content key of this matrix: identical matrices dedup to
        // one factorization, and the key is the data edge XLA orders solves
        // against, so no optimization_barrier is needed.
        uint64_t key = 0;
        bool deduped = false;
        auto entry = make_or_get_numeric<T>(sym, _Ai, _Aj, n_nz, _Ax_i, &key, &deduped);
        if (!entry) {
            return ffi::Error::InvalidArgument("klu_factor/z_factor failed (singular matrix?)");
        }
        _numeric[i] = key;
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
    int32_t* _out_status = nullptr,
    int32_t* _out_rebuild = nullptr) {
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

    // Shared CSC pattern. Values (Bx) and the content key are per left-hand side.
    auto _Bk = std::make_unique<int[]>(n_nz);
    auto _Bi = std::make_unique<int[]>(n_nz);
    auto _Bp = std::make_unique<int[]>(n_col + 1);
    auto _Bx = std::make_unique<T[]>(n_nz);
    bool is_complex = std::is_same<T, Complex>::value;

    coo_to_csc_analyze(n_col, n_nz, _Ai, _Aj, _Bi.get(), _Bp.get(), _Bk.get());

    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
        ContentKey ck = hash_content('n', is_complex, n_col, n_nz, _Bp.get(), _Bi.get(),
                                     _Bx.get(), sizeof(T) * (size_t)n_nz);
        uint64_t out_key = 0;
        int32_t klu_status = KLU_OK;
        RebuildReason reason = RebuildReason::NONE;
        ffi::Error err = ffi::Error::Success();
        auto entry = refactor_to_entry<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Bp.get(), _Bi.get(),
                                          _Bx.get(), _Ax_i, ck, &out_key, &klu_status, &reason, err);
        if (err.failure()) return err;
        if (!entry && _out_status == nullptr) {
            return ffi::Error::InvalidArgument("klu_refactor/z_refactor failed (singular matrix?)");
        }
        // Re-key: the new content is a new handle, so a stale alias of the old id
        // can never be handed these new values (it rebuilds instead).
        _out_numeric[i] = out_key;
        if (_out_status != nullptr) _out_status[i] = klu_status;
        if (_out_rebuild != nullptr) _out_rebuild[i] = (int32_t)reason;
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
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return refactor_impl<double>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric, Ax.typed_data(),
                                 out_numeric->typed_data(), out_status->typed_data(),
                                 out_rebuild->typed_data());
}

ffi::Error refactor_status_c128(
    int64_t n_col,
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::U64>> out_numeric,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return refactor_impl<Complex>(n_col, Ai, Aj, Ax.dimensions(), symbolic, numeric,
                                  reinterpret_cast<const Complex*>(Ax.typed_data()),
                                  out_numeric->typed_data(), out_status->typed_data(),
                                  out_rebuild->typed_data());
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
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (new content keys)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_status (KLU status per batch element)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_rebuild (RebuildReason per element)
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
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (new content keys)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_status (KLU status per batch element)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_rebuild (RebuildReason per element)
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
    int32_t* _out_status = nullptr,
    int32_t* _out_rebuild = nullptr) {
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

    bool is_complex = std::is_same<T, Complex>::value;
    for (int i = 0; i < n_lhs; i++) {
        const T* _Ax_i = _Ax + (size_t)i * n_nz;
        int n = i * n_rhs * n_col;

        for (int k = 0; k < n_nz; k++) _Bx[k] = _Ax_i[_Bk[k]];
        ContentKey ck = hash_content('n', is_complex, n_col, n_nz, _Bp.get(), _Bi.get(),
                                     _Bx.get(), sizeof(T) * (size_t)n_nz);
        uint64_t out_key = 0;
        int32_t klu_status = KLU_OK;
        RebuildReason reason = RebuildReason::NONE;
        ffi::Error err = ffi::Error::Success();
        // Bring the numeric up to the new values (in-place refactor + re-key when
        // possible, else a fresh factor). Re-keying means a stale alias of the old
        // id can never be handed these values.
        auto entry = refactor_to_entry<T>(_numeric[i], _Ai, _Aj, n_nz, n_col, _Bp.get(), _Bi.get(),
                                          _Bx.get(), _Ax_i, ck, &out_key, &klu_status, &reason, err);
        if (err.failure()) return err;

        _out_numeric[i] = out_key;
        if (_out_rebuild != nullptr) _out_rebuild[i] = (int32_t)reason;

        // A singular refactor/factor fills this element's output with NaN rather
        // than aborting the whole call.
        if (!entry || klu_status < KLU_OK) {
            fill_nan(&_x_temp[n], n_rhs * n_col);
            if (_out_status != nullptr) _out_status[i] = (int32_t)klu_status;
            continue;
        }

        Common.status = KLU_OK;
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
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return refactor_and_solve_impl<double>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        Ax.typed_data(), b.typed_data(), x->typed_data(), out_numeric->typed_data(),
        out_status->typed_data(), out_rebuild->typed_data());
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
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_status,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return refactor_and_solve_impl<Complex>(
        Ai, Aj, Ax.dimensions(), b.dimensions(), symbolic, numeric,
        reinterpret_cast<const Complex*>(Ax.typed_data()),
        reinterpret_cast<const Complex*>(b.typed_data()),
        reinterpret_cast<Complex*>(x->typed_data()),
        out_numeric->typed_data(), out_status->typed_data(), out_rebuild->typed_data());
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
        .Ret<ffi::Buffer<ffi::DataType::U64>>()  // out_numeric (new content keys)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_status (KLU status per batch element)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()  // out_rebuild (RebuildReason per element)
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
        .Ret<ffi::Buffer<ffi::DataType::U64>>()   // out_numeric (new content keys)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_status (KLU status per batch element)
        .Ret<ffi::Buffer<ffi::DataType::S32>>()   // out_rebuild (RebuildReason per element)
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
    T* _x,
    int32_t* _out_rebuild = nullptr) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_numeric = numeric.element_count();
    if (_out_rebuild != nullptr) {
        for (int i = 0; i < n_numeric; i++) _out_rebuild[i] = (int32_t)RebuildReason::NONE;
    }
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
        RebuildReason reason = RebuildReason::NONE;
        auto entry = resolve_numeric<T>(_numeric[num_i], _Ai, _Aj, n_nz, n_col, _Ax_i, err, &reason);
        if (_out_rebuild != nullptr) _out_rebuild[num_i] = (int32_t)reason;
        if (!entry) {
            if (err.failure()) return err;
            // Rebuild hit a singular matrix: NaN this element instead of aborting
            // the whole call, same as a direct singular solve below.
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }

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

// solve_with_numeric_status: as solve_with_numeric, plus a per-numeric-handle
// RebuildReason so a caller can see whether (and why) a handle had to be rebuilt.
ffi::Error solve_with_numeric_status_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return solve_with_numeric_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                           symbolic, numeric, Ax.typed_data(), b.typed_data(),
                                           x->typed_data(), out_rebuild->typed_data());
}

ffi::Error solve_with_numeric_status_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return solve_with_numeric_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                            symbolic, numeric,
                                            reinterpret_cast<const Complex*>(Ax.typed_data()),
                                            reinterpret_cast<const Complex*>(b.typed_data()),
                                            reinterpret_cast<Complex*>(x->typed_data()),
                                            out_rebuild->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_status_f64_handler, solve_with_numeric_status_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // numeric
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (per numeric handle)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    solve_with_numeric_status_c128_handler, solve_with_numeric_status_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // numeric
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()   // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (per numeric handle)

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
    T* _x,
    int32_t* _out_rebuild = nullptr) {
    (void)symbolic;  // numeric handle is self-contained, uses its own analysis

    int n_numeric = numeric.element_count();
    if (_out_rebuild != nullptr) {
        for (int i = 0; i < n_numeric; i++) _out_rebuild[i] = (int32_t)RebuildReason::NONE;
    }
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
        RebuildReason reason = RebuildReason::NONE;
        auto entry = resolve_numeric<T>(_numeric[num_i], _Ai, _Aj, n_nz, n_col, _Ax_i, err, &reason);
        if (_out_rebuild != nullptr) _out_rebuild[num_i] = (int32_t)reason;
        if (!entry) {
            if (err.failure()) return err;
            // Rebuild hit a singular matrix: NaN this element instead of aborting
            // the whole call, same as a direct singular solve below.
            fill_nan(&_x_temp[n], n_rhs * n_col);
            continue;
        }

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

// tsolve_with_numeric_status: as tsolve_with_numeric, plus a per-numeric-handle
// RebuildReason.
ffi::Error tsolve_with_numeric_status_f64(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::F64> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::F64> b,
    ffi::Result<ffi::Buffer<ffi::DataType::F64>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return tsolve_with_numeric_impl<double>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                            symbolic, numeric, Ax.typed_data(), b.typed_data(),
                                            x->typed_data(), out_rebuild->typed_data());
}

ffi::Error tsolve_with_numeric_status_c128(
    const ffi::Buffer<ffi::DataType::S32> Ai,
    const ffi::Buffer<ffi::DataType::S32> Aj,
    const ffi::Buffer<ffi::DataType::C128> Ax,
    const ffi::Buffer<ffi::DataType::U64> symbolic,
    const ffi::Buffer<ffi::DataType::U64> numeric,
    const ffi::Buffer<ffi::DataType::C128> b,
    ffi::Result<ffi::Buffer<ffi::DataType::C128>> x,
    ffi::Result<ffi::Buffer<ffi::DataType::S32>> out_rebuild) {
    return tsolve_with_numeric_impl<Complex>(Ai, Aj, Ax.dimensions(), b.dimensions(), x->dimensions(),
                                             symbolic, numeric,
                                             reinterpret_cast<const Complex*>(Ax.typed_data()),
                                             reinterpret_cast<const Complex*>(b.typed_data()),
                                             reinterpret_cast<Complex*>(x->typed_data()),
                                             out_rebuild->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_status_f64_handler, tsolve_with_numeric_status_f64,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // numeric
        .Arg<ffi::Buffer<ffi::DataType::F64>>()    // b
        .Ret<ffi::Buffer<ffi::DataType::F64>>()    // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (per numeric handle)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    tsolve_with_numeric_status_c128_handler, tsolve_with_numeric_status_c128,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Ai
        .Arg<ffi::Buffer<ffi::DataType::S32>>()    // Aj
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // Ax
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // symbolic
        .Arg<ffi::Buffer<ffi::DataType::U64>>()    // numeric
        .Arg<ffi::Buffer<ffi::DataType::C128>>()   // b
        .Ret<ffi::Buffer<ffi::DataType::C128>>()   // x
        .Ret<ffi::Buffer<ffi::DataType::S32>>());  // out_rebuild (per numeric handle)

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
        if (_numeric[i] != 0) r.erase(_numeric[i], RebuildReason::FREED);
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
    CacheRegistry::instance().erase(sym_id, RebuildReason::FREED);
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
    *symbolic->typed_data() = CacheRegistry::instance().insert(entry, sym->key);
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
    m.def("solve_with_symbol_status_f64",
          []() { return py::capsule((void*)&solve_with_symbol_status_f64_handler); });
    m.def("solve_with_symbol_status_c128",
          []() { return py::capsule((void*)&solve_with_symbol_status_c128_handler); });
    m.def("tsolve_with_symbol_status_f64",
          []() { return py::capsule((void*)&tsolve_with_symbol_status_f64_handler); });
    m.def("tsolve_with_symbol_status_c128",
          []() { return py::capsule((void*)&tsolve_with_symbol_status_c128_handler); });
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
    m.def("solve_with_numeric_status_f64",
          []() { return py::capsule((void*)&solve_with_numeric_status_f64_handler); });
    m.def("solve_with_numeric_status_c128",
          []() { return py::capsule((void*)&solve_with_numeric_status_c128_handler); });
    m.def("tsolve_with_numeric_status_f64",
          []() { return py::capsule((void*)&tsolve_with_numeric_status_f64_handler); });
    m.def("tsolve_with_numeric_status_c128",
          []() { return py::capsule((void*)&tsolve_with_numeric_status_c128_handler); });
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
          []() {
              auto& r = CacheRegistry::instance();
              r.rebuilds.store(0);
              for (auto& c : r.reason_counts) c.store(0);
          });
    // Count of rebuilds recorded for a single RebuildReason (see RebuildReason
    // in klujax.py). Lets rebuild_stats() report a per-reason breakdown.
    m.def("rebuild_reason_count", [](int reason) {
        auto& r = CacheRegistry::instance();
        if (reason < 0 || reason >= kNumRebuildReasons) return (long)0;
        return (long)r.reason_counts[reason].load();
    });
}