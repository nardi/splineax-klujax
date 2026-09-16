---
title: Factorization Safety — Fix Plan
summary: Make a handle name a matrix (a content hash), not a mutable cache slot
---

# Fix plan: content-addressed factorization handles

Goal: kill the failure modes in [Factorization Safety](factorization-safety.md) by
making a handle **name the matrix it represents** instead of naming a mutable cache
slot. The handle becomes a content hash of `(dtype, pattern, values)`. A holder can
then never be handed the wrong matrix — at worst it rebuilds. Identical matrices
hash equal, so the cache also deduplicates.

The design keeps the hot path free of extra hashing.

## The invariant we want

> **A handle equals `hash(content)`, and whatever the cache returns for key `K` was
> built from content whose hash is `K`.**

Everything follows from enforcing this at the two points where content enters the
cache (`build_*`) and never trusting a slot that violates it.

## Core scheme (64-bit content key)

### 1. A hash utility (C++)

Add an inline, dependency-free hash (FNV-1a/wyhash-style 64-bit mixing is enough;
vendor xxHash only if profiling later demands it). It hashes raw bytes and folds in
a type tag.

```cpp
uint64_t content_key(bool is_complex, int n_col, int n_nz,
                     const int* Bp, const int* Bi, const void* Bx, size_t bx_bytes);
```

Hash over the **canonical CSC buffers** `Bp` (n_col+1), `Bi` (n_nz), and `Bx`
(n_nz values) — plus `is_complex`, `n_col`, `n_nz`. `factor`/`refactor` already
compute `Bp/Bi/Bx` via `coo_to_csc_analyze` (`klujax.cpp:94`), so we hash a buffer
that is already in cache — one extra O(n_nz) pass, no extra allocation. For a fixed
`(Ai, Aj)` the CSC order is stable (that function groups by column in input order),
so the key is stable across calls in the reuse loop. Fully sorting within a column
(for cross-ordering dedup) is an optional extra, not required for correctness.

`dtype` in the key is what closes the real/complex hole (failure case 3): a complex
and a real matrix can never share a key.

### 2. Key the cache by content

- `CacheRegistry`: the map stays `uint64 -> shared_ptr<CacheEntry>`, but the key is
  now the content hash. Delete `next_id` / `fresh_id` (`klujax.cpp:245,271`);
  `insert` always takes an explicit key.
- `CacheEntry` gains `bool is_complex` and stores its own key. `SymbolicObj`
  similarly gets a pattern key `hash(n_col, Bp, Bi)`.
- `build_symbolic` / `build_numeric` (`klujax.cpp:314,331`) return the key alongside
  the object.

### 3. Resolve heals under the *true* key

`resolve_symbolic` / `resolve_numeric` (`klujax.cpp:352,377`): on a miss, compute the
key from the caller's arrays and insert under **that** key — never under a
caller-supplied id. This means a forged or desynced token (failure case 2) can only
ever populate the cache with a self-consistent entry; a later hit on that key is
provably the matrix the key describes.

### 4. `factor` returns content keys

`factor_impl` (`klujax.cpp:909`): per left-hand side, compute the key from the CSC
buffers and return it as the numeric id. Before building, look up the key: on a hit
with matching dtype, **reuse** the entry (dedup — skip `klu_factor` entirely). This
is a pure win for repeated identical matrices.

### 5. `refactor` re-keys instead of aliasing

This is the heart of the fix (failure case 1). `refactor_impl`
(`klujax.cpp:1002`) and `refactor_and_solve_impl` (`klujax.cpp:1174`):

1. Compute the new key `K1` from the new values.
2. If `K1` is already resident (dtype matches) → reuse it; done.
3. Else take the old entry for `K0`. If it is **uniquely referenced**
   (`shared_ptr.use_count() == 1` under the lock) → `klu_refactor` it in place
   (keeps pivot reuse — the whole speed reason to call refactor), then **move** it:
   erase `K0`, insert the same object under `K1`. If it is shared, or `K0` is absent
   → build a fresh numeric under `K1` (correct, just no pivot reuse).
4. Return `K1` as the out id.

Why this is correct for the reported case: after refactor, `K0` no longer resolves.
A stale alias still holding `K0` **misses and rebuilds from its own carried `Ax0`**
→ the original matrix, exactly right. The refactored token holds `K1` → the new
matrix. The two can never be confused, because they have different names. Under
`jit` the ordering hazard also disappears: the alias's solve depends only on `K0`,
which the refactor no longer touches.

The `use_count`-guarded in-place path is also what removes the data race (failure
case 5): we mutate a `klu_numeric` only when no other handle can be looking at it;
otherwise we allocate a fresh one.

### 6. Solve trusts the key, dispatches on stored dtype

`solve_with_numeric_impl` (`klujax.cpp:1593`) and the tsolve twin: unchanged hot
path — look up by the token's id (already the content key, computed at factor time),
solve against the resident entry. **No hashing on the solve path.** Add one guard:
if the resolved entry's `is_complex` disagrees with the handler's type, treat it as a
miss and rebuild (the carried `Ax` matches the handler's type), or return
`InvalidArgument`. That is the second half of the case-3 fix — the entry's own dtype,
not `b.dtype`, decides the KLU entry point. Do the same in `rcond` (stop reading
dtype from a user argument; read it off the entry).

## Optional hardening: a second check hash (128-bit safety)

The core scheme's only residual risk is a genuine 64-bit collision between two
*distinct, simultaneously live* matrices — with an ≤8-entry cache, ~2⁻⁵⁹. If we want
that gone with zero hot-path cost:

- Compute a second, independent 64-bit hash `check` at build time (same buffer pass).
- Store it on the entry and **carry it on the token** as one extra `uint64` pytree
  leaf (same per-lhs shape as `id`).
- On a solve/refactor **hit**, compare `entry.check == token.check` — an O(1) integer
  compare, no re-hashing. Mismatch ⇒ treat as a miss ⇒ rebuild (always correct).

That is effectively a 128-bit key while the primary map key and all existing
`astype(uint64)` plumbing stay 64-bit. Cost: one extra FFI output on `factor`/
`refactor`/`refactor_and_solve` (the `check` array) and one extra token field. If
that plumbing isn't wanted now, ship the core first; the token field can be added
later without breaking the id.

## Python changes

- `NumericToken` / `SymbolToken` (`klujax.py:251,303`): no structural change for the
  core scheme — `id` is still a uint64 array, now content-derived. (Add the optional
  `check` leaf + register it in the pytree if adopting the hardening.)
- `refactor` / `refactor_and_solve` docstrings (`klujax.py:648,960`): drop the
  "same cache id" language; document that the returned id reflects the new values and
  that stale aliases of the old handle stay correct (rebuild at worst). The XLA
  dependency edge still holds — the out id is a real, data-dependent output threaded
  into the next solve.
- `rcond` (`klujax.py:765`): read dtype from the token/entry, not a `dtype=` arg
  (keep the arg as an override at most).

## Performance summary

| Path | Added cost |
| --- | --- |
| `factor` | one O(n_nz) hash over a buffer already built; may **save** a full `klu_factor` on dedup |
| `refactor` | one O(n_nz) hash; keeps `klu_refactor` pivot reuse via the in-place-and-move path |
| `solve` / `tsolve` (hit) | none (core); one integer compare (with hardening) |
| `solve` (miss) | unchanged rebuild, now filed under the true key |

No asymptotic change on any path, and a new dedup fast-path. That meets the
"stay performant" bar: hashing rides on work already O(n_nz), and the solve hot loop
gains nothing heavier than an integer compare.

## Rollout

1. C++ hash util + `content_key` unit-tested in isolation (same bytes → same key;
   dtype/pattern/value change → different key).
2. Content-key the registry, `build_*`, `resolve_*`, `factor` (+dedup). Keep behavior
   identical for the single-holder case; all existing tests should pass.
3. Re-key `refactor` / `refactor_and_solve` (in-place-and-move with `use_count`
   guard). Update docstrings and any test asserting id stability.
4. dtype dispatch fix in solve/tsolve/rcond.
5. (Optional) check-hash hardening.

## Regression tests to add

- **Aliasing:** factor → hand the token to two paths → one refactors → the other
  solves → assert it gets the *original* matrix's solution. This is the reported bug;
  it must pass regardless of cache residency.
- **Residency invariance:** solve one handle while resident and again after forcing
  eviction (`KLUJAX_FACTOR_CACHE=1` + churn); assert identical results.
- **Dedup:** factor the same matrix twice; assert `rebuild_count()` and a KLU-call
  counter show the second was reused.
- **dtype guard:** solving a complex-factored handle with a real `b` must not call the
  real entry point on complex memory — assert it errors or rebuilds, never garbage.
- **In-place safety:** a refactor while an alias is outstanding leaves the alias
  correct (covered by the aliasing test under `use_count > 1`).
