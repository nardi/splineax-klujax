---
title: Rebuild Visibility — Plan
summary: Surface per-call rebuild events as a RebuildReason status flag
---

# Plan: per-call rebuild visibility

> **Status: implemented.** `RebuildReason`, the tombstone-backed reasons, the
> `solve/tsolve_with_numeric_with_status` twins, the extended `refactor_status` /
> `refactor_and_solve_status`, and `rebuild_stats()` have landed. See the
> [Changelog](../../CHANGELOG.md).

Under the [content-addressed handle fix](factorization-safety-fix-plan.md) a rebuild
is the mechanism that keeps a stale handle *correct* — but it is silent per call
(only the process-global `rebuild_count()` moves). This plan makes each call able to
report **whether it rebuilt, and why**, as a per-left-hand-side `RebuildReason` flag,
without adding cost to the default path.

## What "why" can actually mean

A rebuild happens when `resolve_*` cannot use the resident entry for a handle. With
content keys, the distinguishable causes are:

- The key **isn't in the cache** — it was evicted (LRU overflow), freed
  (`free_numeric`), or **superseded** (a `refactor` elsewhere re-keyed this handle —
  this is the reported aliasing pattern), or simply never seen.
- The key **is** in the cache but the entry's **dtype** disagrees (real `b` against a
  complex factorization).
- The key is in the cache but a **content check** disagrees (a genuine hash collision
  or a hand-forged token — only detectable if the optional check-hash hardening is
  adopted).

A bare key lookup collapses evicted / freed / superseded into one "missing." To tell
them apart — and the superseded case is precisely the one worth seeing — we keep a
small **tombstone** of retired keys with the reason they retired.

## RebuildReason enum

Mirror `KLUStatus` (klujax.py:210): a C++ enum plus a Python `enum.IntEnum`.

```python
class RebuildReason(enum.IntEnum):
    """Why a call rebuilt a factorization from the token's carried arrays.

    NONE means the cached factorization was used directly (no rebuild).
    Everything else means the handle's KLU object was not usable and was
    rebuilt — always correct, but it costs the rebuild.
    """
    NONE       = 0   # cache hit, no rebuild
    EVICTED    = 1   # fell out of the bounded cache (raise KLUJAX_FACTOR_CACHE)
    FREED      = 2   # free_numeric / close() had dropped it (expected)
    SUPERSEDED = 3   # a refactor re-keyed this handle — a stale alias was used
    DTYPE      = 4   # entry present but factored with the other dtype
    UNKNOWN    = 5   # never resident, or its tombstone was itself evicted
    # STALE   = 6    # (only with the check-hash hardening) collision / forged token
```

`SUPERSEDED` is the direct signal for the bug that started this: a handle handed to
two places, one refactors, the other still uses the old token.

## C++ changes

1. **Tombstone map in `CacheRegistry`** (klujax.cpp:241). A bounded
   `std::map<uint64_t,uint8_t>` plus its own LRU list, capped like the main cache.
   Written on the cold paths only:
   - `evict()` (klujax.cpp:282) records the victim key → `EVICTED`.
   - `erase()` / `free_numeric` path (klujax.cpp:306) records → `FREED`.
   - the re-key/move in the new `refactor` records the old key → `SUPERSEDED`.
   No hot-path cost: it is touched only when an entry leaves, and read only on a miss.

2. **`resolve_*` returns a reason** (klujax.cpp:352,377). Replace the existing
   `bool* rebuilt` out-param (klujax.cpp:379) with `RebuildReason* reason`:
   - hit, dtype ok → `NONE`;
   - hit, dtype mismatch → rebuild, `DTYPE`;
   - miss → consult the tombstone: `EVICTED` / `FREED` / `SUPERSEDED`, else `UNKNOWN`.
   The existing internal callers already thread `rebuilt` to decide refactor-vs-not,
   so this is a type change, not new call sites.

3. **Optional per-reason counters** alongside `rebuilds` (klujax.cpp:246): a small
   `std::atomic<long>` per reason, exposed as a dict (see below). Cheap aggregate
   visibility even for callers who don't switch to the status ops.

## Surfacing it per call

Follow the existing `_status`-twin pattern: the plain ops stay silent and
zero-overhead; the observable variants carry an extra `rebuild` output. The op is
already computing the reason internally, so the only addition is one `int32[n_lhs]`
result buffer.

- **Extend the existing status ops** (pre-1.0, advanced surface — acceptable break,
  changelog-noted):
  - `refactor_status` → also returns `rebuild[]` (klujax.cpp:1115, py:698).
  - `refactor_and_solve_status` → also returns `rebuild[]` (klujax.cpp:1335, py:1000).
- **Add status twins where none exist**, for the ops where a rebuild is the
  interesting event:
  - `solve_with_numeric_status` → `(x, rebuild[])` (klujax.cpp:1593, py:871).
  - `tsolve_with_numeric_status` → `(x, rebuild[])` (klujax.cpp:1769, py:910).
- **Phase 2 (same pattern, optional):** `factor` (reports its symbolic rebuild and a
  `DEDUP` hit), `solve_with_symbol` / `tsolve_with_symbol` (symbolic rebuild),
  `rcond` / `condest`. Add only if there's demand — the numeric four cover the
  reported case.

Each new/extended handler adds one `.Ret<Buffer<S32>>()` and writes `reason[i]` in the
per-lhs loop it already runs. Under `jit` the `rebuild` array is an ordinary traced
output, so a caller can branch on it (e.g. re-pin a factorization when it sees
`SUPERSEDED`/`EVICTED` churn) — which the global counter can never support.

## Python changes

- `RebuildReason` IntEnum + add to `__all__`.
- New wrappers return the array as a plain leaf next to the result, matching the
  `*_with_status` shape already in the API:

  ```python
  x, rebuild = klujax.solve_with_numeric_with_status(numeric, b, symbolic)
  # rebuild: int32[n_lhs] of RebuildReason
  ```

  Keep the silent `solve_with_numeric` unchanged as the default.
- Optional `klujax.rebuild_stats() -> dict[RebuildReason,int]` reading the per-reason
  counters, complementing the scalar `rebuild_count()` (kept for compat).
- Docstrings: on the status variants, state that a nonzero reason means a rebuild
  occurred (correct but costlier) and what each value implies; call out `SUPERSEDED`
  as the stale-alias signal.

## Zero-overhead guarantee

| Path | Added cost |
| --- | --- |
| plain `solve` / `refactor` / … (default) | none — reason computed but not emitted; tombstone untouched on a hit |
| status variants, cache hit | one `int32` store per lhs (`NONE`) |
| status variants, miss | one bounded tombstone lookup (already off the hot path — it's a rebuild) |
| entry leaving cache (evict/free/refactor-move) | one bounded tombstone insert |

The tombstone is written only when an entry retires and read only when a handle
misses, so steady-state solving that hits the cache pays nothing.

## Interaction with the fix plan

This layers cleanly on the content-addressing work and sharpens it:

- The re-key/move `refactor` is what lets a miss be labeled `SUPERSEDED` rather than a
  vague "gone" — the two plans reinforce each other.
- `DTYPE` here is the observable face of the same dtype-dispatch fix.
- `STALE` only exists if the check-hash hardening is taken; leave it reserved
  otherwise.

Recommended order: land the content-addressing core, then this on top, then the
optional check-hash + `STALE`.

## Regression tests to add

- **SUPERSEDED:** factor → refactor via a second token → solve the *first* token
  through `solve_with_numeric_with_status`; assert the result is the original matrix's
  solution **and** `rebuild == SUPERSEDED`.
- **EVICTED:** `KLUJAX_FACTOR_CACHE=1` + churn, then solve an old handle; assert
  `rebuild == EVICTED`.
- **FREED:** `free_numeric` then solve; assert `rebuild == FREED`.
- **NONE:** resident handle; assert `rebuild == NONE` and `rebuild_count()` unchanged.
- **DTYPE:** complex handle solved with real `b`; assert it does not misdispatch and
  reports `DTYPE` (or errors, per the fix plan's chosen behavior).
- **jit branch:** inside `jax.jit`, branch on `rebuild` and assert the branch is taken
  when a rebuild is forced — proving in-computation visibility the counter can't give.
