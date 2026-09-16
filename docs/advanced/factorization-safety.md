---
title: Factorization Safety
summary: Failure modes of the handle cache beyond use-after-free, and how a content-addressed key fixes them
---

# Factorization safety

The cache was built to make one hazard impossible: **use after free**. A handle
is a `uint64` id into a process-global cache, never a raw pointer, and every call
carries the arrays needed to rebuild the object it names, so a stale id rebuilds
instead of dereferencing freed memory (see [Memory Management](memory-management.md)).

That property holds. But it is not the only way to get a wrong answer out of a
handle, and the remaining failure modes all share one root cause.

## Root cause: an id names a mutable slot, not a value

A handle is supposed to mean *"this numeric factorization."* In the current design
it actually means *"cache slot number N,"* and the contents of slot N can change
under a holder's feet. Two facts combine:

1. **`refactor` / `refactor_and_solve` mutate the slot in place and keep the id.**
   `refactor_impl` calls `klu_refactor` on `entry->numeric` and returns the *same*
   id (`klujax.cpp:1046-1062`). The docstring says so plainly: *"Returns a
   NumericToken with the same cache id as the input."*

2. **On a cache hit, the solve trusts the resident object and ignores the arrays
   the token carries.** `resolve_numeric` returns the resident entry as soon as it
   finds one (`klujax.cpp:382-384`); `solve_with_numeric_impl` then solves against
   `entry->numeric` (`klujax.cpp:1693-1696`). The token's `Ax` is consulted *only*
   on the rebuild (miss) path.

So the invariant the whole scheme leans on — *"the id and the `(Ai, Aj, Ax)` a token
carries describe the same matrix"* — is assumed everywhere and enforced nowhere.
`refactor` deliberately breaks it, and nothing else restores it. Worse, **which
copy wins depends on cache residency**: a resident slot wins on a hit, the carried
arrays win on a miss. Residency is an LRU/timing artifact, so correctness becomes
nondeterministic.

Every failure below is an instance of this one problem.

## Failure cases

### 1. Aliasing through `refactor` (the reported case)

```python
num  = klujax.factor(Ai, Aj, Ax0, sym)     # id H, slot H = factor(Ax0)
# num handed to two places that don't know about each other
num1 = klujax.refactor(Ai, Aj, Ax1, num, sym)   # slot H mutated to factor(Ax1)
x    = klujax.solve_with_numeric(num, b, sym)    # still holds id H, expects Ax0
```

`num` and `num1` share id `H`. The refactor overwrote slot `H` in place. The final
solve hits the cache and solves with `factor(Ax1)` while the caller believes it is
using `Ax0`. **Silent wrong answer.**

The insidious part is the failure is residency-dependent: if slot `H` had been
*evicted* before the last solve, `resolve_numeric` would rebuild from `num.Ax ==
Ax0` and return the *correct* answer. So the same code is right or wrong depending
on how many unrelated factorizations happened to pass through an 8-slot LRU in
between.

Under `jax.jit` it is worse still. The refactor on `num1` and the solve on `num`
share no data edge (they are different tokens as far as XLA can see), so XLA is
free to order the in-place refactor *before or after* the solve. The result then
depends on XLA scheduling, not just residency.

### 2. Desynchronized token fields

`NumericToken` exposes `id`, `Ai`, `Aj`, `Ax` as public pytree leaves. Anything
that rebuilds a token from parts, swaps a leaf, or lines batch dimensions up
wrongly (a `tree_map`, a manual `NumericToken(...)`, a slice of the id array
without the matching slice of `Ax`) produces a token whose id and carried arrays
disagree. Same consequence as case 1: hit uses the resident object, miss uses the
carried arrays, answer depends on residency. The batched `factor` path is the easy
place to trip on this — the id is `(n_lhs,)` and `Ax` is `(n_lhs, n_nz)`, and the
broadcast logic in `solve_with_numeric_impl` (`n_ax == 1` vs per-lhs) assumes they
stay aligned.

### 3. dtype confusion — the one memory-safety hole left

The handle carries no dtype. The real/complex entry point is chosen elsewhere:

- `solve_with_numeric` picks the primitive from **`b.dtype`** (`klujax.py:856`).
- `rcond` picks it from a user-supplied **`dtype=` argument** (`klujax.py:794`).

If a numeric was factored as `complex128` but a later `solve_with_numeric` is
handed a real `b`, then on a cache **hit** the real `klu_solve` is called on a
`klu_z_numeric` object — a real routine reinterpreting a complex factorization's
memory. This is type confusion at the C boundary, not merely a wrong number, and
unlike the other cases it can bite even without a refactor. On a **miss** it
instead rebuilds the numeric from `Ax` cast to real — a *different* factorization —
so once again the two paths disagree. The in-code comment already flags the
fragility: *"It must match the solve's real/complex kind, which b selects."*
Nothing enforces it.

### 4. Stale-pivot degradation is invisible downstream

`refactor` reuses the symbolic pivot order. When values move enough that the old
pivots are numerically poor, the factorization is inaccurate — that is what
`rcond`/`condest` exist to detect. But because the id is unchanged, a consumer
holding "the same" numeric handle has no way to know its contents were replaced by
a lower-quality refactor. Identity reuse hides a quality change that the holder
might have wanted to gate on.

### 5. Concurrency

KLU work runs outside the registry lock by design (the lock only guards the map).
That is fine for immutable entries, but `refactor` **mutates** `entry->numeric`
while `klu_refactor` runs, holding no lock. Two threads refactoring the same
resident id, or one refactoring while another solves the same id, race on one
`klu_numeric*`. That is a data race and undefined behavior, again gated on the slot
being resident (co-tenancy) rather than evicted.

### 6. Heal-under-same-id can resurrect a slot

`resolve_*` heals a miss by re-inserting under the *same* id (`insert(e, id)`,
`klujax.cpp:369, 405`). Combined with in-place mutation this means a freed id is
not gone: a later use rebuilds it from whatever arrays that call happens to carry.
If two tokens ever share an id but carry different arrays (cases 1–2), the last
rebuilder decides what the id means for everyone.

## Assessment of the content-addressed-key idea

The proposal — *key the cache by something that reflects which matrix the
factorization represents, ideally a hash so identical matrices dedupe* — attacks
the root cause directly and is the right direction. It turns an id from "slot N"
into "the value `factor(Ai, Aj, Ax, dtype)`," which restores the assumed
invariant by construction.

What it fixes:

- **Cases 1, 2, 6.** If the key is derived from `(Ai, Aj, Ax, dtype)`, then
  `factor(Ax0)` and a refactor to `Ax1` land on *different* keys. A holder of the
  `Ax0` key always resolves to `factor(Ax0)`; at worst it rebuilds, never returns
  the wrong matrix. Residency stops affecting correctness and only affects speed —
  which is exactly the guarantee the use-after-free design was reaching for,
  extended to *which* matrix rather than just *a valid* matrix.
- **Case 3 (partly).** Fold dtype into the key and the real/complex rebuild becomes
  correct. But keying alone does not fix solve-side dispatch: the resolved entry
  must also *carry its dtype and dispatch the solve off it*, instead of off
  `b.dtype`. Do both.
- **Dedup.** Two call sites (or two loop iterations) with a bitwise-identical
  matrix share one factorization for free. This also makes the "two functions"
  pattern that started this investigation share safely instead of collide.

Sharp edges to design around:

- **Collisions are silently wrong.** A pure 64-bit hash that collides returns the
  wrong factorization — reintroducing exactly the bug class we are removing. Either
  use a wide digest *and* verify `(Ai, Aj, Ax, dtype)` against the stored key on a
  hit, or store the key material and compare on hit. Verify-on-hit costs a full
  compare but is the only way to make "hit" mean "provably the same matrix." Given
  the arrays already travel with every call, the material is in hand.
- **Hashing is on the hot path.** Hashing `Ax` is O(n_nz) per call, on top of the
  O(n_nz) COO→CSC copy already done each call — same order, but it is real overhead
  in a tight refactor loop. Hash over the raw bytes actually used (post-CSC `Bx`
  and the pattern) so it is one pass, and consider caching the key on the Python
  token so a reused token skips re-hashing.
- **Exact bits, not values.** The key must be over bit patterns. `-0.0`/`+0.0` and
  NaN payloads make "equal" matrices hash differently; that only costs a missed
  dedup (still correct), but it means dedup helps only for bitwise-identical inputs
  — and under `jit` re-tracing those may not recur. Treat dedup as a bonus, not a
  guarantee.
- **`refactor` loses its in-place identity.** Today `refactor` returns the same id
  so XLA sees a `refactor -> solve` edge and `klu_refactor` reuses the numeric's
  storage and pivots. Content keys make refactor produce a *new* key. That is more
  correct (the edge now reflects the new value, and stale aliases can't form), and
  token threading still gives XLA the edge — but you can no longer mutate the old
  key's slot, so decide deliberately whether refactor (a) allocates a fresh numeric
  under the new key, or (b) `klu_refactor`s into a copy. You keep the pivot-reuse
  speed either way; you lose only the in-place aliasing that caused case 1.

A cheaper middle option, if hashing overhead or the refactor redesign is
unwelcome: keep monotonic ids but make **`refactor` mint a fresh id** (immutable
slots, copy-on-refactor) and **carry dtype in the entry and dispatch off it**. That
alone kills cases 1, 2, 5, 6 without any hashing; the content hash then becomes a
pure dedup/perf optimization layered on top, not a correctness dependency. Whether
the key is a hash or a fresh monotonic id, the correctness fix is the same move:
**stop reusing an id across a change of contents.**

## Test gap

The suite covers eviction, rebuild, and explicit free thoroughly
(`test_forgotten_handles_are_bounded_and_rebuild`,
`test_strict_mode_turns_a_rebuild_into_an_error`,
`test_released_handle_rebuilds_on_next_solve`), but there is no test where **two
live tokens share an id and one is refactored**, nor one that pins the
residency-dependent divergence (solve the same handle once while resident and once
after forcing eviction, and assert both agree). Those are the regression tests any
fix here should ship with.
