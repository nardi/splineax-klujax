---
title: free_symbolic / free_numeric
summary: Optionally release the cache slot behind a KLU handle
---

# free_symbolic / free_numeric

```python
klujax.free_symbolic(symbolic, dependency=None) -> Array
klujax.free_numeric(numeric, dependency=None) -> Array
```

Release the cache slot behind a KLU token. This is **optional**: a token carries
the arrays it needs, so a freed token rebuilds itself on next use rather than
becoming invalid. Call these only to give the memory back sooner.

`dependency` orders the free after a value, for use inside `jax.jit`. Pass the
solution the free should wait for, so the free runs after that solve rather than
racing it. Outside jit it is unnecessary, since Python already runs calls in
order. `token.track(solution)` does the same by folding the solve into the token.

## Parameters

### free_symbolic

| Parameter    | Type            | Description                          |
| ------------ | --------------- | ------------------------------------ |
| `symbolic`   | SymbolToken     | Token from [analyze](../api/analyze.md) |
| `dependency` | Any             | Value the free is ordered after (optional) |

### free_numeric

| Parameter    | Type            | Description                                                              |
| ------------ | --------------- | ------------------------------------------------------------------------ |
| `numeric`    | NumericToken    | Token from [factor](../api/factor.md) or [refactor](../api/refactor.md) |
| `dependency` | Any             | Value the free is ordered after (optional)                              |

## Freeing is optional

Freeing drops the cache entry. The token stays valid, so reusing it just rebuilds
the KLU object from the arrays the token carries:

```python
numeric = klujax.factor(Ai, Aj, Ax, symbolic)
klujax.free_numeric(numeric)          # releases the cache slot

# Still works: this rebuilds the factorization, then solves.
x = klujax.solve_with_numeric(numeric, b, symbolic)
```

The bounded cache also frees tokens for you: once it is full, the least
recently used entry is evicted. So even if you never call these, memory stays
bounded. See [Memory Management](../advanced/memory-management.md) for the cache
size (`KLUJAX_FACTOR_CACHE`) and how to spot rebuilds with `rebuild_count()`.

## When a free actually frees

Calling these is always safe for correctness, at any point. But a free only
reclaims memory without forcing a wasted rebuild when it runs after the token's
last use. That ordering is automatic when you free eagerly in Python. Inside a
single `jax.jit` trace a bare free and a solve on the same token are unordered,
so pass `dependency=solution` or call `token.track(solution)` to order the free
after the solve. See
[When is it safe to free explicitly?](../advanced/memory-management.md#when-is-it-safe-to-free-explicitly)
for the details.
