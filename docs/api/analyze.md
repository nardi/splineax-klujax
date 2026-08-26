---
title: analyze
summary: Symbolic analysis of sparsity pattern
---

# analyze

```python
klujax.analyze(Ai, Aj, n_col) -> SymbolToken
```

Perform symbolic analysis on the sparsity pattern of a sparse matrix. This is the first (and most expensive) stage of the KLU algorithm. It studies **where** the nonzeros are (not their values) to find optimal orderings and block structures for the subsequent factorization.

## Parameters

| Parameter | Type  | Shape     | Description                          |
| --------- | ----- | --------- | ------------------------------------ |
| `Ai`      | int32 | `(n_nz,)` | Row indices of nonzero entries       |
| `Aj`      | int32 | `(n_nz,)` | Column indices of nonzero entries    |
| `n_col`   | int   | scalar    | Number of rows/columns in the matrix |

## Returns

| Type               | Description                                                     |
| ------------------ | --------------------------------------------------------------- |
| `SymbolToken` | A handle to the symbolic analysis: a cache id plus the `Ai`, `Aj` it rebuilds from |

## How It Fits In

```mermaid
flowchart TD
    AN["analyze#40;Ai, Aj, n_col#41;"]:::active --> SYM["symbolic handle"]
    SYM --> F["factor#40;Ai, Aj, Ax, symbolic#41;"]
    SYM --> SWS["solve_with_symbol#40;Ai, Aj, Ax, b, symbolic#41;"]
    F --> NUM["numeric handle"]
    NUM --> SWN["solve_with_numeric#40;numeric, b, symbolic#41;"]

    classDef active fill:#f59e0b,color:#fff,stroke:none
```

The symbolic handle is used by:

- [factor](factor.md) — to compute LU decomposition
- [refactor](refactor.md) — to re-compute LU decomposition with new values
- [solve_with_symbol](solve-with-symbol.md) — to solve while skipping the analyze step
- [solve_with_numeric](solve-with-numeric.md) — also needs the symbolic handle

## Example

```python
import klujax
import jax.numpy as jnp

Ai = jnp.array([0, 1, 2], dtype=jnp.int32)
Aj = jnp.array([0, 1, 2], dtype=jnp.int32)
n_col = 3

# Analyze the sparsity pattern once
symbolic = klujax.analyze(Ai, Aj, n_col)

# Use it many times with different Ax and b values
for Ax_t, b_t in simulation_data:
    x = klujax.solve_with_symbol(Ai, Aj, Ax_t, b_t, symbolic)
```

## Memory Management

The returned `SymbolToken` is a handle into a bounded cache, not a raw pointer, so there is
nothing you must do with it. Let it sit unused and the cache evicts it once it overflows
(`KLUJAX_FACTOR_CACHE`, eight entries by default). If you want the memory back earlier, free
it sooner with `symbolic.close()`, a `with` block, or [free_symbolic](free.md). See
[Memory Management](../advanced/memory-management.md) for the full picture, including calling
`analyze` inside `jax.jit`.

```python
with klujax.analyze(Ai, Aj, n_col) as symbolic:
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
# Freed on exit. Still fine to reuse `symbolic` afterwards, it just rebuilds.
```

!!! note "Not JIT-compiled itself"
    `analyze` is a Python-side function — it runs eagerly on the CPU. Call it **outside** your JIT-compiled loops.
