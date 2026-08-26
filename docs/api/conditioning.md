---
title: rcond / condest
summary: Cheap conditioning checks on an existing factorization
---

# rcond / condest

```python
klujax.rcond(symbolic, numeric, *, dtype=jnp.float64) -> Array
klujax.condest(Ai, Aj, Ax, symbolic, numeric) -> Array
```

Two ways to ask how trustworthy a factorization is, both reading an existing numeric object rather than redoing work.

`rcond` returns the reciprocal pivot growth estimate `min|Uii| / max|Uii|`, computed by `klu_rcond` in O(n). A value near 1 is well conditioned, a tiny value means the pivots have gone bad, and 0 means the factorization is singular.

`condest` returns a 1-norm condition number estimate from `klu_condest`, using Hager's method as modified by Higham and Tisseur, the same method as MATLAB's `condest`. It is more accurate and more expensive, so the usual pattern is to reach for it only when `rcond` is borderline. A singular factorization gives infinity.

## Why this matters for refactor

[refactor](refactor.md) reuses the pivot order chosen for the original matrix. As the values drift, those pivots can become a poor choice long before the matrix is actually singular, and the solve quietly loses accuracy. `rcond` detects that directly, for a fraction of the cost of solving with a probe vector and measuring the residual.

```python
numeric = klujax.refactor(Ai, Aj, Ax_new, numeric, symbolic)
if klujax.rcond(symbolic, numeric)[0] < 1e-10:
    # pivots have degraded: throw the factorization away and start over
    klujax.free_numeric(numeric)
    numeric = klujax.factor(Ai, Aj, Ax_new, symbolic)
```

## The dtype argument

KLU needs the matching entry point (`klu_rcond` or `klu_z_rcond`), so `rcond` takes the
dtype the factorization was built with rather than reading it off the token. Passing a
dtype that does not match what `numeric.Ax` was factored with calls the wrong entry point,
so keep it in sync with whatever dtype [factor](factor.md) was called with. `condest` avoids
this: it takes `Ax` directly and reads the dtype from it.

## Returns

Both return `float64` of shape `(n_lhs,)`, one value per left-hand side.

## JAX Features

| Feature    | Supported             |
| ---------- | --------------------- |
| `jax.jit`  | Yes                   |
| `jax.vmap` | Yes                   |
| `jax.grad` | No                    |

Neither call writes to the numeric object, so neither is marked as having a side effect.
