---
title: refactor_with_status
summary: Re-do LU factorization and report failure as a status code
---

# refactor_with_status

```python
klujax.refactor_with_status(Ai, Aj, Ax, numeric, symbolic) -> tuple[NumericToken, Array]
klujax.refactor_and_solve_with_status(Ai, Aj, Ax, b, numeric, symbolic) -> tuple[Array, NumericToken, Array]
```

Same work as [refactor](refactor.md), but a failed refactorization is reported through a status code instead of raised as an error. Use it when the caller wants to fall back to a fresh [factor](factor.md) rather than lose the whole computation.

`refactor` raises on a singular matrix. Inside `jax.jit` that error aborts the entire computation and cannot be caught or branched on, so there is no way to recover. `refactor_with_status` always succeeds and hands back one status code per left-hand side, which works as a traced value.

## Status codes

`klujax.KLUStatus` mirrors the `KLU_*` constants in SuiteSparse.

| Name            | Value | Meaning                            |
| --------------- | ----- | ---------------------------------- |
| `OK`            | 0     | Refactorization succeeded          |
| `SINGULAR`      | 1     | Zero pivot under the reused order  |
| `OUT_OF_MEMORY` | -2    | Allocation failed                  |
| `INVALID`       | -3    | Invalid input                      |
| `TOO_LARGE`     | -4    | Integer overflow                   |

## Contract on failure

!!! warning "The numeric object after a failed refactor"

    When `status != KLUStatus.OK` the numeric object may be partially overwritten and must not be used for a solve. It remains valid to pass to [free_numeric](free.md), and the symbolic object is unaffected and may be reused for a fresh [factor](factor.md).

## Returns

`refactor_with_status` returns `(numeric, status)`. The returned `NumericToken` carries the
same cache id as the input, refactored in place exactly as [refactor](refactor.md) does.
`status` is `int32` of shape `(n_lhs,)`.

`refactor_and_solve_with_status` returns `(x, numeric, status)`. A failed left-hand side still gets its slice of `x` filled with NaN, so either signal can be used.

## Example: Fall Back Inside JIT

```python
import jax
import jax.numpy as jnp
import klujax

@jax.jit
def solve_step(Ax_new):
    numeric2, status = klujax.refactor_with_status(Ai, Aj, Ax_new, numeric, symbolic)
    ok = status[0] == klujax.KLUStatus.OK
    return jax.lax.cond(
        ok,
        lambda: klujax.solve_with_numeric(numeric2, b, symbolic),
        lambda: jnp.full_like(b, jnp.nan),
    )
```

Note that the branch above cannot itself call `factor`, since both sides of a `lax.cond` allocate. The usual pattern is to return the status to the caller and re-factor outside the jitted region.

## JAX Features

| Feature    | Supported                      |
| ---------- | ------------------------------ |
| `jax.jit`  | Yes                            |
| `jax.vmap` | Yes (status batches along the same axis) |
| `jax.grad` | No                             |
