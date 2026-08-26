# KLUJAX

> version: 0.5.0.post3

A sparse linear solver for JAX based on the efficient [KLU algorithm](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf).

> This is a fork of the original
> [`klujax`](https://github.com/gdsfactory/klujax) package, meant for use in
> [`splineax`](https://github.com/nardi/splineax). The aim is to eventually
> merge the changes here into the upstream package and remove this fork. In the
> meantime, the version number will stay as 0.5.0.postN, and N will be
> incremented sequentially on each new published version.

## CPU & float64

This library is a wrapper around the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) KLU
algorithms. This means the algorithm is only implemented for
C-arrays and hence is **only available for CPU
arrays with double precision**, i.e. float64 or complex128.

Note that `float32`/`complex64` arrays will be cast to `float64`/`complex128`!

## Basic Usage

The `klujax` library provides a basic function `solve(Ai, Aj, Ax, b)`, which solves for `x` in
the sparse linear system `Ax=b`, where `A` is explicitly given in COO-format (`Ai`, `Aj`, `Ax`).

> NOTE: the sparse matrix represented by (`Ai`, `Aj`, `Ax`) needs to be [coalesced](https://pytorch.org/docs/stable/sparse.html#uncoalesced-sparse-coo-tensors)!
> KLUJAX provides a `coalesce` function (which unfortunately is not jax-jittable).

Supported shapes (`?` suffix means optional):

- `Ai`: `(n_nz,)`
- `Aj`: `(n_nz,)`
- `Ax`: `(n_lhs?, n_nz)`
- `b`: `(n_lhs?, n_col, n_rhs?)`
- `A` (represented by (`Ai`, `Aj`, `Ax`)): (`n_lhs?`, `n_col`, `n_col`)

KLUJAX will automatically select a sensible way to act on underdefined dimensions of Ax
and b:

| dim(Ax) | dim(b) | assumed shape(Ax) | assumed shape(b)      |
| ------- | ------ | ----------------- | --------------------- |
| 1D      | 1D     | n_nz              | n_col                 |
| 1D      | 2D     | n_nz              | n_col x n_rhs         |
| 1D      | 3D     | n_nz              | n_lhs x n_col x n_rhs |
| 2D      | 1D     | n_lhs x n_nz      | n_col                 |
| 2D      | 2D     | n_lhs x n_nz      | n_lhs x n_col         |
| 2D      | 3D     | n_lhs x n_nz      | n_lhs x n_col x n_rhs |

Where the `A` is always acting on the `n_col` dimension of `b`. The `n_lhs` dim is a
shared batch dimension between `A` and `b`.

Additional dimensions can be added with `jax.vmap` (alternatively any higher dimensional
problem can be reduced to the one above by properly transposing and reshaping `Ax` and `b`).

> NOTE: JAX now has an experimental sparse library (`jax.experimental.sparse`). Using
> this natively in KLUJAX is not yet supported (but converting from `BCOO` or `COO` to
> `Ai`, `Aj`, `Ax` is trivial).

## Basic Example

Script:

```python
import klujax
import jax.numpy as jnp

b = jnp.array([8, 45, -3, 3, 19])
A_dense = jnp.array(
    [
        [2, 3, 0, 0, 0],
        [3, 0, 4, 0, 6],
        [0, -1, -3, 2, 0],
        [0, 0, 1, 0, 0],
        [0, 4, 2, 0, 1],
    ]
)
Ai, Aj = jnp.where(jnp.abs(A_dense) > 0)
Ax = A_dense[Ai, Aj]

result_ref = jnp.linalg.inv(A_dense) @ b
result = klujax.solve(Ai, Aj, Ax, b)

print(jnp.abs(result - result_ref) < 1e-12)
print(result)
```

Output:

```
[ True True True True True]
[1. 2. 3. 4. 5.]
```

## Advanced Usage

For high-performance applications like transient simulations or iterative solvers, you should avoid using the high-level `klujax.solve` function. The `klujax.solve` is in fact a wrapper around three distinct parts of the KLU algorithm:

1. **Analyze (Symbolic)**: Inspects the sparsity pattern ($A_i, A_j$) to find optimal permutations and block triangular forms. This depends only on the structure of the matrix.
2. **Factorize (Numeric)**: Performs the actual LU decomposition. This depends on the values ($A_x$) and requires a symbolic handle.
3. **Solve (Numeric)**: Executes forward and backward substitution to find $x$. This depends on the right-hand side ($b$) and requires a numeric handle.

Significant performance gains are achieved by hoisting the "Analysis" or "Factorization" steps out of your inner loops.

### 1. High-Performance Transient Pattern (Reusing Symbolic)

In a simulation where the sparsity pattern is constant but the values ($A_x$) and right-hand side ($b$) change, you should perform the expensive `analyze` step exactly once outside your JIT loop.

```python
import jax
import klujax

# 1. Analyze once in Python (CPU)
# Returns a SymbolToken: a cache handle that is freed automatically
symbolic = klujax.analyze(Ai, Aj, n_col)

@jax.jit
def simulation_step(Ax_t, b_t, sym):
    # 2. Use the symbolic handle inside JIT
    # The solver will perform numeric factorization and solve
    return klujax.solve_with_symbol(Ai, Aj, Ax_t, b_t, sym)

for t in range(steps):
    x_t = simulation_step(Ax[t], b[t], symbolic)

```

### Fine-Grained Control (Numeric Factorization)

If you need to solve the same system with many different $b$ vectors while the matrix $A$ remains constant, you can further split the numeric factorization. This is often performed in a modified Newton-Raphson loop where the computationally expensive jacobian+factorization is only evaluated once and the **solve** stage is deemed "cheap" in comparison

```python
# Factorize the matrix once
numeric = klujax.factor(Ai, Aj, Ax, symbolic)

@jax.jit
def fast_solve(b_t, num, sym):
    # This call is extremely fast as it skips factorization entirely
    return klujax.solve_with_numeric(num, b_t, sym)

for i in range(100):
    x_i = fast_solve(b_batch[i], numeric, symbolic)
```

### Safe Refactorization (Status Codes & Conditioning)

`klujax.refactor` reuses the pivot order picked for the original matrix. That is what makes it fast, but it also means the factorization can fail, or silently lose accuracy, once the values have drifted far enough. Two additions make that recoverable.

`klujax.refactor_with_status` reports a failure through a status code rather than raising, so it can be branched on under `jax.jit` where an error would abort everything. `klujax.refactor_and_solve_with_status` does the same for the fused path.

```python
numeric, status = klujax.refactor_with_status(Ai, Aj, Ax_new, numeric, symbolic)
if status[0] != klujax.KLUStatus.OK:
    # the numeric object is unusable for a solve, but the symbolic one is fine
    klujax.free_numeric(numeric)
    numeric = klujax.factor(Ai, Aj, Ax_new, symbolic)
```

When `status != OK` the numeric object may be partially overwritten and must not be used for a solve. It remains valid to pass to `free_numeric`, and the symbolic object is unaffected and may be reused for a fresh `factor`.

Degradation short of outright failure is caught by `klujax.rcond`, the reciprocal pivot growth estimate `min|Uii| / max|Uii|`. It costs O(n), far less than solving with a probe vector and measuring the residual. `klujax.condest` gives a proper 1-norm condition number estimate and is the usual follow-up when `rcond` is borderline.

```python
if klujax.rcond(symbolic, numeric)[0] < 1e-10:
    ...  # pivots have degraded, re-factor from scratch
```

### Lifecycle & Memory Safety

`klujax.analyze` and `klujax.factor` return handle tokens (`SymbolToken`,
`NumericToken`). A token is a small cache id bundled with the arrays it can
rebuild from, not a raw pointer, so handles are memory-safe by construction.

- **Forgetting to free leaks only a bounded amount.** Handles live in a
  process-wide cache that holds a fixed number of KLU objects (eight by
  default, set by `KLUJAX_FACTOR_CACHE`). When it overflows, the least recently
  used object is evicted.

- **Using a freed or evicted handle is safe.** Every call carries the matrix the
  handle needs, so a call that lands on a missing handle rebuilds it on the spot
  and continues. This is why freeing is optional and there is no ghost-pointer
  hazard: there is no pointer to dereference.

Because of this, creating a token inside `jax.jit` needs no special care. The id
threads through the trace as data. A bare free is unordered against the solve, so
to actually free inside the trace, order it after the solve by tracking the
solution (or passing it as `dependency`):

```python
@jax.jit
def dynamic_solve(Ai, Aj, Ax, b):
    sym = klujax.analyze(Ai, Aj, 5)
    x = klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)
    klujax.free_symbolic(sym.track(x))   # ordered after the solve
    return x
```

#### Summary of Best Practices

1. **Hoist Creations**: for best performance, call `analyze` or `factor` once
   outside JIT loops and reuse the handle.

2. **Freeing is optional**: use `free_symbolic` / `free_numeric` (or the token's
   `close()`, or a `with` block) only to release memory sooner. The token stays
   usable afterwards.

3. **Watch `rebuild_count()`**: a count that climbs during steady-state solving
   means the working set is larger than `KLUJAX_FACTOR_CACHE`. Set
   `KLUJAX_STRICT_CACHE` to turn a rebuild into an error while debugging. See
   the [Memory Management](docs/advanced/memory-management.md) guide.

## Installation

The library is statically linked to the SuiteSparse C++ library. It can be installed on
most platforms as follows:

```bash
pip install splineax-klujax
```

**There exist pre-built wheels for Linux and Windows (python 3.8+).** If no compatible
wheel is found, however, pip will attempt to install the library from source... make
sure you have the necessary build dependencies installed (see [Installing from Source](#installing-from-source))

## Installing from Source

> NOTE: Installing from source should only be necessary when developing the library. If
> you as the user experience an install from source please create an issue.

Before installing, clone the build dependencies:

```sh
git clone --depth 1 --branch v7.2.0 https://github.com/DrTimothyAldenDavis/SuiteSparse suitesparse
git clone --depth 1 --branch main https://github.com/openxla/xla xla
git clone --depth 1 --branch stable https://github.com/pybind/pybind11 pybind11
```

### Linux

On linux, you'll need `gcc` and `g++`, then inside the repo:

```sh
pip install .
```

### MacOs

On MacOS, you'll need `clang`, then inside the repo:

```sh
pip install .
```

### Windows

On Windows, installing from source is a bit more involved as typically the build
dependencies are not installed. To install those, download Visual Studio Community 2017
from [here](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads). During installation, go to Workloads and select the following workloads:

- Desktop development with C++
- Python development

Then go to Individual Components and select the following additional items:

- C++/CLI support
- VC++ 2015.3 v14.00 (v140) toolset for desktop

Then, download and install Microsoft Visual C++ Redistributable from [here](https://aka.ms/vs/16/release/vc_redist.x64.exe).

After these installation steps, run the following commands inside a x64 Native Tools
Command Prompt for VS 2017:

```cmd
set DISTUTILS_USE_SDK=1
pip install .
```

## License & Credits

© Floris Laporte 2022, LGPL-2.1

This library was partly based on:

- [torch_sparse_solve](https://github.com/flaport/torch_sparse_solve), LGPL-2.1
- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3

This library vendors an unmodified version of the
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) libraries in its source
(.tar.gz) distribution to allow for static linking.
This is in accordance with their
[LGPL licence](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/LICENSE.txt).
