"""klujax: a KLU solver for JAX."""

# Metadata ============================================================================

__version__ = "0.5.0"
__author__ = "Floris Laporte"
__all__ = [
    "KLUStatus",
    "NumericToken",
    "RebuildReason",
    "SymbolToken",
    "analyze",
    "coalesce",
    "condest",
    "dot",
    "factor",
    "free_numeric",
    "free_symbolic",
    "rcond",
    "rebuild_count",
    "rebuild_stats",
    "refactor",
    "refactor_and_solve",
    "refactor_and_solve_with_status",
    "refactor_with_status",
    "reset_rebuild_count",
    "solve",
    "solve_with_numeric",
    "solve_with_numeric_with_status",
    "solve_with_symbol",
    "solve_with_symbol_with_status",
    "tsolve_with_numeric",
    "tsolve_with_numeric_with_status",
    "tsolve_with_symbol",
    "tsolve_with_symbol_with_status",
]

# Imports =============================================================================

import contextlib
import enum
import os
import sys
from functools import partial
from typing import Any, Self, cast

import jax
import jax._src.core
import jax._src.effects
import jax.core
import jax.extend.core
import jax.numpy as jnp
import klujax_cpp  # ty: ignore[unresolved-import]
import numpy as np
from jax.core import ShapedArray
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array

# Config ==============================================================================

DEBUG = bool(int(os.environ.get("KLUJAX_DEBUG", "0")))
debug = lambda s: None if not DEBUG else print(s, file=sys.stderr)  # noqa: E731,T201
debug("KLUJAX DEBUG MODE.")


def _require_x64() -> None:
    """Raise if JAX 64-bit mode is disabled.

    klujax casts inputs to float64/complex128; without x64 these silently
    truncate to 32-bit, giving wrong results. So we require x64 explicitly.
    """
    if not jax.config.jax_enable_x64:  # ty:ignore[unresolved-attribute]
        msg = (
            "klujax requires JAX 64-bit mode, but it is disabled. "
            "Enable it globally:\n"
            "    import jax\n"
            "    jax.config.update('jax_enable_x64', True)\n"
            "or scope it with the context manager:\n"
            "    with jax.experimental.enable_x64():\n"
            "        ...  # klujax calls here"
        )
        raise RuntimeError(msg)


# Constants ===========================================================================

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)

# Main Functions ======================================================================


@jax.jit
def solve(Ai: Array, Aj: Array, Ax: Array, b: Array) -> Array:
    """Solve for x in the sparse linear system Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector

    Returns:
        x: the result (x≈A^-1b)

    """
    _require_x64()
    debug("solve")
    Ai, Aj, Ax, b, shape = validate_args(Ai, Aj, Ax, b, x_name="b")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, b)):
        debug("solve-complex128")
        x = solve_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            b.astype(jnp.complex128),
        )
    else:
        debug("solve-float64")
        x = solve_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            b.astype(jnp.float64),
        )

    return x.reshape(*shape)


@jax.jit
def dot(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    """Multiply a sparse matrix with a vector: Ax=b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        x:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the vector multiplied by A

    Returns:
        b: the result (b=A@x)

    """
    _require_x64()
    debug("dot")
    Ai, Aj, Ax, x, shape = validate_args(Ai, Aj, Ax, x, x_name="x")
    if any(x.dtype in COMPLEX_DTYPES for x in (Ax, x)):
        debug("dot-complex128")
        b = dot_c128.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.complex128),
            x.astype(jnp.complex128),
        )
    else:
        debug("dot-float64")
        b = dot_f64.bind(
            Ai.astype(jnp.int32),
            Aj.astype(jnp.int32),
            Ax.astype(jnp.float64),
            x.astype(jnp.float64),
        )
    return b.reshape(*shape)


def coalesce(
    Ai: Array,
    Aj: Array,
    Ax: Array,
) -> tuple[Array, Array, Array]:
    """Coalesce a sparse matrix by summing duplicate indices.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [... x n_nz; float64|complex128]: the values of the sparse matrix A

    Returns:
        coalesced Ai, Aj, Ax

    """
    with jax.ensure_compile_time_eval():
        shape = Ax.shape

        order = jnp.lexsort((Aj, Ai))
        Ai = Ai[order]
        Aj = Aj[order]

        # Compute unique indices
        unique_mask = jnp.concatenate(
            [jnp.array([True]), (Ai[1:] != Ai[:-1]) | (Aj[1:] != Aj[:-1])],
        )
        unique_idxs = jnp.where(unique_mask)[0]

        # Assign each entry to a unique group
        groups = jnp.cumsum(unique_mask) - 1

        # Sum Ax values over groups
        Ai = Ai[unique_idxs]
        Aj = Aj[unique_idxs]

    Ax = Ax.reshape(-1, shape[-1])
    Ax = Ax[:, order]
    Ax = jax.vmap(jax.ops.segment_sum, [0, None], 0)(Ax, groups)

    return Ai, Aj, Ax.reshape(*shape[:-1], -1)


# Split Solve handle tokens ==========================================================


class KLUStatus(enum.IntEnum):
    """KLU status codes, mirroring the KLU_* constants in SuiteSparse.

    Returned by refactor_with_status and refactor_and_solve_with_status. Values
    above OK are warnings, values below it are errors.
    """

    OK = 0
    SINGULAR = 1
    OUT_OF_MEMORY = -2
    INVALID = -3
    TOO_LARGE = -4


class RebuildReason(enum.IntEnum):
    """Why a call rebuilt a factorization from the token's carried arrays.

    A content-addressed handle names a matrix, not a mutable slot, so a call
    whose handle is no longer resident rebuilds it from the arrays the token
    carries. That is always correct but costs the rebuild. The status variants
    (`solve_with_numeric_with_status`, `refactor_with_status`, ...) report one
    value per numeric handle.

    The values mirror the `RebuildReason` enum in `klujax.cpp`:

    - `NONE`: cache hit, the resident factorization was used.
    - `EVICTED`: fell out of the bounded cache. Raise `KLUJAX_FACTOR_CACHE`.
    - `FREED`: `free_numeric` or `free_symbolic` had dropped it. This is expected.
    - `SUPERSEDED`: a `refactor` re-keyed this handle, so a stale alias was used.
    - `DTYPE`: reserved. Entry present but factored with the other dtype.
    - `UNKNOWN`: never resident, or its tombstone was itself evicted.
    - `STALE`: reserved. Content-check mismatch from a collision or forged token.
    """

    NONE = 0
    EVICTED = 1
    FREED = 2
    SUPERSEDED = 3
    DTYPE = 4
    UNKNOWN = 5
    STALE = 6


# A handle is a token: a cache id plus the arrays needed to rebuild the KLU
# object it names. The id is the cache key. The arrays travel with every call so
# a native cache miss (the object was evicted or freed) rebuilds it rather than
# reading freed memory. Because a token carries everything, freeing is optional
# and use after free is safe. The arrays are pytree children so the token flows
# through jit, vmap, and grad. n_col is static aux.

_ZERO_DEPS = jnp.zeros((), jnp.int32)


def _ordering_witness(solution: Array) -> Array:
    """Return a zero-valued int32 that XLA cannot fold away, so it forces ordering.

    Multiplying by 0.0 keeps the value zero but stays live, since 0.0 times a
    NaN or infinity is NaN under IEEE 754. Reading one element makes the result
    depend on the whole solve. See free_numeric.
    """
    return (0.0 * jnp.real(jnp.ravel(solution)[0])).astype(jnp.int32)


def _track(count: Array, solutions: tuple[Array, ...]) -> Array:
    """Advance a token's solve counter, once per solution, entangled with each."""
    for solution in solutions:
        count = count + jnp.int32(1) + _ordering_witness(solution)
    return count


class SymbolToken:
    """A handle to a KLU symbolic analysis: its cache id plus (Ai, Aj).

    n_dependent_solutions counts the solutions passed to track. free_symbolic
    consumes it so a free is ordered after those solves even inside a jit trace.
    """

    def __init__(
        self,
        id: Array,  # noqa: A002
        Ai: Array,
        Aj: Array,
        n_col: int,
        n_dependent_solutions: Array,
    ) -> None:
        """Store the cache id, the pattern arrays, the static n_col, and the counter."""
        self.id = id
        self.Ai = Ai
        self.Aj = Aj
        self.n_col = n_col
        self.n_dependent_solutions = n_dependent_solutions

    @property
    def handle(self) -> Array:
        """Return the raw cache id, for callers that reach past the token."""
        return self.id

    def track(self, *solutions: Array) -> Self:
        """Return a token whose free is ordered after these solutions."""
        return type(self)(
            self.id,
            self.Ai,
            self.Aj,
            self.n_col,
            _track(self.n_dependent_solutions, solutions),
        )

    def __enter__(self) -> Self:
        """Enter a context. The token is usable inside it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Free the cache slot on exit. Optional, reusing the token rebuilds."""
        with contextlib.suppress(Exception):
            free_symbolic(self)

    def close(self) -> None:
        """Free the cache slot early. Optional, the token stays usable."""
        with contextlib.suppress(Exception):
            free_symbolic(self)


class NumericToken:
    """A handle to a KLU numeric factorization: its cache id plus (Ai, Aj, Ax).

    n_dependent_solutions counts the solutions passed to track. free_numeric
    consumes it so a free is ordered after those solves even inside a jit trace.
    """

    def __init__(
        self,
        id: Array,  # noqa: A002
        Ai: Array,
        Aj: Array,
        Ax: Array,
        n_col: int,
        n_dependent_solutions: Array,
    ) -> None:
        """Store the cache ids, the matrix arrays, the static n_col, and the counter."""
        self.id = id
        self.Ai = Ai
        self.Aj = Aj
        self.Ax = Ax
        self.n_col = n_col
        self.n_dependent_solutions = n_dependent_solutions

    @property
    def handle(self) -> Array:
        """Return the raw cache ids (one per left-hand side)."""
        return self.id

    def track(self, *solutions: Array) -> Self:
        """Return a token whose free is ordered after these solutions."""
        return type(self)(
            self.id,
            self.Ai,
            self.Aj,
            self.Ax,
            self.n_col,
            _track(self.n_dependent_solutions, solutions),
        )

    def __enter__(self) -> Self:
        """Enter a context. The token is usable inside it."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Free the cache slot on exit. Optional, reusing the token rebuilds."""
        with contextlib.suppress(Exception):
            free_numeric(self)

    def close(self) -> None:
        """Free the cache slot early. Optional, the token stays usable."""
        with contextlib.suppress(Exception):
            free_numeric(self)


jax.tree_util.register_pytree_node(
    SymbolToken,
    lambda t: ((t.id, t.Ai, t.Aj, t.n_dependent_solutions), (t.n_col,)),
    lambda aux, ch: SymbolToken(ch[0], ch[1], ch[2], aux[0], ch[3]),
)

jax.tree_util.register_pytree_node(
    NumericToken,
    lambda t: ((t.id, t.Ai, t.Aj, t.Ax, t.n_dependent_solutions), (t.n_col,)),
    lambda aux, ch: NumericToken(ch[0], ch[1], ch[2], ch[3], aux[0], ch[4]),
)


def rebuild_count() -> int:
    """Return the count of factorizations rebuilt because their handle was gone.

    A handle whose KLU object was evicted from the bounded cache
    (KLUJAX_FACTOR_CACHE) or freed rebuilds it from the token's own arrays on
    next use. That is always correct but costs the rebuild, so a count that
    climbs during steady-state solving means the cache is too small.
    """
    return int(klujax_cpp.rebuild_count())


def reset_rebuild_count() -> None:
    """Reset the rebuild counter (and the per-reason breakdown) to zero."""
    klujax_cpp.reset_rebuild_count()


def rebuild_stats() -> dict[RebuildReason, int]:
    """Return the per-reason breakdown of rebuilds so far.

    Complements rebuild_count() (the scalar total) with a count for each
    RebuildReason, so a rising SUPERSEDED count points at stale-alias use while a
    rising EVICTED count points at cache pressure. Reset with
    reset_rebuild_count().
    """
    return {
        reason: int(klujax_cpp.rebuild_reason_count(int(reason)))
        for reason in RebuildReason
        if reason != RebuildReason.NONE
    }


def _ordering_operand(token: Any, dependency: Any) -> Array:  # noqa: ANN401
    """Return the int32 a free consumes to order itself after the wanted solves."""
    if dependency is not None:
        witness = jnp.zeros((), jnp.int32)
        for leaf in jax.tree_util.tree_leaves(dependency):
            array = jnp.asarray(leaf)
            # Only float or complex leaves can carry an edge XLA will not fold.
            if jnp.issubdtype(array.dtype, jnp.inexact):
                witness = witness + _ordering_witness(array)
        return witness
    if isinstance(token, (SymbolToken, NumericToken)):
        return token.n_dependent_solutions
    return jnp.zeros((), jnp.int32)


def free_symbolic(symbolic: SymbolToken | Array, dependency: Any = None) -> Array:  # noqa: ANN401
    """Free the cache slot for a symbolic analysis. Optional, never required.

    Freeing only drops the cache entry. The token stays valid: a later use
    rebuilds it from its own arrays, so calling this is always safe for
    correctness. It only reclaims memory without a wasted rebuild when it runs
    after the token's last use. Inside a jit trace a bare free and a solve on the
    same token are unordered, so pass that solve's result as dependency, or call
    symbolic.track(solution) first, to order the free after it. See the memory
    management guide.
    """
    handle = symbolic.id if isinstance(symbolic, SymbolToken) else symbolic
    return free_symbolic_p.bind(handle, _ordering_operand(symbolic, dependency))


def free_numeric(numeric: NumericToken | Array, dependency: Any = None) -> Array:  # noqa: ANN401
    """Free the cache slot for a numeric factorization. Optional, never required.

    Like free_symbolic, this only drops the cache entry and the token self-heals
    on next use, so it is always safe for correctness. To order it after a solve
    inside a jit trace, pass that solve's result as dependency or call
    numeric.track(solution) first. See the memory management guide.
    """
    handle = numeric.id if isinstance(numeric, NumericToken) else numeric
    return free_numeric_p.bind(handle, _ordering_operand(numeric, dependency))


# Split Solve routines =============================================================


def analyze(Ai: Array, Aj: Array, n_col: int) -> SymbolToken:
    """Analyze the sparsity pattern of a matrix A.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        n_col: [int]: the number of columns in the sparse matrix A

    Returns:
        symbolic: [SymbolToken]: the symbolic analysis handle

    """
    _require_x64()
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    sym_id = analyze_p.bind(Ai, Aj, jnp.int32(n_col))
    # The token carries Ai/Aj so the analysis can be rebuilt if it is evicted.
    return SymbolToken(sym_id, Ai, Aj, int(n_col), _ZERO_DEPS)


def validate_numeric_solve(
    Ai: Array, Aj: Array, Ax: Array, b: Array
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    """Reduced set of validate_args for use with solve_with_symbol."""
    order = jnp.lexsort((Aj, Ai))
    Ai, Aj = Ai[order], Aj[order]
    Ax = Ax[..., order] if Ax.ndim == 2 else Ax[order]

    shape = b.shape

    # 2. Dimension expansion to base case: (n_lhs, n_nz) and (n_lhs, n_col, n_rhs)
    if Ax.ndim == 1 and b.ndim == 1:
        Ax, b = Ax[None, :], b[None, :, None]
    elif Ax.ndim == 1 and b.ndim == 2:
        Ax, b = Ax[None, :], b[None, :, :]
    elif Ax.ndim == 1 and b.ndim == 3:
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and b.ndim == 1:
        b = b[None, :, None]
        shape = (Ax.shape[0], shape[0])
    elif Ax.ndim == 2 and b.ndim == 2:
        if Ax.shape[0] != b.shape[0] and Ax.shape[0] != 1 and b.shape[0] != 1:
            msg = f"Batch mismatch: {Ax.shape=} vs {b.shape=}"
            raise ValueError(msg)
        b = b[:, :, None]
        if b.shape[0] == 1 and Ax.shape[0] > 1:
            shape = (Ax.shape[0], *shape[1:])

    # 3. Final broadcasting for C++ FFI
    n_lhs = max(Ax.shape[0], b.shape[0])
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    b = jnp.broadcast_to(b, (n_lhs, b.shape[1], b.shape[2]))

    if len(shape) == 3 and shape[0] != b.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])

    return Ai, Aj, Ax, b, shape


@jax.jit
def _solve_with_symbol_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> Array:
    # Use the robust validator
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = solve_with_symbol_c128 if is_complex else solve_with_symbol_f64

    # Pass standardized arrays to the C++ extension
    x = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )

    return x.reshape(*out_shape)


def solve_with_symbol(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: SymbolToken | Array
) -> Array:
    """Solve Ax=b using a pre-computed symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        x: the result (x≈A^-1b)

    """
    _require_x64()
    handle = getattr(symbolic, "handle", symbolic)
    return _solve_with_symbol_jit(Ai, Aj, Ax, b, handle)


@jax.jit
def _tsolve_with_symbol_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> Array:
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = tsolve_with_symbol_c128 if is_complex else tsolve_with_symbol_f64

    x = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )

    return x.reshape(*out_shape)


def tsolve_with_symbol(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: SymbolToken | Array
) -> Array:
    """Solve A^T x=b (transpose solve) using a pre-computed symbolic analysis.

    Factors A numerically, then solves the transposed system using klu_tsolve.
    The symbolic handle describes the sparsity pattern of A (not A^T), and is
    reused as-is — KLU's triangular transpose solver handles the direction internally.

    For complex matrices, this solves A^T x = b (plain transpose, not conjugate).
    Use the conjugate transpose if you need A^H x = b.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        x: the result (x≈(A^T)^-1 b)

    """
    _require_x64()
    handle = getattr(symbolic, "handle", symbolic)
    return _tsolve_with_symbol_jit(Ai, Aj, Ax, b, handle)


@jax.jit
def _solve_with_symbol_status_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)
    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = solve_with_symbol_status_c128 if is_complex else solve_with_symbol_status_f64
    x, rebuild = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )
    return x.reshape(*out_shape), rebuild


def solve_with_symbol_with_status(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: SymbolToken | Array
) -> tuple[Array, Array]:
    """Like solve_with_symbol(), but also reports whether the analysis was rebuilt.

    Returns the solution and a scalar `RebuildReason`. It is `NONE` when the
    resident symbolic analysis was used, or `EVICTED` / `FREED` / `SUPERSEDED` /
    `UNKNOWN` when the analysis handle was gone and rebuilt from the carried
    Ai/Aj. A rebuild is still correct, only costlier. The rebuild value is a
    plain array returned alongside the solution, not a Python-side attribute,
    so you can use it in `jax.lax.cond` or similar under `jax.jit`.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        (x, rebuild): the solution and a scalar RebuildReason value

    """
    _require_x64()
    handle = getattr(symbolic, "handle", symbolic)
    return _solve_with_symbol_status_jit(Ai, Aj, Ax, b, handle)


@jax.jit
def _tsolve_with_symbol_status_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)
    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = (
        tsolve_with_symbol_status_c128 if is_complex else tsolve_with_symbol_status_f64
    )
    x, rebuild = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
    )
    return x.reshape(*out_shape), rebuild


def tsolve_with_symbol_with_status(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: SymbolToken | Array
) -> tuple[Array, Array]:
    """Like tsolve_with_symbol(), but also reports the analysis rebuild reason.

    See solve_with_symbol_with_status for the RebuildReason meaning.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        (x, rebuild): the solution and a scalar RebuildReason value

    """
    _require_x64()
    handle = getattr(symbolic, "handle", symbolic)
    return _tsolve_with_symbol_status_jit(Ai, Aj, Ax, b, handle)


def _n_col_of(token: SymbolToken | NumericToken | Any) -> int:  # noqa: ANN401
    """Read n_col off a token. It is static, so it can key the FFI as an attr.

    The split-solve calls need n_col to rebuild an evicted analysis, and the
    matrix values alone do not give it (an all-zero trailing column is legal).
    So it rides the token rather than being guessed.
    """
    n_col = getattr(token, "n_col", None)
    if n_col is None:
        msg = (
            "expected a SymbolToken or NumericToken carrying n_col; got a raw "
            "handle. Pass the token returned by analyze()/factor() instead."
        )
        raise TypeError(msg)
    return int(n_col)


def _as_batched_values(Ax: Array) -> Array:
    """Shape values as (n_lhs, n_nz) so they line up with the numeric id array."""
    Ax = jnp.asarray(Ax)
    return Ax if Ax.ndim == 2 else Ax[None, :]


@partial(jax.jit, static_argnames=("n_col",))
def _factor_jit(Ai: Array, Aj: Array, Ax: Array, sym_h: Array, *, n_col: int) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = factor_c128 if Ax.dtype in COMPLEX_DTYPES else factor_f64
    return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h, n_col=n_col)


def factor(Ai: Array, Aj: Array, Ax: Array, symbolic: SymbolToken) -> NumericToken:
    """Compute the numeric factorization of a matrix A given its symbolic analysis.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        symbolic: [SymbolToken]: the symbolic analysis handle

    Returns:
        numeric: [NumericToken]: the numeric factorization handle

    """
    _require_x64()
    sym_h = getattr(symbolic, "handle", symbolic)
    n_col = _n_col_of(symbolic)
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    num_id = cast("Any", _factor_jit)(Ai, Aj, Ax, sym_h, n_col=n_col)
    # The token carries the matrix so an evicted factorization rebuilds from it.
    return NumericToken(num_id, Ai, Aj, _as_batched_values(Ax), n_col, _ZERO_DEPS)


@partial(jax.jit, static_argnames=("n_col",))
def _refactor_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, *, n_col: int
) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = refactor_c128 if Ax.dtype in COMPLEX_DTYPES else refactor_f64
    return prim.bind(
        Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h, num_h, n_col=n_col
    )


def refactor(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    numeric: NumericToken,
    symbolic: SymbolToken,
) -> NumericToken:
    """Re-factorize matrix A numerically, reusing the symbolic analysis.

    Use when the sparsity pattern is unchanged but values have changed. Reuses the
    old factorization's pivots (klu_refactor) when it is safe to, so it is faster
    than a fresh factor().

    Returns a NumericToken whose id is the content key of the new values (a new
    handle, since a handle names a matrix). Thread the returned token into later
    solve_with_numeric calls so XLA sees the edge factor -> refactor -> solve. A
    stale alias still holding the old token stays correct: it names the old matrix
    and rebuilds it if needed, so it can never be handed the new values.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        numeric: [NumericToken]: existing numeric factorization
        symbolic: [SymbolToken]: the symbolic analysis handle

    Returns:
        numeric: [NumericToken]: the updated numeric handle (a new content key)

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    n_col = _n_col_of(symbolic)
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    out_id = cast("Any", _refactor_jit)(Ai, Aj, Ax, sym_h, num_h, n_col=n_col)
    return NumericToken(out_id, Ai, Aj, _as_batched_values(Ax), n_col, _ZERO_DEPS)


@partial(jax.jit, static_argnames=("n_col",))
def _refactor_with_status_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, *, n_col: int
) -> tuple[Array, Array, Array]:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = refactor_status_c128 if Ax.dtype in COMPLEX_DTYPES else refactor_status_f64
    raw_handle, status, rebuild = prim.bind(
        Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h, num_h, n_col=n_col
    )
    return raw_handle, status, rebuild


def refactor_with_status(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    numeric: NumericToken,
    symbolic: SymbolToken,
) -> tuple[NumericToken, Array, Array]:
    """Like refactor(), but reports failure through a status code instead of raising.

    refactor() turns a failed klu_refactor into an error, which aborts the whole
    computation under jax.jit and cannot be caught or branched on. This variant always
    succeeds and returns the KLU status per left-hand side, so a caller can fall back to
    a fresh factor() from inside a jitted function.

    When status != KLUStatus.OK the numeric object may be partially overwritten and must
    not be used for a solve. It remains valid to pass to free_numeric, and the symbolic
    object is unaffected and may be reused for a fresh factor().

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        numeric: [NumericToken|Array]: existing numeric factorization
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        (numeric, status, rebuild): the updated numeric handle (a new content key,
            for XLA dep tracking), [n_lhs; int32] of KLUStatus values, and
            [n_lhs; int32] of RebuildReason values (see RebuildReason)

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    n_col = _n_col_of(symbolic)
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    out_id, status, rebuild = cast("Any", _refactor_with_status_jit)(
        Ai, Aj, Ax, sym_h, num_h, n_col=n_col
    )
    token = NumericToken(out_id, Ai, Aj, _as_batched_values(Ax), n_col, _ZERO_DEPS)
    return token, status, rebuild


@partial(jax.jit, static_argnames=("is_complex", "n_col"))
def _rcond_jit(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    sym_h: Array,
    num_h: Array,
    *,
    is_complex: bool,
    n_col: int,
) -> Array:
    prim = rcond_c128 if is_complex else rcond_f64
    Ax = Ax.astype(jnp.complex128 if is_complex else jnp.float64)
    return prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax,
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
        n_col=n_col,
    )


def rcond(
    symbolic: SymbolToken | Array,
    numeric: NumericToken,
    *,
    dtype: Any = jnp.float64,  # noqa: ANN401
) -> Array:
    """Reciprocal pivot growth estimate min|Uii| / max|Uii| of a factorization.

    Computed by klu_rcond in O(n), so it is far cheaper than a probe solve. Use it
    to detect pivot degradation after a refactor() that reuses an older pivot order.
    A value near 1 is well conditioned, a tiny value means the reused pivots have
    gone bad, and 0 means the factorization is singular.

    KLU needs the matching entry point (klu_rcond or klu_z_rcond), read from dtype
    rather than off the token, so keep it in sync with what numeric.Ax was
    factored with. A mismatch calls the wrong entry point.

    Args:
        symbolic: [SymbolToken|Array]: the symbolic analysis handle
        numeric: [NumericToken]: the numeric factorization handle
        dtype: the dtype the factorization was built with (float64 or complex128)

    Returns:
        rcond: [n_lhs; float64]

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    is_complex = jnp.dtype(dtype) in COMPLEX_DTYPES
    # The numeric token carries the matrix so an evicted factorization rebuilds.
    return cast("Any", _rcond_jit)(
        numeric.Ai,
        numeric.Aj,
        numeric.Ax,
        sym_h,
        num_h,
        is_complex=is_complex,
        n_col=_n_col_of(numeric),
    )


@partial(jax.jit, static_argnames=("n_col",))
def _condest_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, *, n_col: int
) -> Array:
    dummy_b = jnp.zeros((1,), dtype=Ax.dtype)
    Ai, Aj, Ax, _, _ = validate_args(Ai, Aj, Ax, dummy_b)
    prim = condest_c128 if Ax.dtype in COMPLEX_DTYPES else condest_f64
    return prim.bind(
        Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, sym_h, num_h, n_col=n_col
    )


def condest(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    symbolic: SymbolToken | Array,
    numeric: NumericToken | Array,
) -> Array:
    """1-norm condition number estimate of A, given its factorization.

    Uses klu_condest, which applies Hager's method as modified by Higham and
    Tisseur, the same method as MATLAB's condest. More accurate but more expensive
    than rcond(), so the usual pattern is to reach for it only when rcond() is
    borderline. A singular factorization gives infinity.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        symbolic: [SymbolToken|Array]: the symbolic analysis handle
        numeric: [NumericToken|Array]: the numeric factorization of A

    Returns:
        condest: [n_lhs; float64]

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    return cast("Any", _condest_jit)(
        Ai, Aj, Ax, sym_h, num_h, n_col=_n_col_of(symbolic)
    )


@jax.jit
def _solve_with_numeric_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, b: Array
) -> Array:
    # The KLU entry point must match the factorization's dtype, so a complex
    # numeric always dispatches the complex handler even if b is real. Reading
    # only b would down-cast Ax and call the wrong entry point (type confusion).
    is_complex = (b.dtype in COMPLEX_DTYPES) or (Ax.dtype in COMPLEX_DTYPES)
    prim = solve_with_numeric_c128 if is_complex else solve_with_numeric_f64
    dt = jnp.complex128 if is_complex else jnp.float64
    # Ax rides along so an evicted numeric can be rebuilt.
    return prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(dt),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
        b.astype(dt),
    )


def solve_with_numeric(
    numeric: NumericToken,
    b: Array,
    symbolic: SymbolToken | Array,
) -> Array:
    """Solve Ax=b using a pre-computed numeric factorization.

    Args:
        numeric: [NumericToken]: the numeric factorization handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        x: the result (x≈A^-1b)

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    return _solve_with_numeric_jit(numeric.Ai, numeric.Aj, numeric.Ax, sym_h, num_h, b)


@jax.jit
def _tsolve_with_numeric_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, b: Array
) -> Array:
    # As solve_with_numeric: dispatch by the factorization's dtype, not b's.
    is_complex = (b.dtype in COMPLEX_DTYPES) or (Ax.dtype in COMPLEX_DTYPES)
    prim = tsolve_with_numeric_c128 if is_complex else tsolve_with_numeric_f64
    dt = jnp.complex128 if is_complex else jnp.float64
    return prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(dt),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
        b.astype(dt),
    )


def tsolve_with_numeric(
    numeric: NumericToken,
    b: Array,
    symbolic: SymbolToken | Array,
) -> Array:
    """Solve A^T x=b (transpose solve) using a pre-computed numeric factorization.

    Uses klu_tsolve internally. The numeric factorization must have been computed
    for A (not A^T); KLU handles the transposition during the triangular solve.

    For complex matrices, this solves A^T x = b (plain transpose, not conjugate).

    Args:
        numeric: [NumericToken]: the numeric factorization handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        x: the result (x≈(A^T)^-1 b)

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    return _tsolve_with_numeric_jit(numeric.Ai, numeric.Aj, numeric.Ax, sym_h, num_h, b)


@jax.jit
def _solve_with_numeric_status_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, b: Array
) -> tuple[Array, Array]:
    is_complex = (b.dtype in COMPLEX_DTYPES) or (Ax.dtype in COMPLEX_DTYPES)
    prim = (
        solve_with_numeric_status_c128 if is_complex else solve_with_numeric_status_f64
    )
    dt = jnp.complex128 if is_complex else jnp.float64
    return prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(dt),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
        b.astype(dt),
    )


def solve_with_numeric_with_status(
    numeric: NumericToken,
    b: Array,
    symbolic: SymbolToken | Array,
) -> tuple[Array, Array]:
    """Like solve_with_numeric(), but also reports whether the handle was rebuilt.

    Returns the solution and, per numeric handle, a `RebuildReason`. It is `NONE`
    when the resident factorization was used, or `EVICTED` / `FREED` / `SUPERSEDED`
    / `UNKNOWN` when the handle was gone and rebuilt from the token's carried arrays.
    A rebuild is still correct, only costlier. `SUPERSEDED` means a `refactor`
    elsewhere re-keyed this handle, the stale-alias case. The rebuild array is an
    ordinary output, so you can branch on it under `jax.jit`.

    Args:
        numeric: [NumericToken]: the numeric factorization handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        (x, rebuild): the solution and [n_numeric; int32] of RebuildReason values

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    return _solve_with_numeric_status_jit(
        numeric.Ai, numeric.Aj, numeric.Ax, sym_h, num_h, b
    )


@jax.jit
def _tsolve_with_numeric_status_jit(
    Ai: Array, Aj: Array, Ax: Array, sym_h: Array, num_h: Array, b: Array
) -> tuple[Array, Array]:
    is_complex = (b.dtype in COMPLEX_DTYPES) or (Ax.dtype in COMPLEX_DTYPES)
    prim = (
        tsolve_with_numeric_status_c128
        if is_complex
        else tsolve_with_numeric_status_f64
    )
    dt = jnp.complex128 if is_complex else jnp.float64
    return prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(dt),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
        b.astype(dt),
    )


def tsolve_with_numeric_with_status(
    numeric: NumericToken,
    b: Array,
    symbolic: SymbolToken | Array,
) -> tuple[Array, Array]:
    """Like tsolve_with_numeric(), but also reports the rebuild reason per handle.

    See solve_with_numeric_with_status for the RebuildReason meaning.

    Args:
        numeric: [NumericToken]: the numeric factorization handle
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the target vector
        symbolic: [SymbolToken|Array]: the symbolic analysis handle

    Returns:
        (x, rebuild): the solution and [n_numeric; int32] of RebuildReason values

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    return _tsolve_with_numeric_status_jit(
        numeric.Ai, numeric.Aj, numeric.Ax, sym_h, num_h, b
    )


@jax.jit
def _refactor_and_solve_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array, num_h: Array
) -> tuple[Array, Array]:
    # Use validate_numeric_solve to standardize shapes and get the expected out_shape
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = refactor_and_solve_c128 if is_complex else refactor_and_solve_f64

    x, out_num = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
    )

    # Reshape x back to the original dimensions of b
    return x.reshape(*out_shape), out_num


def refactor_and_solve(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    numeric: NumericToken | Array,
    symbolic: SymbolToken | Array,
) -> tuple[Array, NumericToken]:
    """Fused in-place refactorization followed by triangular solve.

    Equivalent to calling refactor() then solve_with_numeric(), but executes as a
    single C++ kernel call. This avoids allocating the COO→CSC work buffer twice
    and saves a JAX dispatch round-trip, which matters in tight iteration loops.

    The returned NumericToken's id is the content key of the new values (same
    behaviour as refactor()); threading it into later calls keeps the XLA
    dependency edge, and a stale alias of the old token stays correct.

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the right-hand side
        numeric: [NumericToken]: existing numeric factorization (updated in place)
        symbolic: [SymbolToken]: the symbolic analysis handle

    Returns:
        (x, numeric): solution array and the updated numeric handle (same id)

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    n_col = _n_col_of(symbolic)
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    x, out_id = _refactor_and_solve_jit(Ai, Aj, Ax, b, sym_h, num_h)
    return x, NumericToken(out_id, Ai, Aj, _as_batched_values(Ax), n_col, _ZERO_DEPS)


@jax.jit
def _refactor_and_solve_with_status_jit(
    Ai: Array, Aj: Array, Ax: Array, b: Array, sym_h: Array, num_h: Array
) -> tuple[Array, Array, Array, Array]:
    Ai, Aj, Ax, b, out_shape = validate_numeric_solve(Ai, Aj, Ax, b)

    is_complex = any(x.dtype in COMPLEX_DTYPES for x in (Ax, b))
    prim = (
        refactor_and_solve_status_c128 if is_complex else refactor_and_solve_status_f64
    )

    x, out_num, status, rebuild = prim.bind(
        Ai.astype(jnp.int32),
        Aj.astype(jnp.int32),
        Ax.astype(jnp.complex128 if is_complex else jnp.float64),
        b.astype(jnp.complex128 if is_complex else jnp.float64),
        sym_h.astype(jnp.uint64),
        num_h.astype(jnp.uint64),
    )

    return x.reshape(*out_shape), out_num, status, rebuild


def refactor_and_solve_with_status(
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    numeric: NumericToken,
    symbolic: SymbolToken,
) -> tuple[Array, NumericToken, Array, Array]:
    """Like refactor_and_solve(), but also reports status and rebuild per element.

    Lets the fused path be used from the same branching code as refactor_with_status().
    A failed element still gets its solution filled with NaN, so either signal works.

    The contract on the numeric object is the same as for refactor_with_status(): when
    status != KLUStatus.OK it must not be used for a solve, but it is still safe to free
    and the symbolic object may be reused for a fresh factor().

    Args:
        Ai: [n_nz; int32]: the row indices of the sparse matrix A
        Aj: [n_nz; int32]: the column indices of the sparse matrix A
        Ax: [n_lhs? x n_nz; float64|complex128]: the values of the sparse matrix A
        b:  [n_lhs? x n_col x n_rhs?; float64|complex128]: the right-hand side
        numeric: [NumericToken]: existing numeric factorization
        symbolic: [SymbolToken]: the symbolic analysis handle

    Returns:
        (x, numeric, status, rebuild): solution array (NaN where the refactor
            failed), the updated numeric handle (a new content key), [n_lhs; int32]
            of KLUStatus values, and [n_lhs; int32] of RebuildReason values

    """
    _require_x64()
    num_h = getattr(numeric, "handle", numeric)
    sym_h = getattr(symbolic, "handle", symbolic)
    n_col = _n_col_of(symbolic)
    Ai = jnp.asarray(Ai, dtype=jnp.int32)
    Aj = jnp.asarray(Aj, dtype=jnp.int32)
    x, out_id, status, rebuild = _refactor_and_solve_with_status_jit(
        Ai, Aj, Ax, b, sym_h, num_h
    )
    token = NumericToken(out_id, Ai, Aj, _as_batched_values(Ax), n_col, _ZERO_DEPS)
    return x, token, status, rebuild


# Primitives ==========================================================================

dot_f64 = jax.extend.core.Primitive("dot_f64")
dot_c128 = jax.extend.core.Primitive("dot_c128")
solve_f64 = jax.extend.core.Primitive("solve_f64")
solve_c128 = jax.extend.core.Primitive("solve_c128")
analyze_p = jax.extend.core.Primitive("analyze")
solve_with_symbol_f64 = jax.extend.core.Primitive("solve_with_symbol_f64")
solve_with_symbol_c128 = jax.extend.core.Primitive("solve_with_symbol_c128")
tsolve_with_symbol_f64 = jax.extend.core.Primitive("tsolve_with_symbol_f64")
tsolve_with_symbol_c128 = jax.extend.core.Primitive("tsolve_with_symbol_c128")
solve_with_symbol_status_f64 = jax.extend.core.Primitive("solve_with_symbol_status_f64")
solve_with_symbol_status_c128 = jax.extend.core.Primitive(
    "solve_with_symbol_status_c128"
)
tsolve_with_symbol_status_f64 = jax.extend.core.Primitive(
    "tsolve_with_symbol_status_f64"
)
tsolve_with_symbol_status_c128 = jax.extend.core.Primitive(
    "tsolve_with_symbol_status_c128"
)
solve_with_symbol_status_f64.multiple_results = True
solve_with_symbol_status_c128.multiple_results = True
tsolve_with_symbol_status_f64.multiple_results = True
tsolve_with_symbol_status_c128.multiple_results = True
free_symbolic_p = jax.extend.core.Primitive("free_symbolic")
factor_f64 = jax.extend.core.Primitive("factor_f64")
factor_c128 = jax.extend.core.Primitive("factor_c128")
solve_with_numeric_f64 = jax.extend.core.Primitive("solve_with_numeric_f64")
solve_with_numeric_c128 = jax.extend.core.Primitive("solve_with_numeric_c128")
tsolve_with_numeric_f64 = jax.extend.core.Primitive("tsolve_with_numeric_f64")
tsolve_with_numeric_c128 = jax.extend.core.Primitive("tsolve_with_numeric_c128")
solve_with_numeric_status_f64 = jax.extend.core.Primitive(
    "solve_with_numeric_status_f64"
)
solve_with_numeric_status_c128 = jax.extend.core.Primitive(
    "solve_with_numeric_status_c128"
)
tsolve_with_numeric_status_f64 = jax.extend.core.Primitive(
    "tsolve_with_numeric_status_f64"
)
tsolve_with_numeric_status_c128 = jax.extend.core.Primitive(
    "tsolve_with_numeric_status_c128"
)
solve_with_numeric_status_f64.multiple_results = True
solve_with_numeric_status_c128.multiple_results = True
tsolve_with_numeric_status_f64.multiple_results = True
tsolve_with_numeric_status_c128.multiple_results = True
free_numeric_p = jax.extend.core.Primitive("free_numeric")
refactor_f64 = jax.extend.core.Primitive("refactor_f64")
refactor_c128 = jax.extend.core.Primitive("refactor_c128")
refactor_and_solve_f64 = jax.extend.core.Primitive("refactor_and_solve_f64")
refactor_and_solve_c128 = jax.extend.core.Primitive("refactor_and_solve_c128")
refactor_and_solve_f64.multiple_results = True
refactor_and_solve_c128.multiple_results = True
refactor_status_f64 = jax.extend.core.Primitive("refactor_status_f64")
refactor_status_c128 = jax.extend.core.Primitive("refactor_status_c128")
refactor_and_solve_status_f64 = jax.extend.core.Primitive(
    "refactor_and_solve_status_f64"
)
refactor_and_solve_status_c128 = jax.extend.core.Primitive(
    "refactor_and_solve_status_c128"
)
refactor_status_f64.multiple_results = True
refactor_status_c128.multiple_results = True
refactor_and_solve_status_f64.multiple_results = True
refactor_and_solve_status_c128.multiple_results = True
rcond_f64 = jax.extend.core.Primitive("rcond_f64")
rcond_c128 = jax.extend.core.Primitive("rcond_c128")
condest_f64 = jax.extend.core.Primitive("condest_f64")
condest_c128 = jax.extend.core.Primitive("condest_c128")

# free_symbolic and free_numeric are called only for what they do to the
# cache, so every caller throws away their return value. jax.jit's own
# dead-code elimination does not know that and silently deletes a call whose
# result is unused, unless the primitive is marked impure. Reuse jax's own
# GenericEffect for that, the same tool jax.ffi.ffi_call's has_side_effect
# uses internally, and register it as lowerable the way that internal use
# does (there is no public API for either of these two calls).
_free_symbolic_effect = jax._src.core.GenericEffect(free_symbolic_p)  # noqa: SLF001
_free_numeric_effect = jax._src.core.GenericEffect(free_numeric_p)  # noqa: SLF001
jax._src.effects.lowerable_effects.add_type(jax._src.core.GenericEffect)  # noqa: SLF001
jax._src.effects.control_flow_allowed_effects.add_type(  # noqa: SLF001
    jax._src.core.GenericEffect  # noqa: SLF001
)

# Implementations ========================================================


@dot_f64.def_impl
def dot_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_f64", Ai, Aj, Ax, x)


@dot_c128.def_impl
def dot_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("dot_c128", Ai, Aj, Ax, x)


@solve_f64.def_impl
def solve_f64_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_f64", Ai, Aj, Ax, x)


@solve_c128.def_impl
def solve_c128_impl(Ai: Array, Aj: Array, Ax: Array, x: Array) -> Array:
    return general_impl("solve_c128", Ai, Aj, Ax, x)


@solve_with_symbol_f64.def_impl
def solve_with_symbol_f64_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("solve_with_symbol_f64", Ai, Aj, Ax, b, symbolic)


@solve_with_symbol_c128.def_impl
def solve_with_symbol_c128_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("solve_with_symbol_c128", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_f64.def_impl
def tsolve_with_symbol_f64_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("tsolve_with_symbol_f64", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_c128.def_impl
def tsolve_with_symbol_c128_impl(
    Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
) -> Array:
    return general_impl("tsolve_with_symbol_c128", Ai, Aj, Ax, b, symbolic)


def _symbol_status_impl(
    name: str, Ai: Array, Aj: Array, Ax: Array, b: Array, symbolic: Array
):
    # As general_impl, but the call also returns a scalar RebuildReason for the
    # one symbolic handle backing this solve.
    call = jax.ffi.ffi_call(
        name,
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((), jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic)


@solve_with_symbol_status_f64.def_impl
def solve_with_symbol_status_f64_impl(Ai, Aj, Ax, b, symbolic):
    return _symbol_status_impl("solve_with_symbol_status_f64", Ai, Aj, Ax, b, symbolic)


@solve_with_symbol_status_c128.def_impl
def solve_with_symbol_status_c128_impl(Ai, Aj, Ax, b, symbolic):
    return _symbol_status_impl("solve_with_symbol_status_c128", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_status_f64.def_impl
def tsolve_with_symbol_status_f64_impl(Ai, Aj, Ax, b, symbolic):
    return _symbol_status_impl("tsolve_with_symbol_status_f64", Ai, Aj, Ax, b, symbolic)


@tsolve_with_symbol_status_c128.def_impl
def tsolve_with_symbol_status_c128_impl(Ai, Aj, Ax, b, symbolic):
    return _symbol_status_impl(
        "tsolve_with_symbol_status_c128", Ai, Aj, Ax, b, symbolic
    )


@analyze_p.def_impl
def analyze_impl(Ai: Array, Aj: Array, n_col: Array) -> Array:
    return jax.ffi.ffi_call("analyze", ShapedArray((), jnp.uint64))(Ai, Aj, n_col)


@factor_f64.def_impl
def factor_f64_impl(Ai, Aj, Ax, symbolic, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, n_col=np.int64(n_col))


@factor_c128.def_impl
def factor_c128_impl(Ai, Aj, Ax, symbolic, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("factor_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, n_col=np.int64(n_col))


@refactor_f64.def_impl
def refactor_f64_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("refactor_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@refactor_c128.def_impl
def refactor_c128_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("refactor_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.uint64))
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@refactor_status_f64.def_impl
def refactor_status_f64_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_status_f64",
        (
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@refactor_status_c128.def_impl
def refactor_status_c128_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_status_c128",
        (
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@solve_with_numeric_status_f64.def_impl
def solve_with_numeric_status_f64_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_status_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct(numeric.shape, jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@solve_with_numeric_status_c128.def_impl
def solve_with_numeric_status_c128_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_status_c128",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct(numeric.shape, jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@tsolve_with_numeric_status_f64.def_impl
def tsolve_with_numeric_status_f64_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_status_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct(numeric.shape, jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@tsolve_with_numeric_status_c128.def_impl
def tsolve_with_numeric_status_c128_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_status_c128",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct(numeric.shape, jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


# rcond and condest only read an existing numeric object, so no has_side_effect here.
@rcond_f64.def_impl
def rcond_f64_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    call = jax.ffi.ffi_call(
        "rcond_f64", jax.ShapeDtypeStruct(numeric.shape, jnp.float64)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@rcond_c128.def_impl
def rcond_c128_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    call = jax.ffi.ffi_call(
        "rcond_c128", jax.ShapeDtypeStruct(numeric.shape, jnp.float64)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@condest_f64.def_impl
def condest_f64_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("condest_f64", jax.ShapeDtypeStruct((n_lhs,), jnp.float64))
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@condest_c128.def_impl
def condest_c128_impl(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call("condest_c128", jax.ShapeDtypeStruct((n_lhs,), jnp.float64))
    return call(Ai, Aj, Ax, symbolic, numeric, n_col=np.int64(n_col))


@solve_with_numeric_f64.def_impl
def solve_with_numeric_f64_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@solve_with_numeric_c128.def_impl
def solve_with_numeric_c128_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "solve_with_numeric_c128", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@tsolve_with_numeric_f64.def_impl
def tsolve_with_numeric_f64_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_f64", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@tsolve_with_numeric_c128.def_impl
def tsolve_with_numeric_c128_impl(Ai, Aj, Ax, symbolic, numeric, b):
    call = jax.ffi.ffi_call(
        "tsolve_with_numeric_c128", jax.ShapeDtypeStruct(b.shape, b.dtype)
    )
    return call(Ai, Aj, Ax, symbolic, numeric, b)


@refactor_and_solve_f64.def_impl
def refactor_and_solve_f64_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


@refactor_and_solve_c128.def_impl
def refactor_and_solve_c128_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_c128",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


@refactor_and_solve_status_f64.def_impl
def refactor_and_solve_status_f64_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_status_f64",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


@refactor_and_solve_status_c128.def_impl
def refactor_and_solve_status_c128_impl(Ai, Aj, Ax, b, symbolic, numeric):
    n_lhs = Ax.shape[0]
    call = jax.ffi.ffi_call(
        "refactor_and_solve_status_c128",
        (
            jax.ShapeDtypeStruct(b.shape, b.dtype),
            jax.ShapeDtypeStruct((n_lhs,), jnp.uint64),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
            jax.ShapeDtypeStruct((n_lhs,), jnp.int32),
        ),
    )
    return call(Ai, Aj, Ax, b, symbolic, numeric)


def general_impl(
    name: str, Ai: Array, Aj: Array, Ax: Array, x: Array, *args: Array
) -> Array:
    call = jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct(x.shape, x.dtype),
    )
    if not callable(call):
        msg = "jax.ffi.ffi_call did not return a callable."
        raise RuntimeError(msg)  # noqa: TRY004
    return call(Ai, Aj, Ax, x, *args)


# Lowerings ===========================================================================

jax.ffi.register_ffi_target(
    "dot_f64",
    klujax_cpp.dot_f64(),
    platform="cpu",
)

dot_f64_low = mlir.lower_fun(dot_f64_impl, multiple_results=False)
mlir.register_lowering(dot_f64, dot_f64_low)

jax.ffi.register_ffi_target(
    "dot_c128",
    klujax_cpp.dot_c128(),
    platform="cpu",
)

dot_c128_low = mlir.lower_fun(dot_c128_impl, multiple_results=False)
mlir.register_lowering(dot_c128, dot_c128_low)

jax.ffi.register_ffi_target(
    "solve_f64",
    klujax_cpp.solve_f64(),
    platform="cpu",
)

solve_f64_low = mlir.lower_fun(solve_f64_impl, multiple_results=False)
mlir.register_lowering(solve_f64, solve_f64_low)

jax.ffi.register_ffi_target(
    "solve_c128",
    klujax_cpp.solve_c128(),
    platform="cpu",
)

solve_c128_low = mlir.lower_fun(solve_c128_impl, multiple_results=False)
mlir.register_lowering(solve_c128, solve_c128_low)

jax.ffi.register_ffi_target(
    "analyze",
    klujax_cpp.analyze(),
    platform="cpu",
)

analyze_low = mlir.lower_fun(analyze_impl, multiple_results=False)
mlir.register_lowering(analyze_p, analyze_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_f64",
    klujax_cpp.solve_with_symbol_f64(),
    platform="cpu",
)

solve_with_symbol_f64_low = mlir.lower_fun(
    solve_with_symbol_f64_impl, multiple_results=False
)
mlir.register_lowering(solve_with_symbol_f64, solve_with_symbol_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_c128",
    klujax_cpp.solve_with_symbol_c128(),
    platform="cpu",
)

solve_with_symbol_c128_low = mlir.lower_fun(
    solve_with_symbol_c128_impl, multiple_results=False
)
mlir.register_lowering(solve_with_symbol_c128, solve_with_symbol_c128_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_f64",
    klujax_cpp.tsolve_with_symbol_f64(),
    platform="cpu",
)

tsolve_with_symbol_f64_low = mlir.lower_fun(
    tsolve_with_symbol_f64_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_symbol_f64, tsolve_with_symbol_f64_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_c128",
    klujax_cpp.tsolve_with_symbol_c128(),
    platform="cpu",
)

tsolve_with_symbol_c128_low = mlir.lower_fun(
    tsolve_with_symbol_c128_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_symbol_c128, tsolve_with_symbol_c128_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_status_f64",
    klujax_cpp.solve_with_symbol_status_f64(),
    platform="cpu",
)
solve_with_symbol_status_f64_low = mlir.lower_fun(
    solve_with_symbol_status_f64_impl, multiple_results=True
)
mlir.register_lowering(solve_with_symbol_status_f64, solve_with_symbol_status_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_symbol_status_c128",
    klujax_cpp.solve_with_symbol_status_c128(),
    platform="cpu",
)
solve_with_symbol_status_c128_low = mlir.lower_fun(
    solve_with_symbol_status_c128_impl, multiple_results=True
)
mlir.register_lowering(solve_with_symbol_status_c128, solve_with_symbol_status_c128_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_status_f64",
    klujax_cpp.tsolve_with_symbol_status_f64(),
    platform="cpu",
)
tsolve_with_symbol_status_f64_low = mlir.lower_fun(
    tsolve_with_symbol_status_f64_impl, multiple_results=True
)
mlir.register_lowering(tsolve_with_symbol_status_f64, tsolve_with_symbol_status_f64_low)

jax.ffi.register_ffi_target(
    "tsolve_with_symbol_status_c128",
    klujax_cpp.tsolve_with_symbol_status_c128(),
    platform="cpu",
)
tsolve_with_symbol_status_c128_low = mlir.lower_fun(
    tsolve_with_symbol_status_c128_impl, multiple_results=True
)
mlir.register_lowering(
    tsolve_with_symbol_status_c128, tsolve_with_symbol_status_c128_low
)

jax.ffi.register_ffi_target(
    "free_symbolic",
    klujax_cpp.free_symbolic(),
    platform="cpu",
)

jax.ffi.register_ffi_target(
    "free_numeric",
    klujax_cpp.free_numeric(),
    platform="cpu",
)


@free_numeric_p.def_impl
def free_numeric_impl(numeric, ordering):
    call = jax.ffi.ffi_call(
        "free_numeric", jax.ShapeDtypeStruct((), jnp.int32), has_side_effect=True
    )
    return call(numeric, ordering)


@free_symbolic_p.def_impl
def free_symbolic_impl(symbolic, ordering):
    call = jax.ffi.ffi_call(
        "free_symbolic", jax.ShapeDtypeStruct((), jnp.int32), has_side_effect=True
    )
    return call(symbolic, ordering)


@free_numeric_p.def_effectful_abstract_eval
def free_numeric_abstract_eval(numeric, ordering):
    return ShapedArray((), jnp.int32), {_free_numeric_effect}


jax.ffi.register_ffi_target("factor_f64", klujax_cpp.factor_f64(), platform="cpu")
factor_f64_low = mlir.lower_fun(factor_f64_impl, multiple_results=False)
mlir.register_lowering(factor_f64, factor_f64_low)

jax.ffi.register_ffi_target("factor_c128", klujax_cpp.factor_c128(), platform="cpu")
factor_c128_low = mlir.lower_fun(factor_c128_impl, multiple_results=False)
mlir.register_lowering(factor_c128, factor_c128_low)

jax.ffi.register_ffi_target("refactor_f64", klujax_cpp.refactor_f64(), platform="cpu")
refactor_f64_low = mlir.lower_fun(refactor_f64_impl, multiple_results=False)
mlir.register_lowering(refactor_f64, refactor_f64_low)

jax.ffi.register_ffi_target("refactor_c128", klujax_cpp.refactor_c128(), platform="cpu")
refactor_c128_low = mlir.lower_fun(refactor_c128_impl, multiple_results=False)
mlir.register_lowering(refactor_c128, refactor_c128_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_f64", klujax_cpp.solve_with_numeric_f64(), platform="cpu"
)
solve_with_numeric_f64_low = mlir.lower_fun(
    solve_with_numeric_f64_impl, multiple_results=False
)
mlir.register_lowering(solve_with_numeric_f64, solve_with_numeric_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_c128", klujax_cpp.solve_with_numeric_c128(), platform="cpu"
)
solve_with_numeric_c128_low = mlir.lower_fun(
    solve_with_numeric_c128_impl, multiple_results=False
)
mlir.register_lowering(solve_with_numeric_c128, solve_with_numeric_c128_low)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_f64", klujax_cpp.tsolve_with_numeric_f64(), platform="cpu"
)
tsolve_with_numeric_f64_low = mlir.lower_fun(
    tsolve_with_numeric_f64_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_numeric_f64, tsolve_with_numeric_f64_low)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_c128", klujax_cpp.tsolve_with_numeric_c128(), platform="cpu"
)
tsolve_with_numeric_c128_low = mlir.lower_fun(
    tsolve_with_numeric_c128_impl, multiple_results=False
)
mlir.register_lowering(tsolve_with_numeric_c128, tsolve_with_numeric_c128_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_status_f64",
    klujax_cpp.solve_with_numeric_status_f64(),
    platform="cpu",
)
solve_with_numeric_status_f64_low = mlir.lower_fun(
    solve_with_numeric_status_f64_impl, multiple_results=True
)
mlir.register_lowering(solve_with_numeric_status_f64, solve_with_numeric_status_f64_low)

jax.ffi.register_ffi_target(
    "solve_with_numeric_status_c128",
    klujax_cpp.solve_with_numeric_status_c128(),
    platform="cpu",
)
solve_with_numeric_status_c128_low = mlir.lower_fun(
    solve_with_numeric_status_c128_impl, multiple_results=True
)
mlir.register_lowering(
    solve_with_numeric_status_c128, solve_with_numeric_status_c128_low
)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_status_f64",
    klujax_cpp.tsolve_with_numeric_status_f64(),
    platform="cpu",
)
tsolve_with_numeric_status_f64_low = mlir.lower_fun(
    tsolve_with_numeric_status_f64_impl, multiple_results=True
)
mlir.register_lowering(
    tsolve_with_numeric_status_f64, tsolve_with_numeric_status_f64_low
)

jax.ffi.register_ffi_target(
    "tsolve_with_numeric_status_c128",
    klujax_cpp.tsolve_with_numeric_status_c128(),
    platform="cpu",
)
tsolve_with_numeric_status_c128_low = mlir.lower_fun(
    tsolve_with_numeric_status_c128_impl, multiple_results=True
)
mlir.register_lowering(
    tsolve_with_numeric_status_c128, tsolve_with_numeric_status_c128_low
)

jax.ffi.register_ffi_target(
    "refactor_and_solve_f64", klujax_cpp.refactor_and_solve_f64(), platform="cpu"
)
refactor_and_solve_f64_low = mlir.lower_fun(
    refactor_and_solve_f64_impl, multiple_results=True
)
mlir.register_lowering(refactor_and_solve_f64, refactor_and_solve_f64_low)

jax.ffi.register_ffi_target(
    "refactor_and_solve_c128", klujax_cpp.refactor_and_solve_c128(), platform="cpu"
)
refactor_and_solve_c128_low = mlir.lower_fun(
    refactor_and_solve_c128_impl, multiple_results=True
)
mlir.register_lowering(refactor_and_solve_c128, refactor_and_solve_c128_low)

jax.ffi.register_ffi_target(
    "refactor_status_f64", klujax_cpp.refactor_status_f64(), platform="cpu"
)
refactor_status_f64_low = mlir.lower_fun(
    refactor_status_f64_impl, multiple_results=True
)
mlir.register_lowering(refactor_status_f64, refactor_status_f64_low)

jax.ffi.register_ffi_target(
    "refactor_status_c128", klujax_cpp.refactor_status_c128(), platform="cpu"
)
refactor_status_c128_low = mlir.lower_fun(
    refactor_status_c128_impl, multiple_results=True
)
mlir.register_lowering(refactor_status_c128, refactor_status_c128_low)

jax.ffi.register_ffi_target(
    "refactor_and_solve_status_f64",
    klujax_cpp.refactor_and_solve_status_f64(),
    platform="cpu",
)
refactor_and_solve_status_f64_low = mlir.lower_fun(
    refactor_and_solve_status_f64_impl, multiple_results=True
)
mlir.register_lowering(refactor_and_solve_status_f64, refactor_and_solve_status_f64_low)

jax.ffi.register_ffi_target(
    "refactor_and_solve_status_c128",
    klujax_cpp.refactor_and_solve_status_c128(),
    platform="cpu",
)
refactor_and_solve_status_c128_low = mlir.lower_fun(
    refactor_and_solve_status_c128_impl, multiple_results=True
)
mlir.register_lowering(
    refactor_and_solve_status_c128, refactor_and_solve_status_c128_low
)

jax.ffi.register_ffi_target("rcond_f64", klujax_cpp.rcond_f64(), platform="cpu")
rcond_f64_low = mlir.lower_fun(rcond_f64_impl, multiple_results=False)
mlir.register_lowering(rcond_f64, rcond_f64_low)

jax.ffi.register_ffi_target("rcond_c128", klujax_cpp.rcond_c128(), platform="cpu")
rcond_c128_low = mlir.lower_fun(rcond_c128_impl, multiple_results=False)
mlir.register_lowering(rcond_c128, rcond_c128_low)

jax.ffi.register_ffi_target("condest_f64", klujax_cpp.condest_f64(), platform="cpu")
condest_f64_low = mlir.lower_fun(condest_f64_impl, multiple_results=False)
mlir.register_lowering(condest_f64, condest_f64_low)

jax.ffi.register_ffi_target("condest_c128", klujax_cpp.condest_c128(), platform="cpu")
condest_c128_low = mlir.lower_fun(condest_c128_impl, multiple_results=False)
mlir.register_lowering(condest_c128, condest_c128_low)

free_numeric_low = mlir.lower_fun(free_numeric_impl, multiple_results=False)
mlir.register_lowering(free_numeric_p, free_numeric_low)

free_symbolic_low = mlir.lower_fun(free_symbolic_impl, multiple_results=False)
mlir.register_lowering(free_symbolic_p, free_symbolic_low)

# Abstract Evals ======================================================================


@dot_f64.def_abstract_eval
@dot_c128.def_abstract_eval
@solve_f64.def_abstract_eval
@solve_c128.def_abstract_eval
@solve_with_symbol_f64.def_abstract_eval
@solve_with_symbol_c128.def_abstract_eval
@tsolve_with_symbol_f64.def_abstract_eval
@tsolve_with_symbol_c128.def_abstract_eval
def general_abstract_eval(
    Ai: Array, Aj: Array, Ax: Array, b: Array, *args: Array
) -> ShapedArray:
    return ShapedArray(b.shape, b.dtype)


@analyze_p.def_abstract_eval
def analyze_abstract_eval(Ai: Array, Aj: Array, n_col: Array) -> ShapedArray:
    return ShapedArray((), jnp.uint64)


@free_symbolic_p.def_effectful_abstract_eval
def free_symbolic_abstract_eval(
    symbolic: Array, ordering: Array
) -> tuple[ShapedArray, set]:
    return ShapedArray((), jnp.int32), {_free_symbolic_effect}


@factor_f64.def_abstract_eval
@factor_c128.def_abstract_eval
def factor_abstract_eval(Ai, Aj, Ax, symbolic, *, n_col):
    return ShapedArray((Ax.shape[0],), jnp.uint64)


@refactor_f64.def_abstract_eval
@refactor_c128.def_abstract_eval
def refactor_abstract_eval(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    return ShapedArray((Ax.shape[0],), jnp.uint64)


@solve_with_numeric_f64.def_abstract_eval
@solve_with_numeric_c128.def_abstract_eval
@tsolve_with_numeric_f64.def_abstract_eval
@tsolve_with_numeric_c128.def_abstract_eval
def solve_with_numeric_abstract_eval(Ai, Aj, Ax, symbolic, numeric, b):
    # Output has same shape as input b
    return ShapedArray(b.shape, b.dtype)


@refactor_and_solve_f64.def_abstract_eval
@refactor_and_solve_c128.def_abstract_eval
def refactor_and_solve_abstract_eval(Ai, Aj, Ax, b, symbolic, numeric):
    # Returns (x with same shape as b, out_numeric with same shape as numeric)
    return ShapedArray(b.shape, b.dtype), ShapedArray(numeric.shape, jnp.uint64)


@refactor_status_f64.def_abstract_eval
@refactor_status_c128.def_abstract_eval
def refactor_status_abstract_eval(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    # As refactor, plus a KLU status and a RebuildReason per left-hand side
    return (
        ShapedArray((Ax.shape[0],), jnp.uint64),
        ShapedArray((Ax.shape[0],), jnp.int32),
        ShapedArray((Ax.shape[0],), jnp.int32),
    )


@refactor_and_solve_status_f64.def_abstract_eval
@refactor_and_solve_status_c128.def_abstract_eval
def refactor_and_solve_status_abstract_eval(Ai, Aj, Ax, b, symbolic, numeric):
    return (
        ShapedArray(b.shape, b.dtype),
        ShapedArray(numeric.shape, jnp.uint64),
        ShapedArray((Ax.shape[0],), jnp.int32),
        ShapedArray((Ax.shape[0],), jnp.int32),
    )


@solve_with_numeric_status_f64.def_abstract_eval
@solve_with_numeric_status_c128.def_abstract_eval
@tsolve_with_numeric_status_f64.def_abstract_eval
@tsolve_with_numeric_status_c128.def_abstract_eval
def solve_with_numeric_status_abstract_eval(Ai, Aj, Ax, symbolic, numeric, b):
    # Solution (shape of b), plus one RebuildReason per numeric handle
    return (
        ShapedArray(b.shape, b.dtype),
        ShapedArray(numeric.shape, jnp.int32),
    )


@solve_with_symbol_status_f64.def_abstract_eval
@solve_with_symbol_status_c128.def_abstract_eval
@tsolve_with_symbol_status_f64.def_abstract_eval
@tsolve_with_symbol_status_c128.def_abstract_eval
def solve_with_symbol_status_abstract_eval(Ai, Aj, Ax, b, symbolic):
    # Solution (shape of b), plus a scalar RebuildReason for the symbolic handle
    return (
        ShapedArray(b.shape, b.dtype),
        ShapedArray((), jnp.int32),
    )


@rcond_f64.def_abstract_eval
@rcond_c128.def_abstract_eval
def rcond_abstract_eval(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    return ShapedArray(numeric.shape, jnp.float64)


@condest_f64.def_abstract_eval
@condest_c128.def_abstract_eval
def condest_abstract_eval(Ai, Aj, Ax, symbolic, numeric, *, n_col):
    return ShapedArray((Ax.shape[0],), jnp.float64)


# Forward Differentiation =============================================================


def dot_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[dot_f64] = dot_f64_value_and_jvp


def dot_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return dot_value_and_jvp(dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[dot_c128] = dot_c128_value_and_jvp


def solve_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_f64, dot_f64, arg_values, arg_tangents)


ad.primitive_jvps[solve_f64] = solve_f64_value_and_jvp


def solve_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    return solve_value_and_jvp(solve_c128, dot_c128, arg_values, arg_tangents)


ad.primitive_jvps[solve_c128] = solve_c128_value_and_jvp


def solve_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
    x = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
    dA_x = prim_dot.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), dAx, x)
    invA_dA_x = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, dA_x)
    invA_db = prim_solve.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, db)
    return x, -invA_dA_x + invA_db


def dot_value_and_jvp(
    prim: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array],
    arg_tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    Ai, Aj, Ax, b = arg_values
    dAi, dAj, dAx, db = arg_tangents
    if not isinstance(dAi, ad.Zero) or not isinstance(dAj, ad.Zero):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)  # noqa: TRY004
    dAx = dAx if not isinstance(dAx, ad.Zero) else jnp.zeros_like(Ax)
    db = db if not isinstance(db, ad.Zero) else jnp.zeros_like(b)
    x = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
    dA_b = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), dAx, b)
    A_db = prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, db)
    return x, dA_b + A_db


# Batching (vmap) =====================================================================


def dot_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_f64] = dot_f64_vmap


def dot_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(dot_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[dot_c128] = dot_c128_vmap


def solve_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_f64, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_f64] = solve_f64_vmap


def solve_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap(solve_c128, vector_arg_values, batch_axes)


batching.primitive_batchers[solve_c128] = solve_c128_vmap


def solve_with_symbol_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        solve_with_symbol_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_symbol_f64] = solve_with_symbol_f64_vmap


def solve_with_symbol_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        solve_with_symbol_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[solve_with_symbol_c128] = solve_with_symbol_c128_vmap


def tsolve_with_symbol_f64_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        tsolve_with_symbol_f64, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_symbol_f64] = tsolve_with_symbol_f64_vmap


def tsolve_with_symbol_c128_vmap(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    return general_vmap_with_symbol(
        tsolve_with_symbol_c128, vector_arg_values, batch_axes
    )


batching.primitive_batchers[tsolve_with_symbol_c128] = tsolve_with_symbol_c128_vmap


def solve_with_symbol_status_f64_vmap(vals, axes):
    return general_vmap_with_symbol_status(solve_with_symbol_status_f64, vals, axes)


batching.primitive_batchers[solve_with_symbol_status_f64] = (
    solve_with_symbol_status_f64_vmap
)


def solve_with_symbol_status_c128_vmap(vals, axes):
    return general_vmap_with_symbol_status(solve_with_symbol_status_c128, vals, axes)


batching.primitive_batchers[solve_with_symbol_status_c128] = (
    solve_with_symbol_status_c128_vmap
)


def tsolve_with_symbol_status_f64_vmap(vals, axes):
    return general_vmap_with_symbol_status(tsolve_with_symbol_status_f64, vals, axes)


batching.primitive_batchers[tsolve_with_symbol_status_f64] = (
    tsolve_with_symbol_status_f64_vmap
)


def tsolve_with_symbol_status_c128_vmap(vals, axes):
    return general_vmap_with_symbol_status(tsolve_with_symbol_status_c128, vals, axes)


batching.primitive_batchers[tsolve_with_symbol_status_c128] = (
    tsolve_with_symbol_status_c128_vmap
)


# Every batcher forwards **params so the n_col attribute (a primitive param on
# the split-solve calls) rides through vmap to the bind unchanged.
def solve_with_numeric_f64_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric(solve_with_numeric_f64, vals, axes, **params)


batching.primitive_batchers[solve_with_numeric_f64] = solve_with_numeric_f64_vmap


def solve_with_numeric_c128_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric(solve_with_numeric_c128, vals, axes, **params)


batching.primitive_batchers[solve_with_numeric_c128] = solve_with_numeric_c128_vmap


def tsolve_with_numeric_f64_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric(tsolve_with_numeric_f64, vals, axes, **params)


batching.primitive_batchers[tsolve_with_numeric_f64] = tsolve_with_numeric_f64_vmap


def tsolve_with_numeric_c128_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric(tsolve_with_numeric_c128, vals, axes, **params)


batching.primitive_batchers[tsolve_with_numeric_c128] = tsolve_with_numeric_c128_vmap


def solve_with_numeric_status_f64_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric_status(
        solve_with_numeric_status_f64, vals, axes, **params
    )


batching.primitive_batchers[solve_with_numeric_status_f64] = (
    solve_with_numeric_status_f64_vmap
)


def solve_with_numeric_status_c128_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric_status(
        solve_with_numeric_status_c128, vals, axes, **params
    )


batching.primitive_batchers[solve_with_numeric_status_c128] = (
    solve_with_numeric_status_c128_vmap
)


def tsolve_with_numeric_status_f64_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric_status(
        tsolve_with_numeric_status_f64, vals, axes, **params
    )


batching.primitive_batchers[tsolve_with_numeric_status_f64] = (
    tsolve_with_numeric_status_f64_vmap
)


def tsolve_with_numeric_status_c128_vmap(vals, axes, **params: object):
    return general_vmap_with_numeric_status(
        tsolve_with_numeric_status_c128, vals, axes, **params
    )


batching.primitive_batchers[tsolve_with_numeric_status_c128] = (
    tsolve_with_numeric_status_c128_vmap
)


def factor_f64_vmap(vals, axes, **params: object):
    return general_vmap_factor(factor_f64, vals, axes, **params)


batching.primitive_batchers[factor_f64] = factor_f64_vmap


def factor_c128_vmap(vals, axes, **params: object):
    return general_vmap_factor(factor_c128, vals, axes, **params)


batching.primitive_batchers[factor_c128] = factor_c128_vmap


def refactor_f64_vmap(vals, axes, **params: object):
    return general_vmap_refactor(refactor_f64, vals, axes, **params)


batching.primitive_batchers[refactor_f64] = refactor_f64_vmap


def refactor_c128_vmap(vals, axes, **params: object):
    return general_vmap_refactor(refactor_c128, vals, axes, **params)


batching.primitive_batchers[refactor_c128] = refactor_c128_vmap


def refactor_status_f64_vmap(vals, axes, **params: object):
    return general_vmap_refactor_status(refactor_status_f64, vals, axes, **params)


batching.primitive_batchers[refactor_status_f64] = refactor_status_f64_vmap


def refactor_status_c128_vmap(vals, axes, **params: object):
    return general_vmap_refactor_status(refactor_status_c128, vals, axes, **params)


batching.primitive_batchers[refactor_status_c128] = refactor_status_c128_vmap


def condest_f64_vmap(vals, axes, **params: object):
    # condest takes the same arguments as refactor and also returns one value per lhs.
    return general_vmap_refactor(condest_f64, vals, axes, **params)


batching.primitive_batchers[condest_f64] = condest_f64_vmap


def condest_c128_vmap(vals, axes, **params: object):
    return general_vmap_refactor(condest_c128, vals, axes, **params)


batching.primitive_batchers[condest_c128] = condest_c128_vmap


def rcond_f64_vmap(vals, axes, **params: object):
    # rcond now takes (Ai, Aj, Ax, symbolic, numeric), same shape as refactor.
    return general_vmap_refactor(rcond_f64, vals, axes, **params)


batching.primitive_batchers[rcond_f64] = rcond_f64_vmap


def rcond_c128_vmap(vals, axes, **params: object):
    return general_vmap_refactor(rcond_c128, vals, axes, **params)


batching.primitive_batchers[rcond_c128] = rcond_c128_vmap


def general_vmap(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x = vector_arg_values
    aAi, aAj, aAx, ax = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 0
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing "
                f"over Ax. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        ax = 0
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 0
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_rhs
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x).reshape(
            *shape
        ), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_symbol(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[Array, int]:
    Ai, Aj, Ax, x, symbolic = vector_arg_values
    aAi, aAj, aAx, ax, asymbolic = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)

    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)

    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 0
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing "
                f"over Ax. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_lhs
        ax = 0
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 0
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        # vectorize over n_rhs
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        ).reshape(*shape), 3
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_symbol_status(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[tuple[Array, Array], tuple[int, int | None]]:
    # As general_vmap_with_symbol, but the primitive also returns a scalar
    # rebuild reason for the one symbolic handle. That handle is never
    # vectorized, so its reason is shared across the batch and stays unbatched.
    Ai, Aj, Ax, x, symbolic = vector_arg_values
    aAi, aAj, aAx, ax, asymbolic = batch_axes

    if aAi is not None:
        msg = "Ai cannot be vectorized."
        raise ValueError(msg)
    if aAj is not None:
        msg = "Aj cannot be vectorized."
        raise ValueError(msg)
    if asymbolic is not None:
        msg = "symbolic handle cannot be vectorized."
        raise ValueError(msg)

    if aAx is not None and ax is not None:
        if Ax.ndim != 3 or x.ndim != 4:
            msg = (
                "Ax and x should be 3D and 4D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.moveaxis(x, ax, 0)
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        sol, rebuild = prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        )
        return (sol.reshape(*shape), rebuild), (0, None)
    if aAx is not None:
        if Ax.ndim != 3 or x.ndim != 3:
            msg = (
                "Ax and x should both be 3D when vectorizing over Ax. "
                f"Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        x = jnp.broadcast_to(x[None], (Ax.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        shape = x.shape
        Ax = Ax.reshape(Ax.shape[0] * Ax.shape[1], Ax.shape[2])
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        sol, rebuild = prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        )
        return (sol.reshape(*shape), rebuild), (0, None)
    if ax is not None:
        if Ax.ndim != 2 or x.ndim != 4:
            msg = (
                "Ax and x should both be 2D and 4D respectively when vectorizing "
                f"over x. Got: {Ax.shape=}; {x.shape=}."
            )
            raise ValueError(msg)
        x = jnp.moveaxis(x, ax, 3)
        shape = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        sol, rebuild = prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, x, symbolic
        )
        return (sol.reshape(*shape), rebuild), (3, None)
    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_numeric(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, ...],
    **params: Any,  # noqa: ANN401
) -> tuple[Array, int]:
    Ai, Aj, Ax, symbolic, numeric, b = vector_arg_values
    aAi, aAj, aAx, asymbolic, anumeric, ab = batch_axes

    # Ai/Aj/symbolic are shared across the batch, so drop any batch axis on them.
    Ai = _debatch_shared(Ai, aAi).astype(jnp.int32)
    Aj = _debatch_shared(Aj, aAj).astype(jnp.int32)
    symbolic = _debatch_shared(symbolic, asymbolic)

    # numeric and its values Ax come from the same token, so they batch together.
    # Fold that batch axis into n_lhs the way the multi-LHS convention expects.
    if anumeric is not None:
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs = numeric.shape
        numeric = numeric.reshape(batch * n_lhs)
        if aAx is not None:
            Ax = jnp.moveaxis(Ax, aAx, 0).reshape(batch * n_lhs, -1)
        else:
            Ax = jnp.broadcast_to(Ax[None], (batch, *Ax.shape)).reshape(
                batch * n_lhs, -1
            )
        if ab is not None:
            b = jnp.moveaxis(b, ab, 0)
        else:
            b = jnp.broadcast_to(b[None], (batch, *b.shape))
        shape = b.shape
        if b.ndim >= 3:
            b = b.reshape(batch * b.shape[1], *b.shape[2:])
        return prim.bind(Ai, Aj, Ax, symbolic, numeric, b, **params).reshape(*shape), 0

    if ab is not None:
        # Only the right-hand side is batched: it becomes extra multi-RHS columns.
        b = jnp.moveaxis(b, ab, -1)
        shape = b.shape
        if b.ndim >= 3:
            b = b.reshape(*b.shape[:-2], b.shape[-2] * b.shape[-1])
        return prim.bind(Ai, Aj, Ax, symbolic, numeric, b, **params).reshape(
            *shape
        ), len(shape) - 1

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def general_vmap_with_numeric_status(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, ...],
    **params: Any,  # noqa: ANN401
) -> tuple[tuple[Array, Array], tuple[int, int | None]]:
    # As general_vmap_with_numeric, but the primitive also returns a per-handle
    # rebuild array (shape of numeric). x batches like the plain solve. The rebuild
    # array batches with the numeric handle when that is the vectorized axis.
    Ai, Aj, Ax, symbolic, numeric, b = vector_arg_values
    aAi, aAj, aAx, asymbolic, anumeric, ab = batch_axes

    Ai = _debatch_shared(Ai, aAi).astype(jnp.int32)
    Aj = _debatch_shared(Aj, aAj).astype(jnp.int32)
    symbolic = _debatch_shared(symbolic, asymbolic)

    if anumeric is not None:
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs = numeric.shape
        numeric = numeric.reshape(batch * n_lhs)
        if aAx is not None:
            Ax = jnp.moveaxis(Ax, aAx, 0).reshape(batch * n_lhs, -1)
        else:
            Ax = jnp.broadcast_to(Ax[None], (batch, *Ax.shape)).reshape(
                batch * n_lhs, -1
            )
        if ab is not None:
            b = jnp.moveaxis(b, ab, 0)
        else:
            b = jnp.broadcast_to(b[None], (batch, *b.shape))
        shape = b.shape
        if b.ndim >= 3:
            b = b.reshape(batch * b.shape[1], *b.shape[2:])
        x, rebuild = prim.bind(Ai, Aj, Ax, symbolic, numeric, b, **params)
        return (x.reshape(*shape), rebuild.reshape(batch, n_lhs)), (0, 0)

    if ab is not None:
        b = jnp.moveaxis(b, ab, -1)
        shape = b.shape
        if b.ndim >= 3:
            b = b.reshape(*b.shape[:-2], b.shape[-2] * b.shape[-1])
        x, rebuild = prim.bind(Ai, Aj, Ax, symbolic, numeric, b, **params)
        # numeric is not batched, so its rebuild array is shared (unbatched).
        return (x.reshape(*shape), rebuild), (len(shape) - 1, None)

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def _debatch_shared(x: Array, axis: int | None) -> Array:
    """Collapse a spurious batch axis on an array that is shared across the batch.

    A token carries its pattern arrays (Ai, Aj) and its analysis id as pytree
    leaves, so vmapping over a token adds a batch axis to those too even though
    every row is identical. Taking index 0 recovers the single shared value.
    """
    return x if axis is None else jnp.take(x, 0, axis=axis)


def general_vmap_factor(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None],
    **params: Any,  # noqa: ANN401
) -> tuple[Array, int]:
    Ai, Aj, Ax, symbolic = vector_arg_values
    aAi, aAj, aAx, asymbolic = batch_axes

    # Ai/Aj/symbolic are shared across the batch, so drop any batch axis on them.
    Ai = _debatch_shared(Ai, aAi)
    Aj = _debatch_shared(Aj, aAj)
    symbolic = _debatch_shared(symbolic, asymbolic)

    if aAx is not None:
        if Ax.ndim != 3:
            msg = f"Ax should be 3D when vectorizing over it. Got: {Ax.shape=}."
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        batch, n_lhs, n_vals = Ax.shape
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        return prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic, **params
        ).reshape(batch, n_lhs), 0

    msg = "vmap failed. Please select an axis to vectorize over."
    raise ValueError(msg)


def _flatten_refactor_batch(
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
) -> tuple[tuple[Array, Array, Array, Array, Array], tuple[int, int]]:
    """Fold the vmap batch axis of a refactor-shaped call into n_lhs.

    Returns the flattened bind arguments and the (batch, n_lhs) shape every per-lhs
    output has to be reshaped back to. Shared by refactor, refactor_status and condest,
    which all take (Ai, Aj, Ax, symbolic, numeric).
    """
    Ai, Aj, Ax, symbolic, numeric = vector_arg_values
    aAi, aAj, aAx, asymbolic, anumeric = batch_axes

    # Ai/Aj/symbolic are shared across the batch, so drop any batch axis on them.
    Ai = _debatch_shared(Ai, aAi)
    Aj = _debatch_shared(Aj, aAj)
    symbolic = _debatch_shared(symbolic, asymbolic)

    if aAx is not None and anumeric is not None:
        if Ax.ndim != 3 or numeric.ndim != 2:
            msg = (
                "Ax and numeric should be 3D and 2D respectively when vectorizing "
                f"over them simultaneously. Got: {Ax.shape=}; {numeric.shape=}."
            )
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs, n_vals = Ax.shape
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        numeric = numeric.reshape(batch * n_lhs)

    elif aAx is not None:
        if Ax.ndim != 3:
            msg = f"Ax should be 3D when vectorizing over it. Got: {Ax.shape=}."
            raise ValueError(msg)
        Ax = jnp.moveaxis(Ax, aAx, 0)
        batch, n_lhs, n_vals = Ax.shape
        numeric = jnp.broadcast_to(numeric[None], (batch, numeric.shape[0]))
        Ax = Ax.reshape(batch * n_lhs, n_vals)
        numeric = numeric.reshape(batch * n_lhs)

    elif anumeric is not None:
        if numeric.ndim != 2:
            msg = (
                f"numeric should be 2D when vectorizing over it. Got: {numeric.shape=}."
            )
            raise ValueError(msg)
        numeric = jnp.moveaxis(numeric, anumeric, 0)
        batch, n_lhs = numeric.shape
        Ax = jnp.broadcast_to(Ax[None], (batch, Ax.shape[0], Ax.shape[1]))
        Ax = Ax.reshape(batch * n_lhs, Ax.shape[2])
        numeric = numeric.reshape(batch * n_lhs)

    else:
        msg = "vmap failed. Please select an axis to vectorize over."
        raise ValueError(msg)

    args = (Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, symbolic, numeric)
    return args, (batch, n_lhs)


def general_vmap_refactor(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
    **params: Any,  # noqa: ANN401
) -> tuple[Array, int]:
    args, shape = _flatten_refactor_batch(vector_arg_values, batch_axes)
    return prim.bind(*args, **params).reshape(*shape), 0


def general_vmap_refactor_status(
    prim: jax.extend.core.Primitive,
    vector_arg_values: tuple[Array, Array, Array, Array, Array],
    batch_axes: tuple[int | None, int | None, int | None, int | None, int | None],
    **params: Any,  # noqa: ANN401
) -> tuple[tuple[Array, Array, Array], tuple[int, int, int]]:
    # As general_vmap_refactor, but the status and rebuild outputs are batched
    # along the same axis as the numeric handle.
    args, shape = _flatten_refactor_batch(vector_arg_values, batch_axes)
    numeric, status, rebuild = prim.bind(*args, **params)
    return (
        (numeric.reshape(*shape), status.reshape(*shape), rebuild.reshape(*shape)),
        (0, 0, 0),
    )


def free_numeric_p_vmap(
    vector_arg_values: tuple[Array],
    batch_axes: tuple[int | None],
) -> tuple[Array, int | None]:
    return (
        jax.vmap(
            jax.custom_batching.sequential_vmap(free_numeric_p.bind), in_axes=batch_axes
        )(*vector_arg_values),
        batch_axes[0],
    )


batching.primitive_batchers[free_numeric_p] = free_numeric_p_vmap


def free_symbolic_p_vmap(
    vector_arg_values: tuple[Array],
    batch_axes: tuple[int | None],
) -> tuple[Array, int | None]:
    return (
        jax.vmap(
            jax.custom_batching.sequential_vmap(free_symbolic_p.bind),
            in_axes=batch_axes,
        )(*vector_arg_values),
        batch_axes[0],
    )


batching.primitive_batchers[free_symbolic_p] = free_symbolic_p_vmap


# Transposition =======================================================================


def dot_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_f64, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_f64] = dot_f64_transpose


def dot_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    return dot_transpose(dot_c128, ct, Ai, Aj, Ax, x)


ad.primitive_transposes[dot_c128] = dot_c128_transpose


def solve_f64_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_f64, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_f64] = solve_f64_transpose


def solve_c128_transpose(
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    return solve_transpose(solve_c128, ct, Ai, Aj, Ax, b)


ad.primitive_transposes[solve_c128] = solve_c128_transpose


def dot_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    x: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(x):
        # replace x by ct
        return Aj, Ai, Ax, prim.bind(Aj.astype(jnp.int32), Ai.astype(jnp.int32), Ax, ct)

    if ad.is_undefined_primal(Ax):
        # ∂L/∂Ax[m,n] = Σₖ ct[m, Ai[n], k] · x[m, Aj[n], k]
        return Ai, Aj, (ct[:, Ai] * x[:, Aj, :]).sum(-1), x

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_transpose(
    prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
) -> tuple[Array, Array, Array, Array]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    if ad.is_undefined_primal(b):
        b_bar = prim.bind(Aj.astype(jnp.int32), Ai.astype(jnp.int32), Ax, ct)
        return Ai, Aj, Ax, b_bar

    if ad.is_undefined_primal(Ax):
        Ax_bar = -(
            ct * prim.bind(Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b)
        ).sum(-1)
        return Ai, Aj, Ax_bar, b

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


# Validators ==========================================================================


def validate_args(  # noqa: C901,PLR0912
    Ai: Array, Aj: Array, Ax: Array, x: Array, x_name: str = "x"
) -> tuple[Array, Array, Array, Array, tuple[int, ...]]:
    # cases:
    # - (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_lhs, n_nz) x (n_lhs, n_col) --> (n_lhs, n_nz) x (n_lhs, n_col, 1)
    # - (n_nz,) x (n_lhs, n_col, n_rhs) --> (n_lhs, n_nz) x (n_lhs, n_col, n_rhs)
    # - (n_nz,) x (n_col, n_rhs) --> (1, n_nz) x (1, n_col, n_rhs)
    # - (n_nz,) x (n_col,) --> (1, n_nz) x (1, n_col, 1)
    if Ai.ndim != 1:
        msg = f"Ai should be 1D with shape (n_nz). Got: {Ai.shape=}."
        raise ValueError(msg)
    if Aj.ndim != 1:
        msg = f"Aj should be 1D with shape (n_nz,). Got: {Aj.shape=}."
        raise ValueError(msg)
    if Ax.ndim == 0 or Ax.ndim > 2:
        msg = (
            "Ax should be 1D with shape (n_nz,) "
            "or 2D with shape (n_lhs, n_nz). "
            f"Got: {Ax.shape=}."
        )
        raise ValueError(msg)
    if x.ndim == 0 or x.ndim > 3:
        msg = (
            f"{x_name} should be 1D with shape (n_col,) "
            "or 2D with shape (n_col, n_rhs) "
            "or 3D with shape (n_lhs, n_col, n_rhs). "
            f"Got: {x_name}.shape={x.shape}."
        )
        raise ValueError(msg)

    shape = x.shape

    if Ax.ndim == 1 and x.ndim == 1:  # expand Ax and b dims
        debug(f"assuming (n_nz:={Ax.shape[0]},) x (n_col:={x.shape[0]},)")
        Ax = Ax[None, :]
        x = x[None, :, None]
    elif Ax.ndim == 1 and x.ndim == 2:  # expand Ax and b dims
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_col:={x.shape[0]}, n_rhs:={x.shape[1]})"
        )
        Ax = Ax[None, :]
        x = x[None, :, :]
    elif Ax.ndim == 1 and x.ndim == 3:  # expand A dim (broadcast will happen in base)
        debug(
            f"assuming (n_nz:={Ax.shape[0]},) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        Ax = Ax[None, :]
    elif Ax.ndim == 2 and x.ndim == 1:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_col:={x.shape[0]},)"
        )
        x = x[None, :, None]
        shape = (Ax.shape[0], shape[0])  # we need to expand the shape here.
    elif Ax.ndim == 2 and x.ndim == 2:  # expand dims to base case
        debug(
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
        )
        if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
            msg = (
                f"Ax (2D) and {x_name} (2D) should have their first shape "
                f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
                f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
                f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]})"
            )
            raise ValueError(msg)
        x = x[:, :, None]
        if x.shape[0] == 1 and Ax.shape[0] > 0:
            shape = (Ax.shape[0], *shape[1:])

    if Ax.ndim != 2 or x.ndim != 3:
        msg = (
            f"Invalid shapes for Ax and {x_name}. "
            f"Got: {Ax.shape=}; {x_name}.shape={x.shape}. "
            f"Expected: Ax.shape=([n_lhs],n_nz); "
            f"{x_name}.shape=([n_lhs],n_col,[n_rhs])."
        )
        raise ValueError(msg)

    # base case
    debug(
        f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
        f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
    )
    if Ax.shape[0] != x.shape[0] and Ax.shape[0] != 1 and x.shape[0] != 1:
        msg = (
            f"Ax (2D) and {x_name} (3D) should have their first shape "
            f"index `n_lhs` match. Got: {Ax.shape=}; {x_name}.shape={x.shape}."
            f"assuming (n_lhs:={Ax.shape[0]}, n_nz:={Ax.shape[1]}) x "
            f"(n_lhs:={x.shape[0]}, n_col:={x.shape[1]}, n_rhs:={x.shape[2]})"
        )
        raise ValueError(msg)
    n_lhs = max(Ax.shape[0], x.shape[0])  # handle broadcastable 1-index
    Ax = jnp.broadcast_to(Ax, (n_lhs, Ax.shape[1]))
    x = jnp.broadcast_to(x, (n_lhs, x.shape[1], x.shape[2]))
    if len(shape) == 3 and shape[0] != x.shape[0]:
        shape = (Ax.shape[0], shape[1], shape[2])
    return Ai, Aj, Ax, x, shape


# Differentiation rules for solve_with_symbol =================================


def solve_with_symbol_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    prim_dot: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """Jacobian-vector product rule for `solve_with_symbol`.

    The rule for the JVP of a linear solve `x = A^-1 * b` is:
    `dx = A^-1 * (db - dA * x)`

    This function implements this rule efficiently. It first computes the primal
    solution `x`. Then it computes the right-hand side of the tangent system,
    `rhs = db - dA * x`. If `dA` is not zero, `dA * x` is computed using the
    `dot` primitive. If the `rhs` is not zero, `dx` is found by solving the
    system `A * dx = rhs`, reusing the symbolic factorization of A for
    efficiency.

    Args:
        prim_solve: The `solve_with_symbol` primitive.
        prim_dot: The corresponding `dot` primitive.
        arg_values: Primal values for `(Ai, Aj, Ax, b, symbolic)`.
        arg_tangents: Tangent values for `(Ai, Aj, Ax, b, symbolic)`.

    Returns:
        A tuple containing the primal output `x` and the tangent output `dx`.

    """
    Ai, Aj, Ax, b, symbolic = arg_values
    _t_Ai, _t_Aj, t_Ax, t_b, _t_symbolic = arg_tangents

    x = prim_solve.bind(Ai, Aj, Ax, b, symbolic)

    rhs = t_b

    if not isinstance(t_Ax, ad.Zero):
        dAx = prim_dot.bind(Ai, Aj, t_Ax, x)
        rhs = -dAx if isinstance(rhs, ad.Zero) else rhs - dAx

    if isinstance(rhs, ad.Zero):
        dx = jnp.zeros_like(x)
    else:
        dx = prim_solve.bind(Ai, Aj, Ax, rhs, symbolic)

    return x, dx


# Register JVP for Float64
def solve_with_symbol_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """JVP rule for float64 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_f64,
        dot_f64,
        arg_values,
        arg_tangents,
    )


ad.primitive_jvps[solve_with_symbol_f64] = solve_with_symbol_f64_value_and_jvp


# Register JVP for Complex128
def solve_with_symbol_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """JVP rule for complex128 solve_with_symbol."""
    return solve_with_symbol_value_and_jvp(
        solve_with_symbol_c128,
        dot_c128,
        arg_values,
        arg_tangents,
    )


ad.primitive_jvps[solve_with_symbol_c128] = solve_with_symbol_c128_value_and_jvp


def solve_with_symbol_transpose(
    solve_prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    b: Array,
    symbolic: Array,
) -> tuple[Array, Array, Array, Array, None]:
    if ad.is_undefined_primal(Ai) or ad.is_undefined_primal(Aj):
        msg = "Sparse indices Ai and Aj should not require gradients."
        raise ValueError(msg)

    # Pick the right tsolve primitive
    if solve_prim is solve_with_symbol_f64:
        tsolve_prim = tsolve_with_symbol_f64
    else:
        tsolve_prim = tsolve_with_symbol_c128

    if ad.is_undefined_primal(b):
        # FAST PATH: Backpropagate through b using the reused handle!
        b_bar = tsolve_prim.bind(
            Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, ct, symbolic
        )
        return Ai, Aj, Ax, b_bar, None

    if ad.is_undefined_primal(Ax):
        Ax_bar = -(
            ct
            * tsolve_prim.bind(
                Ai.astype(jnp.int32), Aj.astype(jnp.int32), Ax, b, symbolic
            )
        ).sum(-1)
        return Ai, Aj, Ax_bar, b, None

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_with_symbol_f64_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(
        solve_with_symbol_f64, ct, Ai, Aj, Ax, b, symbolic
    )


ad.primitive_transposes[solve_with_symbol_f64] = solve_with_symbol_f64_transpose


def solve_with_symbol_c128_transpose(ct, Ai, Aj, Ax, b, symbolic):
    return solve_with_symbol_transpose(
        solve_with_symbol_c128, ct, Ai, Aj, Ax, b, symbolic
    )


ad.primitive_transposes[solve_with_symbol_c128] = solve_with_symbol_c128_transpose


# Differentiation rules for solve_with_numeric ================================


def solve_with_numeric_value_and_jvp(
    prim_solve: jax.extend.core.Primitive,
    arg_values: tuple[Array, Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    """Jacobian-vector product rule for `solve_with_numeric`.

    Since the matrix values are baked into the numeric handle, we only track
    gradients with respect to `b`. The rule is simply: dx = A^-1 * db.
    """
    Ai, Aj, Ax, symbolic, numeric, b = arg_values
    t_b = arg_tangents[-1]

    x = prim_solve.bind(Ai, Aj, Ax, symbolic, numeric, b)

    if isinstance(t_b, ad.Zero):
        dx = jnp.zeros_like(x)
    else:
        dx = prim_solve.bind(Ai, Aj, Ax, symbolic, numeric, t_b)

    return x, dx


def solve_with_numeric_f64_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    return solve_with_numeric_value_and_jvp(
        solve_with_numeric_f64, arg_values, arg_tangents
    )


ad.primitive_jvps[solve_with_numeric_f64] = solve_with_numeric_f64_value_and_jvp


def solve_with_numeric_c128_value_and_jvp(
    arg_values: tuple[Array, Array, Array, Array, Array, Array],
    arg_tangents: tuple[Any, Any, Any, Any, Any, Any],
) -> tuple[Array, Array]:
    return solve_with_numeric_value_and_jvp(
        solve_with_numeric_c128, arg_values, arg_tangents
    )


ad.primitive_jvps[solve_with_numeric_c128] = solve_with_numeric_c128_value_and_jvp


def solve_with_numeric_transpose(
    solve_prim: jax.extend.core.Primitive,
    ct: Array,
    Ai: Array,
    Aj: Array,
    Ax: Array,
    symbolic: Array,
    numeric: Array,
    b: Array,
) -> tuple[None, None, None, None, None, Array]:
    """Compute the transpose of solve_with_numeric.

    Uses the tsolve primitives to solve A^T x_bar = b_bar with the existing
    numeric factorization. Only b is linear, so only its cotangent is returned.
    """
    tsolve_prim = (
        tsolve_with_numeric_f64
        if solve_prim is solve_with_numeric_f64
        else tsolve_with_numeric_c128
    )

    if ad.is_undefined_primal(b):
        b_bar = tsolve_prim.bind(Ai, Aj, Ax, symbolic, numeric, ct)
        # Only b takes a gradient. The matrix and handle inputs do not.
        return None, None, None, None, None, b_bar

    msg = "No undefined primals in transpose."
    raise ValueError(msg)


def solve_with_numeric_f64_transpose(ct, Ai, Aj, Ax, symbolic, numeric, b):
    return solve_with_numeric_transpose(
        solve_with_numeric_f64, ct, Ai, Aj, Ax, symbolic, numeric, b
    )


ad.primitive_transposes[solve_with_numeric_f64] = solve_with_numeric_f64_transpose


def solve_with_numeric_c128_transpose(ct, Ai, Aj, Ax, symbolic, numeric, b):
    return solve_with_numeric_transpose(
        solve_with_numeric_c128, ct, Ai, Aj, Ax, symbolic, numeric, b
    )


ad.primitive_transposes[solve_with_numeric_c128] = solve_with_numeric_c128_transpose
