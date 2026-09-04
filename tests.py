import sys
from functools import wraps

import jax
import jax.core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from jax import lax

import klujax
from klujax import COMPLEX_DTYPES, coalesce

jax.config.update("jax_enable_x64", val=True)

OPS_DENSE = {  # sparse to dense
    klujax.dot: lax.dot,
    klujax.solve: jsp.linalg.solve,
}


def log_test_name(f):
    @wraps(f)
    def new(*args, **kwargs):
        print(f"\n{f.__name__}", file=sys.stderr)
        if args:
            print(f"args={args}", file=sys.stderr)
        if kwargs:
            print(f"kwargs={kwargs}", file=sys.stderr)
        return f(*args, **kwargs)

    return new


def parametrize_dtypes(func):
    return pytest.mark.parametrize("dtype", [np.float64, np.complex128])(func)


def parametrize_ops(func):
    return pytest.mark.parametrize("op_sparse", [klujax.solve, klujax.dot])(func)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_1d(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_col, n_col), dtype=Ax.dtype).at[Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_2d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    x_sp = jax.vmap(op_sparse, (None, None, 1, 1), 0)(Ai, Aj, Ax.T, b.T)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 2), 8, (n_col := 3), 4, dtype=dtype)
    x_sp = op_sparse(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacfwd(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacfwd(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacfwd(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacfwd(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_jacrev(dtype, op_sparse):
    op_dense = jax.vmap(OPS_DENSE[op_sparse], (0, 0), 0)
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    # jacobian on b
    jac_sp = jax.jacrev(op_sparse, 3, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 1, holomorphic=holomorphic)(A, b)
    _log_and_test_equality(jac_sp, jac)

    # jacobian on A
    jac_sp = jax.jacrev(op_sparse, 2, holomorphic=holomorphic)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    jac = jax.jacrev(op_dense, 0, holomorphic=holomorphic)(A, b)[..., Ai, Aj]
    _log_and_test_equality(jac_sp, jac)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_3d_vmap(dtype, op_sparse):
    op_dense = OPS_DENSE[op_sparse]
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 2, dtype=dtype)
    _log(Ai_shape=Ai.shape, Aj_shape=Aj.shape, Ax_shape=Ax.shape, b_shape=b.shape)
    x_sp = jax.vmap(op_sparse, (None, None, None, -1), -1)(Ai, Aj, Ax, b)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = jax.vmap(op_dense, (0, 0), 0)(A, b)
    _log_and_test_equality(x, x_sp)


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 15, 5, 2, dtype=dtype)
    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax, b[None])

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b)

    with pytest.raises(ValueError):  # noqa: PT011
        op_sparse(Ai, Aj, Ax[None], b[None])


@log_test_name
@parametrize_dtypes
@parametrize_ops
def test_4d_vmap(dtype, op_sparse):
    Ai, Aj, Ax, b = _get_rand_arrs_3d(3, 8, 5, 2, dtype=dtype)
    bb = np.stack([b, (b2 := np.random.RandomState(seed=42).rand(*b.shape))], axis=1)
    r = jax.vmap(op_sparse, (None, None, None, 1), 0)(Ai, Aj, Ax, bb)
    r1 = op_sparse(Ai, Aj, Ax, b)
    r2 = op_sparse(Ai, Aj, Ax, b2)
    _log_and_test_equality(r[0], r1)
    _log_and_test_equality(r[1], r2)


@log_test_name
def test_analyze():
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)

    # 1. Test Eager Analysis. A handle is now a SymbolToken carrying its id and
    # the pattern arrays it can rebuild from, not a raw pointer manager.
    symbolic = klujax.analyze(Ai, Aj, n_col)
    assert isinstance(symbolic, klujax.SymbolToken)
    assert symbolic.handle.dtype == jnp.uint64
    assert symbolic.n_col == n_col

    # Freeing is optional now, it just drops the cache slot.
    klujax.free_symbolic(symbolic)

    @jax.jit
    def jit_analyze_and_solve(Ai, Aj, Ax, b):
        # Created inside JIT. The handle is a traced value threaded as data. No
        # explicit free or dependency barrier is needed: a stale id self-heals.
        sym = klujax.analyze(Ai, Aj, 5)
        return klujax.solve_with_symbol(Ai, Aj, Ax, b, sym)

    x = jit_analyze_and_solve(Ai, Aj, Ax, b)
    assert x.shape == (n_col,)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)

    # Test JIT
    x_sp_jit = jax.jit(klujax.solve_with_symbol)(Ai, Aj, Ax, b, symbolic)
    _log_and_test_equality(x, x_sp_jit)

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.solve_with_numeric(numeric, b, symbolic)

    # Dense verification
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_vmap_1d_b(dtype):
    """vmap over solve_with_numeric where b is 1D (single-system, single-RHS)."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    # Create a batch of different Ax values (different matrix values, same sparsity)
    key = jax.random.PRNGKey(42)
    batch = 4
    Ax_batch = jax.random.normal(key, (batch, *Ax.shape), dtype=dtype) + 10.0

    def solve_one(Ax_i):
        num = klujax.factor(Ai, Aj, Ax_i, symbolic)
        result = klujax.solve_with_numeric(num, b, symbolic)
        klujax.free_numeric(num, dependency=result)
        return result

    x_sp = jax.vmap(solve_one)(Ax_batch)

    # Dense reference
    A_batch = jnp.zeros((batch, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax_batch)
    x = jax.vmap(lambda A: jsp.linalg.solve(A, b))(A_batch)
    _log_and_test_equality(x, x_sp)

    klujax.free_symbolic(symbolic)


def _get_rand_arrs_1d(n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(Axkey, (n_nz,), dtype=dtype)
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones(n_col, dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x])
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_col,), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_2d(n_lhs, n_nz, n_col, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_lhs, n_col), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_rand_arrs_3d(n_lhs, n_nz, n_col, n_rhs, *, dtype, seed=33):
    Axkey, Aikey, Ajkey, bkey = jax.random.split(jax.random.PRNGKey(seed), 4)
    Ai = jax.random.randint(Aikey, (n_nz,), 0, n_col, jnp.int32)
    Aj = jax.random.randint(Ajkey, (n_nz,), 0, n_col, jnp.int32)
    Ax = jax.random.normal(
        Axkey,
        (
            n_lhs,
            n_nz,
        ),
        dtype=dtype,
    )
    # Add diagonal to ensure matrix is invertible
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0
    Ai = jnp.concatenate([Ai, diag_i])
    Aj = jnp.concatenate([Aj, diag_i])
    Ax = jnp.concatenate([Ax, diag_x], axis=1)
    Ai, Aj, Ax = coalesce(Ai, Aj, Ax)
    b = jax.random.normal(bkey, (n_lhs, n_col, n_rhs), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_singular_arrs_1d(n_col, *, dtype, seed=33):
    """A diagonal matrix (10.0 on the diagonal) with the first diagonal entry
    zeroed out, making it exactly singular."""
    bkey = jax.random.PRNGKey(seed)
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = (jnp.ones(n_col, dtype=dtype) * 10.0).at[0].set(0.0)
    Ai, Aj, Ax = coalesce(diag_i, diag_i, diag_x)
    b = jax.random.normal(bkey, (n_col,), dtype=dtype)
    return Ai, Aj, Ax, b


def _get_partially_singular_arrs_2d(n_lhs, n_col, *, dtype, singular_lhs, seed=33):
    """A batched diagonal matrix (10.0 on the diagonal) where only
    `singular_lhs`'s first diagonal entry is zeroed out, so exactly that one
    batch element is singular while the others stay well-posed."""
    bkey = jax.random.PRNGKey(seed)
    diag_i = jnp.arange(n_col, dtype=jnp.int32)
    diag_x = (jnp.ones((n_lhs, n_col), dtype=dtype) * 10.0).at[singular_lhs, 0].set(0.0)
    Ai, Aj, Ax = coalesce(diag_i, diag_i, diag_x)
    b = jax.random.normal(bkey, (n_lhs, n_col), dtype=dtype)
    return Ai, Aj, Ax, b


def _log_and_test_equality(x, x_sp):
    print(f"\nx=\n{x}")
    print(f"\nx_sp=\n{x_sp}")
    print(f"\ndiff=\n{np.round(x_sp - x, 9)}")
    print(f"\nis_equal=\n{_is_almost_equal(x, x_sp)}")
    np.testing.assert_array_almost_equal(x, x_sp)


def _log(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}={v}")


def _is_almost_equal(arr1, arr2):
    try:
        np.testing.assert_array_almost_equal(arr1, arr2)
    except AssertionError:
        return False
    else:
        return True


def _make_ax2(Ai, Aj, Ax, *, dtype, seed=42):
    """New Ax values with the same sparsity pattern as Ax, with a strong diagonal."""
    Ax_raw = jax.random.normal(jax.random.PRNGKey(seed), Ax.shape, dtype=dtype)
    # Ensure diagonal entries are large (invertible) — Ai[k]==Aj[k] marks diagonal positions
    if Ax_raw.ndim == 1:
        Ax_raw = Ax_raw + jnp.where(
            Ai == Aj, jnp.full_like(Ax_raw, 10.0), jnp.zeros_like(Ax_raw)
        )
    else:  # (n_lhs, n_nz) batch case
        diag_mask = (Ai == Aj)[None, :]
        Ax_raw = Ax_raw + jnp.where(
            diag_mask, jnp.full_like(Ax_raw, 10.0), jnp.zeros_like(Ax_raw)
        )
    return Ax_raw


@log_test_name
@parametrize_dtypes
def test_refactor(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)
    assert isinstance(num2, klujax.NumericToken)

    x_sp = klujax.solve_with_numeric(num2, b2, sym)
    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jsp.linalg.solve(A2, b2)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)
    assert isinstance(num2, klujax.NumericToken)

    x_sp = klujax.solve_with_numeric(num2, b2, sym)
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = op_dense(A2, b2)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_vmap(dtype):
    """vmap solve_with_symbol over n_rhs with shared sym used by refactor.

    Tests that vmap composes correctly with the symbolic analysis after refactor,
    and that refactor does not corrupt the symbolic object.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_3d((n_lhs := 3), 15, (n_col := 5), 4, dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)

    # vmap solve_with_symbol over n_rhs axis (same pattern as test_3d_vmap)
    x_sp = jax.vmap(klujax.solve_with_symbol, (None, None, None, -1, None), -1)(
        Ai, Aj, Ax2, b, sym
    )

    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = jax.vmap(jax.vmap(jsp.linalg.solve, (0, 0), 0), (None, -1), -1)(A2, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num2)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_pmap(dtype):
    """pmap solve_with_numeric across devices after refactor."""
    n_dev = jax.device_count()
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    B = jax.random.normal(jax.random.PRNGKey(77), (n_dev, n_col), dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)

    x_sp = jax.pmap(lambda b: klujax.solve_with_numeric(num2, b, sym))(B)

    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jax.vmap(lambda b: jsp.linalg.solve(A2, b))(B)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num2)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_grad(dtype):
    """AD through solve_with_symbol is unaffected by refactor on the shared symbolic analysis."""
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), (n_col,), dtype=dtype)
    holomorphic = dtype in COMPLEX_DTYPES

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    num2 = klujax.refactor(Ai, Aj, Ax2, num, sym)

    # refactor value and solve_with_symbol value must agree
    x_num = klujax.solve_with_numeric(num2, b2, sym)
    x_sym = klujax.solve_with_symbol(Ai, Aj, Ax2, b2, sym)
    _log_and_test_equality(x_num, x_sym)

    # jacfwd through solve_with_symbol w.r.t. Ax: tests that sym is not corrupted by refactor
    jac_sp = jax.jacfwd(
        lambda ax: klujax.solve_with_symbol(Ai, Aj, ax, b2, sym),
        holomorphic=holomorphic,
    )(Ax2)
    jac_dense = jax.jacfwd(
        lambda ax: jnp.linalg.solve(
            jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(ax), b2
        ),
        holomorphic=holomorphic,
    )(Ax2)
    _log_and_test_equality(jac_sp, jac_dense)

    klujax.free_numeric(num2)
    klujax.free_symbolic(sym)


def test_solve_with_symbol_jvp():
    Ai = jnp.array([0, 1], dtype=jnp.int32)
    Aj = jnp.array([0, 1], dtype=jnp.int32)
    Ax = jnp.array([2.0, 4.0], dtype=jnp.float64)  # Values
    b = jnp.array([10.0, 20.0], dtype=jnp.float64)

    # Pre-compute symbolic factorization
    n_col = 2
    symbolic = klujax.analyze(Ai, Aj, n_col)

    def solve_step(Ax_vals, b_vec):
        return klujax.solve_with_symbol(Ai, Aj, Ax_vals, b_vec, symbolic)

    # Tangents (perturbations)
    tangent_Ax = jnp.array([0.1, 0.1], dtype=jnp.float64)
    tangent_b = jnp.array([0.0, 0.0], dtype=jnp.float64)

    # This triggers the JVP rule lookup
    primals, tangents = jax.jvp(solve_step, (Ax, b), (tangent_Ax, tangent_b))
    assert primals.shape == (2,)
    assert tangents.shape == (2,)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_symbol(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # For tsolve, we compare against the transposed dense matrix A.T
    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A.T, b)
    _log_and_test_equality(x, x_sp)

    # Test JIT
    x_sp_jit = jax.jit(klujax.tsolve_with_symbol)(Ai, Aj, Ax, b, symbolic)
    _log_and_test_equality(x, x_sp_jit)

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_symbol_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)

    # Dense verification for batched transpose
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    A_T = jnp.swapaxes(A, -1, -2)
    x = op_dense(A_T, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_numeric(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.tsolve_with_numeric(numeric, b, symbolic)

    A = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)
    x = jsp.linalg.solve(A.T, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_numeric_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)
    numeric = klujax.factor(Ai, Aj, Ax, symbolic)

    x_sp = klujax.tsolve_with_numeric(numeric, b, symbolic)

    # Dense verification for batched transpose
    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    A_T = jnp.swapaxes(A, -1, -2)
    x = op_dense(A_T, b)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(numeric)
    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_refactor_and_solve(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # Verifying specific API order: Ai, Aj, Ax, b, numeric, symbolic
    x_sp, num2 = klujax.refactor_and_solve(Ai, Aj, Ax2, b2, num, sym)

    assert isinstance(num2, klujax.NumericToken)

    A2 = jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax2)
    x = jsp.linalg.solve(A2, b2)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_and_solve_batched(dtype):
    Ai, Aj, Ax, b = _get_rand_arrs_2d((n_lhs := 3), 15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(43), b.shape, dtype=dtype)

    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # Verifying specific API order: Ai, Aj, Ax, b, numeric, symbolic
    x_sp, num2 = klujax.refactor_and_solve(Ai, Aj, Ax2, b2, num, sym)

    assert isinstance(num2, klujax.NumericToken)

    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A2 = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax2)
    x = op_dense(A2, b2)
    _log_and_test_equality(x, x_sp)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


# Singular-matrix handling: `solve`, `solve_with_symbol`, `tsolve_with_symbol`, and
# `refactor_and_solve` must return NaN for a singular system instead of raising, matching
# how LAPACK-backed solvers behave. `solve_with_numeric`/`tsolve_with_numeric` are not
# covered here: they operate on an already-factored numeric handle, and `factor`/`analyze`
# (which would themselves fail first on a singular matrix) are unchanged, so there's no way
# to reach those two with a singular matrix through the public API.


@log_test_name
@parametrize_dtypes
def test_solve_singular(dtype):
    Ai, Aj, Ax, b = _get_singular_arrs_1d((n_col := 5), dtype=dtype)
    x_sp = klujax.solve(Ai, Aj, Ax, b)
    assert x_sp.shape == (n_col,)
    assert jnp.all(jnp.isnan(x_sp))


@log_test_name
@parametrize_dtypes
def test_solve_with_symbol_singular(dtype):
    Ai, Aj, Ax, b = _get_singular_arrs_1d((n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.solve_with_symbol(Ai, Aj, Ax, b, symbolic)
    assert jnp.all(jnp.isnan(x_sp))

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_symbol_singular(dtype):
    Ai, Aj, Ax, b = _get_singular_arrs_1d((n_col := 5), dtype=dtype)
    symbolic = klujax.analyze(Ai, Aj, n_col)

    x_sp = klujax.tsolve_with_symbol(Ai, Aj, Ax, b, symbolic)
    assert jnp.all(jnp.isnan(x_sp))

    klujax.free_symbolic(symbolic)


@log_test_name
@parametrize_dtypes
def test_refactor_and_solve_singular(dtype):
    # `factor` runs on a well-posed matrix (refactor_and_solve needs an existing, valid
    # numeric handle to refactor); the *new* Ax values passed to refactor_and_solve are
    # what's singular.
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # Zero out every entry in row 0, guaranteeing an all-zero row (singular) regardless
    # of the rest of the (otherwise random) matrix.
    row0_mask = Ai == 0
    Ax2 = jnp.where(row0_mask, jnp.zeros_like(Ax), Ax)

    x_sp, _num2 = klujax.refactor_and_solve(Ai, Aj, Ax2, b, num, sym)
    assert jnp.all(jnp.isnan(x_sp))

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_solve_batched_partially_singular(dtype):
    """Only one batch element is singular: its slice must be NaN, but the other
    (well-posed) batch elements must still solve correctly. This is the key behavior
    change from the old code, which aborted the *entire* batch on one singular element."""
    n_lhs, n_col = 3, 5
    singular_lhs = 1
    Ai, Aj, Ax, b = _get_partially_singular_arrs_2d(
        n_lhs, n_col, dtype=dtype, singular_lhs=singular_lhs
    )

    x_sp = klujax.solve(Ai, Aj, Ax, b)
    assert jnp.all(jnp.isnan(x_sp[singular_lhs]))

    op_dense = jax.vmap(jsp.linalg.solve, (0, 0), 0)
    A = jnp.zeros((n_lhs, n_col, n_col), dtype=dtype).at[:, Ai, Aj].add(Ax)
    x = op_dense(A, b)
    well_posed = jnp.array([i for i in range(n_lhs) if i != singular_lhs])
    assert not jnp.any(jnp.isnan(x_sp[well_posed]))
    _log_and_test_equality(x[well_posed], x_sp[well_posed])


# Refactor status and conditioning testing


def parametrize_modes(func):
    """Run a test both eagerly and under jax.jit.

    Branching on a refactor failure is only useful if it survives tracing, so every new
    entry point is checked in both modes.
    """
    return pytest.mark.parametrize("mode", ["eager", "jit"])(func)


def _run(mode, f, *args):
    return f(*args) if mode == "eager" else jax.jit(f)(*args)


def _get_zero_pivot_arrs(*, dtype, n_lhs=None):
    """A 2x2 dense matrix plus value sets that reuse its pivot order.

    Returns (Ai, Aj, Ax_ok, Ax_singular, Ax_near_singular, b). Ax_singular has two equal
    rows, so under the pivot order of Ax_ok the second pivot is exactly zero. The block
    has to be dense: KLU only checks for a zero pivot inside BTF blocks larger than 1x1,
    so a diagonal matrix would refactor without complaint.

    With n_lhs set, the value arrays are batched and only the last left-hand side is
    singular (respectively near-singular).
    """
    Ai = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    Aj = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
    ok = jnp.array([1.0, 1.0, 1.0, 2.0], dtype=dtype)
    singular = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=dtype)
    near = jnp.array([1.0, 1.0, 1.0, 1.0 + 1e-14], dtype=dtype)
    b = jnp.array([3.0, 1.0], dtype=dtype)

    if n_lhs is not None:
        stack = lambda last: jnp.stack([ok] * (n_lhs - 1) + [last])  # noqa: E731
        ok, singular, near = stack(ok), stack(singular), stack(near)
        b = jnp.broadcast_to(b, (n_lhs, b.shape[0]))

    return Ai, Aj, ok, singular, near, b


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_refactor_with_status_ok(dtype, mode):
    Ai, Aj, Ax, _, _, b = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    def f(ax):
        _, status = klujax.refactor_with_status(Ai, Aj, ax, num, sym)
        return status

    status = _run(mode, f, Ax)
    assert status.tolist() == [klujax.KLUStatus.OK]

    x_sp = klujax.solve_with_numeric(num, b, sym)
    A = jnp.zeros((2, 2), dtype=dtype).at[Ai, Aj].add(Ax)
    _log_and_test_equality(jsp.linalg.solve(A, b), x_sp)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_refactor_with_status_singular(dtype, mode):
    """A zero pivot must be reported, not raised. This is the whole point: under jit an
    error aborts the computation and cannot be branched on."""
    Ai, Aj, Ax, Ax_singular, _, _ = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    def f(ax):
        _, status = klujax.refactor_with_status(Ai, Aj, ax, num, sym)
        return status

    status = _run(mode, f, Ax_singular)
    assert status.tolist() == [klujax.KLUStatus.SINGULAR]

    # the plain refactor does raise on the very same input
    with pytest.raises(Exception, match="refactor"):
        klujax.refactor(Ai, Aj, Ax_singular, num, sym)

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_refactor_with_status_branch_under_jit(dtype):
    """The status must be usable as a traced value, so a caller can fall back to a fresh
    factorization from inside a jitted function."""
    Ai, Aj, Ax, Ax_singular, _, b = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    @jax.jit
    def f(ax):
        num2, status = klujax.refactor_with_status(Ai, Aj, ax, num, sym)
        x = klujax.solve_with_numeric(num2, b, sym)
        # on failure, report a zero solution instead of the unusable factorization
        return lax.cond(
            status[0] == klujax.KLUStatus.OK,
            lambda: x,
            lambda: jnp.zeros_like(x),
        )

    assert jnp.all(f(Ax_singular) == 0)
    A = jnp.zeros((2, 2), dtype=dtype).at[Ai, Aj].add(Ax)
    _log_and_test_equality(jsp.linalg.solve(A, b), f(Ax))

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_factor_after_singular_status(dtype):
    """Documented contract: after a failed refactor the symbolic object is unaffected and
    the numeric object is still safe to free."""
    Ai, Aj, Ax, Ax_singular, _, b = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    _, status = klujax.refactor_with_status(Ai, Aj, Ax_singular, num, sym)
    assert status.tolist() == [klujax.KLUStatus.SINGULAR]

    klujax.free_numeric(num)

    num2 = klujax.factor(Ai, Aj, Ax, sym)
    x_sp = klujax.solve_with_numeric(num2, b, sym)
    A = jnp.zeros((2, 2), dtype=dtype).at[Ai, Aj].add(Ax)
    _log_and_test_equality(jsp.linalg.solve(A, b), x_sp)

    klujax.free_numeric(num2)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_refactor_with_status_batched(dtype, mode):
    n_lhs = 3
    Ai, Aj, Ax, Ax_singular, _, _ = _get_zero_pivot_arrs(dtype=dtype, n_lhs=n_lhs)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    def f(ax):
        _, status = klujax.refactor_with_status(Ai, Aj, ax, num, sym)
        return status

    # only the last left-hand side is singular, the rest still refactor fine
    status = _run(mode, f, Ax_singular)
    assert status.tolist() == [klujax.KLUStatus.OK] * (n_lhs - 1) + [
        klujax.KLUStatus.SINGULAR
    ]

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_refactor_with_status_vmap(dtype, mode):
    n_lhs, batch = 2, 3
    Ai, Aj, Ax, Ax_singular, _, _ = _get_zero_pivot_arrs(dtype=dtype, n_lhs=n_lhs)
    sym = klujax.analyze(Ai, Aj, 2)
    num = klujax.factor(Ai, Aj, Ax, sym)

    # one batch element gets the singular values, the others the benign ones
    Ax_batched = jnp.stack([Ax] * (batch - 1) + [Ax_singular])

    def f(ax):
        _, status = klujax.refactor_with_status(Ai, Aj, ax, num, sym)
        return status

    status = _run(mode, jax.vmap(f), Ax_batched)
    assert status.shape == (batch, n_lhs)
    assert jnp.all(status[: batch - 1] == klujax.KLUStatus.OK)
    assert status[batch - 1, n_lhs - 1] == klujax.KLUStatus.SINGULAR

    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_refactor_and_solve_with_status(dtype, mode):
    Ai, Aj, Ax, Ax_singular, _, b = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)

    # The numeric object lives entirely inside f, so the fused call gets a fresh one on
    # every case and (under jit) has to work with a traced handle.
    def f(ax):
        num = klujax.factor(Ai, Aj, Ax, sym)
        x, _, status = klujax.refactor_and_solve_with_status(Ai, Aj, ax, b, num, sym)
        num = num.track(x)
        klujax.free_numeric(num)
        return x, status

    # a failed element reports the status and still gets a NaN solution
    x_sp, status = _run(mode, f, Ax_singular)
    assert status.tolist() == [klujax.KLUStatus.SINGULAR]
    assert jnp.all(jnp.isnan(x_sp))

    x_sp, status = _run(mode, f, Ax)
    assert status.tolist() == [klujax.KLUStatus.OK]
    A = jnp.zeros((2, 2), dtype=dtype).at[Ai, Aj].add(Ax)
    _log_and_test_equality(jsp.linalg.solve(A, b), x_sp)

    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_rcond(dtype, mode):
    Ai, Aj, Ax, _, Ax_near, _ = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)

    def f(ax):
        num = klujax.factor(Ai, Aj, ax, sym)
        value = klujax.rcond(sym, num, dtype=dtype)
        num = num.track(value)
        klujax.free_numeric(num)
        return value

    # well conditioned: min|Uii| / max|Uii| = 1/2 here
    assert _run(mode, f, Ax).tolist() == pytest.approx([0.5])

    # reused-pivot degradation shows up directly, without a probe solve
    assert 0.0 < _run(mode, f, Ax_near)[0] < 1e-13

    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_condest(dtype, mode):
    Ai, Aj, Ax, _, Ax_near, _ = _get_zero_pivot_arrs(dtype=dtype)
    sym = klujax.analyze(Ai, Aj, 2)

    def f(ax):
        num = klujax.factor(Ai, Aj, ax, sym)
        value = klujax.condest(Ai, Aj, ax, sym, num)
        num = num.track(value)
        klujax.free_numeric(num)
        return value

    assert _run(mode, f, Ax)[0] == pytest.approx(9.0)
    assert _run(mode, f, Ax_near)[0] > 1e13

    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
@parametrize_modes
def test_rcond_and_condest_batched(dtype, mode):
    n_lhs = 3
    Ai, Aj, _, _, Ax_near, _ = _get_zero_pivot_arrs(dtype=dtype, n_lhs=n_lhs)
    sym = klujax.analyze(Ai, Aj, 2)

    def f(ax):
        num = klujax.factor(Ai, Aj, ax, sym)
        values = klujax.rcond(sym, num, dtype=dtype)
        conds = klujax.condest(Ai, Aj, ax, sym, num)
        num = num.track(conds)
        klujax.free_numeric(num)
        return values, conds

    # only the last left-hand side is near-singular
    values, conds = _run(mode, f, Ax_near)
    assert values.shape == (n_lhs,)
    assert jnp.all(values[: n_lhs - 1] > 0.1)
    assert values[n_lhs - 1] < 1e-13
    assert conds[n_lhs - 1] > 1e13

    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_rcond_vmap(dtype):
    n_lhs, batch = 2, 3
    Ai, Aj, Ax, _, Ax_near, _ = _get_zero_pivot_arrs(dtype=dtype, n_lhs=n_lhs)
    sym = klujax.analyze(Ai, Aj, 2)
    # one batch element carries the near-singular values, the others the benign ones
    Ax_batched = jnp.stack([Ax] * (batch - 1) + [Ax_near])

    # factor returns a NumericToken. Vmapping it gives a token with batched
    # leaves, which rcond reads back through directly.
    nums = jax.vmap(lambda ax: klujax.factor(Ai, Aj, ax, sym))(Ax_batched)
    values = jax.vmap(lambda num: klujax.rcond(sym, num, dtype=dtype))(nums)

    assert values.shape == (batch, n_lhs)
    assert jnp.all(values[: batch - 1] > 0.1)
    assert values[batch - 1, n_lhs - 1] < 1e-13

    jax.vmap(klujax.free_numeric)(nums)
    klujax.free_symbolic(sym)


# Handle token testing


def test_symbol_token_pytree_roundtrip():
    """A SymbolToken flattens to its arrays and rebuilds unchanged.

    The token must be a pytree whose leaves are the id and pattern arrays and
    whose aux is the static n_col, so it can cross jit and vmap boundaries.
    """
    Ai, Aj, _, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    leaves, treedef = jax.tree_util.tree_flatten(sym)
    rebuilt = treedef.unflatten(leaves)
    assert len(leaves) == 4  # id, Ai, Aj, n_dependent_solutions
    assert int(rebuilt.handle) == int(sym.handle)
    assert rebuilt.n_col == n_col
    assert int(rebuilt.n_dependent_solutions) == 0
    klujax.free_symbolic(sym)


def test_numeric_token_pytree_roundtrip():
    """A NumericToken carries id plus Ai/Aj/Ax and rebuilds unchanged."""
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    leaves, treedef = jax.tree_util.tree_flatten(num)
    rebuilt = treedef.unflatten(leaves)
    assert len(leaves) == 6  # id, Ai, Aj, Ax, n_dependent_solutions, version
    assert rebuilt.n_col == n_col
    assert int(rebuilt.n_dependent_solutions) == 0
    assert int(rebuilt.version[0]) == int(num.version[0])
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


def test_token_flows_through_jit():
    """A token created outside jit is usable inside it, threaded as data.

    Unlike the old pointer manager, the token's id is an ordinary array, so it
    traces into a jitted function and the solve there works normally.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    @jax.jit
    def solve(num, b):
        return klujax.solve_with_numeric(num, b, sym)

    x = solve(num, b)
    A = jnp.zeros((n_col, n_col), dtype=Ax.dtype).at[Ai, Aj].add(Ax)
    _log_and_test_equality(jsp.linalg.solve(A, b), x)
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


# Handle cache: bounded memory, rebuild-on-miss, strict mode ==================
# These pin the memory-safety contract: a handle is a key into a bounded cache,
# never a raw pointer, so forgetting to free leaks only the cache, and a handle
# whose factorization was evicted or freed rebuilds from the arrays it carries
# instead of reading freed memory.


def _dense(Ai, Aj, Ax, n_col, dtype):
    return jnp.zeros((n_col, n_col), dtype=dtype).at[Ai, Aj].add(Ax)


@log_test_name
def test_forgotten_handles_are_bounded_and_rebuild(monkeypatch):
    """Never freeing handles stays bounded, and an evicted one still solves.

    With the cache capped at two, factoring many systems without freeing cannot
    grow the registry without end. Solving through the first handle then rebuilds
    its factorization, which we check stays correct and bumps the rebuild count.
    """
    monkeypatch.setenv("KLUJAX_FACTOR_CACHE", "2")
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)

    first = klujax.factor(Ai, Aj, Ax, sym)
    for _ in range(5):
        klujax.factor(Ai, Aj, Ax, sym)  # evicts older entries, first included

    klujax.reset_rebuild_count()
    x = klujax.solve_with_numeric(first, b, sym)
    assert klujax.rebuild_count() >= 1
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )


@log_test_name
def test_strict_mode_turns_a_rebuild_into_an_error(monkeypatch):
    """Strict mode raises instead of silently rebuilding an evicted handle.

    A silent rebuild is correct but slow, so strict mode is the switch that makes
    a lost factorization loud while debugging. We force an eviction with a tiny
    cache, then check the next solve raises and names strict mode.
    """
    # Capacity 2 holds the symbolic plus one numeric. A second numeric then
    # evicts the first, which strict mode refuses to rebuild.
    monkeypatch.setenv("KLUJAX_FACTOR_CACHE", "2")
    monkeypatch.setenv("KLUJAX_STRICT_CACHE", "1")
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    first = klujax.factor(Ai, Aj, Ax, sym)
    klujax.factor(Ai, Aj, Ax, sym)  # evicts first

    with pytest.raises(Exception, match="strict cache mode"):
        klujax.solve_with_numeric(first, b, sym)


@log_test_name
@parametrize_dtypes
def test_released_handle_rebuilds_on_next_solve(dtype):
    """An explicitly freed handle self-heals when solved through again.

    free_numeric only drops the cache slot; the token stays valid and rebuilds
    from the matrix the solve carries. This is the freeing-is-optional guarantee.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    klujax.free_numeric(num)

    klujax.reset_rebuild_count()
    x = klujax.solve_with_numeric(num, b, sym)
    assert klujax.rebuild_count() >= 1
    _log_and_test_equality(jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, dtype), b), x)


@log_test_name
def test_full_lifecycle_inside_jit():
    """analyze, factor, solve, and free run correctly inside one jit.

    The handle is threaded as data, so XLA orders the lifecycle by data
    dependency. We check the jitted answer matches a dense solve, which is what
    lets the cache scheme be used from inside compiled code.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)

    @jax.jit
    def run(Ai, Aj, Ax, b):
        sym = klujax.analyze(Ai, Aj, n_col)
        num = klujax.factor(Ai, Aj, Ax, sym)
        x = klujax.solve_with_numeric(num, b, sym)
        klujax.free_numeric(num)
        klujax.free_symbolic(sym)
        return x

    x = run(Ai, Aj, Ax, b)
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )


@log_test_name
def test_eager_factor_then_jitted_solve_self_heals(monkeypatch):
    """A handle built eagerly, then evicted, still solves inside a later jit.

    This is the mixed eager/jit case: the factorization is built eagerly, the
    cache is overrun so the handle is evicted, and a jitted solve reaches the
    missing handle and rebuilds. We check the answer is still correct.
    """
    monkeypatch.setenv("KLUJAX_FACTOR_CACHE", "1")
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    klujax.factor(Ai, Aj, Ax, sym)  # evicts num (capacity 1)

    @jax.jit
    def solve(b):
        return klujax.solve_with_numeric(num, b, sym)

    klujax.reset_rebuild_count()
    x = solve(b)
    assert klujax.rebuild_count() >= 1
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )


# Ordering an explicit free after solves ======================================
# A bare free inside a jit trace is unordered against a solve on the same token,
# so it may run first and be undone by the solve's rebuild. track and the
# dependency argument make the free wait. These check the free costs no rebuild,
# which is the observable sign it ran after the solve.


@log_test_name
def test_track_orders_a_free_after_the_solve():
    """num.track makes an in-trace free wait for the solve, so no rebuild."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    @jax.jit
    def step(num, b):
        x = klujax.solve_with_numeric(num, b, sym)
        klujax.free_numeric(num.track(x))
        return x

    klujax.reset_rebuild_count()
    x = step(num, b)
    assert klujax.rebuild_count() == 0
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )
    klujax.free_symbolic(sym)


@log_test_name
def test_free_dependency_orders_after_the_solve():
    """free_numeric(num, dependency=x) orders the free after that solve."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    @jax.jit
    def step(num, b):
        x = klujax.solve_with_numeric(num, b, sym)
        klujax.free_numeric(num, dependency=x)
        return x

    klujax.reset_rebuild_count()
    x = step(num, b)
    assert klujax.rebuild_count() == 0
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )
    klujax.free_symbolic(sym)


@log_test_name
def test_track_counts_solutions():
    """n_dependent_solutions counts the solutions tracked into a token."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    assert int(num.n_dependent_solutions) == 0
    num = num.track(b).track(b).track(b)
    assert int(num.n_dependent_solutions) == 3
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
def test_free_without_ordering_still_frees_and_stays_correct():
    """A free with neither track nor dependency still frees and self-heals.

    Ordering only affects whether an early free wastes a rebuild, never
    correctness, so this checks a bare free eagerly frees and a later solve on
    the same token still gives the right answer.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    klujax.free_numeric(num)
    x = klujax.solve_with_numeric(num, b, sym)
    _log_and_test_equality(
        jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, np.float64), b), x
    )
    klujax.free_symbolic(sym)


@log_test_name
def test_many_solves_in_a_scan_then_free():
    """A token threads through lax.scan, and a free after it costs no rebuild.

    Each step solves and tracks its solution, so after the loop the free is
    ordered behind every solve. We check no rebuild, the counter equals the step
    count, and each solution matches a dense solve. This also confirms the token
    is a valid scan carry, since its shape stays fixed across steps.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    bs = jnp.stack([b * scale for scale in (1.0, 2.0, 3.0)])

    @jax.jit
    def run(num, bs):
        def step(num, b):
            x = klujax.solve_with_numeric(num, b, sym)
            return num.track(x), x

        num, xs = lax.scan(step, num, bs)
        klujax.free_numeric(num)
        return num.n_dependent_solutions, xs

    klujax.reset_rebuild_count()
    count, xs = run(num, bs)
    assert klujax.rebuild_count() == 0
    assert int(count) == bs.shape[0]
    A = _dense(Ai, Aj, Ax, n_col, np.float64)
    _log_and_test_equality(jax.vmap(lambda bb: jsp.linalg.solve(A, bb))(bs), xs)
    klujax.free_symbolic(sym)


@log_test_name
def test_many_solves_in_a_vmap_then_free():
    """A vmapped batch of solves tracked and freed costs no rebuild.

    The batch of solves runs in one vmapped call. Tracking the batched solution
    orders the free after it, so we check no rebuild and a correct batched answer.
    """
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    bs = jnp.stack([b * scale for scale in (1.0, 2.0, 3.0)])

    @jax.jit
    def run(num, bs):
        xs = jax.vmap(lambda bb: klujax.solve_with_numeric(num, bb, sym))(bs)
        klujax.free_numeric(num.track(xs))
        return xs

    klujax.reset_rebuild_count()
    xs = run(num, bs)
    assert klujax.rebuild_count() == 0
    A = _dense(Ai, Aj, Ax, n_col, np.float64)
    _log_and_test_equality(jax.vmap(lambda bb: jsp.linalg.solve(A, bb))(bs), xs)
    klujax.free_symbolic(sym)


# Ordering tokens from reads =========================================================
#
# A read (solve, tsolve, rcond, condest) can hand back a NumericToken so a later
# in-place refactor waits on the read. These tests cover the returned value, the
# ordering under jit, the version staleness check, and grad through a threaded read.


@log_test_name
@parametrize_dtypes
def test_solve_with_numeric_return_token(dtype):
    """return_token gives back the solution and a token with the same id and version."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    x, tok = klujax.solve_with_numeric(num, b, sym, return_token=True)

    x_ref = jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, dtype), b)
    _log_and_test_equality(x_ref, x)
    assert int(tok.id[0]) == int(num.id[0])
    assert int(tok.version[0]) == int(num.version[0])

    x_plain = klujax.solve_with_numeric(tok, b, sym)
    _log_and_test_equality(x_ref, x_plain)
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_tsolve_with_numeric_return_token(dtype):
    """return_token on the transpose solve returns the solution and a usable token."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    x, tok = klujax.tsolve_with_numeric(num, b, sym, return_token=True)

    x_ref = jsp.linalg.solve(_dense(Ai, Aj, Ax, n_col, dtype).T, b)
    _log_and_test_equality(x_ref, x)
    assert int(tok.id[0]) == int(num.id[0])
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
def test_rcond_condest_return_token():
    """rcond and condest hand back a token alongside their estimate."""
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    r, tok_r = klujax.rcond(sym, num, return_token=True)
    r_plain = klujax.rcond(sym, num)
    _log_and_test_equality(np.asarray(r_plain), np.asarray(r))
    assert int(tok_r.id[0]) == int(num.id[0])

    c, tok_c = klujax.condest(Ai, Aj, Ax, sym, num, return_token=True)
    c_plain = klujax.condest(Ai, Aj, Ax, sym, num)
    _log_and_test_equality(np.asarray(c_plain), np.asarray(c))
    assert int(tok_c.id[0]) == int(num.id[0])
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
def test_condest_return_token_needs_token():
    """A raw handle has no arrays to build a token from, so return_token rejects it."""
    Ai, Aj, Ax, _ = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    with pytest.raises(TypeError):
        klujax.condest(Ai, Aj, Ax, sym, num.handle, return_token=True)
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
@parametrize_dtypes
def test_threaded_solve_orders_refactor(dtype):
    """Threading the read token keeps an earlier solve correct across an in-place refactor.

    factor, solve, then refactor the same slot, then solve again, all in one jit.
    The first solve reads the first matrix and the second reads the refactored one.
    Threading the token puts a data edge from the first solve to the refactor.
    """
    Ai, Aj, Ax1, b1 = _get_rand_arrs_1d(15, (n_col := 5), dtype=dtype)
    Ax2 = _make_ax2(Ai, Aj, Ax1, dtype=dtype)
    b2 = jax.random.normal(jax.random.PRNGKey(7), (n_col,), dtype=dtype)
    sym = klujax.analyze(Ai, Aj, n_col)

    @jax.jit
    def run(Ax1, Ax2, b1, b2):
        n0 = klujax.factor(Ai, Aj, Ax1, sym)
        x1, n1 = klujax.solve_with_numeric(n0, b1, sym, return_token=True)
        n2 = klujax.refactor(Ai, Aj, Ax2, n1, sym)
        x2 = klujax.solve_with_numeric(n2, b2, sym)
        return x1, x2

    x1, x2 = run(Ax1, Ax2, b1, b2)
    x1_ref = jsp.linalg.solve(_dense(Ai, Aj, Ax1, n_col, dtype), b1)
    x2_ref = jsp.linalg.solve(_dense(Ai, Aj, Ax2, n_col, dtype), b2)
    _log_and_test_equality(x1_ref, x1)
    _log_and_test_equality(x2_ref, x2)
    klujax.free_symbolic(sym)


@log_test_name
def test_stale_token_reports_mismatch():
    """Reusing a token after a later refactor overwrote its slot is reported, not silent."""
    Ai, Aj, Ax1, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    Ax2 = _make_ax2(Ai, Aj, Ax1, dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)

    n0 = klujax.factor(Ai, Aj, Ax1, sym)
    _, n1 = klujax.solve_with_numeric(n0, b, sym, return_token=True)
    klujax.refactor(Ai, Aj, Ax2, n1, sym)  # slot moves to a new version in place

    with pytest.raises(Exception, match="stale"):
        jax.block_until_ready(klujax.solve_with_numeric(n1, b, sym, return_token=True))
    klujax.free_symbolic(sym)


@log_test_name
def test_return_token_grad_matches_plain():
    """grad through a solve is unchanged by asking for the token as well."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)

    def loss_plain(b):
        return jnp.sum(klujax.solve_with_numeric(num, b, sym) ** 2)

    def loss_token(b):
        x, _tok = klujax.solve_with_numeric(num, b, sym, return_token=True)
        return jnp.sum(x**2)

    g_plain = jax.grad(loss_plain)(b)
    g_token = jax.grad(loss_token)(b)
    _log_and_test_equality(g_plain, g_token)
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)


@log_test_name
def test_return_token_vmap():
    """vmap over a batch of right-hand sides works with return_token set."""
    Ai, Aj, Ax, b = _get_rand_arrs_1d(15, (n_col := 5), dtype=np.float64)
    sym = klujax.analyze(Ai, Aj, n_col)
    num = klujax.factor(Ai, Aj, Ax, sym)
    bs = jnp.stack([b * scale for scale in (1.0, 2.0, 3.0)])

    def solve_one(bb):
        x, _tok = klujax.solve_with_numeric(num, bb, sym, return_token=True)
        return x

    xs = jax.vmap(solve_one)(bs)
    A = _dense(Ai, Aj, Ax, n_col, np.float64)
    _log_and_test_equality(jax.vmap(lambda bb: jsp.linalg.solve(A, bb))(bs), xs)
    klujax.free_numeric(num)
    klujax.free_symbolic(sym)
