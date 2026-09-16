# Changelog

## Unreleased

- Handles are now **content-addressed**: a handle is a hash of the matrix it
  names (dtype + sparsity pattern + values), not a mutable cache slot. A stale
  alias of a handle can never be handed the wrong matrix — at worst it rebuilds.
  Identical matrices hash equal, so the cache now also deduplicates.
- `refactor` / `refactor_and_solve` re-key to the new values instead of
  overwriting the old handle in place (they still reuse pivots via klu_refactor
  when safe). The returned handle is a new content key.
- Fix a dtype-dispatch bug: `solve_with_numeric` / `tsolve_with_numeric` now pick
  the KLU entry point from the factorization's dtype, not just `b`'s, so a complex
  factorization solved with a real right-hand side no longer down-casts.
- New `RebuildReason` enum and per-call rebuild visibility. New
  `solve_with_numeric_with_status` / `tsolve_with_numeric_with_status` return a
  per-handle `RebuildReason` (NONE / EVICTED / FREED / SUPERSEDED / UNKNOWN),
  branchable under `jax.jit`. New `rebuild_stats()` returns the per-reason total.
- **Breaking:** `refactor_with_status` now returns `(numeric, status, rebuild)`
  and `refactor_and_solve_with_status` returns `(x, numeric, status, rebuild)`.

## 0.5.0.post6

- Mark factor as side-effecting so slots stay distinct

## 0.5.0.post5

- Set default n_dependent_solutions explicitly to fix pytree interop (e.g. `eqx.partition`)

## 0.5.0.post4

- Handles are now cache tokens instead of raw pointers
- `KLUHandleManager` replaced by `SymbolToken` and `NumericToken`, which are
  JAX pytrees carrying their rebuild arrays as leaves
- New `KLUJAX_FACTOR_CACHE` env var sets the max live KLU objects (default 8)
- New `KLUJAX_STRICT_CACHE` env var turns a rebuild into an error for debugging
- New `rebuild_count()` and `reset_rebuild_count()` to detect cache pressure
- New `token = token.track(solution)` API to order a free after a solve inside `jax.jit`

## 0.5.0.post3

- New `refactor_with_status` and `refactor_and_solve_with_status` report a failed
  refactorization through a `KLUStatus` code instead of raising, so callers can
  branch on it under `jax.jit` and fall back to a fresh `factor`
- New `KLUStatus` IntEnum mirroring the SuiteSparse `KLU_*` status codes
- New `rcond` and `condest` expose `klu_rcond` and `klu_condest`, giving a cheap
  conditioning check on an existing factorization

## 0.5.0.post2

- `solve`, `solve_with_symbol`, `tsolve_with_symbol`, `solve_with_numeric`,
  `tsolve_with_numeric`, and `refactor_and_solve` now return NaN for a
  singular matrix instead of raising, matching standard LAPACK-style
  behavior

## 0.4.0

- Upgrade to jax>=0.5.0
- Fix issues in vmap
- Fix issues in jacfwd/jacrev
- No more hidden segfaults (hopefully)
- More consistent array shape broadcasting
- Drop support for x86 MacOS (not supported by jax>=0.5.0 either)
- Update GitHub CI

## 0.3.1

- Bugfixes

## 0.3.0

- Implement new FFI API for C++ extension
- Enhance C++ testing and error handling (proper error throwing instead of segfaulting)
- Enable JIT and `vmap` for optimized performance
- Improve shape handling for arrays
- Retrieve array sizes from C++ buffers
- Refactor workflows and remove deprecated implementations
- Remove old notebook files
- Upgrade dependencies and ensure compatibility with C++17 standard

## 0.2.10

- Run tests post wheel build
- Pin exact dependency versions
- Streamline GitHub workflows and update `setup.py`
- Fix issues with `vmap` over array `b`

## 0.2.8

- Refine CI/CD environment variable configuration for `cibuildwheel`

## 0.2.7

- Prevent memory leaks
- Introduce pre-commit configuration
- Update `cibuildwheel` configuration
- Clone specific SuiteSparse version

## 0.2.5

- Address deprecations in XLA translations

## 0.2.4

- Add support for Python 3.12
- Consolidate GitHub workflow files and update package metadata

## 0.2.0

- Vendor SuiteSparse library in source distribution
- Re-enable `PIP_FIND_LINKS`
- Update build recipes and dependencies
- Improve README with setup and build instructions

## 0.1.4

- Add support for Python 3.11

## 0.1.3

- Enable installation on macOS
- Fix issues with static linking on macOS (C++11 requirement)

## 0.1.1

- Publish release on PyPI and include tarball for distribution
- Add support for multiple Python versions and manylinux2014 wheels

## 0.1.0

- Enable custom JVP/VJP rules
- Improve differentiation features with forward-mode JVP and transposition
- Add `pyproject.toml` for better build configuration

## 0.0.6

- Add more library/include paths for builds
- Refine README and setup instructions
- Initial integration of complex value handling in `vmap`

## 0.0.4

- Add `bump2version` configuration for automated versioning
- Bugfix: Correct matrix-vector multiplication in COO format (`mul_coo_vec`)

## 0.0.3

- Set up core functionality with sparse matrix multiplication (COO format)
- Integrate `vmap` for float64 and complex128 arrays
- Initial setup with Makefile, test suites, and Docker configuration
- Begin development of XLA translations and gradient support
