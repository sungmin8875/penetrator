"""
_foundry_shim.py — Palantir `transforms.api` shim for the MLWB port.
================================================================================
The ported engine modules (`net_production_demand`, `monthly_fulfillment`,
`allocation_engine`) are *byte-identical* copies of the Palantir Foundry source.
Their original `from transforms.api import ...` lines were mechanically rewritten
to import from THIS module, so the unchanged Foundry logic runs outside Foundry.

This module has two jobs:

  1.  No-op stand-ins for the Foundry DECORATORS and type hints, so the modules
      import and define their functions normally (the decorators do nothing).

  2.  `InMemoryInput` / `InMemoryOutput` adapters so the orchestrator can CALL a
      module's `compute(...)` directly — passing in-memory polars DataFrames where
      Foundry passed dataset handles. Inputs expose `.polars()`; outputs capture
      whatever `compute()` writes via `.write_table()`.

Nothing here contains business logic. If you ever move this package back into
Foundry, delete this file and restore the original `transforms.api` imports.
================================================================================
"""
from __future__ import annotations

import polars as pl


# ==============================================================================
# 1.  Decorator stand-ins  (all no-ops — they just return the function)
# ==============================================================================

def _identity(fn):
    return fn


class _DecoratorFactory:
    """Returned by ``transform(...)`` and ``transform.using(...)``.

    Acts as a no-op decorator and also supports the chained ``.with_resources(...)``
    form used by the allocation engine:  ``@transform.using(...).with_resources(...)``.
    """

    def __call__(self, fn):
        return fn

    def with_resources(self, *args, **kwargs):
        return self  # still a no-op decorator


class _Transform:
    """Stand-in for ``transforms.api.transform``.

    Supports both ``@transform(...)`` and ``@transform.using(...)``.
    """

    def __call__(self, *args, **kwargs):
        return _DecoratorFactory()

    def using(self, *args, **kwargs):
        return _DecoratorFactory()


transform = _Transform()


def lightweight(*args, **kwargs):
    """Supports both bare ``@lightweight`` and parametrized ``@lightweight(cpu_cores=...)``."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return _identity


def incremental(*args, **kwargs):
    return _identity


def configure(*args, **kwargs):
    return _identity


# ==============================================================================
# 2.  Decorator-argument placeholders  (Input("rid")/Output("path") — ignored)
# ==============================================================================

class Input:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class Output:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


# ==============================================================================
# 3.  Type-hint aliases  (only ever referenced in function annotations)
# ==============================================================================

class LightweightContext:
    """Foundry execution context. Only ``abort_job`` is referenced; it's a no-op
    here because the single-run port always has work to do."""

    def abort_job(self):
        raise RuntimeError(
            "LightweightContext.abort_job() called — the single-run port should not "
            "reach Foundry's incremental early-exit. Call the engine wrapper instead "
            "of the module's compute()."
        )


# These are used purely as annotations on the original compute() signatures.
LightweightInput = object
LightweightOutput = object
IncrementalLightweightInput = object
IncrementalLightweightOutput = object
TransformInput = object
TransformOutput = object


# ==============================================================================
# 4.  Runtime adapters  —  let the orchestrator drive a module's compute()
#     with in-memory polars DataFrames instead of Foundry dataset handles.
# ==============================================================================

def _to_polars(df) -> pl.DataFrame:
    """Coerce whatever a compute() wrote into a polars DataFrame."""
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    try:
        import pandas as pd  # local import: pandas may be absent until pycelonis is wired
        if isinstance(df, pd.DataFrame):
            return pl.from_pandas(df)
    except Exception:
        pass
    return df


class InMemoryInput:
    """Stand-in for a Foundry input dataset handle, backed by an in-memory DataFrame.

    Reproduces the read calls the ported modules make:
        .polars()              -> the DataFrame
        .polars(lazy=True)     -> a LazyFrame
        .polars("previous")    -> empty frame with the same schema (single-run: no
                                   prior incremental snapshot exists)
        .pandas()              -> a pandas DataFrame
    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def polars(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode")
        if mode == "previous":
            return self._df.clear()  # empty, same schema — first-run semantics
        lazy = kwargs.get("lazy", False)
        return self._df.lazy() if lazy else self._df

    def pandas(self, *args, **kwargs):
        return self._df.to_pandas()


class InMemoryOutput:
    """Stand-in for a Foundry output dataset handle; captures whatever compute() writes.

    After calling a module's compute(), read the captured table from ``.result``.
    """

    def __init__(self):
        self.result: pl.DataFrame | None = None

    def write_table(self, df, *args, **kwargs):
        self.result = _to_polars(df)

    def write_pandas(self, df, *args, **kwargs):
        self.result = _to_polars(df)

    def write_dataframe(self, df, *args, **kwargs):
        self.result = _to_polars(df)

    def set_mode(self, *args, **kwargs):
        pass  # write-mode (replace/modify) is meaningless for an in-memory single run
