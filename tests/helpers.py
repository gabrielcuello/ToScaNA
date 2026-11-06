"""Shared helpers for the test suite."""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import tempfile
from typing import Any, Callable, Iterator

import _pkg  # noqa: F401

import numpy as np
from scipy import integrate


def quiet(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call `func` with stdout suppressed (legacy code prints a lot)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


@contextlib.contextmanager
def in_tempdir() -> Iterator[pathlib.Path]:
    """Run inside a fresh temporary `work/` directory.

    Legacy code writes to CWD-relative paths such as `../logfiles`, so the
    working directory is a subfolder of the temporary root.
    """
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp) / "work"
        work.mkdir()
        os.chdir(work)
        try:
            yield work
        finally:
            os.chdir(old)


def synthetic_liquid(rho: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S(Q) of a model liquid: g(r)=0 below 2.5 Å, one Gaussian shell at 2.8 Å.

    Returns (SQ as an N x 2 array, r, g).
    """
    r = np.arange(0, 30, 0.005)
    g = np.where(r < 2.5, 0.0, 1.0) + np.where(
        r >= 2.5, 0.8 * np.exp(-((r - 2.8) ** 2) / 0.02), 0.0
    )
    q = np.arange(0.0, 30.0, 0.01)
    s = np.ones_like(q)
    for i, qq in enumerate(q[1:], start=1):
        s[i] = 1 + 4 * np.pi * rho / qq * integrate.simpson(
            r * (g - 1) * np.sin(qq * r), x=r
        )
    s[0] = s[1]
    return np.column_stack([q, s]), r, g
