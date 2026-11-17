"""Shared fixtures for the unit tests."""
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

import pytest

from atomsci.ddm.pipeline import dist_metrics


@pytest.fixture
def thread_dist_pool(monkeypatch):
    """Run dist_metrics' "parallel" distance calculations on a thread pool.

    dist_metrics builds a fresh multiprocessing.Pool(cpu_count()) for every distance matrix. On
    platforms whose default start method is "spawn" (macOS, Windows) each of those workers re-imports
    the whole AMPL/DeepChem stack, which costs tens of seconds per call no matter how small the
    inputs are. SplitStats makes several such calls per instance (two per fold for k-fold splits), so
    the plotting tests were spending nearly all of their time starting worker processes. A thread pool
    gives identical results for these tiny inputs. The real process pool is exercised by the
    integrative tests.
    """
    monkeypatch.setattr(dist_metrics, 'multiprocessing', SimpleNamespace(Pool=ThreadPool))
