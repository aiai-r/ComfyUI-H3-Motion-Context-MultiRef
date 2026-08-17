"""Pytest entry point for the standalone repository regression suite."""

from run_tests import run_all


def test_repo_regressions():
    assert run_all() > 0
