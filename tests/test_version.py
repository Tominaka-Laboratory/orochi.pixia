"""Unit tests for __version__.py."""

import orochi.pdfsim


def test_package_version():
    """Ensure the package version is defined and not set to the initial
    placeholder."""
    assert hasattr(orochi.pdfsim, "__version__")
    assert orochi.pdfsim.__version__ != "0.0.0"
