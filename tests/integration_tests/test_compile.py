"""Placeholder test used to ensure integration tests compile without running them.

This test is marked with the 'compile' marker and is used in CI to verify
that integration tests have valid syntax and imports without requiring API keys.
"""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Placeholder test for compilation check."""
    pass
