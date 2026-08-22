"""Pytest configuration — custom markers."""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that hit real external services (GitHub API, git clone)",
    )
