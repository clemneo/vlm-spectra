"""Acceptance tests for cache hooks across all registered models."""

import pytest

from tests.helpers.cache_hook_suites import (
    CacheHookRegistrationSuite,
    CacheHookShapesSuite,
    CacheHookNonInvasivenessSuite,
    CacheHookCleanupSuite,
    collect_cache_metadata,
)


@pytest.fixture(scope="session")
def vlm_model(model):
    """Alias the acceptance model fixture to the shared suite name."""
    model.model.eval()
    return model


@pytest.fixture(scope="class")
def cache_all_hooks(vlm_model):
    """Collect cached activations once for each suite class."""
    return collect_cache_metadata(vlm_model)


class TestCacheHookRegistration(CacheHookRegistrationSuite):
    """Run cache hook registration tests on every acceptance model."""


class TestCacheHookShapes(CacheHookShapesSuite):
    """Run cache hook shape tests on every acceptance model."""


class TestCacheNonInvasiveness(CacheHookNonInvasivenessSuite):
    """Ensure cache hooks stay non-invasive for every acceptance model."""


class TestCacheCleanup(CacheHookCleanupSuite):
    """Ensure cache hooks clean up correctly on every acceptance model."""
