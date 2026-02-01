"""Integration tests for cache hook registration mechanics.

Integration still exercises the shared cache hook suites using the smallest
model (SmolVLM-256M) to keep runtime low.
"""

import pytest

from tests.helpers.cache_hook_suites import (
    CacheHookRegistrationSuite,
    CacheHookShapesSuite,
    CacheHookNonInvasivenessSuite,
    CacheHookCleanupSuite,
    collect_cache_metadata,
)


@pytest.fixture(scope="module")
def vlm_model(tiny_model):
    """Alias the integration tiny model to the shared suite fixture name."""
    return tiny_model


@pytest.fixture(scope="class")
def cache_all_hooks(vlm_model):
    """Collect cached activations once per test class."""
    return collect_cache_metadata(vlm_model)


class TestCacheHookRegistration(CacheHookRegistrationSuite):
    """Reuse shared registration tests for the integration fixture."""


class TestCacheHookShapes(CacheHookShapesSuite):
    """Reuse shared shape tests for the integration fixture."""


class TestCacheNonInvasiveness(CacheHookNonInvasivenessSuite):
    """Validate cache hooks do not affect outputs on the integration model."""


class TestCacheCleanup(CacheHookCleanupSuite):
    """Ensure cache hook cleanup works with the integration model."""
