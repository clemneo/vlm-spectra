"""Integration tests for the tuple-based patch hook API.

Integration runs the shared patch hook suites against the smallest model to
keep runtime low while validating mechanics end-to-end.
"""

import pytest

from tests.helpers.patch_hook_suites import (
    PatchTupleAPISuite,
    PatchValidHookPointsSuite,
    PatchValidateHookTypeSuite,
    PatchHelperClassesSuite,
    PatchHookCleanupSuite,
    PatchCacheInteractionSuite,
)


@pytest.fixture(scope="module")
def vlm_model(tiny_model):
    """Alias the integration tiny model to the shared suite fixture name."""
    return tiny_model


class TestTupleBasedAPI(PatchTupleAPISuite):
    """Reuse tuple-based API tests with the integration fixture."""


class TestValidHookPoints(PatchValidHookPointsSuite):
    """Reuse hook point validation tests with the integration fixture."""


class TestValidatePatchHookType(PatchValidateHookTypeSuite):
    """Test validate_patch_hook_type using shared suites."""


class TestPatchHelperClasses(PatchHelperClassesSuite):
    """Reuse helper class tests with the integration fixture."""


class TestHookCleanup(PatchHookCleanupSuite):
    """Ensure hook cleanup tests run against the integration fixture."""


class TestCacheAndHooksInteraction(PatchCacheInteractionSuite):
    """Exercise cache + patch interaction tests on the integration fixture."""
