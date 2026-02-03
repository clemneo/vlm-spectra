"""Acceptance tests for tuple-based patch hooks across all models."""

import pytest

from tests.helpers.patch_hook_suites import (
    PatchTupleAPISuite,
    PatchValidHookPointsSuite,
    PatchValidateHookTypeSuite,
    PatchHelperClassesSuite,
    PatchHookCleanupSuite,
    PatchCacheInteractionSuite,
    PatchPreHookSuite,
    PatchPreHookCleanupSuite,
)


@pytest.fixture(scope="session")
def vlm_model(model):
    """Alias the acceptance model fixture to the shared suite name."""
    model.model.eval()
    return model


class TestTupleBasedAPI(PatchTupleAPISuite):
    """Run tuple-based patch hook API tests on every acceptance model."""


class TestValidHookPoints(PatchValidHookPointsSuite):
    """Ensure valid/invalid hook point coverage on acceptance models."""


class TestValidatePatchHookType(PatchValidateHookTypeSuite):
    """Validate hook type checking logic independently of model."""


class TestPatchHelperClasses(PatchHelperClassesSuite):
    """Run helper class coverage for all acceptance models."""


class TestHookCleanup(PatchHookCleanupSuite):
    """Ensure hook cleanup works across acceptance models."""


class TestCacheAndHooksInteraction(PatchCacheInteractionSuite):
    """Exercise cache + patch interactions for every acceptance model."""


class TestPreHookPatching(PatchPreHookSuite):
    """Run pre-hook patching tests on acceptance models."""


class TestPreHookCleanup(PatchPreHookCleanupSuite):
    """Ensure pre-hook cleanup works across acceptance models."""
