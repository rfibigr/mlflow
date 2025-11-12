"""
Test to reproduce the bug where MLflow creates a duplicate TracerProvider when used with
OpenTelemetry auto-instrumentation.

Bug: mlflow.set_tracking_uri() calls reset() which resets the global _TRACER_PROVIDER_SET_ONCE
flag, causing MLflow to create a new TracerProvider instead of using the existing one from
auto-instrumentation.

Expected: MLflow should use the existing TracerProvider from auto-instrumentation
Actual: MLflow creates a second TracerProvider, breaking trace context and span linking
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import mlflow
from mlflow.tracing.provider import provider as mlflow_provider


@pytest.fixture(autouse=True)
def reset_mlflow_tracing():
    """Reset MLflow tracing state before and after each test."""
    yield
    # Cleanup after test
    try:
        mlflow.tracing.disable()
    except Exception:
        pass


@pytest.fixture
def mock_auto_instrumentation():
    """
    Simulate OpenTelemetry auto-instrumentation by setting up a TracerProvider
    before MLflow is initialized, similar to what happens with sitecustomize.py
    """
    # Create a TracerProvider like auto-instrumentation would
    auto_tracer_provider = TracerProvider()

    # Set it as the global provider (this is what auto-instrumentation does)
    trace.set_tracer_provider(auto_tracer_provider)

    # Verify the Once flag was set (this happens automatically in set_tracer_provider)
    assert trace._TRACER_PROVIDER_SET_ONCE._done is True, (
        "Auto-instrumentation should have set the Once flag"
    )

    return auto_tracer_provider


def test_reset_preserves_global_once_flag(mock_auto_instrumentation, monkeypatch):
    """
    Test that mlflow.set_tracking_uri() doesn't reset the global _TRACER_PROVIDER_SET_ONCE flag.

    This is the core bug: reset() sets _TRACER_PROVIDER_SET_ONCE._done = False,
    which causes MLflow to create a new TracerProvider instead of using the existing one.
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    # Capture the initial state after auto-instrumentation
    initial_once_flag = trace._TRACER_PROVIDER_SET_ONCE._done
    assert initial_once_flag is True, "Once flag should be True after auto-instrumentation"

    # This call triggers reset() which is where the bug occurs
    mlflow.set_tracking_uri("http://localhost:5000")

    # BUG: The Once flag gets reset to False
    # EXPECTED: The flag should remain True
    after_reset_once_flag = trace._TRACER_PROVIDER_SET_ONCE._done

    assert after_reset_once_flag is True, (
        "BUG: reset() should NOT reset the global _TRACER_PROVIDER_SET_ONCE flag when "
        "using global TracerProvider (MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false)"
    )


def test_single_tracer_provider_created(mock_auto_instrumentation, monkeypatch):
    """
    Test that only one TracerProvider exists when using auto-instrumentation.

    With the bug: Two TracerProviders are created (one from auto-instrumentation, one from
    MLflow)
    Expected: Only the auto-instrumentation TracerProvider should exist
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    auto_provider = mock_auto_instrumentation
    auto_provider_id = id(auto_provider)

    # Trigger MLflow initialization
    mlflow.set_tracking_uri("http://localhost:5000")

    # Get a tracer from MLflow - this is where it might create a new TracerProvider
    mlflow_provider.get_tracer(__name__)

    # Get the current global TracerProvider
    current_provider = trace.get_tracer_provider()
    current_provider_id = id(current_provider)

    # BUG: MLflow creates a new TracerProvider instead of using the existing one
    # EXPECTED: The provider should be the same instance
    assert current_provider_id == auto_provider_id, (
        f"BUG: MLflow created a new TracerProvider (id={current_provider_id}) "
        f"instead of using the existing auto-instrumentation (id={auto_provider_id})"
    )


def test_trace_decorator_with_auto_instrumentation(mock_auto_instrumentation, monkeypatch):
    """
    Test that @mlflow.trace() decorator works correctly with auto-instrumentation.

    With the bug: AttributeError: 'NoneType' object has no attribute 'set_span_type'
    Expected: Decorator should work without errors
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("test")

    # This should not raise an error
    @mlflow.trace()
    def test_function():
        return {"result": "success"}

    # BUG: Calling the decorated function may fail with AttributeError
    # EXPECTED: Function executes successfully
    try:
        result = test_function()
        assert result == {"result": "success"}
    except AttributeError as e:
        if "'NoneType' object has no attribute 'set_span_type'" in str(e):
            pytest.fail(
                "BUG: @mlflow.trace() decorator failed because MLflow created a duplicate "
                "TracerProvider instead of using the one from auto-instrumentation"
            )
        else:
            raise


def test_processors_added_to_existing_provider(mock_auto_instrumentation, monkeypatch):
    """
    Test that MLflow adds its processors to the existing TracerProvider from
    auto-instrumentation.

    With the bug: Processors are added to a new TracerProvider
    Expected: Processors should be added to the auto-instrumentation TracerProvider
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    auto_provider = mock_auto_instrumentation
    initial_processor_count = len(auto_provider._active_span_processor._span_processors)

    mlflow.set_tracking_uri("http://localhost:5000")

    # Trigger MLflow to initialize its tracer (lazy initialization)
    mlflow_provider.get_tracer(__name__)

    # Get the current provider
    current_provider = trace.get_tracer_provider()

    # BUG: If a new TracerProvider was created, it won't be the same object
    # EXPECTED: Processors should be added to the original auto-instrumentation provider
    assert isinstance(current_provider, TracerProvider), (
        "Current provider should be a TracerProvider instance"
    )

    if id(current_provider) == id(auto_provider):
        # If providers are the same, check that MLflow's processors were added
        final_processor_count = len(current_provider._active_span_processor._span_processors)
        assert final_processor_count > initial_processor_count, (
            "MLflow should have added its processors to the existing TracerProvider"
        )
    else:
        pytest.fail(
            "BUG: MLflow created a new TracerProvider instead of adding processors "
            "to the existing auto-instrumentation TracerProvider"
        )


def test_span_context_preservation(mock_auto_instrumentation, monkeypatch):
    """
    Test that span context is preserved across auto-instrumentation and MLflow spans.

    With the bug: Spans are created in different TracerProviders, breaking trace context
    Expected: All spans should be in the same trace with proper parent-child relationships
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    mlflow.set_tracking_uri("http://localhost:5000")

    auto_provider = mock_auto_instrumentation
    auto_tracer = auto_provider.get_tracer("auto_instrumentation")

    # Create a span from auto-instrumentation
    with auto_tracer.start_as_current_span("auto_span") as auto_span:
        auto_trace_id = auto_span.get_span_context().trace_id

        # Create a span from MLflow inside the auto-instrumentation span
        @mlflow.trace()
        def mlflow_function():
            current_span = trace.get_current_span()
            return current_span.get_span_context().trace_id

        # BUG: MLflow span may be in a different TracerProvider, breaking the trace
        # EXPECTED: MLflow span should be a child of the auto-instrumentation span
        try:
            mlflow_trace_id = mlflow_function()

            assert mlflow_trace_id == auto_trace_id, (
                "BUG: MLflow span has a different trace_id, indicating it's in a "
                "separate TracerProvider and not properly linked to the parent span"
            )
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                pytest.fail("BUG: MLflow span creation failed because of duplicate TracerProvider")
            else:
                raise


def test_reset_does_not_affect_global_provider_once_flag(monkeypatch):
    """
    Test that calling reset() when MLFLOW_USE_DEFAULT_TRACER_PROVIDER=false
    does not reset the global _TRACER_PROVIDER_SET_ONCE flag.
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    # Set up a TracerProvider like auto-instrumentation would
    auto_provider = TracerProvider()
    trace.set_tracer_provider(auto_provider)

    # Verify the flag is set
    assert trace._TRACER_PROVIDER_SET_ONCE._done is True

    # Call MLflow's reset function (this is called by set_tracking_uri)
    from mlflow.tracing.provider import reset

    reset()

    # BUG: The flag gets reset to False
    # EXPECTED: The flag should remain True because we're using the global provider
    assert trace._TRACER_PROVIDER_SET_ONCE._done is True, (
        "BUG: reset() should not reset _TRACER_PROVIDER_SET_ONCE when using global provider"
    )


def test_reset_only_resets_isolated_provider_once_flag(monkeypatch):
    """
    Test that reset() only resets the isolated provider Once flag when
    MLFLOW_USE_DEFAULT_TRACER_PROVIDER=true (isolated mode).
    """
    monkeypatch.setenv("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "true")
    # Initialize MLflow's isolated provider
    mlflow_provider.get_tracer(__name__)

    # Verify the isolated once flag is set
    assert mlflow_provider._isolated_tracer_provider_once._done is True

    # Call reset
    from mlflow.tracing.provider import reset

    reset()

    # In isolated mode, it's OK to reset the isolated Once flag
    # because it doesn't affect the global OpenTelemetry state
    assert mlflow_provider._isolated_tracer_provider_once._done is False, (
        "reset() should reset the isolated provider Once flag in isolated mode"
    )
