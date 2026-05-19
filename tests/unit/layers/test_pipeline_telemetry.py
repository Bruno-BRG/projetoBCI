from brainbridge_v2.application.pipeline_telemetry import (
    PipelineTelemetry,
    SampleRateMonitor,
)
import pytest


def test_sample_rate_monitor_reports_rate_after_interval():
    monitor = SampleRateMonitor(interval_seconds=1.0)

    assert monitor.observe(now_seconds=10.0) is None
    assert monitor.observe(now_seconds=10.25) is None
    assert monitor.observe(now_seconds=10.50) is None
    rate = monitor.observe(now_seconds=11.0)

    assert rate == 4.0
    assert monitor.latest_rate_hz == 4.0


def test_pipeline_telemetry_records_recent_events_and_eeg_rate():
    telemetry = PipelineTelemetry(max_events=3, sample_rate_interval_seconds=1.0)

    telemetry.record("TASK_SENT", marker="T1")
    telemetry.observe_eeg_sample(now_seconds=1.0)
    telemetry.observe_eeg_sample(now_seconds=1.5)
    telemetry.observe_eeg_sample(now_seconds=2.0)
    telemetry.record("PREDICTION_DONE", predicted_index=0)

    events = telemetry.latest_events(10)

    assert [event.name for event in events] == [
        "TASK_SENT",
        "EEG_RATE",
        "PREDICTION_DONE",
    ]
    assert events[1].details["rate_hz"] == 3.0


def test_sample_rate_monitor_reset_clears_timing_state():
    monitor = SampleRateMonitor(interval_seconds=1.0)
    monitor.observe(now_seconds=1.0)
    monitor.observe(now_seconds=2.0)

    monitor.reset()

    assert monitor.window_start_seconds is None
    assert monitor.samples_in_window == 0
    assert monitor.latest_rate_hz == 0.0
    assert monitor.observe(now_seconds=3.0) is None


def test_pipeline_telemetry_keeps_only_recent_events():
    telemetry = PipelineTelemetry(max_events=2)

    telemetry.record("ONE")
    telemetry.record("TWO")
    telemetry.record("THREE")

    assert [event.name for event in telemetry.latest_events(10)] == ["TWO", "THREE"]


def test_pipeline_telemetry_latest_events_handles_non_positive_count():
    telemetry = PipelineTelemetry()
    telemetry.record("ONE")

    assert telemetry.latest_events(0) == []
    assert telemetry.latest_events(-1) == []


def test_pipeline_telemetry_reset_clears_events_and_rate_state():
    telemetry = PipelineTelemetry()
    telemetry.record("TASK_SENT")
    telemetry.observe_eeg_sample(now_seconds=1.0)

    telemetry.reset()

    assert telemetry.latest_events() == []
    assert telemetry.sample_rate.window_start_seconds is None


def test_pipeline_telemetry_disabled_does_not_store_events_or_sample_rate():
    telemetry = PipelineTelemetry(enabled=False)

    event = telemetry.record("TASK_SENT", marker="T1")
    rate = telemetry.observe_eeg_sample(now_seconds=1.0)

    assert event.name == "TASK_SENT"
    assert rate is None
    assert telemetry.latest_events() == []
    assert telemetry.sample_rate.window_start_seconds is None


def test_pipeline_telemetry_set_enabled_resets_when_turning_off():
    telemetry = PipelineTelemetry(enabled=True)
    telemetry.record("TASK_SENT")

    telemetry.set_enabled(False)

    assert telemetry.enabled is False
    assert telemetry.latest_events() == []

    telemetry.set_enabled(True)
    telemetry.record("AI_WINDOW_OPENED")

    assert [event.name for event in telemetry.latest_events()] == ["AI_WINDOW_OPENED"]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SampleRateMonitor(interval_seconds=0),
        lambda: PipelineTelemetry(max_events=0),
    ],
)
def test_telemetry_rejects_invalid_config(factory):
    with pytest.raises(ValueError):
        factory()
