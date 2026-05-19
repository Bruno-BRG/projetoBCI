from brainbridge_v2.application.runtime_config import DEFAULT_RUNTIME_CONFIG


def test_default_runtime_config_matches_two_second_inference_window():
    assert DEFAULT_RUNTIME_CONFIG.sample_rate_hz == 125
    assert DEFAULT_RUNTIME_CONFIG.window_size == 250
    assert DEFAULT_RUNTIME_CONFIG.channels == 16
    assert DEFAULT_RUNTIME_CONFIG.ai_window_duration_ms == 2000
    assert DEFAULT_RUNTIME_CONFIG.tensorflow_warmup_enabled is True
