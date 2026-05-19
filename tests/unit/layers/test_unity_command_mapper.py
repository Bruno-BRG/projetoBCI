import pytest

from brainbridge_v2.application.unity_command_mapper import UnityCommandMapper


def test_unity_command_mapper_maps_left_and_right_predictions():
    left = UnityCommandMapper.from_prediction(0)
    right = UnityCommandMapper.from_prediction(1)

    assert left.direction == "esquerda"
    assert right.direction == "direita"


def test_unity_command_mapper_rejects_unknown_prediction_index():
    with pytest.raises(ValueError):
        UnityCommandMapper.from_prediction(2)
