"""
Mapping between model predictions and runtime actuator directions.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeAction:
    predicted_index: int
    direction: str


class UnityCommandMapper:
    LEFT_INDEX = 0
    RIGHT_INDEX = 1
    LEFT_DIRECTION = "esquerda"
    RIGHT_DIRECTION = "direita"

    @classmethod
    def from_prediction(cls, predicted_index: int) -> RuntimeAction:
        normalized_index = int(predicted_index)
        if normalized_index == cls.LEFT_INDEX:
            return RuntimeAction(
                predicted_index=normalized_index,
                direction=cls.LEFT_DIRECTION,
            )
        if normalized_index == cls.RIGHT_INDEX:
            return RuntimeAction(
                predicted_index=normalized_index,
                direction=cls.RIGHT_DIRECTION,
            )
        raise ValueError(f"Indice de predicao sem mapeamento: {predicted_index}")
