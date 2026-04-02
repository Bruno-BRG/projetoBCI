"""
Port definition for model catalog discovery.
"""

from typing import List, Protocol

from brainbridge_v2.domain.entities.model_metadata import ModelMetadata


class ModelCatalogGateway(Protocol):
    def list_models(self) -> List[ModelMetadata]:
        ...
