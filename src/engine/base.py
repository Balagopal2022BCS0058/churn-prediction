from abc import ABC, abstractmethod

from src.api.schemas.request import PredictRiskRequest
from src.api.schemas.response import PredictRiskResponse


class RiskEngine(ABC):
    @abstractmethod
    async def evaluate(self, request: PredictRiskRequest) -> PredictRiskResponse:
        ...
