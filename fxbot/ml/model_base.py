from abc import ABC, abstractmethod


class MLModel(ABC):
    @abstractmethod
    def predict(self, symbol: str, features: dict) -> float:
        """Retorna sfxbot.core 0..1 (probabilidade de sucesso)"""
        ...
