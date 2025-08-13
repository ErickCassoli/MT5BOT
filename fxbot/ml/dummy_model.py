from .model_base import MLModel


class DummyModel(MLModel):
    def __init__(self, threshold=0.5): self.th = threshold

    def predict(self, symbol: str, features: dict) -> float:
        # brinque com o score — ex: favorece ema_slope positiva e ATR moderado
        slope = float(features.get("ema_slope", 0.0))
        atrv = float(features.get("atr", 0.0005))
        base = 0.55 + 0.15*(1 if slope > 0 else -1) - 0.05*(atrv > 0.004)
        return max(0.0, min(1.0, base))
