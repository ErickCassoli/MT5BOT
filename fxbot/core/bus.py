# fxbot/core/bus.py  (opcional; não usado agora, mas deixo pronto)
from collections import defaultdict
from typing import Callable, Dict, List, Any


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event: str, fn: Callable[[Any], None]):
        self._subs[event].append(fn)

    def publish(self, event: str, payload: Any):
        for fn in self._subs.get(event, []):
            try:
                fn(payload)
            except Exception as e:
                print(f"[bus:{event}] handler error: {e}")
