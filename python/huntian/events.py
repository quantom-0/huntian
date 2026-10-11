"""A tiny synchronous publish/subscribe event bus.

The bus is the observability spine of a run: the registry announces
registrations and hot-swaps, the context announces stage boundaries, and the
executor announces pipeline progress. Subscribers are synchronous and must not
raise; handler failures are captured on the bus instead of breaking a run.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Iterable, Mapping

__all__ = ["Event", "EventBus", "Handler"]

Handler = Callable[["Event"], None]


@dataclass(frozen=True)
class Event:
    """An immutable notification published on the bus."""

    topic: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    seq: int = 0
    at: float = 0.0

    def __str__(self) -> str:
        detail = ", ".join(f"{k}={v!r}" for k, v in self.payload.items())
        return f"<{self.topic} #{self.seq} {detail}>"


@dataclass(frozen=True)
class _Subscription:
    pattern: str
    handler: Handler
    token: int


@dataclass(frozen=True)
class HandlerError:
    pattern: str
    topic: str
    exception: BaseException


def _topic_matches(pattern: str, topic: str) -> bool:
    """Segment-wise glob match: ``*`` matches one segment, ``**`` matches rest."""
    if pattern in ("*", "**"):
        return True
    pat = pattern.split(".")
    top = topic.split(".")
    i = 0
    for i, segment in enumerate(pat):
        if segment == "**":
            return True
        if i >= len(top):
            return False
        if segment != "*" and segment != top[i]:
            return False
    return len(pat) == len(top)


class EventBus:
    """Synchronous, in-process event bus with a bounded history."""

    def __init__(self, *, history_size: int = 512) -> None:
        self._subscriptions: list[_Subscription] = []
        self._history: Deque[Event] = deque(maxlen=history_size)
        self._seq = 0
        self._token = 0
        self._handler_errors: list[HandlerError] = []
        self._publishing = False

    # -- subscription ---------------------------------------------------
    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        """Register ``handler`` for ``pattern``; returns an unsubscribe callable."""
        self._token += 1
        subscription = _Subscription(pattern=pattern, handler=handler, token=self._token)
        self._subscriptions.append(subscription)

        def _unsubscribe() -> None:
            self._subscriptions = [s for s in self._subscriptions if s.token != subscription.token]

        return _unsubscribe

    def unsubscribe_all(self, pattern: str | None = None) -> int:
        """Drop subscriptions (all, or only those matching ``pattern``)."""
        before = len(self._subscriptions)
        if pattern is None:
            self._subscriptions = []
        else:
            self._subscriptions = [s for s in self._subscriptions if s.pattern != pattern]
        return before - len(self._subscriptions)

    # -- publishing -----------------------------------------------------
    def publish(self, topic: str, **payload: Any) -> Event:
        """Publish an event and dispatch it to matching subscribers."""
        self._seq += 1
        event = Event(topic=topic, payload=dict(payload), seq=self._seq, at=time.time())
        self._history.append(event)
        # Re-entrant publishes are allowed but must not corrupt iteration.
        self._publishing = True
        try:
            for subscription in tuple(self._subscriptions):
                if _topic_matches(subscription.pattern, topic):
                    try:
                        subscription.handler(event)
                    except Exception as exc:  # noqa: BLE001 - observability must not break runs
                        self._handler_errors.append(
                            HandlerError(subscription.pattern, topic, exc)
                        )
        finally:
            self._publishing = False
        return event

    # -- inspection -----------------------------------------------------
    @property
    def handler_errors(self) -> tuple[HandlerError, ...]:
        return tuple(self._handler_errors)

    def history(
        self, *, patterns: Iterable[str] | None = None, limit: int | None = None
    ) -> list[Event]:
        events = list(self._history)
        if patterns is not None:
            pats = tuple(patterns)
            events = [e for e in events if any(_topic_matches(p, e.topic) for p in pats)]
        if limit is not None:
            events = events[-limit:]
        return events

    def clear(self) -> None:
        self._history.clear()
        self._handler_errors.clear()

    @property
    def published(self) -> int:
        """Total number of events published on this bus.

        Deliberately *not* ``__len__``: a bus must stay truthy even when empty,
        otherwise ``bus or EventBus()`` silently discards a caller-supplied bus.
        """
        return self._seq

    def __repr__(self) -> str:
        return (
            f"EventBus(subscriptions={len(self._subscriptions)}, published={self._seq}, "
            f"history={len(self._history)})"
        )
