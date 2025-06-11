import asyncio
import heapq
from dataclasses import dataclass, field
from typing import AsyncIterable, Callable, Iterable, TypeVar

T = TypeVar("T")


@dataclass(order=True)
class TimedItem[T]:
    time: float
    item: T = field(compare=False)

    def as_tuple(self) -> tuple[float, T]:
        return self.time, self.item


class RealtimeQueue[T]:
    """A data structure that accumulates timestamped items and releases them at the given times.

    Implemented as a heap, so it doesn't have to be FIFO.
    """

    def __init__(self, get_time: Callable[[], float] | None = None):
        self.queue: list[TimedItem] = []
        self.start_time: float | None = None

        if get_time is None:
            self.get_time = lambda: asyncio.get_event_loop().time()
        else:
            # Use an external time function to support use cases where "real time"
            # means something different
            self.get_time = get_time

    def start_if_not_started(self):
        if self.start_time is None:
            self.start_time = self.get_time()

    def put(self, item: T, time: float):
        heapq.heappush(self.queue, TimedItem(time, item))

    async def get(self) -> AsyncIterable[tuple[float, T]]:
        """Get all items that are past due. If none is, wait for the next one."""

        if self.start_time is None:
            return
        if not self.queue:
            return

        time_since_start = self.get_time() - self.start_time
        while self.queue:
            delta = self.queue[0].time - time_since_start

            if delta > 0:
                await asyncio.sleep(delta)

            yield heapq.heappop(self.queue).as_tuple()

    def get_nowait(self) -> Iterable[tuple[float, T]]:
        if self.start_time is None:
            return None

        time_since_start = self.get_time() - self.start_time

        while self.queue and self.queue[0].time <= time_since_start:
            yield heapq.heappop(self.queue).as_tuple()

    async def __aiter__(self):
        if self.start_time is None or not self.queue:
            return

        while self.queue:
            time_since_start = self.get_time() - self.start_time
            delta = self.queue[0].time - time_since_start

            if delta > 0:
                await asyncio.sleep(delta)

            yield heapq.heappop(self.queue).as_tuple()

    def empty(self):
        return not self.queue
