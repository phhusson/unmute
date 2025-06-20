import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

import unmute.openai_realtime_api_events as ora

RECORDINGS_DIR = Path(__file__).parents[1] / "recordings"
SAVE_EVERY_N_EVENTS = 10

logger = logging.getLogger(__name__)

EventSender = Literal["client", "server"]


class RecorderEvent(BaseModel):
    timestamp_wall: float
    event_sender: EventSender
    data: Annotated[ora.Event, Field(discriminator="type")]


class Recorder:
    def __init__(self):
        self.path = RECORDINGS_DIR / (make_filename() + ".jsonl")
        RECORDINGS_DIR.mkdir(exist_ok=True)
        self._events = []
        self.queue = asyncio.Queue()
        # The lock lets us know if the recorder is running.
        self.loop_lock = asyncio.Lock()

    async def run(self):
        logger.info(f"Starting recording into {self.path}")
        await self._loop()

    async def add_event(self, event_sender: EventSender, data: ora.Event):
        """If the recorder is not actually running, the event will be ignored."""
        if not self.loop_lock.locked():
            return

        await self.queue.put(
            RecorderEvent(
                timestamp_wall=datetime.now().timestamp(),
                event_sender=event_sender,
                data=data,
            )
        )

    async def _loop(self):
        async with self.loop_lock:
            while True:
                event = await self.queue.get()
                self._events.append(event)

                if len(self._events) % SAVE_EVERY_N_EVENTS == 0:
                    with self.path.open("a") as f:
                        for e in self._events[-10:]:
                            f.write(e.model_dump_json() + "\n")


def make_filename() -> str:
    """Create a unique filename based on the current timestamp and a short UUID, without a suffix."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4().hex[:4]
    return f"{timestamp}_{unique_id}"
