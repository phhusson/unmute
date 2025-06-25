"""A dummy speech-to-text that never sends any words.

Useful for testing, like checking if not running the STT on the same GPU can reduce
latency.
"""

import asyncio
from logging import getLogger
from typing import AsyncIterator, Literal

import numpy as np

from unmute.kyutai_constants import FRAME_TIME_SEC, STT_DELAY_SEC, STT_SERVER
from unmute.service_discovery import ServiceWithStartup
from unmute.stt.exponential_moving_average import ExponentialMovingAverage
from unmute.stt.speech_to_text import STTMarkerMessage, STTWordMessage
from unmute.websocket_utils import WebsocketState

logger = getLogger(__name__)

TranscriptionStatus = Literal[
    "should_transcribe", "has_transcribed", "should_not_transcribe"
]


class DummySpeechToText(ServiceWithStartup):
    def __init__(
        self, stt_instance: str = STT_SERVER, delay_sec: float = STT_DELAY_SEC
    ):
        self.stt_instance = stt_instance
        self.sent_samples = 0
        self.received_words = 0
        self.delay_sec = delay_sec
        self.current_time = -STT_DELAY_SEC

        # We just keep this at 1.0 = user is not speaking
        self.pause_prediction = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )
        self.should_shutdown = asyncio.Event()

    def state(self) -> WebsocketState:
        return "connected"

    async def send_audio(self, audio: np.ndarray) -> None:
        self.current_time += FRAME_TIME_SEC

    async def send_marker(self, id: int) -> None:
        return

    async def start_up(self):
        logger.info("Starting dummy STT")

    async def shutdown(self):
        logger.info("Shutting down dummy STT")
        self.should_shutdown.set()

    async def __aiter__(
        self,
    ) -> AsyncIterator[STTWordMessage | STTMarkerMessage]:
        while self.should_shutdown.is_set() is False:
            await asyncio.sleep(1.0)

        # Just to satisfy the type checker
        yield STTMarkerMessage(type="Marker", id=0)
