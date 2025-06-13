import asyncio
import random
from logging import getLogger
from typing import AsyncIterator, Literal, Union

import msgpack
import numpy as np
import websockets
from fastrtc import audio_to_float32
from pydantic import BaseModel, TypeAdapter

from unmute import metrics as mt
from unmute.exceptions import MissingServiceAtCapacity
from unmute.kyutai_constants import (
    FRAME_TIME_SEC,
    HEADERS,
    SAMPLE_RATE,
    SPEECH_TO_TEXT_PATH,
    STT_DELAY_SEC,
    STT_SERVER,
)
from unmute.service_discovery import ServiceWithStartup
from unmute.stt.exponential_moving_average import ExponentialMovingAverage
from unmute.timer import Stopwatch
from unmute.websocket_utils import WebsocketState

logger = getLogger(__name__)


class STTWordMessage(BaseModel):
    type: Literal["Word"]
    text: str
    start_time: float


class STTEndWordMessage(BaseModel):
    type: Literal["EndWord"]
    stop_time: float


class STTMarkerMessage(BaseModel):
    type: Literal["Marker"]
    id: int


class STTStepMessage(BaseModel):
    type: Literal["Step"]
    step_idx: int
    prs: list[float]


class STTErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str


class STTReadyMessage(BaseModel):
    type: Literal["Ready"]


STTMessage = Union[
    STTWordMessage,
    STTEndWordMessage,
    STTMarkerMessage,
    STTStepMessage,
    STTErrorMessage,
    STTReadyMessage,
]
STTMessageAdapter = TypeAdapter(STTMessage)


class SpeechToText(ServiceWithStartup):
    def __init__(
        self, stt_instance: str = STT_SERVER, delay_sec: float = STT_DELAY_SEC
    ):
        self.stt_instance = stt_instance
        self.delay_sec = delay_sec
        self.websocket: websockets.ClientConnection | None = None
        self.sent_samples = 0
        self.received_words = 0
        self.current_time = -STT_DELAY_SEC
        self.time_since_first_audio_sent = Stopwatch(autostart=False)
        self.waiting_first_step: bool = True

        # In our case, attack  = from speaking to not speaking
        #              release = from not speaking to speaking
        self.pause_prediction = ExponentialMovingAverage(
            attack_time=0.01, release_time=0.01, initial_value=1.0
        )

        self.shutdown_complete = asyncio.Event()

    def state(self) -> WebsocketState:
        if not self.websocket:
            return "not_created"
        else:
            d: dict[websockets.protocol.State, WebsocketState] = {
                websockets.protocol.State.CONNECTING: "connecting",
                websockets.protocol.State.OPEN: "connected",
                websockets.protocol.State.CLOSING: "closing",
                websockets.protocol.State.CLOSED: "closed",
            }
            return d[self.websocket.state]

    async def send_audio(self, audio: np.ndarray) -> None:
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D array, got {audio.shape=}")

        if audio.dtype != np.float32:
            audio = audio_to_float32(audio)

        self.sent_samples += len(audio)
        self.time_since_first_audio_sent.start_if_not_started()
        mt.STT_SENT_FRAMES.inc()

        await self._send({"type": "Audio", "pcm": audio.tolist()})

    async def send_marker(self, id: int) -> None:
        await self._send({"type": "Marker", "id": id})

    async def _send(self, data: dict) -> None:
        """Send an arbitrary message to the STT server."""
        to_send = msgpack.packb(data, use_bin_type=True, use_single_float=True)

        if self.websocket:
            await self.websocket.send(to_send)
        else:
            logger.warning("STT websocket not connected")

    async def start_up(self):
        logger.info(f"Connecting to STT {self.stt_instance}...")
        self.websocket = await websockets.connect(
            self.stt_instance + SPEECH_TO_TEXT_PATH, additional_headers=HEADERS
        )
        logger.info("Connected to STT")

        try:
            message_bytes = await self.websocket.recv()
            message_dict = msgpack.unpackb(message_bytes)  # type: ignore
            message = STTMessageAdapter.validate_python(message_dict)
            if isinstance(message, STTReadyMessage):
                mt.STT_ACTIVE_SESSIONS.inc()
                return
            elif isinstance(message, STTErrorMessage):
                raise MissingServiceAtCapacity("stt")
            else:
                raise RuntimeError(
                    f"Expected ready or error message, got {message.type}"
                )
        except Exception as e:
            logger.error(f"Error during STT startup: {repr(e)}")
            # Make sure we don't leave a dangling websocket connection
            await self.websocket.close()
            self.websocket = None
            raise

    async def shutdown(self):
        logger.info("Shutting down STT, receiving last messages")
        if self.shutdown_complete.is_set():
            return

        mt.STT_ACTIVE_SESSIONS.dec()
        if self.time_since_first_audio_sent.started:
            mt.STT_SESSION_DURATION.observe(self.time_since_first_audio_sent.time())
            mt.STT_AUDIO_DURATION.observe(self.sent_samples / SAMPLE_RATE)
            mt.STT_NUM_WORDS.observe(self.received_words)

        if not self.websocket:
            raise RuntimeError("STT websocket not connected")
        await self.websocket.close()
        await self.shutdown_complete.wait()

        logger.info("STT shutdown() finished")

    async def __aiter__(
        self,
    ) -> AsyncIterator[STTWordMessage | STTMarkerMessage]:
        if not self.websocket:
            raise RuntimeError("STT websocket not connected")

        my_id = random.randint(1, int(1e9))

        # The pause prediction is all over the place in the first few steps, so ignore.
        n_steps_to_wait = 12

        try:
            async for message_bytes in self.websocket:
                data = msgpack.unpackb(message_bytes)  # type: ignore
                logger.debug(f"{my_id} {self.pause_prediction.value} got {data}")
                message: STTMessage = STTMessageAdapter.validate_python(data)

                match message:
                    case STTWordMessage():
                        num_words = len(message.text.split())
                        mt.STT_RECV_WORDS.inc(num_words)
                        self.received_words += 1
                        yield message
                    case STTEndWordMessage():
                        continue
                    case STTStepMessage():
                        self.current_time += FRAME_TIME_SEC
                        mt.STT_RECV_FRAMES.inc()
                        if (
                            self.waiting_first_step
                            and self.time_since_first_audio_sent.started
                        ):
                            self.waiting_first_step = False
                            mt.STT_TTFT.observe(self.time_since_first_audio_sent.time())
                        if n_steps_to_wait > 0:
                            n_steps_to_wait -= 1
                        else:
                            self.pause_prediction.update(
                                dt=FRAME_TIME_SEC, new_value=message.prs[2]
                            )
                    case STTMarkerMessage():
                        yield message
                    case STTReadyMessage():
                        continue
                    case _:
                        # Not sure why Pyright complains about non-exhaustive match
                        raise ValueError(f"Unknown message: {message}")

        except websockets.ConnectionClosedOK:
            # The server closes the connection once we send \0, and this actually shows
            # up as a websockets.ConnectionClosedError.
            pass
        finally:
            self.shutdown_complete.set()
