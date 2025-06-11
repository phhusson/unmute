import asyncio
import time

import numpy as np
import websockets
from fastrtc import AsyncStreamHandler, Stream, wait_for_item

from unmute.tts.text_to_speech import TextToSpeech, TTSAudioMessage

SAMPLE_RATE = 24000
OUTPUT_FRAME_SIZE = 480


class TTSHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE,
            output_frame_size=OUTPUT_FRAME_SIZE,
        )
        self.tts = TextToSpeech()
        self.output_queue = asyncio.Queue()
        self.go = False
        self.cur_time_samples = 0

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        pass

    async def emit(self) -> tuple[int, np.ndarray]:
        # if not self.output_queue.empty():
        #     return await self.output_queue.get()

        # times = np.arange(
        #     self.cur_time_samples,
        #     self.cur_time_samples + OUTPUT_FRAME_SIZE,
        # )
        # x = np.sin(2 * np.pi * 440 / SAMPLE_RATE * times) * 0.3
        # x = x.astype(np.float32)
        # self.cur_time_samples += OUTPUT_FRAME_SIZE

        # await asyncio.sleep(0.01)

        # return (SAMPLE_RATE, x)

        return await wait_for_item(self.output_queue)
        # return await self.output_queue.get()

    def copy(self):
        return TTSHandler()

    async def start_up(self) -> None:
        asyncio.create_task(self._tts_loop())

    async def _tts_loop(self):
        await self.tts.start_up()

        await self.tts.send(" ".join(["Hello, world! "] * 10))

        try:
            audio_started = None

            async for message in self.tts:
                if audio_started is not None:
                    time_since_start = time.time() - audio_started
                    time_received = self.tts.received_samples / self.input_sample_rate
                    ratio = time_received / time_since_start
                    assert self.input_sample_rate == SAMPLE_RATE
                    print(
                        f"{time_received=:.2f}, {time_since_start=:.2f}, "
                        f"ratio {ratio:.2f}"
                    )

                if isinstance(message, TTSAudioMessage):
                    audio = np.array(message.pcm, dtype=np.float32)
                    assert self.output_sample_rate == SAMPLE_RATE

                    assert len(audio) % OUTPUT_FRAME_SIZE == 0, (
                        "Audio length must be a multiple of the frame size."
                    )
                    for i in range(0, len(audio), OUTPUT_FRAME_SIZE):
                        await self.output_queue.put(
                            (SAMPLE_RATE, audio[i : i + OUTPUT_FRAME_SIZE])
                        )
                    # await self.output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = time.time()

        except websockets.ConnectionClosed:
            print("TTS connection closed while receiving messages.")


if __name__ == "__main__":
    stream = Stream(
        handler=TTSHandler(),
        modality="audio",
        mode="send-receive",
    )

    stream.ui.launch(debug=True)
