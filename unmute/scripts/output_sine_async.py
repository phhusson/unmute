import asyncio

import numpy as np
from fastrtc import AsyncStreamHandler, Stream

from unmute.audio_stream_saver import AudioStreamSaver

SAMPLE_RATE = 24000
OUTPUT_FRAME_SIZE = 1920


class SineHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(input_sample_rate=SAMPLE_RATE)
        self.cur_time_samples = 0
        self.saver = AudioStreamSaver()

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        pass

    async def emit(self) -> tuple[int, np.ndarray]:
        times = np.arange(
            self.cur_time_samples,
            self.cur_time_samples + OUTPUT_FRAME_SIZE,
        )
        x = np.sin(2 * np.pi * 440 / SAMPLE_RATE * times) * 0.3
        x = x.astype(np.float32)
        self.cur_time_samples += OUTPUT_FRAME_SIZE

        self.saver.add(x)

        await asyncio.sleep(0.01)

        return (SAMPLE_RATE, x)

    def copy(self):
        return SineHandler()

    async def start_up(self) -> None:
        pass


if __name__ == "__main__":
    stream = Stream(
        handler=SineHandler(),
        modality="audio",
        mode="send-receive",
    )

    stream.ui.launch(debug=True)
