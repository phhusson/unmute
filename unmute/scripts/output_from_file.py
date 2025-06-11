import argparse
import asyncio
from pathlib import Path

import numpy as np
import sphn
from fastrtc import AsyncStreamHandler, Stream, wait_for_item

SAMPLE_RATE = 24000
# 480 works but the default 960 doesn't!!!
OUTPUT_FRAME_SIZE = 480


class FilePlaybackHandler(AsyncStreamHandler):
    def __init__(self, audio_path: Path) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE,
            output_frame_size=OUTPUT_FRAME_SIZE,
        )
        self.output_queue = asyncio.Queue()
        self.audio_path = audio_path

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        pass

    async def emit(self) -> tuple[int, np.ndarray]:
        return await wait_for_item(self.output_queue)

    def copy(self):
        return FilePlaybackHandler(self.audio_path)

    async def start_up(self) -> None:
        data, _sr = sphn.read(self.audio_path, sample_rate=SAMPLE_RATE)
        data = data[0]  # Take first channel to make it mono

        simulated_ratio = 1.5

        for i in range(0, len(data), OUTPUT_FRAME_SIZE):
            await self.output_queue.put((SAMPLE_RATE, data[i : i + OUTPUT_FRAME_SIZE]))
            await asyncio.sleep(OUTPUT_FRAME_SIZE / SAMPLE_RATE / simulated_ratio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    stream = Stream(
        handler=FilePlaybackHandler(args.file),
        modality="audio",
        mode="send-receive",
    )

    stream.ui.launch(debug=True)
