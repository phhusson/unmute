"""Transcribe audio from the microphone in real-time."""

import asyncio
from typing import Any

import numpy as np

try:
    # We don't need this for anything else so it's not in the dependencies
    import sounddevice as sd  # type: ignore
except ImportError as e:
    raise ImportError(
        "Please install sounddevice to run this example: pip install sounddevice "
        "(or uv pip install sounddevice if you're using uv)."
    ) from e
import tqdm

from unmute.kyutai_constants import SAMPLES_PER_FRAME
from unmute.stt.speech_to_text import (
    SpeechToText,
    STTMarkerMessage,
    STTWordMessage,
)


async def receive_loop(stt: SpeechToText):
    delay = None
    async for msg in stt:
        if isinstance(msg, STTWordMessage):
            print(f"Word: {msg.text} ({msg.start_time:.2f}s). Delay: {delay:.2f}s")
        elif isinstance(msg, STTMarkerMessage):  # type: ignore
            marker_time = msg.id / 1000
            time = asyncio.get_event_loop().time()
            delay = time - marker_time


async def main():
    stt = SpeechToText()
    await stt.start_up()
    audio_queue = asyncio.Queue()

    duration_sec = 30

    receive_task = asyncio.create_task(receive_loop(stt))

    def callback(indata: np.ndarray, frames: int, time: Any, status: sd.CallbackFlags):
        mono_audio = indata[:, 0]
        audio_queue.put_nowait(mono_audio.copy())

    start_time = asyncio.get_event_loop().time()

    audio_buffer = np.zeros((0,), dtype=np.float32)

    with sd.InputStream(callback=callback, blocksize=1024, samplerate=24000):
        pbar = tqdm.tqdm(total=duration_sec, desc="Recording", unit="s")
        while asyncio.get_event_loop().time() - start_time < duration_sec:
            try:
                audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            pbar.set_postfix(
                volume=np.mean(np.abs(audio_chunk)),
            )
            # Updating this is a bit annoying
            # pbar.update(audio_chunk.shape[0] / 24000)

            audio_buffer = np.concatenate((audio_buffer, audio_chunk), axis=0)
            while audio_buffer.shape[0] > SAMPLES_PER_FRAME:
                audio_chunk = audio_buffer[:SAMPLES_PER_FRAME]
                audio_buffer = audio_buffer[SAMPLES_PER_FRAME:]

                await stt.send_marker(int(asyncio.get_event_loop().time() * 1000))
                await stt.send_audio(audio_chunk)

    receive_task.cancel()
    print(f"Quit after {duration_sec} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
