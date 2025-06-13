"""Run speech-to-text on an audio file in a non-streaming way."""

import asyncio
import logging
from pathlib import Path

import numpy as np
import sphn
import tqdm

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage, STTWordMessage

TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono
logging.basicConfig(level=logging.INFO)


def load_and_process_audio(audio_path: Path):
    data, _sr = sphn.read(audio_path, sample_rate=SAMPLE_RATE)
    data = data[0]  # Take first channel to make it mono
    return data


async def main(audio_path: Path):
    stt = SpeechToText()
    await stt.start_up()

    audio_data = load_and_process_audio(audio_path)

    for i in tqdm.trange(0, len(audio_data), SAMPLES_PER_FRAME, desc="Sending audio"):
        chunk = audio_data[i : i + SAMPLES_PER_FRAME]
        await stt.send_audio(chunk)
        await asyncio.sleep(SAMPLES_PER_FRAME / SAMPLE_RATE)

    # When we get the marker back from the server, we know it has processed the audio
    await stt.send_marker(0)

    # Send extra audio to make sure the marker is processed
    for _ in range(25):
        await stt.send_audio(np.zeros(SAMPLES_PER_FRAME, dtype=np.int16))

    words = []

    with tqdm.tqdm() as pbar:
        async for msg in stt:
            if isinstance(msg, STTWordMessage):
                words.append(msg)
                pbar.set_postfix(n_words=len(words))
            elif isinstance(msg, STTMarkerMessage):  # pyright: ignore[reportUnnecessaryIsInstance]
                break

            pbar.update()

    print("\n".join(str(s) for s in words))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    args = parser.parse_args()

    asyncio.run(main(args.audio_path))
