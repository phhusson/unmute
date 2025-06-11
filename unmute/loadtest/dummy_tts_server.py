import asyncio
import logging
import random

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME

TEXT_TO_SPEECH_PATH = "/api/tts_streaming"

app = FastAPI()

logger = logging.getLogger(__name__)


def generate_sine_wave(
    duration_s: float, frequency: float = 440.0
) -> list[list[float]]:
    """Generate a sine wave with the given duration and frequency.
    Returns a list of chunks, where each chunk contains exactly CHUNK_SIZE samples,
    except possibly the last chunk.
    """
    num_samples = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)

    # Generate sine wave
    sine_wave = 0.5 * np.sin(2 * np.pi * frequency * t)

    # Apply envelope for smooth start and end
    envelope = np.ones_like(sine_wave)
    fade_samples = min(
        int(0.05 * SAMPLE_RATE), num_samples // 4
    )  # 50ms fade or 1/4 of sound
    if fade_samples > 0 and num_samples > 2 * fade_samples:
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    amplitude = 0.3
    envelope = amplitude * envelope

    # Apply envelope to sine wave
    audio_data = sine_wave * envelope

    # Split into chunks of CHUNK_SIZE
    chunks = []
    for i in range(0, len(audio_data), SAMPLES_PER_FRAME):
        chunk = audio_data[i : i + SAMPLES_PER_FRAME]

        # If we have a partial chunk at the end, pad it with zeros
        if len(chunk) < SAMPLES_PER_FRAME:
            padding = np.zeros(SAMPLES_PER_FRAME - len(chunk))
            chunk = np.concatenate([chunk, padding])

        chunks.append(chunk.tolist())

    return chunks


@app.get("/api/build_info")
def get_build_info():
    return {"note": "this is a dummy build info"}


@app.websocket(TEXT_TO_SPEECH_PATH)
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        current_time = 0.0

        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                logger.info(f"Received message type: {type(message)}")
            except asyncio.TimeoutError:
                # This prevents the loop from completely blocking signals
                await asyncio.sleep(0.01)
                continue

            # message = await websocket.receive()
            logger.info(message)

            if "text" in message:
                text = message["text"]
            else:
                if message["bytes"] == b"\0":
                    break
                else:
                    raise ValueError(f"Invalid message: {message}")

            if not text.strip():
                continue

            words = text.strip().split()

            frame_length = SAMPLES_PER_FRAME / SAMPLE_RATE

            for word in words:
                # Sounds more fun if the lengths are uneven
                word_duration = frame_length * len(word)

                start_time = current_time
                stop_time = current_time + word_duration

                # Send text message with timing information
                text_message = {
                    "type": "Text",
                    "text": word,
                    "start_s": start_time,
                    "stop_s": stop_time,
                }
                await websocket.send_bytes(msgpack.packb(text_message))

                # Generate audio (sine wave) for this word, split into fixed-size chunks
                note = random.randint(0, 12)
                frequency = 440 * (2 ** (note / 12))
                audio_chunks = generate_sine_wave(word_duration, frequency=frequency)

                # Calculate time for each chunk (for consistent pacing)
                chunk_duration = SAMPLES_PER_FRAME / SAMPLE_RATE
                chunk_count = len(audio_chunks)

                # Send each audio chunk with proper timing
                for chunk_idx, pcm_data in enumerate(audio_chunks):
                    audio_message = {"type": "Audio", "pcm": pcm_data}
                    await websocket.send_bytes(msgpack.packb(audio_message))

                    # Only sleep between chunks (not after the last chunk)
                    if chunk_idx < chunk_count - 1:
                        await asyncio.sleep(chunk_duration)

                # Calculate remaining time to wait to maintain 0.5s per word
                # We've already waited (chunk_count-1) * chunk_duration seconds
                remaining_wait = word_duration - (chunk_count - 1) * chunk_duration
                if remaining_wait > 0:
                    await asyncio.sleep(remaining_wait)

                current_time += word_duration

    except WebSocketDisconnect:
        print("Client disconnected")

    await websocket.close()


if __name__ == "__main__":
    import sys

    print(f"Run this via:\nfastapi dev {sys.argv[0]}")
    exit(1)
