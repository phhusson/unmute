import argparse
import asyncio
import base64
import json
import os
from pathlib import Path

import numpy as np
import pydub
import pydub.playback
import sphn
import websockets
from fastrtc import audio_to_int16

from unmute.kyutai_constants import SAMPLE_RATE

INPUT_FRAME_SIZE = 960
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono


def base64_encode_audio(audio: np.ndarray):
    pcm_bytes = audio_to_int16(audio)
    encoded = base64.b64encode(pcm_bytes).decode("ascii")
    return encoded


async def send_messages(websocket: websockets.ClientConnection, audio_path: Path):
    data, _sr = sphn.read(audio_path, sample_rate=SAMPLE_RATE)
    data = data[0]  # Take first channel to make it mono

    try:
        while True:
            chunk_size = 1920  # Send data in chunks
            for i in range(0, len(data), chunk_size):
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_encode_audio(data[i : i + chunk_size]),
                }

                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.01)  # Simulate real-time streaming

            await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await websocket.send(json.dumps({"type": "response.create"}))

            for _ in range(0, len(data), chunk_size):
                event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64_encode_audio(
                        np.zeros(chunk_size, dtype=np.float32)
                    ),
                }

                await websocket.send(json.dumps(event))
                await asyncio.sleep(0.01)  # Simulate real-time streaming
    except websockets.ConnectionClosed:
        print("Connection closed while sending messages.")


async def receive_messages(websocket: websockets.ClientConnection):
    buffer = []
    transcript = ""

    try:
        async for message in websocket:
            message = json.loads(message)
            if message["type"] == "response.audio.delta":
                base64_audio = message["delta"]
                binary_audio_data = base64.b64decode(base64_audio)
                buffer.append(binary_audio_data)
            elif message["type"] == "response.audio.done":
                print("Received `response.audio.done` message.")
                break
            elif message["type"] == "response.audio_transcript.delta":
                transcript += message["delta"]
                print(message["delta"], end="", flush=True)
            else:
                print(f"Received message: {message}")
    except websockets.ConnectionClosed:
        print("Connection closed while receiving messages.")

    # save and play using pydub
    audio = pydub.AudioSegment(
        data=b"".join(buffer),
        sample_width=2,
        frame_rate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
    )
    audio.export("output.wav", format="wav")
    pydub.playback.play(audio)


async def main(audio_path: Path, server_url: str):
    if "openai.com" in server_url:
        additional_headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "OpenAI-Beta": "realtime=v1",
        }
        query_string = "model=gpt-4o-realtime-preview"
    else:
        additional_headers = {}
        query_string = ""

    async with websockets.connect(
        f"{server_url}/v1/realtime?{query_string}",
        additional_headers=additional_headers,
        subprotocols=[websockets.Subprotocol("realtime")],
    ) as websocket:
        send_task = asyncio.create_task(send_messages(websocket, audio_path))
        receive_task = asyncio.create_task(receive_messages(websocket))
        await asyncio.gather(send_task, receive_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default="wss://api.openai.com")
    parser.add_argument("audio_path", type=Path)
    args = parser.parse_args()

    asyncio.run(main(args.audio_path, server_url=args.server_url))
