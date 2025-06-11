import argparse
import asyncio
from pathlib import Path

import numpy as np
import sphn
import tqdm

from unmute.loadtest.loadtest_client import preview_audio
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSClientEosMessage,
    TTSTextMessage,
)
from unmute.tts.voice_cloning import clone_voice


async def main(voice_file: Path | None):
    if voice_file:
        voice = clone_voice(voice_file.read_bytes())
    else:
        voice = None

    tts = TextToSpeech(voice=voice)
    await tts.start_up()

    for _ in range(10):
        await tts.send("hello")
        await asyncio.sleep(0.1)

    await tts.send(TTSClientEosMessage())

    audio_chunks = []
    n_words = 0

    with tqdm.tqdm() as pbar:
        async for msg in tts:
            if isinstance(msg, TTSTextMessage):
                pbar.set_postfix(n_words=n_words)
                n_words += 1
            else:
                audio_chunks.append(msg.pcm)
                pbar.update(len(msg.pcm) / 24000)

    all_audio = np.concat(audio_chunks).astype(np.float32)
    preview_audio(all_audio)

    output_path = "out.wav"
    sphn.write_wav(output_path, all_audio, 24000)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice-file",
        type=Path,
    )
    args = parser.parse_args()

    asyncio.run(main(voice_file=args.voice_file))
