import argparse
import asyncio
from pathlib import Path

import numpy as np
import sphn
import tqdm

from unmute.loadtest.loadtest_client import preview_audio
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
    TTSTextMessage,
)
from unmute.tts.voice_cloning import clone_voice


async def main(
    text: str, voice_file: Path | None = None, output_path: Path | None = None
):
    if voice_file:
        voice = clone_voice(voice_file.read_bytes())
    else:
        voice = None

    tts = TextToSpeech(voice=voice)
    await tts.start_up()

    for word in text.split(" "):
        await tts.send(word)
        await asyncio.sleep(0.1)

    await tts.send(TTSClientEosMessage())

    audio_chunks = []
    n_words = 0

    with tqdm.tqdm() as pbar:
        async for msg in tts:
            if isinstance(msg, TTSTextMessage):
                pbar.set_postfix(n_words=n_words)
                n_words += 1
            elif isinstance(msg, TTSAudioMessage):
                audio_chunks.append(msg.pcm)
                pbar.update(len(msg.pcm) / 24000)

    all_audio = np.concat(audio_chunks).astype(np.float32)
    preview_audio(all_audio)

    sphn.write_wav(output_path, all_audio, 24000)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice-file",
        type=Path,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("out.wav"),
        help="Path to save the audio to, .wav or .ogg file (default: out.wav)",
    )
    parser.add_argument(
        "text",
        type=str,
        nargs="?",
        default="Did you know that the author of Octavia "
        "based one character on a former lover?",
    )
    args = parser.parse_args()

    asyncio.run(
        main(args.text, voice_file=args.voice_file, output_path=args.output_path)
    )
