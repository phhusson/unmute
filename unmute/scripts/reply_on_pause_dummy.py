import asyncio

import numpy as np
from fastrtc import ReplyOnPause, Stream


async def response(audio: tuple[int, np.ndarray]):  #
    chunk_size = 1920
    n_chunks = 50
    n_samples = chunk_size * n_chunks
    sr = 24000

    data = np.linspace(0, n_samples / sr, n_samples, endpoint=False)
    sine_wave = np.sin(2 * np.pi * 440 * data, dtype=np.float32) * 0.1

    for chunk in range(n_chunks):
        yield (sr, sine_wave[chunk * chunk_size : (chunk + 1) * chunk_size])
        await asyncio.sleep(1)


stream = Stream(handler=ReplyOnPause(response), modality="audio", mode="send-receive")

stream.ui.launch(debug=True)
