from logging import getLogger
from pathlib import Path

import numpy as np
import sphn

from unmute.kyutai_constants import SAMPLE_RATE

DEBUG_DIR = Path(__file__).parents[1] / "debug"
logger = getLogger(__name__)


class AudioStreamSaver:
    """Collect and save an audio stream. For debugging"""

    def __init__(
        self,
        interval_sec: float = 1.0,
        output_path: str | Path | None = None,
        max_saves: int | None = 1,
    ):
        self.interval_sec = interval_sec
        self.max_saves = max_saves
        self.n_saves_done = 0

        if output_path is None:
            self.output_path = DEBUG_DIR / "out.wav"
        else:
            self.output_path = Path(output_path)

        self.buffer = []

    def add(self, audio_chunk: np.ndarray):
        """Add a chunk of audio. Save if we've collected enough."""
        if self.max_saves is not None and self.n_saves_done >= self.max_saves:
            return

        assert audio_chunk.dtype == np.float32
        assert audio_chunk.ndim == 1

        self.buffer.append(audio_chunk)

        if sum(len(x) for x in self.buffer) / SAMPLE_RATE >= self.interval_sec:
            output_path = self.output_path
            if self.max_saves != 1:  # None is ok too
                output_path = output_path.with_stem(
                    output_path.stem + f"_{self.n_saves_done + 1}"
                )

            sphn.write_wav(
                output_path,
                np.concatenate(self.buffer).astype(np.float32),
                SAMPLE_RATE,
            )
            self.n_saves_done += 1
            self.buffer.clear()
            logger.info(f"Saved audio to {output_path}")
