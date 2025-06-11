from collections import deque

import librosa
import numpy as np
from fastrtc import Stream, StreamHandler

from unmute.audio_stream_saver import AudioStreamSaver

SAMPLE_RATE = 24000
OUTPUT_FRAME_SIZE = 1920


class PitchDetectionHandler(StreamHandler):
    def __init__(self) -> None:
        super().__init__(input_sample_rate=SAMPLE_RATE, output_frame_size=480)
        self.cur_time_samples = 0
        self.saver = AudioStreamSaver()
        self.frequency_queue = deque()
        self.last_phase = 0
        self.last_frequency = 100

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        mono_audio = frame[1][0]
        assert mono_audio.dtype == np.int16
        mono_audio = mono_audio.astype(np.float32) / np.iinfo(np.int16).max

        freqs = librosa.yin(
            mono_audio,
            fmin=float(librosa.note_to_hz("E2")),
            fmax=float(librosa.note_to_hz("E5")),
            sr=SAMPLE_RATE,
            frame_length=len(mono_audio),
            hop_length=len(mono_audio),
            center=False,
        )
        assert len(freqs) == 1
        self.frequency_queue.append(freqs[0])

    def emit(self) -> tuple[int, np.ndarray] | None:
        if not self.frequency_queue:
            return None
        else:
            frequency = self.frequency_queue.popleft()

        phase = self.last_phase + np.cumsum(
            np.linspace(self.last_frequency, frequency, OUTPUT_FRAME_SIZE) / SAMPLE_RATE
        )
        self.last_phase = phase[-1] % 1.0
        amplitude = 0.1
        x = np.sin(2 * np.pi * phase) * amplitude
        x = x.astype(np.float32)

        self.cur_time_samples += OUTPUT_FRAME_SIZE
        self.saver.add(x)
        self.last_frequency = frequency

        return (SAMPLE_RATE, x)

    def copy(self):
        return PitchDetectionHandler()


if __name__ == "__main__":
    stream = Stream(
        handler=PitchDetectionHandler(), modality="audio", mode="send-receive"
    )

    stream.ui.launch(debug=True)
