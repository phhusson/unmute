import numpy as np
from fastrtc import Stream, StreamHandler, get_hf_turn_credentials

from unmute.audio_stream_saver import AudioStreamSaver

SAMPLE_RATE = 24000
OUTPUT_FRAME_SIZE = 1920

# logging.basicConfig(level=logging.DEBUG)


class SineHandler(StreamHandler):
    def __init__(self) -> None:
        super().__init__(input_sample_rate=SAMPLE_RATE, output_frame_size=960)
        self.cur_time_samples = 0
        self.saver = AudioStreamSaver()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        pass

    def emit(self) -> tuple[int, np.ndarray]:
        times = np.arange(
            self.cur_time_samples,
            self.cur_time_samples + OUTPUT_FRAME_SIZE,
        )
        x = np.sin(2 * np.pi * 440 / SAMPLE_RATE * times) * 0.1
        x = x.astype(np.float32)
        self.cur_time_samples += OUTPUT_FRAME_SIZE

        self.saver.add(x)

        return (SAMPLE_RATE, x)

    def copy(self):
        return SineHandler()

    def shutdown(self):
        pass

    def start_up(self) -> None:
        pass


if __name__ == "__main__":
    # rtc_configuration = get_cloudflare_rtc_configuration()
    rtc_configuration = get_hf_turn_credentials()
    stream = Stream(
        handler=SineHandler(),
        modality="audio",
        mode="send-receive",
        rtc_configuration=rtc_configuration,
    )

    stream.ui.launch(debug=True)
