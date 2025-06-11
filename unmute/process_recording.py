"""Process a .msgpack recording for displaying on the project page.

The recording is done by recorder.py, which just records the JSON messages sent back
and forth between the client and the server.

This script converts the recording into a time-aligned format that's easier for
visualization.

It can also extract the audio from the recording.
"""

import argparse
import base64
import json
import logging
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import msgpack
import numpy as np
import sphn
from pydantic import BaseModel

from unmute import openai_realtime_api_events as ora
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.recorder import RecorderEvent
from unmute.tts.text_to_speech import prepare_text_for_tts

# Note this is not the ASR/TTS frame size; for that, see SAMPLES_PER_FRAME.
# It's the number of samples we get from the user per step.
SAMPLES_PER_STEP = 960
SAMPLES_PER_WAVEFORM = 240

logger = logging.getLogger(__name__)


class AudioFrame(BaseModel):
    amplitude_rms: list[float]
    n_samples: int
    created_at_samples: int

    def split(self, n_samples_start: int) -> tuple["AudioFrame", "AudioFrame"]:
        assert 0 < n_samples_start < self.n_samples, (
            f"{self.n_samples=}, {n_samples_start=}"
        )

        fraction = n_samples_start / self.n_samples
        amplitude_index_float = fraction * len(self.amplitude_rms)
        amplitude_index = int(amplitude_index_float)
        if amplitude_index_float - amplitude_index > 0.1:
            logger.warning(
                "Amplitude RMS split unevenly, "
                f"{fraction=}, {amplitude_index_float=} {len(self.amplitude_rms)}."
            )

        return (
            AudioFrame(
                amplitude_rms=self.amplitude_rms[:amplitude_index],
                n_samples=n_samples_start,
                created_at_samples=self.created_at_samples,
            ),
            AudioFrame(
                amplitude_rms=self.amplitude_rms[amplitude_index:],
                n_samples=self.n_samples - n_samples_start,
                created_at_samples=self.created_at_samples,
            ),
        )


class TextFrame(BaseModel):
    text: str
    created_at_samples: int
    # Using 0 as "unknown" is a bit hacky but it makes addition simpler
    duration_samples: int = 0


class AudioAndText(BaseModel):
    audio: AudioFrame | None = None
    text: TextFrame | None = None


class StepEvents(BaseModel):
    samples_since_start: int
    received: AudioAndText
    emitted: AudioAndText = AudioAndText(audio=None, text=None)
    other_events: list[ora.Event] = []


def get_audio_volume_rms(arr: np.ndarray) -> list[float]:
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / np.iinfo(np.int16).max

    if len(arr) % SAMPLES_PER_WAVEFORM != 0:
        raise ValueError(
            f"Array length {len(arr)} is not a multiple of SAMPLES_PER_WAVEFORM ({SAMPLES_PER_WAVEFORM})"
        )

    rms_list = []
    for i in range(0, len(arr), SAMPLES_PER_WAVEFORM):
        chunk = arr[i : i + SAMPLES_PER_WAVEFORM]
        rms = np.sqrt(np.mean(chunk**2))
        rms_list.append(rms)
    return rms_list


def round_to_multiple(value: float, multiple: int) -> int:
    """Round `value` to the nearest multiple of `multiple`."""
    return round(value / multiple) * multiple


def with_samples_since_start(
    recorder_events: list[RecorderEvent],
) -> Iterable[tuple[int, RecorderEvent]]:
    pass
    """Yield (timestamp_samples, recorder_event) pairs from the recorder events."""
    stream_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    samples_since_start = -SAMPLES_PER_STEP
    for recorder_event in recorder_events:
        ora_event = recorder_event.data

        if isinstance(ora_event, ora.InputAudioBufferAppend):
            audio_data = stream_reader.append_bytes(base64.b64decode(ora_event.audio))
            if not audio_data.size:
                logger.warning(
                    f"At {samples_since_start=}, received empty audio data. Skipping."
                )
                continue

            n = len(audio_data)
            if n != SAMPLES_PER_STEP:
                # Related to Opus. Seems to only happen at the beginning
                logger.warning(
                    f"At {samples_since_start=}, received audio data with {n} samples, "
                    f"expected {SAMPLES_PER_STEP} samples. Skipping."
                )
                continue

            samples_since_start += n
            yield samples_since_start, recorder_event
        else:
            yield samples_since_start, recorder_event


def process_events(recorder_events: list[RecorderEvent]) -> list[StepEvents]:
    step_events: dict[int, StepEvents] = {}
    # other_events for a given timestamp might be created before we've created the
    # corresponding step event, so collect them in a separate dict and then merge them
    other_events: defaultdict[int, list[ora.Event]] = defaultdict(list)

    # There are actually two levels of buffering, so use two queues
    tts_server_audio_queued: deque[AudioFrame] = deque()
    tts_client_audio_queued: deque[AudioFrame] = deque()

    tts_text_ready: deque[TextFrame] = deque()

    client_opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    server_opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    for samples_since_start, recorder_event in with_samples_since_start(
        recorder_events
    ):
        recorder_event = deepcopy(recorder_event)
        ora_event = recorder_event.data

        if isinstance(ora_event, ora.InputAudioBufferAppend):
            # Received audio from the client
            audio_data = client_opus_reader.append_bytes(
                base64.b64decode(ora_event.audio)
            )
            if not audio_data.size:
                continue

            assert samples_since_start not in step_events

            n = len(audio_data)

            step_events[samples_since_start] = StepEvents(
                samples_since_start=samples_since_start,
                received=AudioAndText(
                    audio=AudioFrame(
                        amplitude_rms=get_audio_volume_rms(audio_data),
                        n_samples=n,
                        # For received audio, the creation time is the same as the
                        # receive time.
                        # We add SAMPLES_PER_STEP as a hack so that the waveform
                        # visualization of the received audio doesn't show parts of the
                        # audio as being created but not received yet (because the step
                        # is shown as multiple rectangles since amplitude_rms is a list)
                        created_at_samples=samples_since_start + SAMPLES_PER_STEP,
                    ),
                ),
                emitted=AudioAndText(audio=None, text=None),
            )

            if tts_client_audio_queued:
                audio = tts_client_audio_queued.popleft()
                if audio.n_samples == n:
                    step_events[samples_since_start].emitted.audio = audio
                elif audio.n_samples > n:
                    head, tail = audio.split(n)
                    step_events[samples_since_start].emitted.audio = head
                    tts_client_audio_queued.appendleft(tail)
                else:
                    raise RuntimeError(
                        "Unexpected: output audio frame size is not "
                        "a multiple of the input frame size. "
                        f"{n=}, {audio.n_samples=}"
                    )
        elif isinstance(ora_event, ora.UnmuteResponseAudioDeltaReady):
            tts_server_audio_queued.append(
                AudioFrame(
                    amplitude_rms=[],  # Will be set later
                    n_samples=ora_event.number_of_samples,
                    created_at_samples=samples_since_start,
                )
            )
        elif isinstance(ora_event, ora.ResponseAudioDelta):
            # The server emitted TTS audio that it queued up previously

            audio_data = server_opus_reader.append_bytes(
                base64.b64decode(ora_event.delta)
            )
            assert audio_data.size > 0, "Received empty audio delta"

            if not tts_server_audio_queued:
                # Not sure why this happens, maybe some off-by one? Something related
                # to Opus?
                logger.warning(
                    f"Received TTS audio delta at timestamp {samples_since_start} "
                    "but no audio frame was queued on the server side."
                )
                continue

            # Move from the server-side queue to the client-side queue
            assert tts_server_audio_queued[0].n_samples == len(audio_data)
            audio_frame = tts_server_audio_queued.popleft()
            audio_frame.amplitude_rms = get_audio_volume_rms(audio_data)
            tts_client_audio_queued.append(audio_frame)
        elif isinstance(ora_event, ora.UnmuteResponseTextDeltaReady):
            tts_text_ready.append(
                TextFrame(
                    text=ora_event.delta,
                    created_at_samples=samples_since_start,
                    duration_samples=0,  # We don't know yet
                )
            )
            other_events[samples_since_start].append(ora_event)
        elif isinstance(ora_event, ora.ResponseTextDelta):
            assert tts_text_ready
            prepared_text = tts_text_ready.popleft()
            assert ora_event.delta == prepare_text_for_tts(prepared_text.text), (
                f"Expected TTS text delta to be '{prepared_text.text}', "
                f"but got '{ora_event.delta}'"
            )
            step_events[samples_since_start].emitted.text = prepared_text
        elif isinstance(ora_event, ora.ConversationItemInputAudioTranscriptionDelta):
            # The STT transcribed something in the past, so we need to compute the
            # timestamp and retroactively add it to the existing step event
            ts_in_question = round_to_multiple(
                ora_event.start_time * SAMPLE_RATE, SAMPLES_PER_STEP
            )
            assert step_events[ts_in_question].received.text is None

            step_events[ts_in_question].received.text = TextFrame(
                text=ora_event.delta,
                created_at_samples=samples_since_start,
                duration_samples=0,  # We don't know
            )
        elif isinstance(ora_event, ora.ResponseCreated):
            # There might be text that the TTS queued up before but it got interrupted
            # before it could be emitted, so remove that text when we start generating
            # a new response.
            tts_text_ready.clear()
            other_events[samples_since_start].append(ora_event)
        else:
            ignored_event_types = [ora.UnmuteAdditionalOutputs]
            if not isinstance(ora_event, tuple(ignored_event_types)):
                other_events[samples_since_start].append(ora_event)

    # Merge other_events into step_events
    for samples_since_start, step_event in step_events.items():
        step_event.other_events = other_events[samples_since_start]

    step_events_list = list(step_events.values())
    step_events_list.sort(key=lambda x: x.samples_since_start)

    # Sanity checks
    samples_per_step = (
        step_events_list[1].samples_since_start
        - step_events_list[0].samples_since_start
    )
    for i, step_event in enumerate(step_events_list):
        assert step_event.samples_since_start == i * samples_per_step

    assert step_events_list[0].samples_since_start == 0

    return step_events_list


def slice_processed_events(
    processed_events: list[StepEvents], start_samples: int
) -> list[StepEvents]:
    filtered = [
        # Copy because we'll be modifying the events later
        deepcopy(event)
        for event in processed_events
        if start_samples <= event.samples_since_start
    ]

    # Fix the timestamps
    for event in filtered:
        event.samples_since_start -= start_samples
        if event.received.audio:
            event.received.audio.created_at_samples -= start_samples
        if event.received.text:
            event.received.text.created_at_samples -= start_samples
        if event.emitted.audio:
            event.emitted.audio.created_at_samples -= start_samples
        if event.emitted.text:
            event.emitted.text.created_at_samples -= start_samples

    return filtered


def extract_audios(
    recorder_events: list[RecorderEvent],
) -> np.ndarray:
    """Return a 2d NumPy array containing user and assistant audio.

    User audio is on the first channel, assistant audio is on the second channel.
    They are time-aligned and trimmed
    """
    user_pcm_chunks = []
    user_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    user_n_samples = 0

    assistant_pcm_chunks = []
    assistant_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    assistant_n_samples = 0

    for e in recorder_events:
        if isinstance(e.data, ora.InputAudioBufferAppend):
            pcm = user_reader.append_bytes(base64.b64decode(e.data.audio))
            user_pcm_chunks.append(pcm)
            user_n_samples += len(pcm)

        elif isinstance(e.data, ora.ResponseAudioDelta):
            pcm = assistant_reader.append_bytes(base64.b64decode(e.data.delta))
            assistant_pcm_chunks.append(pcm)
            assistant_n_samples += len(pcm)

        # The assistant is not emitting audio all the time, so add silence so that the
        # lengths match
        if user_n_samples > assistant_n_samples:
            assistant_pcm_chunks.append(
                np.zeros(user_n_samples - assistant_n_samples, dtype=np.float32)
            )
            assistant_n_samples = user_n_samples

    user_audio = np.concatenate(user_pcm_chunks)
    assistant_audio = np.concatenate(assistant_pcm_chunks)
    length = max(len(user_audio), len(assistant_audio))

    def pad(audio: np.ndarray):
        """Pad the audio to the given length with zeros."""
        if len(audio) < length:
            return np.pad(audio, (0, length - len(audio)), mode="constant")
        return audio

    return np.array([pad(user_audio), pad(assistant_audio)])


def main(
    input_path: Path,
    output_path: Path,
    audio_output_path: Path | None,
    discard_first_assistant_message: bool = False,
):
    with input_path.open("rb") as f:
        events_raw = msgpack.load(f)
        recorder_events = [RecorderEvent(**e) for e in events_raw]

    processed = process_events(recorder_events)

    slice_from_sample = 0
    if discard_first_assistant_message:
        user_speech_start = None
        for e in processed:
            if e.received.text is not None:
                user_speech_start = e
                break

        assert user_speech_start is not None, "No user speech found in the recording."

        padding_samples = SAMPLE_RATE * 0.2  # A bit arbitrary here
        slice_from_sample = user_speech_start.samples_since_start - int(padding_samples)

    if slice_from_sample > 0:
        processed = slice_processed_events(processed, slice_from_sample)

    with open(output_path, "w") as f:
        json.dump([e.model_dump() for e in processed], f, indent=2)
        len_sec = len(processed) * SAMPLES_PER_STEP / SAMPLE_RATE
        print(
            f"Saved processed recording with {len(processed)} steps ({len_sec:.1f}s) "
            f"to {output_path}"
        )

    if audio_output_path is not None:
        audio = extract_audios(recorder_events)
        audio = np.mean(audio, axis=0)  # Combine channels into one
        audio = audio[slice_from_sample:]

        sphn.write_opus(audio_output_path, audio, SAMPLE_RATE)
        print(
            f"Saved {len(audio) / SAMPLE_RATE:.1f}s of user and assistant audio "
            f"to {audio_output_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path", type=Path, help="The .msgpack file of the raw recording"
    )
    parser.add_argument(
        "output_path", type=Path, help="The path to which the output JSON will be saved"
    )
    parser.add_argument(
        "--discard-first-assistant-message",
        action="store_true",
    )
    parser.add_argument(
        "--audio-output-path",
        type=Path,
        help="Save the combined audio to this path. Supports .ogg and .wav.",
    )
    args = parser.parse_args()

    main(
        args.input_path,
        args.output_path,
        args.audio_output_path,
        args.discard_first_assistant_message,
    )
