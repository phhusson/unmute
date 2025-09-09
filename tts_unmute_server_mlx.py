# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "numpy",
#     "msgpack",
#     "uvicorn",
#     "coremltools",
#     "mlx",
#     "websockets",
#     "fastrtc>=0.0.32",
#     "rustymimi",
#     "sentencepiece",
#     "fastapi",
#     "sounddevice",
# ]
# ///

import asyncio
import json
import msgpack
import argparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dataclasses import dataclass
import json
import queue
import sys
import time
import librosa
import threading
import concurrent.futures
import cProfile

import mlx
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sounddevice as sd
import sphn
import typing as tp
from moshi_mlx import models
from moshi_mlx.models.generate import LmGen
from moshi_mlx.client_utils import make_log
from moshi_mlx.modules.conditioner import (
    ConditionAttributes,
    ConditionTensor,
    dropout_all_conditions,
)
from moshi_mlx.utils.sampling import Sampler
from moshi_mlx.models.tts import (
    Entry,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    script_to_entries,
)
from moshi_mlx.utils.loaders import hf_get


def prepare_script(model: TTSModel, script: str, first_turn: bool) -> list[Entry]:
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


def _make_null(
    all_attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)
@dataclass
class TTSGen:
    tts_model: TTSModel
    attributes: tp.Sequence[ConditionAttributes]
    on_frame: tp.Optional[tp.Callable[[mx.array], None]] = None

    def __post_init__(self):
        self.lastts = None
        tts_model = self.tts_model
        attributes = self.attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])
        self.times = []

        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        self.ct = None
        self.cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if self.ct is None:
                    self.ct = _ct
                else:
                    self.ct = ConditionTensor(self.ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if self.cross_attention_src is None:
                    self.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        def _on_audio_hook(audio_tokens):
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=tts_model.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            cfg_coef=tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            # TODO(laurent):
            # cfg_is_masked_until=cfg_is_masked_until,
            # cfg_is_no_text=cfg_is_no_text,
        )

    async def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            await self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            await self._step()

    async def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            await self._step()

    def __step_sync(self):
        startts = time.time()
        if self.lastts:
            print("AA", time.time() - self.lastts)
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )
        self.lm_gen.step(
            input_tokens, ct=self.ct, cross_attention_src=self.cross_attention_src
        )
        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1
        self.times.append(time.time() - startts)
        print("AZ", time.time() - startts)
        if len(self.times) > 2:
            print("	A%", sum(self.times[-20:]) / len(self.times[-20:]))
        self.lastts = time.time()
        return frame

    async def _step(self):
        if len(self.times) == 35:
            #mx.metal.start_capture("mlx_trace.gputrace")
            frame = await asyncio.to_thread(self.__step_sync)
            #mx.metal.stop_capture()
        else:
            frame = await asyncio.to_thread(self.__step_sync)

        if frame is not None:
            if self.on_frame is not None:
                await self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


def log(level: str, msg: str):
    print(make_log(level, msg))


parser = argparse.ArgumentParser(
    description="Run Kyutai TTS using the MLX implementation"
)
parser.add_argument(
    "--hf-repo",
    type=str,
    default=DEFAULT_DSM_TTS_REPO,
    help="HF repo in which to look for the pretrained models.",
)
parser.add_argument(
    "--voice-repo",
    default=DEFAULT_DSM_TTS_VOICE_REPO,
    help="HF repo in which to look for pre-computed voice embeddings.",
)
parser.add_argument(
    "--voice", default="expresso/ex03-ex01_happy_001_channel1_334s.wav"
)
parser.add_argument(
    "--quantize",
    type=int,
    help="The quantization to be applied, e.g. 8 for 8 bits.",
)
args = parser.parse_args()

mx.random.seed(299792458)

log("info", "retrieving checkpoints")

raw_config = hf_get("config.json", args.hf_repo)
with open(hf_get(raw_config), "r") as fobj:
    raw_config = json.load(fobj)

mimi_weights = hf_get(raw_config["mimi_name"], args.hf_repo)
moshi_name = raw_config.get("moshi_name", "model.safetensors")
moshi_weights = hf_get(moshi_name, args.hf_repo)
tokenizer = hf_get(raw_config["tokenizer_name"], args.hf_repo)
lm_config = models.LmConfig.from_config_dict(raw_config)
model = models.Lm(lm_config)

log("info", f"loading model weights from {moshi_weights}")
model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)
# bfloat16 & float16 does 92ms
# float32 does 83ms
# In Metal profiling there are a lot of copybfloat16float32
# I'm guessing some operations aren't supported in bfloat16 
model.set_dtype(mx.float32)
#model.set_dtype(mx.bfloat16)

log("info", f"quantizing model to {args.quantize} bits")
# Note: pruning doesn't support already quantized layers
for x in model.depformer.slices:
    nn.quantize(x.linear_in, bits=4)
    nn.quantize(x.linear_out, bits=4)
    nn.quantize(x.emb, bits=4)
    for l in x.transformer.layers:
        nn.quantize(l.gating, bits=4)
        l.self_attn.in_proj = l.self_attn.in_proj.to_quantized(group_size=64, bits = 4)
        l.self_attn.out_proj = l.self_attn.out_proj.to_quantized(group_size=64, bits = 4)
        if l.cfg.cross_attention:
            l.cross_attn.quantize()

# model.transformer is the "voice-cloning" part, it is very small, there isn't much gain
for layer in model.transformer.layers:
    nn.quantize(layer.gating, bits=4)
    layer.self_attn.in_proj = layer.self_attn.in_proj.to_quantized(group_size=64, bits = 4)
    layer.self_attn.out_proj = layer.self_attn.out_proj.to_quantized(group_size=64, bits = 4)

    if layer.cfg.cross_attention:
        layer.cross_attention.quantize()
    if layer.self_attn.cfg.positional_embedding == 'rope':
        nn.quantize(layer.self_attn.rope, bits=4)
print(model.leaf_modules())

log("info", f"loading the text tokenizer from {tokenizer}")
text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))  # type: ignore

log("info", f"loading the audio tokenizer {mimi_weights}")
generated_codebooks = lm_config.generated_codebooks
audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)
del audio_tokenizer.encoder
del audio_tokenizer.decoder
del audio_tokenizer.encoder_transformer
del audio_tokenizer.decoder_transformer
#model.set_dtype(mx.bfloat16)
#nn.quantize(audio_tokenizer, bits=8)

cfg_coef_conditioning = None
tts_model = TTSModel(
    model,
    audio_tokenizer,
    text_tokenizer,
    voice_repo=args.voice_repo,
    temp=0.6,
    cfg_coef=1,
    max_padding=8,
    initial_padding=2,
    final_padding=2,
    padding_bonus=-2,
    raw_config=raw_config,
)
if tts_model.valid_cfg_conditionings:
    # Model was trained with CFG distillation.
    cfg_coef_conditioning = tts_model.cfg_coef
    tts_model.cfg_coef = 1.0
mimi = tts_model.mimi

log("info", "reading input from stdin")

if tts_model.multi_speaker:
    #voices = [tts_model.get_voice_path(args.voice)]
    voices = [tts_model.get_voice_path('unmute-prod-website/developpeuse-3.wav')]
else:
    voices = []
all_attributes = [
    tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
]

async def _on_frame(frame):
    if (frame == -1).any():
        return
    _pcm = tts_model.mimi.decode_step(frame[:, :, None])
    _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1))

gen = TTSGen(tts_model, all_attributes, on_frame=_on_frame)

app = FastAPI()


@app.get("/api/build_info")
def get_build_info():
    return {"note": "moshi-mlx"}

async def send(ws: WebSocket, data: dict) -> None:
    to_send = msgpack.packb(data, use_bin_type=True, use_single_float=True)
    await ws.send_bytes(to_send)



@app.websocket("/api/tts_streaming")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await send(websocket, {"type": "Ready"})
    print("A")
    audio_tokenizer.reset_state()
    model.reset_state()

    q = asyncio.Queue()
    async def tts_coroutine():
        startts = time.time()
        async def _on_frame(frame):
            if (frame == -1).any():
                return
            _pcm = tts_model.mimi.decode_step(frame[:, :, None])
            _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1)).tolist()
            print("Sending audio")
            audio_message = {"type": "Audio", "pcm": _pcm}
            await websocket.send_bytes(msgpack.packb(audio_message))
            print("Done sending audio")

        print("Start of tts_corountine")
        gen = TTSGen(tts_model, all_attributes, on_frame=_on_frame)
        while True:
            print("Getting element from queue...")
            entry = await q.get()
            print("Got from queue", entry)
            if not entry:
                break
            print("Entry", entry)
            gen.append_entry(entry)
            await gen.process()
            text_message = {"type": "Text", "text": entry.text, "start_s": time.time() - startts, "stop_s" : time.time() - startts + 0.2}
            await websocket.send_bytes(msgpack.packb(text_message))
        await gen.process_last()

    async def websocket_receive_coroutine():
        first_turn = True
        while True:
            print("D")
            message = await websocket.receive()
            print("msg", message)
            if message['type'] == 'websocket.disconnect':
                return
            message = msgpack.unpackb(message['bytes'])
            print("msg", message)
            if message['type'] == 'Text':
                entries = prepare_script(tts_model, message['text'], first_turn=first_turn)
                for entry in entries:
                    await q.put(entry)
                first_turn = False
            if message['type'] == 'Eos':
                await q.put(None)
                return
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(websocket_receive_coroutine())
        task2 = tg.create_task(tts_coroutine())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = '0.0.0.0', port=8091)
