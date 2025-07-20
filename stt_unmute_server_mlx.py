# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.12",
#     "numpy",
#     "msgpack",
#     "uvicorn",
#     "mlx",
#     "websockets",
#     "fastrtc",
#     "rustymimi",
#     "sentencepiece",
#     "fastapi",
#     "sounddevice",
# ]
# ///


import asyncio
import logging
import random
import json

import msgpack
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from unmute.kyutai_constants import SAMPLE_RATE, SAMPLES_PER_FRAME

import queue

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
import numpy as np
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

TEXT_TO_SPEECH_PATH = "/api/asr-streaming"

app = FastAPI()

logger = logging.getLogger(__name__)



# Copy/pasted from unmute/stt/speech_to_text.py {
from typing import Literal
from pydantic import BaseModel
class STTWordMessage(BaseModel):
    type: Literal["Word"]
    text: str
    start_time: float


class STTEndWordMessage(BaseModel):
    type: Literal["EndWord"]
    stop_time: float


class STTMarkerMessage(BaseModel):
    type: Literal["Marker"]
    id: int


class STTStepMessage(BaseModel):
    type: Literal["Step"]
    step_idx: int
    prs: list[float]


class STTErrorMessage(BaseModel):
    type: Literal["Error"]
    message: str


class STTReadyMessage(BaseModel):
    type: Literal["Ready"]
#}


hf_repo = 'kyutai/stt-1b-en_fr-candle'

lm_config = hf_hub_download(hf_repo, "config.json")
with open(lm_config, "r") as fobj:
    lm_config = json.load(fobj)
mimi_weights = hf_hub_download(hf_repo, lm_config["mimi_name"])
moshi_name = lm_config.get("moshi_name", "model.safetensors")
moshi_weights = hf_hub_download(hf_repo, moshi_name)
tokenizer = hf_hub_download(hf_repo, lm_config["tokenizer_name"])

lm_config = models.LmConfig.from_config_dict(lm_config)
model = models.Lm(lm_config)
model.set_dtype(mx.bfloat16)
if moshi_weights.endswith(".q4.safetensors"):
    nn.quantize(model, bits=4, group_size=32)
elif moshi_weights.endswith(".q8.safetensors"):
    nn.quantize(model, bits=8, group_size=64)

model.load_pytorch_weights(moshi_weights, lm_config, strict=True)

nn.quantize(model.depformer, bits=4)
for layer in model.transformer.layers:
    nn.quantize(layer.self_attn, bits=4)
    nn.quantize(layer.gating, bits=4)

text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)
generated_codebooks = lm_config.generated_codebooks
other_codebooks = lm_config.other_codebooks
mimi_codebooks = max(generated_codebooks, other_codebooks)
audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore
print("warming up the model")
model.warmup()

blocksize = 1920
samplerate = 24000

@app.get("/api/build_info")
def get_build_info():
    return {"note": "moshi-mlx"}

async def send(ws: WebSocket, data: dict) -> None:
    to_send = msgpack.packb(data, use_bin_type=True, use_single_float=True)
    await ws.send_bytes(to_send)

@app.websocket(TEXT_TO_SPEECH_PATH)
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await send(websocket, {"type":"Ready"})
    gen = models.LmGen(
        model=model,
        max_steps=4096,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    try:
        current_time = 0.0

        buffer = []
        # Note this implementation is completely blocking
        current_word = ""
        step = 0
        while True:
            msg = await websocket.receive_bytes()
            msg = msgpack.unpackb(msg)
            if msg['type'] == 'Audio':
                buffer += msg['pcm']
                current_time += len(msg['pcm']) / samplerate
            else:
                print(json.dumps(msg, indent=2))
            while len(buffer) >= blocksize:
                step += 1
                block = buffer[:blocksize]
                buffer = buffer[blocksize:]
                block = np.array([block], dtype='f')
                other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
                other_audio_tokens = mx.array(other_audio_tokens).transpose(0,2,1)[:,:,:other_codebooks]
                text_token, vad_heads = gen.step_with_extra_heads(other_audio_tokens[0])
                text_token = text_token[0].item()
                audio_tokens = gen.last_audio_tokens()
                _text = None
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token)
                    _text = _text.replace("▁", " ")
                    # I think this start_time is wrong? idk
                    # Also we don't send EndWord (not used on unmute currently?)
                    if current_word and _text.startswith(" "):
                        await send(websocket, {"type":"Word", "text": current_word, "start_time": current_time})
                        current_word = ''
                    current_word += _text
                if text_token == 3 and current_word:
                    await send(websocket, {"type":"Word", "text": current_word, "start_time": current_time})
                    current_word = ''

                pr_vad = [x[0, 0, 0].item() for x in vad_heads]
                await send(websocket, {"type":"Step", "step_idx": step, "prs": pr_vad})


    except WebSocketDisconnect:
        print("Client disconnected")
        return

    await websocket.close()


if __name__ == "__main__":
    import sys
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8090)
