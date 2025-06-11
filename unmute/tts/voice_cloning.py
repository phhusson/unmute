import logging
import uuid

import requests

from unmute.cache import get_cache
from unmute.kyutai_constants import VOICE_CLONING_SERVER

logger = logging.getLogger(__name__)


voice_embeddings_cache = get_cache(prefix="voice", ttl_seconds=60 * 60 * 1)  # 1 hour


def clone_voice(audio_data: bytes) -> str:
    # Generate a unique voice name
    voice_name = "custom:" + str(uuid.uuid4())

    # Call the voice cloning server
    response = requests.post(
        f"{VOICE_CLONING_SERVER}/api/voice",
        data=audio_data,
        headers={"Content-Type": "application/octet-stream"},
    )
    response.raise_for_status()
    msgpack_data = response.content

    logger.info(f"Received voice embedding of size: {len(msgpack_data)} bytes")

    voice_embeddings_cache.set(voice_name, msgpack_data)
    voice_embeddings_cache.cleanup()

    return voice_name
