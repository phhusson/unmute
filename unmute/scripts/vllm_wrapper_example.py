# https://github.com/gabrielchua/async-stream-openai-st/blob/824eab8f3ab600d3689d8d946526e48e0e0310c2/app.py
# https://qwen.readthedocs.io/en/latest/deployment/vllm.html#openai-compatible-api-service

from typing import Any, cast

from unmute.llm.llm_utils import VLLMStream, rechunk_to_words

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8111/v1"

# Predefined message
PREDEFINED_MESSAGE = "Explain the second law of thermodynamics"


async def main():
    s = VLLMStream()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Write a {200} word essay on 'bear vs shark'. "
            "The first line is a 2-3 word title with an emoji and then include "
            "2 line breaks. For example 'TITLE <emoji> \n \n ' ",
        },
    ]

    async for message in rechunk_to_words(s.chat_completion(cast(Any, messages))):
        print(message, end="\n", flush=True)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
