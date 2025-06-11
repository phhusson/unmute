# https://github.com/gabrielchua/async-stream-openai-st/blob/824eab8f3ab600d3689d8d946526e48e0e0310c2/app.py
# https://qwen.readthedocs.io/en/latest/deployment/vllm.html#openai-compatible-api-service

from openai import AsyncOpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8111/v1"

# Predefined message
PREDEFINED_MESSAGE = "Explain the second law of thermodynamics"


async def main():
    # Initialize the async OpenAI client
    client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)

    stream = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Write a {200} word essay on 'bear vs shark'. "
                "The first line is a 2-3 word title with an emoji and then include "
                "2 line breaks. For example 'TITLE <emoji> \n \n ' ",
            },
        ],
        stream=True,
    )
    streamed_text = "# "
    async for chunk in stream:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content is not None:
            streamed_text = streamed_text + chunk_content
            print(chunk_content, end="", flush=True)

    print("\n\n\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
