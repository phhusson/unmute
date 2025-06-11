import os

from mistralai import Mistral

if __name__ == "__main__":
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable must be set")

    model = "mistral-small-latest"

    client = Mistral(api_key=mistral_api_key)

    res = client.chat.stream(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Keep your responses to at most a few sentences. "
                "They will be spoken out loud, so don't worry about formatting. "
                "Write as a human would speak.",
            },
            {
                "role": "user",
                "content": "What is the best French cheese?",
            },
        ],
    )

    with res as event_stream:
        for event in event_stream:
            content = event.data.choices[0].delta.content
            print(content, flush=True, end="")

    print("")
