"""Check that a Hugging Face token does not have write access."""

import argparse

import requests


def abbreviate_token(token: str) -> str:
    """Abbreviate the token for display."""
    assert len(token) > 10
    return f"{token[:4]}...{token[-4:]}"


def main(token: str):
    response = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    response.raise_for_status()

    # Example response:
    # {
    #     [...]
    #     "auth": {
    #         "type": "access_token",
    #         "accessToken": {
    #             "displayName": "foo",
    #             "role": "write",
    #             "createdAt": "2025-03-18T10:40:56.186Z"
    #         }
    #     }
    # }

    data = response.json()
    if data["auth"]["type"] != "access_token":
        raise ValueError(f"Unexpected auth type: {data['auth']['type']}.")

    role = data["auth"]["accessToken"]["role"]
    if role == "fineGrained":
        # Harder to test. As a heuristic, just look for "write" somewhere in the JSON.
        if "write" in str(data["auth"]["accessToken"]["fineGrained"]).lower():
            raise ValueError(
                "The provided fine-grained Hugging Face token "
                f"{abbreviate_token(token)} has write access. "
                "Use a read-only token to deploy. "
                "It has the following permissions: "
                f"{data['auth']['accessToken']['fineGrained']}"
            )
    elif role == "write":
        raise ValueError(
            f"The provided Hugging Face token {abbreviate_token(token)} has write "
            "access. Use a read-only token to deploy."
        )
    else:
        if role != "read":
            raise ValueError(
                f"Unknown token role: {role}. Use a read-only token to deploy."
            )

    print("Ok, Hugging Face token has no write access.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check that the Hugging Face token does not have write access. "
        "This is because we don't want the deployed containers to have write access "
        "in case they are compromised. "
        "Exits with non-zero exit code if the token has write access or something else "
        "goes wrong."
    )
    parser.add_argument("token", type=str, help="Hugging Face token to check. ")

    args = parser.parse_args()
    main(args.token)
