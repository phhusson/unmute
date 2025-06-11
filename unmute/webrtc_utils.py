import os

import requests


def get_cloudflare_rtc_configuration():
    # see: https://fastrtc.org/deployment/#cloudflare-calls-api
    turn_key_id = os.environ.get("TURN_KEY_ID")
    turn_key_api_token = os.environ.get("TURN_KEY_API_TOKEN")
    ttl = 86400  # Can modify TTL, here it's set to 24 hours

    response = requests.post(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{turn_key_id}/credentials/generate-ice-servers",
        headers={
            "Authorization": f"Bearer {turn_key_api_token}",
            "Content-Type": "application/json",
        },
        json={"ttl": ttl},
    )
    if response.ok:
        return response.json()
    else:
        raise Exception(
            f"Failed to get TURN credentials: {response.status_code} {response.text}"
        )
