from typing import Literal

WebsocketState = Literal["not_created", "connecting", "connected", "closing", "closed"]


def http_to_ws(url_string: str):
    """
    Converts an HTTP(S) URL string to a WebSocket (WS/WSS) URL string.

    Args:
        url_string: The input URL string starting with http:// or https://.

    Returns:
        The corresponding WebSocket URL string starting with ws:// or wss://.
        Returns the original string if it doesn't start with http:// or https://.
    """
    if url_string.startswith("http://"):
        return "ws://" + url_string[7:]
    elif url_string.startswith("https://"):
        return "wss://" + url_string[8:]
    else:
        return url_string


def ws_to_http(url_string: str):
    """
    Converts a WebSocket (WS/WSS) URL string to an HTTP(S) URL string.

    Args:
        url_string: The input URL string starting with ws:// or wss://.

    Returns:
        The corresponding HTTP URL string starting with http:// or https://.
        Returns the original string if it doesn't start with ws:// or wss://.
    """
    if url_string.startswith("ws://"):
        return "http://" + url_string[5:]
    elif url_string.startswith("wss://"):
        return "https://" + url_string[6:]
    else:
        return url_string
