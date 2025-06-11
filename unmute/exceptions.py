import unmute.openai_realtime_api_events as ora


class NoTracebackError(Exception):
    """Error for which no traceback should be shown."""


class MissingServiceError(Exception):
    """Some service is missing, we might expect this to happen or not."""


class MissingServiceAtCapacity(NoTracebackError, MissingServiceError):
    """A service is operating at capacity, but no serious error."""

    def __init__(self, service: str):
        self.service = service
        super().__init__(f"{service} is not available.")


class MissingServiceTimeout(NoTracebackError, MissingServiceError):
    """A service timed out."""

    def __init__(self, service: str):
        self.service = service
        super().__init__(f"{service} timed out.")


class WebSocketClosedError(NoTracebackError):
    """Remote web socket is closed, let's move on."""


def make_ora_error(type: str, message: str) -> ora.Error:
    details = ora.ErrorDetails(type=type, message=message)
    return ora.Error(error=details)
