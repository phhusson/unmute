# Browser-backend communication protocol

This document explains how the browser frontend and backend service communicate through WebSocket connections in the Unmute system.

## Overview

Unmute uses a WebSocket-based protocol inspired by the [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) for real-time voice conversations. The protocol handles:

- Real-time audio streaming (bidirectional)
- Voice conversation transcription
- Session configuration
- Error handling and debugging

## WebSocket connection

### Endpoint
- **URL**: `/v1/realtime`
- **Protocol**: `realtime` (specified in WebSocket subprotocol)
- **Port**: 8000 (development), routed through Traefik in Docker Swarm and Compose. Traefik uses http (port 80) and https (port 443).

### Connection setup

The WebSocket connection is established using the `realtime` subprotocol. See implementation details in:
- **Frontend**: [`frontend/src/app/Unmute.tsx`](../frontend/src/app/Unmute.tsx)
- **Backend**: [`unmute/main_websocket.py`](../unmute/main_websocket.py)

## Message protocol

All messages are JSON-encoded with a common structure defined in [`unmute/openai_realtime_api_events.py`](../unmute/openai_realtime_api_events.py).

### Base message structure

All messages inherit from [`BaseEvent`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L32-L50) which provides a common type and event_id structure.

## Client → server messages

### 1. Audio input streaming

**Message Type**: `input_audio_buffer.append`

**Purpose**: Stream real-time audio data from microphone to backend

**Model**: [`InputAudioBufferAppend`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L80-L81)

**Audio Format**:
- **Codec**: Opus
- **Sample Rate**: 24kHz
- **Channels**: Mono
- **Encoding**: Base64-encoded bytes

### 2. Session configuration

**Message Type**: `session.update`

**Purpose**: Configure voice character and conversation instructions. The backend will not start sending messages until it gets a session.update message that sets its instructions.

**Models**:
- [`SessionUpdate`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L72-L73)
- [`SessionConfig`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L66-L69)

## Server → client messages

### 1. Audio response streaming

**Message Type**: `response.audio.delta`

**Purpose**: Stream generated speech audio to frontend

**Model**: [`ResponseAudioDelta`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L133-L134)

### 2. Speech transcription

**Message Type**: `conversation.item.input_audio_transcription.delta`

**Purpose**: Real-time transcription of user speech

**Model**: [`ConversationItemInputAudioTranscriptionDelta`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L147-L151)

### 3. Text response streaming

**Message Type**: `response.text.delta`

**Purpose**: Stream generated text responses (for display/debugging)

**Model**: [`ResponseTextDelta`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L125-L126)

### 4. Speech detection events

**Message Types**:
- `input_audio_buffer.speech_started`
- `input_audio_buffer.speech_stopped`

**Purpose**: Indicate when user starts/stops speaking (for UI feedback). In Unmute we actually just ignore these events at the moment, even though we report them.

**Models**:
- [`InputAudioBufferSpeechStarted`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L95-L105)
- [`InputAudioBufferSpeechStopped`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L108-L111)

### 5. Response status updates

**Message Type**: `response.created`

**Purpose**: Indicate when assistant starts generating a response

**Models**:
- [`ResponseCreated`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L121-L122)
- [`Response`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L114-L118)

### 6. Error handling

**Message Type**: `error`

**Purpose**: Communicate errors and warnings

**Models**:
- [`Error`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L62-L63)
- [`ErrorDetails`](https://github.com/kyutai-labs/unmute/blob/main/unmute/openai_realtime_api_events.py#L53-L59)

## Connection lifecycle

1. **Health Check**: Frontend checks `/v1/health` endpoint
2. **WebSocket Connection**: Establish connection with `realtime` protocol
3. **Session Setup**: Send `session.update` with voice and instructions
4. **Audio Streaming**: Bidirectional real-time audio communication
5. **Graceful Shutdown**: Handle disconnection and cleanup

