from prometheus_client import Counter, Gauge, Histogram, Summary

SESSION_DURATION_BINS = [1.0, 10.0, 30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0]
TURN_DURATION_BINS = [0.5, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0]
GENERATION_DURATION_BINS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

PING_BINS_MS = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
PING_BINS = [x / 1000 for x in PING_BINS_MS]

# Time to first token.
TTFT_BINS_STT_MS = [
    10.0,
    15.0,
    25.0,
    50.0,
    75.0,
    100.0,
]
TTFT_BINS_STT = [x / 1000 for x in TTFT_BINS_STT_MS]

TTFT_BINS_TTS_MS = [
    200.0,
    250.0,
    300.0,
    350.0,
    400.0,
    450.0,
    500.0,
    550.0,
]
TTFT_BINS_TTS = [x / 1000 for x in TTFT_BINS_TTS_MS]


TTFT_BINS_VLLM_MS = [
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    750.0,
    1000.0,
]
TTFT_BINS_VLLM = [x / 1000 for x in TTFT_BINS_VLLM_MS]

NUM_WORDS_REQUEST_BINS = [
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    6000.0,
    8000.0,
]
NUM_WORDS_STT_BINS = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 4000.0]
NUM_WORDS_REPLY_BINS = [5.0, 10.0, 25.0, 50.0, 100.0, 200.0]

SESSIONS = Counter("worker_sessions", "")
SERVICE_MISSES = Counter("worker_service_misses", "")
HARD_SERVICE_MISSES = Counter("worker_hard_service_misses", "")
FORCE_DISCONNECTS = Counter("worker_force_disconnects", "")
FATAL_SERVICE_MISSES = Counter("worker_fatal_service_misses", "")
HARD_ERRORS = Counter("worker_hard_errors", "")
ACTIVE_SESSIONS = Gauge("worker_active_sessions", "")
SESSION_DURATION = Histogram(
    "worker_session_duration", "", buckets=SESSION_DURATION_BINS
)
HEALTH_OK = Summary("worker_health_ok", "")

STT_SESSIONS = Counter("worker_stt_sessions", "")
STT_ACTIVE_SESSIONS = Gauge("worker_stt_active_sessions", "")
STT_MISSES = Counter("worker_stt_misses", "")
STT_HARD_MISSES = Counter("worker_stt_hard_misses", "")
STT_SENT_FRAMES = Counter("worker_stt_sent_frames", "")
STT_RECV_FRAMES = Counter("worker_stt_recv_frames", "")
STT_RECV_WORDS = Counter("worker_stt_recv_words", "")
STT_PING_TIME = Histogram("worker_stt_ping_time", "", buckets=PING_BINS)
STT_FIND_TIME = Histogram("worker_stt_find_time", "", buckets=PING_BINS)
STT_SESSION_DURATION = Histogram(
    "worker_stt_session_duration", "", buckets=SESSION_DURATION_BINS
)
STT_AUDIO_DURATION = Histogram(
    "worker_stt_audio_duration", "", buckets=SESSION_DURATION_BINS
)
STT_NUM_WORDS = Histogram("worker_stt_num_words", "", buckets=NUM_WORDS_STT_BINS)
STT_TTFT = Histogram("worker_stt_ttft", "", buckets=TTFT_BINS_STT)

TTS_SESSIONS = Counter("worker_tts_sessions", "")
TTS_ACTIVE_SESSIONS = Gauge("worker_tts_active_sessions", "")
TTS_MISSES = Counter("worker_tts_misses", "")
TTS_HARD_MISSES = Counter("worker_hard_tts_misses", "")
TTS_INTERRUPT = Counter("worker_tts_interrupt", "")
TTS_SENT_FRAMES = Counter("worker_tts_sent_frames", "")
TTS_RECV_FRAMES = Counter("worker_tts_recv_frames", "")
TTS_RECV_WORDS = Counter("worker_tts_recv_words", "")
TTS_PING_TIME = Histogram("worker_tts_ping_time", "", buckets=PING_BINS)
TTS_FIND_TIME = Histogram("worker_tts_find_time", "", buckets=PING_BINS)
TTS_TTFT = Histogram("worker_tts_ttft", "", buckets=TTFT_BINS_TTS)
TTS_AUDIO_DURATION = Histogram(
    "worker_tts_audio_duration", "", buckets=TURN_DURATION_BINS
)
TTS_GEN_DURATION = Histogram(
    "worker_tts_gen_duration", "", buckets=GENERATION_DURATION_BINS
)

VLLM_SESSIONS = Counter("worker_vllm_sessions", "")
VLLM_ACTIVE_SESSIONS = Gauge("worker_vllm_active_sessions", "")
VLLM_INTERRUPTS = Counter("worker_vllm_interrupt", "")
VLLM_HARD_ERRORS = Counter("worker_vllm_hard_errors", "")
VLLM_SENT_WORDS = Counter("worker_vllm_sent_words", "")
VLLM_RECV_WORDS = Counter("worker_vllm_recv_words", "")
VLLM_TTFT = Histogram("worker_vllm_ttft", "", buckets=TTFT_BINS_VLLM)
VLLM_REQUEST_LENGTH = Histogram(
    "worker_vllm_request_length", "", buckets=NUM_WORDS_REQUEST_BINS
)
VLLM_REPLY_LENGTH = Histogram(
    "worker_vllm_reply_length", "", buckets=NUM_WORDS_REPLY_BINS
)
VLLM_GEN_DURATION = Histogram(
    "worker_vllm_gen_duration", "", buckets=GENERATION_DURATION_BINS
)

VOICE_DONATION_SUBMISSIONS = Counter("worker_voice_donation_submissions", "")
