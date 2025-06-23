FROM ghcr.io/astral-sh/uv:0.6.17-debian AS build

RUN uv run --with vllm==0.9.1 echo hello

COPY run_bench.sh /


CMD ["sleep", "infinity"]
