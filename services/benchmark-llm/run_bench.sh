set -ex

uv run --with vllm==0.9.1 vllm bench serve --model google/gemma-3-12b-it --host llm --request-rate 20 --random-prefix-len 3