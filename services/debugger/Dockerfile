FROM ubuntu:22.04

# Install many tools useful for debugging the network
RUN apt-get update && \
    apt-get install -y \
    iputils-ping \
    iproute2 \
    net-tools \
    curl \
    wget \
    dnsutils \
    traceroute \
    tcpdump \
    nmap \
    telnet \
    vim \
    less \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


COPY --from=ghcr.io/astral-sh/uv:0.7.2 /uv /uvx /bin/

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
