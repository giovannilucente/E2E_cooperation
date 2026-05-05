FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Proxy support (IMPORTANT for IT network)
# -----------------------------
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY

# -----------------------------
# system deps (now works through proxy)
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# upgrade pip (also uses proxy automatically)
# -----------------------------
RUN python -m pip install --upgrade pip

# -----------------------------
# copy repo
# -----------------------------
WORKDIR /workspace
COPY . /workspace

# -----------------------------
# python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r opencood/requirements.txt
RUN pip install --no-cache-dir -r simulation/requirements.txt