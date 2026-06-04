FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip

RUN git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git /rvc

WORKDIR /rvc

RUN pip install torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install RVC deps (except fairseq — broken dep chain)
RUN grep -v "^fairseq" requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt

# fairseq needs yanked omegaconf==2.0.6 — use pip<24.1 which allows it
RUN pip install "pip<24.1" && \
    pip install fairseq==0.12.2

# RVC requirements upgrades torch to cu130 which needs CUDA 13.0.
# Host driver supports CUDA 12.9 max — pin to cu128.
RUN pip install --upgrade pip && \
    pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128

# Pre-download hubert so it's baked into image (saves ~360MB download per run)
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os, shutil; \
os.makedirs('assets/hubert', exist_ok=True); \
src = hf_hub_download('lj1995/VoiceConversionWebUI', 'hubert_base.pt'); \
shutil.copy(src, 'assets/hubert/hubert_base.pt'); \
print('hubert_base.pt saved')"

COPY run.py /app/run.py
WORKDIR /app

VOLUME ["/input", "/output", "/models"]
ENTRYPOINT ["python", "/app/run.py"]
