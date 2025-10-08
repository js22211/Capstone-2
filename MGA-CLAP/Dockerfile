# CUDA 12.8 런타임 베이스 이미지 (Step-Audio2와 동일 환경 유지)
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 기본 유틸 및 오디오 관련 라이브러리 설치
RUN set -eux; \
    for f in /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list; do \
        [ -f "$f" ] && mv "$f" "${f}.disabled"; \
    done; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends software-properties-common ca-certificates; \
    add-apt-repository -y universe; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git git-lfs \
        ffmpeg libsndfile1 sox; \
    rm -rf /var/lib/apt/lists/*; \
    git lfs install

# 최신 pip
RUN pip3 install --no-cache-dir --upgrade pip

# PyTorch 2.8 (CUDA 12.8 빌드)
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# NumPy는 faiss와 호환되는 1.x로 고정(예: 1.26.x)
RUN pip3 install --no-cache-dir "numpy<2,>=1.26.0"

# FAISS 설치 (CPU)
RUN pip3 install --no-cache-dir faiss-cpu==1.8.0

# 나머지 파이썬 패키지 (numpy는 이미 설치되어 있으므로 제외)
RUN pip3 install --no-cache-dir \
    scipy pandas pyarrow \
    pyyaml tqdm matplotlib seaborn \
    librosa soundfile \
    sentence-transformers==3.2.1 transformers==4.49.0 \
    torchlibrosa wandb \
    hyperpyyaml

# Hugging Face 캐시 디렉터리 고정 (Step-Audio2와 동일 전략)
ENV HF_HOME=/hf_cache
RUN mkdir -p /hf_cache && chmod 777 /hf_cache

# 로그/경고 감소를 위해 기본 환경 설정
ENV HF_USE_FLASH_ATTENTION=0 FLASH_ATTENTION_FORCE_DISABLE=1

WORKDIR /workspace

# 설치 검증용 간단한 버전 확인
RUN python3 - <<'PY'
import torch, faiss, numpy as np
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("faiss:", faiss.__version__)
print("numpy:", np.__version__)
PY

CMD ["bash"]
