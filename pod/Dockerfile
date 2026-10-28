# Mission4 GRPO pod image: the RunPod base the 2026-10-27 pod ran, plus the exact trainer
# venv (Python 3.12) and llama-server/llama-quantize built for SM120 (RTX PRO 6000 Blackwell).
# Weights and code are NOT baked: pod-init.sh pulls the pinned HF snapshot, push_code.py sends code.
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ARG LLAMA_COMMIT=505b1ed15ca80e2a19f12ff4ac365e40fb374053
ENV G=/opt/grpo UV_PYTHON_INSTALL_DIR=/opt/grpo/python UV_NO_CACHE=1

COPY requirements-lock.txt model.sha256 /opt/grpo/
# causal-conv1d is in the lock but never imported on the training pod (broken build); the
# trained run used the PyTorch conv fallback, so it is left out to match.
RUN pip install --no-cache-dir uv \
 && uv venv --python 3.12 --seed $G/venv \
 && grep -v '^causal-conv1d==' $G/requirements-lock.txt > /tmp/req.txt \
 && uv pip install --python $G/venv/bin/python --index-strategy unsafe-best-match \
      --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/req.txt \
 && $G/venv/bin/python -c "import torch, transformers, peft, fla; print(torch.__version__, transformers.__version__)"

# Same flags as the pod build (build_server.py) except GGML_NATIVE=OFF: the build machine's CPU is not the pod's.
RUN git clone --filter=blob:none https://github.com/ggml-org/llama.cpp $G/llama.cpp \
 && cd $G/llama.cpp && git checkout $LLAMA_COMMIT \
 && export PATH=$G/venv/bin:/usr/local/cuda/bin:$PATH CUDACXX=/usr/local/cuda/bin/nvcc \
 && cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=OFF \
 && cmake --build build --target llama-server llama-quantize -j 4 \
 && rm -rf build/CMakeFiles build/ggml/src/CMakeFiles .git

COPY pod-init.sh run-mission4.sh /opt/grpo/bin/
RUN chmod +x /opt/grpo/bin/*.sh
