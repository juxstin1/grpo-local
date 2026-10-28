# grpo-pod

RunPod image for the Mission4 GRPO run on an RTX PRO 6000 Blackwell (SM120). It contains:
- the runpod/pytorch CUDA 12.8 base;
- the exact trainer venv at `/opt/grpo/venv`: Python 3.12, torch 2.9.1+cu128, transformers 5.17.0, peft 0.21.0, FLA 0.6.0;
- llama-server and llama-quantize from llama.cpp `505b1ed`, at `/opt/grpo/llama.cpp/build/bin`.

Weights and code are not baked in.

To start a new pod, use image `ghcr.io/juxstin1/grpo-pod:mission4` with a volume at /workspace:

1. On the pod, run `/opt/grpo/bin/pod-init.sh`. It pulls Qwen3.5-4B @ 851bf6e (about 25 s), checks the 11 sha256 hashes, and builds the Q4 (about 1 min).
2. On the PC (PowerShell, in grpo-local), run `.venv\Scripts\python.exe pod\push_code.py <ip> <ssh_port>`. It sends the code, gym and SFT adapters (about 95 MB) and checks their hashes.
3. On the pod, run `nohup /opt/grpo/bin/run-mission4.sh /workspace/grpo sanity groups train eval > /workspace/grpo/results/run.log 2>&1 &`


