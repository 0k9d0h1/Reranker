export CUDA_VISIBLE_DEVICES=4,5
export HF_HOME=/home/kdh0901/.cache/huggingface

vllm serve Qwen/Qwen2.5-3B-Instruct --tensor-parallel-size 2 --port 8001