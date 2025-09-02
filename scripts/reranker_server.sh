export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6,7

python -m utils.reranker_server \
    --model-name-or-path "Qwen/Qwen2.5-3B-Instruct" \
    --max-output-tokens 8192 \
    --max-score 5 \
    --concurrency 32 \
    --retriever-url "http://localhost:7999/retrieve" \
    --retriever-initial-topk 50 \
    --vllm-url "http://localhost:8001/v1" \
    --temperature 0.6 \
    --top-p 0.95