export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1

python -m utils.reranker_server \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --max_output_tokens 8192 \
    --max_score 5 \
    --concurrency 32 \
    --retriever_url "http://localhost:8888/retrieve" \
    --retriever_initial_topk 50 \
    --vllm_url "http://localhost:8001/retrieve"