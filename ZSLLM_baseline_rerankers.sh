#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the list of rerankers you want to test
RERANKERS=(
    # "rank_t5" \
    # "rank_zephyr" \
    # "rearank" \
    # "reasonrank" \
    # "rank1" \
    "rank_r1"
) # Add all your reranker names here

# --- First, run the baseline case which doesn't need a reranker server ---
echo "========================================================="
echo "RUNNING EXPERIMENT FOR: baseline"
echo "========================================================="
# python3 -m baseline_eval.ZSLLM_baseline_rerankers --reranker-name "baseline"
# echo "Baseline experiment finished."
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
# --- Now, loop through each reranker that needs a server ---
for reranker_name in "${RERANKERS[@]}"; do
    echo "========================================================="
    echo "STARTING EXPERIMENT FOR: ${reranker_name}"
    echo "========================================================="
    case "$reranker_name" in
        rank_t5|rank_zephyr)
            CONDA_ENV="rankify_env"
            SERVER_SCRIPT="baseline_eval.rankify_server"
            ;;
        rearank)
            CONDA_ENV="rearank_env"
            SERVER_SCRIPT="baseline_eval.rearank_server"
            ;;
        rank_r1)
            CONDA_ENV="rank_r1_env"
            SERVER_SCRIPT="baseline_eval.rank_r1_server"
            ;;
        reasonrank)
            CONDA_ENV="reasonrank_env"
            SERVER_SCRIPT="baseline_eval.reasonrank_server"
            ;;
        rank1)
            CONDA_ENV="rank1_env"
            SERVER_SCRIPT="baseline_eval.rank1_server"
            ;;
        *)
            echo "ERROR: No configuration found for reranker '${reranker_name}'"
            exit 1
            ;;
    esac

    # 1. Activate the correct environment and start the correct server
    echo "--> Activating conda env: ${CONDA_ENV}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate "${CONDA_ENV}"
    
    echo "--> Starting server script: ${SERVER_SCRIPT} for ${reranker_name}..."
    # The server runs within its specific activated environment
    CUDA_VISIBLE_DEVICES=0,1 python3 -m "${SERVER_SCRIPT}" --model-name "${reranker_name}" &
    
    # Get the Process ID (PID) of the background server
    SERVER_PID=$!
    
    # Give the server time to load the model (adjust as needed)
    echo "--> Waiting for server to load... (PID: ${SERVER_PID})"
    sleep 90

    # 2. Run the experiment client
    echo "--> Running RAG client for ${reranker_name}..."
    CUDA_VISIBLE_DEVICES=2,3 python3 -m baseline_eval.ZSLLM_baseline_rerankers --reranker-name "${reranker_name}"
    echo "--> Client finished."

    # 3. Stop the server GRACEFULLY using the API endpoint
    echo "--> Sending shutdown request to server (PID: ${SERVER_PID})..."
    curl -X POST http://localhost:8001/shutdown
    
    # Wait for the process to actually terminate
    wait "${SERVER_PID}" 2>/dev/null
    
    conda deactivate
    echo "Experiment for ${reranker_name} finished."
    echo ""
done

echo "========================================================="
echo "All experiments complete."
echo "========================================================="