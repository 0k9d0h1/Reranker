export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
export VLLM_USE_V1=0
set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG="$HOME/Desktop/Reranker/utils/single_reranker_tool.yaml"
REWARD_FUNCTION_PATH="$HOME/Desktop/Reranker/verl/verl/utils/reward_score/planner_em_format.py"

TRAIN_DATA="./data/test/train.parquet"
VAL_DATA="./data/test/test.parquet"

MAX_TURNS=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.val_batch_size=16 \
    data.max_prompt_length=8192 \
    data.max_response_length=3000 \
    data.max_start_length=3072 \
    data.max_obs_length=5000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.reward_fn_key='data_source' \
    data.filter_overlong_prompts_workers=32 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.temperature=1.0 \
    reward_model.reward_manager='planner' \
    reward_model.reward_kwargs.score=1.0 \
    reward_model.reward_kwargs.structure_format_score=0.2 \
    reward_model.reward_kwargs.final_format_score=0.1 \
    reward_model.reward_kwargs.retrieval_score=0.1 \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name='test' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.validation_data_dir='./validation_data' \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    max_turns=$MAX_TURNS \
    reranker.url="http://localhost:8002/rerank/batch" \
    reranker.topk=3 \
    reranker.return_full_documents=true \
    trainer.total_epochs=1 $@

