# Rank1 Model Training Configurations

This directory contains configuration files for training and exporting Rank1 models using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

## Getting Started

### 1. Setup Training Data

We use the data in `jhu-clsp/rank1-training-data` for training. Follow these steps to set up:

1. Download the training data and store it in LLaMA-Factory's data folder (called `train.json`)
2. Update the `dataset_info.json` file to reference your data:

```json
{
  "r1_reranking_self_filtered_negs_only": {
    "file_name": "train.json"
  }
}
```

### 2. Available Training Configurations

The following configuration files are available:

- `train_lora_llama.yaml` - LoRA training configuration for Llama-3 8B
- `train_lora_mistral.yaml` - LoRA training configuration for Mistral 24B
- `train_lora_qwen_7b.yaml` - LoRA training configuration for Qwen 7B
- `train_lora_qwen_14b.yaml` - LoRA training configuration for Qwen 14B
- `train_lora_qwen_32b.yaml` - LoRA training configuration for Qwen 32B

### 3. Training Process

After installing LLaMA-Factory requirements, start training with:

```bash
llamafactory-cli train train_configs/train_lora_XXX.yaml
```

Replace `XXX` with the specific model you want to train (e.g., `llama`, `mistral`, `qwen_7b`).

### 4. Exporting Model Weights

After training, export the merged LoRA weights using:

```bash
llamafactory-cli export train_configs/export_model.yaml
```

Before running the export command, modify the `export_model.yaml` file with your specific values:

```yaml
model_name_or_path: BASE_MODEL_HERE  # Path to the base model
adapter_name_or_path: INPUT_DIR      # Path to the trained adapter
export_dir: EXPORT_DIR               # Where to save the exported model
```

## Configuration Parameters

Each YAML file contains configuration sections for:
- Model settings
- Training method (LoRA parameters)
- Dataset configuration
- Output settings
- Training hyperparameters
- Evaluation settings

Refer to each specific YAML file for detailed configuration options.

