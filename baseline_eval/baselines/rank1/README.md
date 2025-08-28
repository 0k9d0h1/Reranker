<h1 align="center">Rank1: Test-Time Compute for Reranking in Information Retrieval</h1>

<h4 align="center">
    <p>
        <a href="#links">Model/Data Links</a> |
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="#citing">Citation</a>
    <p>
</h4>

Official repository for [rank1, a reasoning reranker model that "thinks"](http://arxiv.org/abs/2502.18418). Rank1 leverages test-time compute to generate reasoning chains before making relevance judgments.

## Links
#### Models
| Resource | Description |
|:---------|:------------|
| [rank1-0.5b](https://huggingface.co/jhu-clsp/rank1-0.5b) | Trained from Qwen2.5-0.5B base |
| [rank1-1.5b](https://huggingface.co/jhu-clsp/rank1-1.5b) | Trained from Qwen2.5-1.5B base |
| [rank1-3b](https://huggingface.co/jhu-clsp/rank1-3b) | Trained from Qwen2.5-3B base |
| [rank1-7b](https://huggingface.co/jhu-clsp/rank1-7b) | Trained from Qwen2.5-7B base |
| [rank1-14b](https://huggingface.co/jhu-clsp/rank1-14b) | Trained from Qwen2.5-14B base |
| [rank1-32b](https://huggingface.co/jhu-clsp/rank1-32b) | Trained from Qwen2.5-32B base |
| [rank1-mistral-2501-24b](https://huggingface.co/jhu-clsp/rank1-mistral-2501-24b) | Trained from Mistral-Small 2501 24B base |
| [rank1-llama3-8b](https://huggingface.co/jhu-clsp/rank1-llama3-8b) | Trained from Llama 3.1 8B base |

#### Quantized Models (fits in 24GB GPUs)
| Resource | Description |
|:---------|:------------|
| [rank1-7b-awq](https://huggingface.co/jhu-clsp/rank1-7b-awq) | Quantized version of rank1-7b  |
| [rank1-14b-awq](https://huggingface.co/jhu-clsp/rank1-14b-awq) | Quantized version of rank1-14b  |
| [rank1-32b-awq](https://huggingface.co/jhu-clsp/rank1-32b-awq) | Quantized version of rank1-32b  |
| [rank1-mistral-2501-24b-awq](https://huggingface.co/jhu-clsp/rank1-mistral-2501-24b-awq) | Quantized version of rank1-mistral-24b  |
| [rank1-llama3-8b-awq](https://huggingface.co/jhu-clsp/rank1-llama3-8b-awq) | Quantized version of rank1-llama3-8b  |

#### Datasets
| Resource | Description |
|:---------|:------------|
| [rank1-r1-msmarco](https://huggingface.co/datasets/jhu-clsp/rank1-R1-MSMARCO) | All R1 output examples from MS MARCO |
| [rank1-training-data](https://huggingface.co/datasets/jhu-clsp/rank1-training-data) | Training data used for rank1 models |
| [rank1-run-files](https://huggingface.co/datasets/jhu-clsp/rank1-Run-Files) | Pre-computed run files for use in top 100 doc reranking |

## Installation 
To reproduce the experiments, you can use the following code with uv for fast, reliable dependency management:

```bash
git clone https://github.com/orionw/rank1.git
cd rank1/
git submodule update --init --recursive

# Install uv if you don't have it already
curl -fsSL https://pkg.uv.dev/install.sh | sh

# Create and activate virtual environment with uv
uv venv env --python=3.10
source env/bin/activate 

# Install dependencies with uv
uv pip install -r requirements.txt
uv pip install -e mteb_branch/
uv pip install --no-build-isolation xformers==0.0.28.post3
uv pip install vllm==0.7.2

# Recommended: download a flash attention wheel from https://github.com/Dao-AILab/flash-attention/releases and `uv pip install` it
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# uv pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Download the Rank1-Run-Files repository (required for evaluation)
git lfs install # if you don't have it already
git clone https://huggingface.co/datasets/jhu-clsp/rank1-run-files
```

## Usage
### Tips
**Reproducibility** Depending on your batch size for evaluation you will get minorly different results due to non-determinisms in vLLM. For our experiments we processed all instances in one batch (e.g. batch_size=99999999999). We also found that using the flag `enforce_eager` sped up inference for the smaller models but not for the larger models.

**Adapting to New Tasks** You may want to use these models on tasks where the relevance definition is different from MS MARCO. For these you will want a custom prompt to let the model know. You can see these in `prompts.py` various datasets. 


### Running Evaluations
To run an evaluation with the rank1 model on a specific dataset:

```bash
bash launch_job.sh jhu-clsp/rank1-7b NevIR default 1
```

Parameters:
- `jhu-clsp/rank1-7b`: Model name or path
- `NevIR`: Dataset name
- `default`: Subtask name (use "default" if no subtask)
- `1`: Number of GPUs to use


### Using Rank1 in Your Own Code
You can integrate rank1 into your code:

```python
from rank1 import rank1

# Initialize the model
model = rank1(
    model_name_or_path="jhu-clsp/rank1-7B",
    num_gpus=1,
    device="cuda",
    context_size=16000,
    max_output_tokens=8192,
    fp_options="float16"
)

# Rerank documents
query = "Your query / prompt here"
corpus = ["Document 1 content", "Document 2 content", ...]
queries = [query] * len(corpus)
results = model.predict(list(zip(query, corpus)))
```

### MTEB Integration
Rank1 is compatible with the MTEB benchmarking framework. To evaluate your model:

```python
from mteb import MTEB
from rank1 import rank1

# Initialize your model
model = rank1(
    model_name_or_path="jhu-clsp/rank1-7b",
    num_gpus=1,
    device="cuda"
)

# Select tasks (or use specific task names)
evaluation = MTEB(tasks=["NevIR"])

# Run evaluation
results = evaluation.run(model)
```

## Citing
If you use rank1 you can cite:

```bibtex
@misc{weller2025rank1testtimecomputereranking,
      title={Rank1: Test-Time Compute for Reranking in Information Retrieval}, 
      author={Orion Weller and Kathryn Ricci and Eugene Yang and Andrew Yates and Dawn Lawrie and Benjamin Van Durme},
      year={2025},
      eprint={2502.18418},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2502.18418}, 
}
```

## License
[MIT](LICENSE)
