# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def make_instruction(question):
    return f"""You are an expert in retrieval who specializes in thinking, searching, and answering.

Before performing any search or providing an answer, you must first reason about your next action, including before your initial search.

After reasoning, if you require more information to answer the question, you may call the reranker function by placing your query between <search> and </search>. The reranker function will return the top results between <tool_response> and </tool_response>.

If the retrieved results do not contain enough information for answering the question, perform additional searches to gather more context.
Continue searching iteratively until you have gathered sufficient information to respond accurately.

Once you determine that no further external information is required, provide your answer directly within <answer> and </answer>, without detailed explanations. For example:
<answer> Beijing </answer>.

Question: {question}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/hotpotQA")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "RUC-NLPIR/FlashRAG_datasets"
    dataset = datasets.load_dataset(data_source, "hotpotqa")

    train_dataset = dataset["train"]
    test_dataset = dataset["dev"]

    instruction_following = (
        "Let's think step by step and output the final answer after `####`."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            golden_answers = example.pop("golden_answers")
            question = make_instruction(question_raw)

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": golden_answers},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": golden_answers,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "reranker": {
                            "create_kwargs": {},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": golden_answers,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
