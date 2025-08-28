# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import listwiserank


def load_qrels(qrels_path):
    """Load qrels directly into the format needed for pytrec_eval"""
    qrels_dict = {}
    try:
        # Use a more efficient file reading approach
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[:4]
                    qrels_dict.setdefault(qid, {})[docid] = int(rel)
    except Exception as e:
        print(f"Error loading qrels file {qrels_path}: {e}")
    
    return qrels_dict

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or listwiserank.compute_score
        self.reward_fn_key = reward_fn_key
        self.qrels_dict = load_qrels("combined_qrels.txt")

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        all_scores = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            score = self.compute_score(
                predict_str=response_str,
                item=data_item.non_tensor_batch,
                qrels_dict=self.qrels_dict
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            all_scores.append(score['ndcg_reward'])
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"="*100)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)
                print(f"="*100)

        # print scores max, min, mean
        all_scores = torch.tensor(all_scores)
        print("*" * 50, "Scores for the batch", "*" * 50)
        print("[ndcg_reward] max:", all_scores.max().item())
        print("[ndcg_reward] min:", all_scores.min().item())
        print("[ndcg_reward] mean:", all_scores.mean().item())
        # Calculate score intervals with a step of 0.1
        interval_counts = defaultdict(int)
        for score in all_scores:
            interval = round(score.item() // 0.1 * 0.1, 1)
            interval_counts[interval] += 1
        print("Score intervals (step=0.1):")
        for interval, count in sorted(interval_counts.items()):
            print(f"[{interval:.1f}-{interval + 0.1:.1f}): {count}")
        print("*" * 50, "End of scores for the batch", "*" * 50)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor