import re
from .trec_eval import eval_rerank
import copy

def remove_duplicate(response, permutation):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print(f"duplicate {c} in response: {response}")
            print(f"permutation: {permutation}")
    return new_response


def clean_response(response: str):
    # More efficient string processing
    chars = []
    for c in response:
        if c.isdigit():
            chars.append(c)
        else:
            chars.append(' ')
    return ''.join(chars).strip()

def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    print(f"original answer: {permutation}")
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response, permutation)
    print(f"cleaned answer: {response}")
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    print(f"final permutation index: {response}")
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item

def compute_ranking_format_reward(predict_str: str) -> float:
    """
    Checks if the prediction string contains a properly formatted ranking 
    in the <answer> tag with format like [1] > [2] > [3]
    
    Args:
        predict_str: The prediction string to check
    
    Returns:
        1.0 if the format matches, 0.0 otherwise
    """
    try:
        # Extract content between <answer> tags
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        if not content_match:
            return 0.0
            
        content = content_match.group(1).strip()
        
        # Pattern to match rankings like [1] > [2] > [3]
        # This matches one or more items in square brackets separated by '>' symbols
        ranking_pattern = re.compile(r'(\[\d+\]\s*>\s*)*\[\d+\]')
        
        # Check if the content contains this pattern
        if ranking_pattern.search(content):
            return 1.0
        
        return 0.0
    except Exception:
        return 0.0
    
def compute_response_format_reward(predict_str: str) -> float:
    # Precompile pattern for better performance
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0

def compute_ndcg_reward(predict_str: str, item: dict, qrels_dict: dict = None) -> float:
    try:
        # Extract content once using precompiled pattern
        print('-'*50, f"query: {item['query']}", '-'*50)
        print(f"full response: {predict_str}")
        content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
        permutation = content_match.group(1).strip() if content_match else predict_str.strip()
        
        # Create a shallow copy of item to avoid modifying the original
        
        item = receive_permutation(item, permutation, rank_start=0, rank_end=len(item['hits']))
        all_metric = eval_rerank('', item, qrels_dict=qrels_dict)
        rerank_score = all_metric.get('NDCG@10', 0.0)
        init_score = item['metrics'].get('NDCG@10', 0.0)
        best_score = item['best_metrics'].get('NDCG@10', 0.0)
        score = (rerank_score - init_score) / (best_score - init_score) if best_score != init_score else 0.0
        # print(f" rerank score: {rerank_score}, init score: {init_score}, best score: {best_score}")
        # print(f"final score: {score}")
        return score

    except Exception as e:
        print(f"Error in compute_ndcg_reward: {e}")
        return 0.0


def compute_score(predict_str: str, item: dict, qrels_dict: dict) -> float:
    """
    Computes the score based on the response format, ranking format, and NDCG.
    The final score is a weighted sum of the individual scores.
    """
    response_format_reward = compute_response_format_reward(predict_str)
    ndcg_reward = compute_ndcg_reward(predict_str, item, qrels_dict)
    ranking_format_reward = compute_ranking_format_reward(predict_str)
    
    return {
        "score": 0.8 * ndcg_reward + 0.1 * response_format_reward + 0.1 * ranking_format_reward,
        "ranking_format_reward": ranking_format_reward,
        "response_format_reward": response_format_reward,
        "ndcg_reward": ndcg_reward
    }

