import copy
import json
import time
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import get_topics, get_qrels
from .run_evaluation import THE_TOPICS, THE_INDEX, DLV2, BRIGHT
from .agent import get_agent
from .trec_eval import eval_rerank
from typing import List, Dict
import re
import os
from .utils import get_hits_from_run_bright
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def run_retriever(topics, searcher, qrels=None, k=100, qid=None):
    ranks = []
    if isinstance(topics, str):
        hits = searcher.search(topics, k=k)
        ranks.append({'query': topics, 'hits': []})
        rank = 0
        for hit in hits:
            rank += 1
            content = json.loads(searcher.doc(hit.docid).raw())
            if 'title' in content:
                content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
            else:
                content = content['contents']
            content = ' '.join(content.split())
            ranks[-1]['hits'].append({
                'content': content,
                'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
        return ranks[-1]

    for qid in tqdm(topics):
        if qid in qrels:
            query = topics[qid]['title']
            ranks.append({'query': query, 'hits': []})
            hits = searcher.search(query, k=k)
            rank = 0
            for hit in hits:
                rank += 1
                content = json.loads(searcher.doc(hit.docid).raw())
                if 'title' in content:
                    content = 'Title: ' + content['title'] + ' ' + 'Content: ' + content['text']
                elif 'passage' in content:
                    content = content['passage']
                else:
                    content = content['contents']
                content = ' '.join(content.split())
                ranks[-1]['hits'].append({
                    'content': content,
                    'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
                    
    return ranks


def get_prefix_prompt(query, num):
    return [{'role': 'system',
             'content': "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]


def get_post_prompt(query, num):
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only response the ranking results, do not say any word or explain."


def create_permutation_instruction(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])


    messages = get_prefix_prompt(query, num)
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        # For Japanese should cut by character: content = content[:int(max_length)]
        content = ' '.join(content.split())
        messages.append({'role': 'user', 'content': f"[{rank}] {content[:300]}"})
        # messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt(query, num)})

    return messages


def create_permutation_instruction_rearank(item=None, rank_start=0, rank_end=100):
    query = item['query']
    num = len(item['hits'][rank_start: rank_end])
    instruction =  (
            f"I will provide you with passages, each indicated by number identifier []. Rank the passages based on their relevance to the search query."
            f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query."
            f"The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be <answer> [] > [] </answer>, e.g., <answer> [1] > [2] </answer>."
        )
    
    messages = [
            {"role": "system", "content": "You are DeepRerank, an intelligent assistant that can rank passages based on their relevancy to the search query. You first thinks about the reasoning process in the mind and then provides the user with the answer."},
            {"role": "user","content": instruction},
            {"role": "assistant", "content": "Okay, please provide the passages."}
        ]
    rank = 0
    for hit in item['hits'][rank_start: rank_end]:
        rank += 1
        content = hit['content']
        content = content.replace('Title: Content: ', '')
        content = content.strip()
        content = ' '.join(content.split())
        messages.append({"role": "user", "content": f"[{rank}] {content[:400]}"})
        messages.append({"role": "assistant", "content": f"Received passage [{rank}]."})
                
    messages.append({
        "role": "user",
        "content": f"""Please rank these passages according to their relevance to the search query: "{query}"
            Follow these steps exactly:
            1. First, within <think> tags, analyze EACH passage individually:
            - Evaluate how well it addresses the query
            - Note specific relevant information or keywords

            2. Then, within <answer> tags, provide ONLY the final ranking in descending order of relevance using the format: [X] > [Y] > [Z]"""
    })

    return messages



def clean_response(response: str):
    if "<answer>" in response:
        content_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        response = content_match.group(1).strip() if content_match else response.strip()
    # process for Qwen3 with think tags but no answer tags
    if "<think>" in response:
        response = response.split("</think>")[-1]
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100, verbose=False):
    clean_permutation = clean_response(permutation)
    if verbose:
        print("="*100)
        print(f"query: {item['query']}")
        print(f"rank_start: {rank_start}, rank_end: {rank_end}")
        print(f"model response: {permutation}")

    response = []
    for x in clean_permutation.split():
        try:
            response.append(int(x) - 1)
        except:
            pass

    if verbose:
        print(f"cleaned response: {response}")
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def sliding_windows_batch(agent, items, rank_start=0, rank_end=100, window_size=20, step=10, verbose=False, enable_thinking=False):
    """Process multiple items with sliding windows using batched inference."""
    items = [copy.deepcopy(item) for item in items]
    
    # Initialize positions for all items
    all_positions = []
    for _ in items:
        end_pos = rank_end
        start_pos = rank_end - window_size
        item_positions = []
        
        while start_pos >= rank_start:
            start_pos = max(start_pos, rank_start)
            item_positions.append((start_pos, end_pos))
            end_pos = end_pos - step
            start_pos = start_pos - step
            
        all_positions.append(item_positions)
    
    # Process each window position across all items in batches
    max_windows = max(len(positions) for positions in all_positions)
    
    for window_idx in range(max_windows):
        # Collect messages for this window position across all items
        batch_messages = []
        batch_metadata = []  # Store (item_idx, start_pos, end_pos) for each message
        
        for item_idx, (item, positions) in enumerate(zip(items, all_positions)):
            if window_idx < len(positions):
                start_pos, end_pos = positions[window_idx]
                if 'le723z' in agent.model_name and enable_thinking:
                    # rearank prompt
                    messages = create_permutation_instruction_rearank(item=item, rank_start=start_pos, rank_end=end_pos)
                else:
                    # rankgpt prompt
                    messages = create_permutation_instruction(item=item, rank_start=start_pos, rank_end=end_pos)
                batch_messages.append(messages)
                batch_metadata.append((item_idx, start_pos, end_pos))
        
        if not batch_messages:
            continue
            
        # Process this batch of messages
        batch_permutations = agent.batch_chat(batch_messages, return_text=True, enable_thinking=enable_thinking)
        
        # Apply permutations to respective items
        for (item_idx, start_pos, end_pos), permutation in zip(batch_metadata, batch_permutations):
            if "ERROR::" in permutation:
                print(f"Error in batch processing: {permutation}")
                continue
            items[item_idx] = receive_permutation(items[item_idx], permutation, 
                                                 rank_start=start_pos, rank_end=end_pos, verbose=verbose)
    
    return items


def write_eval_file(rank_results, file):
    with open(file, 'w') as f:
        for i in range(len(rank_results)):
            rank = 1
            hits = rank_results[i]['hits']
            for hit in hits:
                f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                rank += 1
    return True


def process_rank_results_in_batches(agent, rank_results: List[Dict], batch_size=8, window_size=20, step=10, verbose=False, enable_thinking=False):
    """
    Process ranking results in batches to improve throughput.
    
    Args:
        agent: The LLM agent to use for reranking
        rank_results: List of ranking results to process
        batch_size: Number of items to process in each batch
        window_size: Size of sliding window for reranking
        step: Step size for sliding window
        
    Returns:
        List of processed ranking results
    """
    new_results = []
    total_start_time = time.time()
    
    for i in tqdm(range(0, len(rank_results), batch_size)):
        batch_items = rank_results[i:i+batch_size]
        print(f"Processing batch: {i//batch_size + 1}/{(len(rank_results) + batch_size - 1)//batch_size}")
        
        # Process entire batch at once
        processed_items = sliding_windows_batch(agent, batch_items, 
                                               rank_start=0, rank_end=min(len(batch_items[0]['hits']), 100), 
                                               window_size=window_size, step=step, verbose=verbose, enable_thinking=enable_thinking)
        new_results.extend(processed_items)
        
        print(f"Completed {len(new_results)}/{len(rank_results)} items")
    
    total_end_time = time.time()
    print(f"Total execution time: {total_end_time - total_start_time:.2f}s")
    
    return new_results

def bm25_retrieve(data, top_k_retrieve=100):
    assert data in THE_INDEX, f"Data {data} not found in THE_INDEX"
    searcher = LuceneSearcher.from_prebuilt_index(THE_INDEX[data])
    topics = get_topics(THE_TOPICS[data] if data not in DLV2 else data)
    qrels = get_qrels(THE_TOPICS[data])
    rank_results = run_retriever(topics, searcher, qrels, k=top_k_retrieve)
    return rank_results



   
if __name__ == '__main__':
    
    model_name = "le723z/Rearank-7B"
    enable_thinking = True
    agent = get_agent(model_name=model_name, api_key=None)

    for data in ['dl19']:
        if data in BRIGHT:
            base_dir = os.getcwd()
            bm25_results = get_hits_from_run_bright(base_dir, data)
        else:
            bm25_results = bm25_retrieve(data, top_k_retrieve=100)
    
        original_metrics, _ = eval_rerank(data, bm25_results)
        rerank_results = process_rank_results_in_batches(agent, bm25_results, batch_size=16, window_size=20, step=10, enable_thinking=enable_thinking)
        rerank_metrics, _ = eval_rerank(data, rerank_results)
        print(f"data: {data}")
        print(f"original metrics: {original_metrics}")
        print(f"rerank metrics:   {rerank_metrics}")
