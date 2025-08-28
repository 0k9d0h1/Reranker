import random
import numpy as np
import torch
import os
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_results_file(results_file):
    # Load existing results if file exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    return all_results


def read_bright_topics(path):
    query_map = {}
    with open(path, 'r') as f:
        for line in f:
            qid, query = line.strip().split('\t')
            query_map[qid] = query
    return query_map

def read_bright_qrels(qrels_path):
    qrels_dict = {}
    # Use a more efficient file reading approach
    with open(qrels_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
                qrels_dict.setdefault(qid, {})[docid] = int(rel)
    return qrels_dict

def read_bm25_run(run_path):
    run_dict = {}
    with open(run_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docid, rank, score = parts[:5]
                if qid not in run_dict:
                    run_dict[qid] = {}
                run_dict[qid][docid] = float(score)
    return run_dict

def read_corpus_from_jsonl(corpus_path):
    corpus = {}
    with open(corpus_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            docid = doc['id']
            text = doc['contents']
            corpus[docid] = text
    return corpus

def get_hits_from_run_bright(base_dir, data):
    topic_dict = read_bright_topics(f"{base_dir}/data/bright/data/pyserini_queries/{data}.tsv")
    run_dict = read_bm25_run(f"{base_dir}/data/bright/runs_bm25/bm25.{data}.filtered.trec")
    corpus = read_corpus_from_jsonl(f"{base_dir}/data/bright/data/pyserini_corpus/{data}/{data}.jsonl")
    rank_results = []
    for qid, docids in run_dict.items():
        result = {}
        result['query'] = topic_dict[qid]
        hits = []
        rank = 1
        for docid, score in docids.items():
            if docid in corpus:
                hits.append({'docid': docid, 'score': score, 'content': corpus[docid], 'qid': qid, 'rank': rank})
                rank += 1
        result['hits'] = hits
        rank_results.append(result)
    return rank_results