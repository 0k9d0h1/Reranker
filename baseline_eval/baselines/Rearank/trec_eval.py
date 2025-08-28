import pandas as pd
import tempfile
import os
import copy
from typing import Dict, Tuple
import pytrec_eval
import datasets
from .utils import read_bright_qrels
from .run_evaluation import BRIGHT

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'dl21': 'dl21-passage',
    'dl22': 'dl22-passage',
    'dl23': 'dl23-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'arguana': 'beir-v1.0.0-arguana-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'scidocs': 'beir-v1.0.0-scidocs-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'quora': 'beir-v1.0.0-quora-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'fever': 'beir-v1.0.0-fever-test',
    'robust04': 'beir-v1.0.0-robust04-test',
    'signal': 'beir-v1.0.0-signal1m-test',

    'mrtydi-ar': 'mrtydi-v1.1-arabic-test',
    'mrtydi-bn': 'mrtydi-v1.1-bengali-test',
    'mrtydi-fi': 'mrtydi-v1.1-finnish-test',
    'mrtydi-id': 'mrtydi-v1.1-indonesian-test',
    'mrtydi-ja': 'mrtydi-v1.1-japanese-test',
    'mrtydi-ko': 'mrtydi-v1.1-korean-test',
    'mrtydi-ru': 'mrtydi-v1.1-russian-test',
    'mrtydi-sw': 'mrtydi-v1.1-swahili-test',
    'mrtydi-te': 'mrtydi-v1.1-telugu-test',
    'mrtydi-th': 'mrtydi-v1.1-thai-test',

}

def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    # Pre-allocate dictionaries with zeros
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}

    # Join strings once instead of in a loop
    map_string = "map_cut." + ",".join(map(str, k_values))
    ndcg_string = "ndcg_cut." + ",".join(map(str, k_values))
    recall_string = "recall." + ",".join(map(str, k_values))
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    # Use more efficient iteration
    num_queries = len(scores)
    for query_scores in scores.values():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += query_scores["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += query_scores["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += query_scores["recall_" + str(k)]

    # Normalize all metrics at once
    for metric_dict in (ndcg, _map, recall):
        for k in metric_dict:
            metric_dict[k] = round(metric_dict[k] / num_queries, 5)

    # Combine metrics more efficiently
    all_metrics = {}
    all_metrics.update(ndcg)
    all_metrics.update(_map)
    all_metrics.update(recall)

    return all_metrics, scores


def get_qrels_file(name):
    
    if os.path.exists(name):
        return name
    split = name.replace('-test', '.test')
    path = 'data/label_file/qrels.' + split + '.txt'  # try to use cache
    if not os.path.exists(path):  # updated to check the correct path variable
        from pyserini.search import get_qrels_file
        try:
            return get_qrels_file(name)  # download from pyserini using the correct path
        except:
            print(f'no qrels file for {name}')
            return None
    return path  # return the correct path



def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
        else:
            print('duplicate')
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


class EvalFunction:

    @staticmethod
    def convert_to_trec_format(rank_results):
        """
        Convert rank_results directly to the format needed for pytrec_eval
        input:
            rank_results: list of dicts, each dict contains 'qid', 'hits'
            [{'qid': 1, 'hits': [{'docid': 1, 'score': 0.5}, {'docid': 2, 'score': 0.4}]},
             {'qid': 2, 'hits': [{'docid': 1, 'score': 0.5}, {'docid': 2, 'score': 0.4}]}]

        return:
            {
                'qid': {
                    'docid': score,
                    'docid': score
            }
        
        
        """
        run_dict = {}
        for result in rank_results:
            for hit in result['hits']:
                qid = str(hit['qid'])  # Ensure qid is a string
                docid = str(hit['docid'])  # Ensure docid is a string
                score = float(hit['score'])  # Ensure score is a float
                
                if qid not in run_dict:
                    run_dict[qid] = {}
                run_dict[qid][docid] = score
        
        return run_dict

    @staticmethod
    def write_file(rank_results, file):
        with open(file, 'w') as f:
            for i in range(len(rank_results)):
                rank = 1
                hits = rank_results[i]['hits']
                for hit in hits:
                    f.write(f"{hit['qid']} Q0 {hit['docid']} {rank} {hit['score']} rank\n")
                    rank += 1
        return True

    @staticmethod
    def load_qrels(qrels_name):
        """Load qrels directly into the format needed for pytrec_eval"""
        qrels_path = get_qrels_file(qrels_name)
        if not qrels_path:
            return None
            
        qrels_dict = {}
        # Use a more efficient file reading approach
        with open(qrels_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[:4]
                    qrels_dict.setdefault(qid, {})[docid] = int(rel)
        
        return qrels_dict

    @staticmethod
    def main(args_qrel, args_run):
        if isinstance(args_run, str):
            # If args_run is a file path, read it
            assert os.path.exists(args_run)
            with open(args_run, 'r') as f_run:
                pred = pytrec_eval.parse_run(f_run)
        else:
            # If args_run is already the rank_results structure
            pred = EvalFunction.convert_to_trec_format(args_run)
        
        # Load qrels
        if isinstance(args_qrel, str):
            qrels_dict = EvalFunction.load_qrels(args_qrel)
        else:
            qrels_dict = args_qrel
            
        all_metrics = trec_eval(qrels_dict, pred, k_values=(1, 5, 10))
        print(all_metrics)
        return all_metrics

def eval_rerank(name, results, qrels_dict=None):
    """Evaluate reranking results without writing to temporary files"""
    if not isinstance(results, (list, datasets.arrow_dataset.Dataset)):
        results = [results]
    if qrels_dict is None:
        if name in BRIGHT:
            basedir = os.path.dirname(os.path.abspath(__file__))
            qrels_dict = read_bright_qrels(f"{basedir}/data/bright/data/pyserini_qrels/{name}.tsv")
        else:
            qrels_dict = EvalFunction.load_qrels(THE_TOPICS.get(name, name))
    run_dict = EvalFunction.convert_to_trec_format(results)
    return trec_eval(qrels_dict, run_dict, k_values=(1, 5, 10))


if __name__ == '__main__':
    pass

