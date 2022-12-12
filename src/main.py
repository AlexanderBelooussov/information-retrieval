import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from document_retrieval import get_relevant_documents_query
from load_data import read_metadata, read_queries, read_qrels, read_podcasts, read_transcripts
from segment_retrieval import split_transcript, retrieve_segments
from tqdm import tqdm
from sklearn.metrics import ndcg_score, dcg_score, PrecisionRecallDisplay


def print_result(result):
    print(f"Results for: {result['query']}")
    print(f"Total number retrieved: {result['total retrieved']}")
    print(f"Total relevant: {result['total relevant']}")
    print(f"Total relevant retrieved: {result['total relevant retrieved']}")
    print(f"Mean NDCG: {result['ndcg']}")
    print()
    for k in [5, 10, 25, 20, 30]:
        try:
            print(f"Document Precision@{k} = {result[f'document precision@{k}']}")
        except KeyError:
            pass
    print()
    for k in [5, 10, 25, 20, 30]:
        try:
            print(f"Segment Precision@{k} = {result[f'segment precision@{k}']}")
        except KeyError:
            pass
    print("\n")
    # display = PrecisionRecallDisplay.from_predictions([1 if rel > 0 else 0 for rel in result["true relevances"]], result["scores"])
    # _ = display.ax_.set_title(f"{result['query']} Precision-Recall curve")
    # plt.show()


def process_query(query, transcripts, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    results = {}
    retrieved_docs, _ = get_relevant_documents_query(query, k=doc_k, use_keybert=True, use_description=True,
                                                     index_type='show_episode_info', n_grams=n_grams)
    retrieved_docs = [transcripts[doc] for doc in retrieved_docs]
    relevant_docs = [qrel['episode_filename_prefix'] for qrel in q_rels if qrel['relevance'] >= 1]
    for k in [5, 10, 25, 20, 30]:
        if k <= doc_k:
            relevant_retrieved = [doc['episode_filename_prefix'] for doc in retrieved_docs[:k] if
                                  doc['episode_filename_prefix'] in relevant_docs]
            results[f'document precision@{k}'] = len(relevant_retrieved) / k

    split_transcripts = []
    for doc in retrieved_docs:
        split_transcripts.extend(split_transcript(doc))
    top_k = retrieve_segments(query, split_transcripts, k=seg_k, n_grams=n_grams, verbose=verbose)

    # evaluate results
    scores = []
    true_relevances = []
    for segment, score in top_k:
        scores.append(score)
        relevance = 0
        for qrel in q_rels:
            if qrel['episode_filename_prefix'] == segment['episode_filename_prefix'] \
                    and qrel['time'] == segment['start_time']:
                relevance = qrel['relevance']
                break
        true_relevances.append(relevance)

    for k in [5, 10, 25, 20, 30]:
        if k <= seg_k:
            relevant_retrieved_seg = true_relevances[:k]
            precision = len([rel for rel in relevant_retrieved_seg if rel >= 1]) / k
            results[f'segment precision@{k}'] = precision

    # for q_rel in q_rels:
    #     true_relevances.append(q_rel['relevance'])
    #     score = 0
    #     for pair in top_k:
    #         segment = pair[0]
    #         if q_rel['episode_filename_prefix'] == segment['episode_filename_prefix'] \
    #                 and q_rel['time'] == segment['start_time']:
    #             score = pair[1]
    #             top_k.remove(pair)
    #             break
    #     scores.append(score)
    # true_relevances.extend([0] * len(top_k))
    # scores.extend([pair[1] for pair in top_k])

    if verbose > 0:
        print(f"True relevance: {true_relevances}")
        print(f"Predicted relevance: {scores}")
    ndcg = ndcg_score([true_relevances], [scores])
    ndcg10 = ndcg_score([true_relevances], [scores], k=10)
    ndcg20 = ndcg_score([true_relevances], [scores], k=20)

    results['query'] = f"{query['id']}: {query['query']}"
    results['top_k'] = top_k
    results['ndcg'] = ndcg
    results['ndcg10'] = ndcg10
    results['ndcg20'] = ndcg20
    results['total relevant'] = len([qrel for qrel in q_rels if qrel['relevance'] > 0])
    results['total relevant retrieved'] = len([rel for rel in true_relevances if rel > 0])
    results['total retrieved'] = len(true_relevances)
    results['true relevances'] = true_relevances
    results['scores'] = scores

    if verbose:
        print_result(results)
    return results


def process_query_type(type, podcasts, queries, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    ndcgs = []
    ndcg10s = []
    ndcg20s = []
    precisions = [[] for _ in range(5)]
    segment_precisions = [[] for _ in range(5)]
    result = {
        'total relevant': 0,
        'total relevant retrieved': 0,
        'total retrieved': 0,
        'scores': [],
        'true relevances': []
    }
    relevant_queries = [query for query in queries if query['type'] == type]
    for query in relevant_queries:
        relevant_q_rels = [q_rel for q_rel in q_rels if q_rel['id'] == query['id']]
        if verbose == 2:
            print(f"Query: {query}")
            print(f"Relevant q_rels: {relevant_q_rels}")
        query_result = process_query(query, podcasts, relevant_q_rels,
                                     doc_k=doc_k,
                                     seg_k=seg_k,
                                     verbose=verbose,
                                     n_grams=n_grams)
        result['total relevant'] += query_result['total relevant']
        result['total relevant retrieved'] += query_result['total relevant retrieved']
        result['total retrieved'] += query_result['total retrieved']
        result['scores'].extend(query_result['scores'])
        result['true relevances'].extend(query_result['true relevances'])
        ndcgs.append(query_result['ndcg'])
        ndcg10s.append(query_result['ndcg10'])
        ndcg20s.append(query_result['ndcg20'])
        for i, k in enumerate([5, 10, 25, 20, 30]):
            try:
                precisions[i].append(query_result[f'document precision@{k}'])
            except KeyError:
                pass
            try:
                segment_precisions[i].append(query_result[f'segment precision@{k}'])
            except KeyError:
                pass

    result['ndcg'] = np.mean(ndcgs)
    result['ndcg10'] = np.mean(ndcg10s)
    result['ndcg20'] = np.mean(ndcg20s)
    result['ndcgs'] = ndcgs
    result['ndcg10s'] = ndcg10s
    result['ndcg20s'] = ndcg20s
    for i, k in enumerate([5, 10, 25, 20, 30]):
        result[f'document precision@{k}'] = np.mean(precisions[i])
        result[f'document precisions@{k}'] = precisions[i]
        result[f'segment precision@{k}'] = np.mean(segment_precisions[i])
        result[f'segment precisions@{k}'] = segment_precisions[i]
    result['query'] = type

    print_result(result)
    return result


def process_all_queries(podcasts, queries, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    now = datetime.now()
    ndcgs_all = []
    ndcg10s_all = []
    ndcg20s_all = []
    precisions = [[] for _ in range(5)]
    segment_precisions = [[] for _ in range(5)]
    result = {
        'total relevant': 0,
        'total relevant retrieved': 0,
        'total retrieved': 0,
        'scores': [],
        'true relevances': []
    }
    for type in ['topical', 'refinding', 'known item']:
        query_result = process_query_type(type, podcasts, queries, q_rels,
                                                     doc_k=doc_k,
                                                     seg_k=seg_k,
                                                     verbose=verbose,
                                                     n_grams=n_grams)
        ndcgs_all.extend(query_result['ndcgs'])
        ndcg10s_all.extend(query_result['ndcg10s'])
        ndcg20s_all.extend(query_result['ndcg20s'])
        result['true relevances'].extend(query_result['true relevances'])
        result['scores'].extend(query_result['scores'])
        for i, k in enumerate([5, 10, 25, 20, 30]):
            precisions[i].extend(query_result[f'document precisions@{k}'])
            segment_precisions[i].extend(query_result[f'segment precisions@{k}'])
        result['total relevant'] += query_result['total relevant']
        result['total relevant retrieved'] += query_result['total relevant retrieved']
        result['total retrieved'] += query_result['total retrieved']

    result['ndcg'] = np.mean(ndcgs_all)
    result['ndcg10'] = np.mean(ndcg10s_all)
    result['ndcg20'] = np.mean(ndcg20s_all)
    for i, k in enumerate([5, 10, 25, 20, 30]):
        result[f'document precision@{k}'] = np.mean(precisions[i])
        result[f'segment precision@{k}'] = np.mean(segment_precisions[i])

    result['query'] = 'all'
    print_result(result)
    print(f"Finished in {datetime.now() - now}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_k', type=int, default=100)
    parser.add_argument('--seg_k', type=int, default=50)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--query_type', type=str, default='all')
    parser.add_argument('--n_grams', type=int, default=3)
    args = parser.parse_args()

    assert args.query_type in ['all', 'topical', 'refinding', 'known item']
    assert args.verbose in [0, 1, 2]

    verbose = args.verbose
    podcasts = read_podcasts().to_dict("index")
    all_queries = read_queries(test=args.test)
    q_rels = read_qrels(test=args.test)

    if args.query > 0:
        for query in all_queries:
            if query['id'] == args.query:
                relevant_q_rels = [q_rel for q_rel in q_rels if q_rel['id'] == query['id']]
                result = process_query(query, podcasts, relevant_q_rels,
                                       doc_k=args.doc_k,
                                       seg_k=args.seg_k,
                                       verbose=min(verbose, 1),
                                       n_grams=args.n_grams)
                break

    elif args.query_type != 'all':
        process_query_type(args.query_type, podcasts, all_queries, q_rels,
                           doc_k=args.doc_k,
                           seg_k=args.seg_k,
                           verbose=verbose,
                           n_grams=args.n_grams)
    else:
        process_all_queries(podcasts, all_queries, q_rels,
                            doc_k=args.doc_k,
                            seg_k=args.seg_k,
                            verbose=verbose,
                            n_grams=args.n_grams)
