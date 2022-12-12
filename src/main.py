import argparse
from datetime import datetime

import numpy as np

from document_retrieval import get_relevant_documents_query
from load_data import read_metadata, read_queries, read_qrels, read_podcasts, read_transcripts
from segment_retrieval import split_transcript, retrieve_segments
from tqdm import tqdm
from sklearn.metrics import ndcg_score, dcg_score


def process_query(query, transcripts, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    relevant_docs, _ = get_relevant_documents_query(query, k=doc_k, use_keybert=True, use_description=True,
                                                    index_type='show_episode_info', n_grams=n_grams)
    relevant_docs = [transcripts[doc] for doc in relevant_docs]
    split_transcripts = []
    for doc in relevant_docs:
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

    if verbose > 0:
        print(f"Query {query['id']}, {query['query']}: {ndcg}, {ndcg10}, {ndcg20}")
    return ndcg, ndcg10, ndcg20


def process_query_type(type, podcasts, queries, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    ndcgs = []
    ndcg10s = []
    ndcg20s = []
    relevant_queries = [query for query in queries if query['type'] == type]
    for query in relevant_queries:
        relevant_q_rels = [q_rel for q_rel in q_rels if q_rel['id'] == query['id']]
        if verbose == 2:
            print(f"Query: {query}")
            print(f"Relevant q_rels: {relevant_q_rels}")
        ndcg, ndcg10, ndcg20 = process_query(query, podcasts, relevant_q_rels,
                                             doc_k=doc_k,
                                             seg_k=seg_k,
                                             verbose=verbose,
                                             n_grams=n_grams)
        ndcgs.append(ndcg)
        ndcg10s.append(ndcg10)
        ndcg20s.append(ndcg20)

    print(f"Results for type: {type}")
    print(f"Average NDCG: {sum(ndcgs) / len(ndcgs)}")
    print(f"Average NDCG@10: {sum(ndcg10s) / len(ndcg10s)}")
    print(f"Average NDCG@20: {sum(ndcg20s) / len(ndcg20s)}")
    print()
    return ndcgs, ndcg10s, ndcg20s


def process_all_queries(podcasts, queries, q_rels, doc_k=100, seg_k=50, n_grams=1, verbose=0):
    now = datetime.now()
    ndcgs_all = []
    ndcg10s_all = []
    ndcg20s_all = []
    for type in ['topical', 'refinding', 'known item']:
        ndcgs, ndcg10s, ndcg20s = process_query_type(type, podcasts, queries, q_rels,
                                                     doc_k=doc_k,
                                                     seg_k=seg_k,
                                                     verbose=verbose,
                                                     n_grams=n_grams)
        ndcgs_all.extend(ndcgs)
        ndcg10s_all.extend(ndcg10s)
        ndcg20s_all.extend(ndcg20s)

    print(f"Overall results")
    print(f"Average NDCG: {sum(ndcgs_all) / len(ndcgs_all)}")
    print(f"Average NDCG@10: {sum(ndcg10s_all) / len(ndcg10s_all)}")
    print(f"Average NDCG@20: {sum(ndcg20s_all) / len(ndcg20s_all)}")
    print(f"Finished in {datetime.now() - now}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_k', type=int, default=100)
    parser.add_argument('--seg_k', type=int, default=50)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--query_type', type=str, default='all')
    parser.add_argument('--n_grams', type=int, default=1)
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
                ndcg, ndcg10, ndcg20 = process_query(query, podcasts, relevant_q_rels,
                                                     doc_k=args.doc_k,
                                                     seg_k=args.seg_k,
                                                     verbose=verbose,
                                                     n_grams=args.n_grams)
                print(f"Query {query['id']}, {query['query']}: {ndcg}, {ndcg10}, {ndcg20}")
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
