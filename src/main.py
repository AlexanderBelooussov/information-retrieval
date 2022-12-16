import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from bert_reranker import rerank_segments
from document_retrieval import get_relevant_documents_query
from load_data import read_queries, read_qrels, read_podcasts, make_segments, \
    read_segments
from keybert_rank import rank_segments_keybert


def normalize(l):
    m = min(l)
    s = max(l) - m
    if s == 0: return [0] * len(l)
    norm = [(float(i) - m) / s for i in l]
    return norm


def combine_scores(l, method):
    if method == "rerank":
        combined_score = l[-1]
    elif method == "mean":
        combined_score = np.mean(l)
    elif method == "min":
        combined_score = np.min(l)
    elif method == "max":
        combined_score = np.max(l)
    else:
        raise ValueError(f"{method} is not a valid method of combining scores")
    return combined_score


def evaluate_result(query, top_k, q_rels, result=None, seg_k=1000, verbose=0):
    if result is None:
        result = {}

    # evaluate results
    scores = []
    true_relevances = []
    used = []
    for segment, score in top_k:
        scores.append(score)
        relevance = 0
        for qrel in q_rels:
            if qrel['episode_filename_prefix'] == segment['episode_filename_prefix'] \
                    and qrel['time'] == segment['start_time']:
                relevance = qrel['relevance']
                used.append(qrel)
                break
        true_relevances.append(relevance)

    # query_id = query['id']
    # for qrel in q_rels:
    #     if qrel['id'] == query_id and qrel not in used:
    #         relevance = qrel['relevance']
    #         true_relevances.append(relevance)
    #         scores.append(0)

    paired = [(scores[i], true_relevances[i]) for i in range(len(true_relevances))]
    for k in [5, 10, 15, 20, 30]:
        if k <= seg_k:
            relevant_retrieved_seg = sorted(paired, reverse=True)[:k]
            precision = len([x[1] for x in relevant_retrieved_seg if x[1] >= 1]) / k
            result[f'segment precision@{k}'] = precision

    ndcg = ndcg_score([true_relevances], [scores])
    ndcg10 = ndcg_score([true_relevances], [scores], k=10)
    ndcg20 = ndcg_score([true_relevances], [scores], k=20)

    result['query'] = f"{query['id']}: {query['query']}"
    result['top_k'] = top_k
    result['ndcg'] = ndcg
    result['ndcg10'] = ndcg10
    result['ndcg20'] = ndcg20
    result['total relevant'] = len([qrel for qrel in q_rels if qrel['relevance'] > 0])
    result['total relevant retrieved'] = len([rel for rel in true_relevances if rel > 0])
    result['total retrieved'] = len(true_relevances)
    result['true relevances'] = true_relevances
    result['scores'] = scores
    result["segments"] = [x[0] for x in top_k]

    if verbose > 0:
        paired = [(scores[i], true_relevances[i]) for i in range(len(true_relevances))]
        print(f"True relevance: {sorted(paired, key=lambda x: (x[1], x[0]), reverse=True)}")
        print(f"Predicted relevance: {sorted(paired, reverse=True)}")

    return result


def print_result(result):
    print(f"Results for: {result['query']}")
    print(f"Total number retrieved: {result['total retrieved']}")
    print(f"Total relevant: {result['total relevant']}")
    print(f"Total relevant retrieved: {result['total relevant retrieved']}")
    print(f"Mean NDCG: {result['ndcg']}")
    print(f"Mean NDCG short: {result['ndcg']:.4f}")
    print()
    for k in [5, 10, 15, 20, 30]:
        try:
            if not np.isnan([result[f'document precision@{k}']])[0]:
                print(f"Document Precision@{k} = {result[f'document precision@{k}']}")
        except KeyError:
            pass
    print()
    for k in [5, 10, 15, 20, 30]:
        try:
            print(f"Segment Precision@{k} = {result[f'segment precision@{k}']}")
        except KeyError:
            pass
    print("\n")
    # display = PrecisionRecallDisplay.from_predictions([1 if rel > 0 else 0 for rel in result["true relevances"]], result["scores"])
    # _ = display.ax_.set_title(f"{result['query']} Precision-Recall curve")
    # plt.show()


def process_query_segments_index(query, all_segments, q_rels, bm_k=1000, seg_k=1000, n_grams=1, verbose=0,
                                 query_expansion=False, index='text_show_ep', keybert=False, mix="rerank", rerank=True):
    """
    Process a query starting from an index on segments
    """
    retrieved_segments, bm_score = get_relevant_documents_query(query, k=bm_k, use_keybert=keybert,
                                                                use_description=keybert,
                                                                index_type=index, n_grams=n_grams,
                                                                query_expansion=query_expansion)
    bm_score = normalize(bm_score)
    all_scores = {}
    relevant_segments = []
    for i, retrieved_segment in enumerate(retrieved_segments):
        all_scores[retrieved_segment] = [bm_score[i]]
        relevant_segments.append(all_segments[retrieved_segment])

    # keybert scores
    if keybert and mix != "rerank":
        for seg in relevant_segments:
            seg["text"] = seg["contents"]
        kb = rank_segments_keybert(query, relevant_segments, k=len(relevant_segments), n_grams=n_grams, verbose=verbose,
                                   query_expansion=False, use_description=True)
        keybert_segments = [k[0] for k in kb]
        keybert_scores = normalize([k[1] for k in kb])
        for i, segment in enumerate(keybert_segments):
            all_scores[segment["id"]].append(keybert_scores[i])

    # reranking
    if rerank:
        rerank_scores = rerank_segments(relevant_segments, query["query"] + ". " + query["description"])
        # rerank_scores = rerank_segments(relevant_segments, query["query"])

        for i, key in enumerate(all_scores.keys()):
            all_scores[key].append(rerank_scores[i])

    final_scores = []
    for key, value in all_scores.items():
        segment = all_segments[key]
        final_score = combine_scores(value, mix)
        final_scores.append((segment, final_score))

    top_k = list(final_scores)
    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:seg_k]

    result = evaluate_result(query, top_k, q_rels, None, seg_k, verbose)
    if verbose:
        print_result(result)

    return result


def process_query(query, transcripts, q_rels, bm_k=100, seg_k=50, n_grams=1, verbose=0, query_expansion=False,
                  index='text_show_ep', keybert=True, mix="rerank", rerank=True):
    """
    Process a query starting with an index on entire podcast transcripts
    Defer to the segment index function if index on segments is passed
    """
    if index in ["all_segments", "segments_ep", "segments_show_ep"]:
        return process_query_segments_index(query, transcripts, q_rels, bm_k, seg_k, n_grams, verbose, query_expansion,
                                            index, keybert, mix=mix, rerank=rerank)
    result = {}
    retrieved_docs, doc_scores = get_relevant_documents_query(query, k=bm_k, use_keybert=keybert,
                                                              use_description=True,
                                                              index_type=index, n_grams=n_grams,
                                                              query_expansion=query_expansion)
    retrieved_docs = [transcripts[doc] for doc in retrieved_docs]
    relevant_docs = [qrel['episode_filename_prefix'] for qrel in q_rels if qrel['relevance'] >= 1]
    for k in [5, 10, 15, 20, 30]:
        if k <= bm_k:
            relevant_retrieved = [doc['episode_filename_prefix'] for doc in retrieved_docs[:k] if
                                  doc['episode_filename_prefix'] in relevant_docs]
            result[f'document precision@{k}'] = len(relevant_retrieved) / k

    split_transcripts = []
    for doc in retrieved_docs:
        split_transcripts.extend(make_segments(doc))

    if mix != "rerank" or not rerank:
        top_k = rank_segments_keybert(query, split_transcripts,
                                      k=seg_k if not rerank else len(split_transcripts),
                                      n_grams=n_grams,
                                      verbose=verbose,
                                      # query_expansion=query_expansion)
                                      query_expansion=False)
    else:
        top_k = []
        for doc in split_transcripts:
            top_k.append((doc, 0))

    if rerank:
        segments = []
        for s in top_k:
            segment = s[0]
            segment["contents"] = segment["text"] + ". " + str(segment["show_name"] or "") + ". " + str(
                segment["show_description"] or "") + \
                                  ". " + str(segment["episode_name"] or "") + ". " + str(
                segment["ep_description"] or "")
            segments.append(segment)

        rerank_scores = rerank_segments(segments, query["query"] + ". " + query["description"])

        segment_scores = normalize([seg[1] for seg in top_k])  # TODO should probably be done in segment retrieval
        combined_scores = []
        for i, seg in enumerate(top_k):
            score = segment_scores[i]
            rerank_score = rerank_scores[i]
            combined_score = combine_scores([score, rerank_score], mix)
            combined_scores.append((seg[0], combined_score))
        top_k = combined_scores

    top_k = sorted(top_k, key=lambda x: x[1], reverse=True)[:seg_k]

    result = evaluate_result(query, top_k, q_rels, result, seg_k, verbose)

    if verbose:
        print_result(result)
    return result


def process_query_type(type, podcasts, queries, q_rels, bm_k=100, seg_k=50, n_grams=1, verbose=0,
                       query_expansion=False, type_aware=False, index="text_show_ep", keybert=True, mix="min",
                       rerank=True):
    """
    Process all queries of a type
    """
    if type_aware and type == 'topical':
        query_expansion = False
    elif type_aware and type in ['refinding', 'known item']:
        query_expansion = True
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
                                     bm_k=bm_k,
                                     seg_k=seg_k,
                                     verbose=verbose,
                                     n_grams=n_grams,
                                     query_expansion=query_expansion,
                                     index=index,
                                     keybert=keybert,
                                     mix=mix,
                                     rerank=rerank)
        result['total relevant'] += query_result['total relevant']
        result['total relevant retrieved'] += query_result['total relevant retrieved']
        result['total retrieved'] += query_result['total retrieved']
        result['scores'].extend(query_result['scores'])
        result['true relevances'].extend(query_result['true relevances'])
        ndcgs.append(query_result['ndcg'])
        ndcg10s.append(query_result['ndcg10'])
        ndcg20s.append(query_result['ndcg20'])
        for i, k in enumerate([5, 10, 15, 20, 30]):
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
    for i, k in enumerate([5, 10, 15, 20, 30]):
        result[f'document precision@{k}'] = np.mean(precisions[i])
        result[f'document precisions@{k}'] = precisions[i]
        result[f'segment precision@{k}'] = np.mean(segment_precisions[i])
        result[f'segment precisions@{k}'] = segment_precisions[i]
    result['query'] = type

    print_result(result)
    return result


def process_all_queries(podcasts, queries, q_rels, bm_k=100, seg_k=50, n_grams=1, verbose=0, query_expansion=False,
                        type_aware=False, index="text_show_ep", keybert=True, mix="min", rerank=True):
    """
    Process all queries
    """
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
                                          bm_k=bm_k,
                                          seg_k=seg_k,
                                          verbose=verbose,
                                          n_grams=n_grams,
                                          query_expansion=query_expansion,
                                          type_aware=type_aware,
                                          index=index,
                                          keybert=keybert,
                                          mix=mix,
                                          rerank=rerank)
        ndcgs_all.extend(query_result['ndcgs'])
        ndcg10s_all.extend(query_result['ndcg10s'])
        ndcg20s_all.extend(query_result['ndcg20s'])
        result['true relevances'].extend(query_result['true relevances'])
        result['scores'].extend(query_result['scores'])
        for i, k in enumerate([5, 10, 15, 20, 30]):
            precisions[i].extend(query_result[f'document precisions@{k}'])
            segment_precisions[i].extend(query_result[f'segment precisions@{k}'])
        result['total relevant'] += query_result['total relevant']
        result['total relevant retrieved'] += query_result['total relevant retrieved']
        result['total retrieved'] += query_result['total retrieved']

    result['ndcg'] = np.mean(ndcgs_all)
    result['ndcg10'] = np.mean(ndcg10s_all)
    result['ndcg20'] = np.mean(ndcg20s_all)
    for i, k in enumerate([5, 10, 15, 20, 30]):
        result[f'document precision@{k}'] = np.mean(precisions[i])
        result[f'segment precision@{k}'] = np.mean(segment_precisions[i])

    result['query'] = 'all'
    print_result(result)
    print(f"Finished in {datetime.now() - now}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bm_k', type=int, default=30)
    parser.add_argument('--seg_k', type=int, default=1000)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--query', type=int, default=0)
    parser.add_argument('--query_type', type=str, default='all')
    parser.add_argument('--n_grams', type=int, default=3)
    parser.add_argument('--query_expansion', action='store_true', default=False)
    parser.add_argument('--type_aware', action='store_true', default=False)
    parser.add_argument('--keybert', action='store_true', default=False)
    parser.add_argument('--index', type=str, default="text_show_ep")
    parser.add_argument('--mix', type=str, default="min")
    parser.add_argument('--rerank', action='store_true', default=False)

    args = parser.parse_args()

    assert args.query_type in ['all', 'topical', 'refinding', 'known item']
    assert args.verbose in [0, 1, 2]
    assert args.index in ["all_segments", "all_text", "segments_ep", "segments_show_ep", "text_ep", "text_show_ep"]
    assert not (args.mix == "rerank" and not args.rerank)

    verbose = args.verbose
    if verbose:
        print(f"Loading data...", end='')
    if "text" in args.index:
        podcasts = read_podcasts().to_dict("index")
    else:
        segment_metadata = pd.read_feather("data/feathers/segment_metadata.feather").set_index("id").to_dict(
            orient="index")
        all_segments = read_segments()
        for i in tqdm(range(len(all_segments)), desc="Combining segment data", leave=False):
            assert i + 1 == all_segments[i]['id']
            segment_metadata[i + 1].update(all_segments[i])
        all_segments = segment_metadata
        podcasts = all_segments

    all_queries = read_queries(test=args.test)
    q_rels = read_qrels(test=args.test)
    if verbose:
        print(f"\r", end='')

    if args.query > 0:
        for query in all_queries:
            if query['id'] == args.query:
                relevant_q_rels = [q_rel for q_rel in q_rels if q_rel['id'] == query['id']]
                result = process_query(query, podcasts, relevant_q_rels,
                                       bm_k=args.bm_k,
                                       seg_k=args.seg_k,
                                       verbose=max(verbose, 1),
                                       n_grams=args.n_grams,
                                       query_expansion=args.query_expansion,
                                       index=args.index,
                                       keybert=args.keybert,
                                       mix=args.mix,
                                       rerank=args.rerank)
                break

    elif args.query_type != 'all':
        process_query_type(args.query_type, podcasts, all_queries, q_rels,
                           bm_k=args.bm_k,
                           seg_k=args.seg_k,
                           verbose=verbose,
                           n_grams=args.n_grams,
                           query_expansion=args.query_expansion,
                           type_aware=args.type_aware,
                           index=args.index,
                           keybert=args.keybert,
                           mix=args.mix,
                           rerank=args.rerank)
    else:
        process_all_queries(podcasts, all_queries, q_rels,
                            bm_k=args.bm_k,
                            seg_k=args.seg_k,
                            verbose=verbose,
                            n_grams=args.n_grams,
                            query_expansion=args.query_expansion,
                            type_aware=args.type_aware,
                            index=args.index,
                            keybert=args.keybert,
                            mix=args.mix,
                            rerank=args.rerank)

    print(f"args: index={args.index}, bm_k={args.bm_k}, seg_k={args.seg_k}, n_grams={args.n_grams}, "
          f"query_expansion={args.query_expansion}, type_aware={args.type_aware}, keybert={args.keybert}, "
          f"rerank={args.rerank}, mix={args.mix}")
