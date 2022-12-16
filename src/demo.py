import argparse

import pandas as pd
from tqdm import tqdm

from load_data import read_queries, read_segments, read_qrels, read_podcasts
from main import process_query

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bm_k', type=int, default=50)
    parser.add_argument('--seg_k', type=int, default=5)
    parser.add_argument('--n_grams', type=int, default=3)
    parser.add_argument('--query_expansion', action='store_true', default=False)
    parser.add_argument('--type_aware', action='store_true', default=False)
    parser.add_argument('--keybert', action='store_true', default=False)
    parser.add_argument('--index', type=str, default="segments_show_ep")
    parser.add_argument('--mix', type=str, default="mean")
    parser.add_argument('--rerank', action='store_true', default=False)

    args = parser.parse_args()

    assert args.index in ["all_segments", "all_text", "segments_ep", "segments_show_ep", "text_ep", "text_show_ep"]
    assert not (args.mix == "rerank" and not args.rerank)

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
    train_queries = read_queries()
    test_queries = read_queries(True)
    query = None
    print(f"1 - Query from number")
    print(f"2 - New query")
    x = int(input(f"> "))
    if x == 1:
        qid = int(input(f"Query id: "))
        for q in test_queries + train_queries:
            if q['id'] == qid:
                query = q
                print(f"Chosen query: {query}")
    elif x == 2:
        qid = 0
        qtype = "new"
        qquery = input(f"Short query: ")
        qdescription = input(f"Query description: ")
        query = {"id": qid,
                 "query": qquery,
                 "description": qdescription,
                 "type": qtype}
    else:
        raise ValueError("Invalid input")


    relevant_q_rels = []
    result = process_query(query, podcasts, relevant_q_rels,
                           bm_k=args.bm_k,
                           seg_k=args.seg_k,
                           verbose=0,
                           n_grams=args.n_grams,
                           query_expansion=args.query_expansion,
                           index=args.index,
                           keybert=args.keybert,
                           mix=args.mix,
                           rerank=args.rerank)
    segments = result["segments"]
    for seg in segments:
        print(f"{seg}\n")


