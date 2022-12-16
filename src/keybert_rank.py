"""
Score segments using KeyBERT
"""

from keybert import KeyBERT
from tqdm import tqdm

from load_data import read_transcripts, make_segments
from query_expansion import expand_query


def rank_segments_keybert(query, split_transcripts, use_description=True, k=50, n=25, n_grams=1, verbose=0,
                          query_expansion=True):
    if query_expansion:
        n = min(n, 10)  # limit keywords if using query expansion
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    if use_description:
        q = query['query'] + '. ' + query['description']
    else:
        q = query['query']
    q_keywords = dict(kw_model.extract_keywords(q, keyphrase_ngram_range=(1, n_grams), stop_words='english', top_n=n))
    if verbose == 2:
        print(f"Query keywords: {q_keywords}")
        print(f"Amount of transcripts: {len(split_transcripts)}")
    if query_expansion:
        expanded_query = {}
        for keyword in q_keywords.keys():
            expanded_query.update(expand_query(keyword, q_keywords[keyword], n_grams))
        if verbose == 2:
            print(f"Expanded query: {expanded_query}")
        # get best n * 10 keywords
        expanded_query = dict(sorted(expanded_query.items(), key=lambda x: x[1], reverse=True)[:n * 10])
        q_keywords = expanded_query
    scores = []
    for transcript in tqdm(split_transcripts, desc=f"Ranking using KeyBERT for query {query['id']}", leave=verbose == 2,
                           disable=verbose == 0):
        text = transcript['text']
        tc_keywords = dict(
            kw_model.extract_keywords(text, keyphrase_ngram_range=(1, n_grams), stop_words='english', top_n=n,
                                      candidates=q_keywords.keys()))
        keyword_intersection = set(q_keywords.keys()).intersection(set(tc_keywords.keys()))
        score = sum([q_keywords[k] * tc_keywords[k] for k in keyword_intersection])
        scores.append((transcript, score))

    # find top k using np.argmax
    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    if verbose == 2:
        for segment, score in top_k[:10]:
            print(f"Score: {score}")
            print(f"Start time: {segment['start_time']}")
            print(f"Text: {segment['text']}")
            print(f"show_filename_prefix: {segment['show_filename_prefix']}")
            print(f"episode_filename_prefix: {segment['episode_filename_prefix']}")
            print()

    return top_k


if __name__ == '__main__':
    transcripts = read_transcripts().to_dict("records")
    for transcript in tqdm(transcripts):
        make_segments(transcript)
