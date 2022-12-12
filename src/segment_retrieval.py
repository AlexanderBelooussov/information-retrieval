from tqdm import tqdm

from load_data import read_metadata, read_transcripts
import numpy as np
from keybert import KeyBERT


def split_transcript(transcript):
    duration = transcript['start_times'][-1]
    start_times = transcript['start_times']
    words = transcript['words']
    splits = []
    for start in range(0, int(duration - 60), 60):
        end = start + 120
        start_index = np.where(start_times > start)[0][0]
        start_index = max(0, start_index - 1)
        end_index = np.where(start_times > end)
        end_index = end_index[0][0] if len(end_index[0]) else len(start_times)
        text = ' '.join(words[start_index:end_index])
        split_dict = transcript.copy()
        split_dict['text'] = text
        split_dict['start_time'] = start

        splits.append(split_dict)
    return splits


def retrieve_segments(query, split_transcripts, use_description=True, k=50, n=25, verbose=0):
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    if use_description:
        q = query['query'] + '. ' + query['description']
    else:
        q = query['query']
    q_keywords = dict(kw_model.extract_keywords(q, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=n))
    # make relevance of query keywords 1
    # for word in query['query'].split():
    #     q_keywords[word] = 1
    if verbose == 2:
        print(f"Query keywords: {q_keywords}")
        print(f"Amount of transcripts: {len(split_transcripts)}")
    scores = []
    for transcript in tqdm(split_transcripts, desc=f"Retrieving segments for query {query['id']}", leave=verbose == 2, disable=verbose == 0):
        text = transcript['text']
        # tc_keywords = dict(kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=n))
        # TODO: test these options
        tc_keywords = dict(kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=n,
                                                     candidates=q_keywords.keys()))
        # tc_keywords = dict(kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=n,
        #                                              seed_keywords=q_keywords.keys()))
        keyword_intersection = set(q_keywords.keys()).intersection(set(tc_keywords.keys()))
        score = sum([q_keywords[k] * tc_keywords[k] for k in keyword_intersection])
        scores.append((transcript, score))

    # find top k using np.argmax
    top_k = sorted(scores, key=lambda x: x[1], reverse=True)[:k]
    if verbose == 2:
        for segment, score in top_k:
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
        split_transcript(transcript)
