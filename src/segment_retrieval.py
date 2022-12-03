from tqdm import tqdm

from load_data import read_metadata, read_transcripts
import numpy as np
from keybert import KeyBERT


def split_transcript(transcript):
    id = transcript['id']
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
        splits.append({
            'text': text,
            'start_time': start,
            'transcript_id': id
        })
    return splits


def retrieve_segments(query, split_transcripts):
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    q_keywords = kw_model.extract_keywords(query['query'], keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
    print(q_keywords)
    for transcript in tqdm(split_transcripts, desc='Retrieving segments'):
        text = transcript['text']
        tc_keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10)
        print(tc_keywords)
        break
        # TODO continue here


if __name__ == '__main__':
    transcripts = read_transcripts().to_dict("records")
    for transcript in tqdm(transcripts):
        split_transcript(transcript)
