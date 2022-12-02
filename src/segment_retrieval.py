from tqdm import tqdm

from load_data import read_metadata, read_transcripts
import numpy as np


def split_transcript(transcript):
    id = transcript['id']
    duration = transcript['start_times'][-1]
    start_times = transcript['start_times']
    words = transcript['words']
    splits = []
    for start in range(0, int(duration-60), 60):
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
    print(f"Split {id} into {len(splits)} segments")
    return splits


if __name__ == '__main__':
    transcripts = read_transcripts().to_dict("records")
    for transcript in tqdm(transcripts):
        split_transcript(transcript)

