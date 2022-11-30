import math

import pandas as pd
import os
import json
from tqdm import tqdm
import pdcast as pdc


pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None
data_dir = 'A:/Pycharm Projects/information-retrieval/data/'


# read metadata.tsv
def read_metadata():
    if os.path.exists(data_dir + 'feathers/metadata.feather'):
        metadata = pd.read_feather(data_dir + 'feathers/metadata.feather')
    else:
        metadata = pd.read_csv(data_dir + 'spotify-podcasts-2020/metadata.tsv', sep='\t')
        metadata.to_feather(data_dir + 'feathers/metadata.feather')
    return metadata


def read_transcripts():
    if os.path.exists(data_dir + 'feathers/transcripts.feather'):
        transcripts = pd.read_feather(data_dir + 'feathers/transcripts.feather')
        return transcripts
    raise Exception('Transcripts not found')


def read_transcript(show, episode):
    folder1 = str(show[5]).upper()
    folder2 = str(show[6]).upper()
    path = data_dir + 'spotify-podcasts-2020/podcasts-transcripts/' \
           + folder1 + '/' \
           + folder2 + '/' \
           + show + '/' \
           + episode + '.json'
    transcript = json.load(open(path, 'r'))
    transcript = transcript['results']
    start_times = []
    words = []
    text = ''
    for ts in transcript:
        sentence = ts['alternatives'][0]
        if 'transcript' in sentence.keys():
            text += sentence['transcript']
            for word in sentence['words']:
                start_times.append(float(word['startTime'][0:-1]))
                words.append(word['word'])
    return {
        'text': text,
        'start_times': start_times,
        'words': words
    }


def concatenate_podcasts():
    all_parts = [file for file in os.listdir(data_dir + 'feathers') if file.startswith('podcasts_')]
    while len(all_parts) > 1:
        print(all_parts)
        for i in range(0, len(all_parts), 2):
            if i + 1 < len(all_parts):
                print(f"Concatenating {all_parts[i]} and {all_parts[i+1]}")
                concatenation = pd.concat([pd.read_feather(data_dir + 'feathers/' + all_parts[i]),
                                        pd.read_feather(data_dir + 'feathers/' + all_parts[i + 1])]).reset_index(drop=True)
                name1 = all_parts[i].split('.')[0]
                name2 = all_parts[i + 1].split('.')[0]
                concatenation.to_feather(data_dir + 'feathers/' + name1 + '_' + name2 + '.feather', chunksize=1000)
                os.remove(data_dir + 'feathers/' + all_parts[i])
                os.remove(data_dir + 'feathers/' + all_parts[i + 1])
        all_parts = [file for file in os.listdir(data_dir + 'feathers') if file.startswith('podcasts_')]

    # rename the last file
    os.rename(data_dir + 'feathers/' + all_parts[0], data_dir + 'feathers/podcasts.feather')
    podcasts = pd.read_feather(data_dir + 'feathers/podcasts.feather')
    return podcasts


def read_podcasts(n_parts=10):
    if os.path.exists(data_dir + 'feathers/podcasts.feather'):
        podcasts = pd.read_feather(data_dir + 'feathers/podcasts.feather')
        return podcasts
    metadata = read_metadata()
    podcasts = []
    n_parts = 10
    podcasts_per_part = math.ceil(len(metadata) / n_parts)
    part = 0
    for row in tqdm(metadata.to_numpy(), desc='Reading podcasts'):
        show_filename_prefix = row[10]
        episode_filename_prefix = row[11]
        podcast_info = read_transcript(show_filename_prefix, episode_filename_prefix)
        podcast_info['show_name'] = row[1]
        podcast_info['show_description'] = row[2]
        podcast_info['episode_name'] = row[7]
        podcast_info['ep_description'] = row[8]
        podcast_info['show_filename_prefix'] = show_filename_prefix
        podcast_info['episode_filename_prefix'] = episode_filename_prefix
        podcasts.append(podcast_info)
        if len(podcasts) == podcasts_per_part:
            podcasts_df = pd.DataFrame(podcasts)
            podcasts = []
            podcasts_df.to_feather(data_dir + 'feathers/podcasts_' + str(part) + '.feather')
            part += 1
            del podcasts_df
    return concatenate_podcasts()


def split_metadata():
    """
    Only keeps relevant information from the metadata
    :return:
    """
    podcasts = read_podcasts()
    metadata = podcasts[['show_name', 'show_description', 'episode_name', 'ep_description', 'show_filename_prefix',
                         'episode_filename_prefix']]
    metadata = metadata.reset_index(drop=False).rename(columns={'index': 'id'})
    metadata['id'] = pd.to_numeric(metadata['id'], downcast='integer')
    del podcasts
    metadata.to_feather(data_dir + 'feathers/metadata.feather', chunksize=1000)


def split_transcripts():
    """
    Removes some columns and adds id column
    :return:
    """
    podcasts = read_podcasts()
    transcripts = podcasts[['text', 'start_times', 'words']].reset_index(drop=False).rename(columns={'index': 'id'})
    transcripts['id'] = pd.to_numeric(transcripts['id'], downcast='integer')
    del podcasts
    print(f"Saving {len(transcripts)} transcripts")
    transcripts.to_feather(data_dir + 'feathers/transcripts.feather', chunksize=1000)


def split_transcripts_metadata():
    split_metadata()
    split_transcripts()


if __name__ == '__main__':
    read_podcasts(10)
    split_transcripts_metadata()