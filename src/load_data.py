"""
Functions for loading data, generating feathers, making indices
"""

import json
import math
import os

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = 100
pd.options.display.width = None
data_dir = 'data/'


# read metadata.tsv
def read_metadata():
    if os.path.exists(data_dir + 'feathers/metadata.feather'):
        metadata = pd.read_feather(data_dir + 'feathers/metadata.feather')
    else:
        metadata = pd.read_csv(data_dir + 'spotify-podcasts-2020/metadata.tsv', sep='\t')
        metadata = metadata.reset_index(drop=False).rename(
            columns={'index': 'id', "episode_description": "ep_description"})
        metadata = metadata[
            ['id', 'show_name', 'show_description', 'episode_name', 'ep_description', 'show_filename_prefix',
             'episode_filename_prefix']]
        metadata['id'] = pd.to_numeric(metadata['id'], downcast='integer')
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
                print(f"Concatenating {all_parts[i]} and {all_parts[i + 1]}")
                concatenation = pd.concat([pd.read_feather(data_dir + 'feathers/' + all_parts[i]),
                                           pd.read_feather(data_dir + 'feathers/' + all_parts[i + 1])]).reset_index(
                    drop=True)
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


def read_podcasts(n_parts=10, concatenate=False):
    if os.path.exists(data_dir + 'feathers/podcasts.feather'):
        podcasts = pd.read_feather(data_dir + 'feathers/podcasts.feather')
        return podcasts
    elif not concatenate and os.path.exists(data_dir + 'feathers/podcasts_0.feather'):
        parts = [pd.read_feather(data_dir + 'feathers/podcasts_' + str(i) + '.feather') for i in range(n_parts)]
        podcasts = pd.DataFrame()
        for part in parts:
            podcasts = pd.concat([podcasts, part])
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
    if concatenate:
        return concatenate_podcasts()
    return read_podcasts(n_parts, False)


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


def make_indices():
    if not os.path.exists('data/all_text_index'):
        try:
            os.mkdir(data_dir + 'jsonl/all_text')
        except FileExistsError:
            pass
        if not os.path.exists('data/jsonl/all_text/all_text.json'):
            transcripts = read_transcripts()
            transcripts = transcripts[['id', 'text']]
            transcripts.rename(columns={'id': 'id', 'text': 'contents'}, inplace=True)
            with open(data_dir + 'jsonl/all_text/all_text.json', 'w', encoding='utf-8') as f:
                for record in tqdm(transcripts.to_dict(orient='records')):
                    json.dump(record, f)
                    f.write('\n')
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/all_text  --index data/all_text_index --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw")

    if not os.path.exists('data/text_ep_desc_title_index'):
        try:
            os.mkdir(data_dir + 'jsonl/text_ep_title_desc')
        except FileExistsError:
            pass
        if not os.path.exists('data/jsonl/text_ep_title_desc/text_ep_title_desc.json'):
            transcripts = read_podcasts()
            transcripts = transcripts[['text', 'ep_description', 'episode_name']]
            transcripts['ep_description'] = transcripts['ep_description'].fillna('')
            transcripts['episode_name'] = transcripts['episode_name'].fillna('')
            transcripts['contents'] = transcripts['episode_name'] + ' | ' + transcripts['ep_description'] + ' | ' + \
                                      transcripts['text']
            transcripts = transcripts[['contents']].reset_index(drop=False).rename(columns={'index': 'id'})
            with open(data_dir + 'jsonl/text_ep_title_desc/text_ep_title_desc.json', 'w', encoding='utf-8') as f:
                for record in tqdm(transcripts.to_dict(orient='records')):
                    json.dump(record, f)
                    f.write('\n')
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/text_ep_title_desc  --index data/text_ep_desc_title_index --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw")

    if not os.path.exists('data/text_show_ep_title_desc_index'):
        try:
            os.mkdir(data_dir + 'jsonl/text_show_ep_title_desc')
        except FileExistsError:
            pass
        if not os.path.exists('data/jsonl/text_show_ep_title_desc/text_show_ep_title_desc.json'):
            transcripts = read_podcasts()
            transcripts = transcripts[['text', 'ep_description', 'episode_name', 'show_name', 'show_description']]
            transcripts['ep_description'] = transcripts['ep_description'].fillna('')
            transcripts['episode_name'] = transcripts['episode_name'].fillna('')
            transcripts['show_description'] = transcripts['show_description'].fillna('')
            transcripts['show_name'] = transcripts['show_name'].fillna('')
            transcripts['contents'] = transcripts['show_name'] + ' | ' + transcripts['show_description'] + ' | ' + \
                                      transcripts['episode_name'] + ' | ' + transcripts['ep_description'] + ' | ' + \
                                      transcripts['text']
            transcripts = transcripts[['contents']].reset_index(drop=False).rename(columns={'index': 'id'})
            with open(data_dir + 'jsonl/text_show_ep_title_desc/text_show_ep_title_desc.json', 'w',
                      encoding='utf-8') as f:
                for record in tqdm(transcripts.to_dict(orient='records')):
                    json.dump(record, f)
                    f.write('\n')
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/text_show_ep_title_desc  --index data/text_show_ep_title_desc_index --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw")


def make_segment_indices():
    if not os.path.exists('data/all_segments_index'):
        try:
            os.mkdir(data_dir + 'jsonl/all_segments')
        except FileExistsError:
            pass
        try:
            os.mkdir(data_dir + 'jsonl/segments_ep_title_desc')
        except FileExistsError:
            pass
        try:
            os.mkdir(data_dir + 'jsonl/segments_show_ep_title_desc')
        except FileExistsError:
            pass
        if not os.path.exists('data/jsonl/all_segments/all_segments.json'):
            transcripts = read_podcasts()
            transcripts['episode_name'] = transcripts['episode_name'].fillna('')
            transcripts['ep_description'] = transcripts['ep_description'].fillna('')
            transcripts['show_name'] = transcripts['show_name'].fillna('')
            transcripts['show_description'] = transcripts['show_description'].fillna('')
            transcripts = transcripts.to_dict(orient='records')
            id_count = 1
            segment_metadata = []
            dumps1 = []
            dumps2 = []
            dumps3 = []
            for record in tqdm(transcripts, desc='Making segment indices'):
                segments = make_segments(record)
                for segment in segments:
                    segment['id'] = id_count
                    id_count += 1
                    metadata = {
                        'id': segment['id'],
                        'episode_filename_prefix': record['episode_filename_prefix'],
                        'start_time': segment['start_time'],
                    }
                    segment_metadata.append(metadata)
                    to_dump = {
                        'id': segment['id'],
                        'contents': segment['text']}
                    dumps1.append(to_dump)
                    to_dump2 = {
                        'id': segment['id'],
                        'contents': segment['episode_name'] + ' | ' + segment['ep_description'] + ' | ' + segment[
                            'text']}
                    dumps2.append(to_dump2)
                    to_dump3 = {
                        'id': segment['id'],
                        'contents': segment['show_name'] + ' | ' + segment['show_description'] + ' | ' + segment[
                            'episode_name'] + ' | ' + segment['ep_description'] + ' | ' + segment['text']}
                    dumps3.append(to_dump3)
            segment_metadata = pd.DataFrame(segment_metadata)
            segment_metadata.to_feather(data_dir + 'feathers/segment_metadata.feather')
            with open(data_dir + 'jsonl/all_segments/all_segments.json', 'w', encoding='utf-8') as f:
                for record in tqdm(dumps1, desc='Writing all segments'):
                    json.dump(record, f)
                    f.write('\n')
            with open(data_dir + 'jsonl/segments_ep_title_desc/segments_ep_title_desc.json', 'w',
                      encoding='utf-8') as f:
                for record in tqdm(dumps2, desc='Writing segments with episode title and description'):
                    json.dump(record, f)
                    f.write('\n')
            with open(data_dir + 'jsonl/segments_show_ep_title_desc/segments_show_ep_title_desc.json', 'w',
                      encoding='utf-8') as f:
                for record in tqdm(dumps3, desc='Writing segments with show title and description'):
                    json.dump(record, f)
                    f.write('\n')
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/all_segments  --index data/all_segments_index --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw")
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/segments_ep_title_desc  --index data/segments_ep_title_desc_index --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw")
        os.system(
            f"python -m pyserini.index.lucene --collection JsonCollection --input data/jsonl/segments_show_ep_title_desc  --index data/segments_show_ep_title_desc_index --generator DefaultLuceneDocumentGenerator --threads 4 --storePositions --storeDocvectors --storeRaw")


def read_queries(test=False):
    if test:
        location = data_dir + 'queries/podcasts_2020_topics_test.xml'
    else:
        location = data_dir + 'queries/podcasts_2020_topics_train.xml'
    with open(location, 'r', encoding='utf-8') as f:
        data = f.read()
    bs_data = BeautifulSoup(data, "xml")
    topics = bs_data.find_all('topic')

    topic_dicts = []
    for topic in topics:
        id = int(topic.find('num').text)
        query = topic.find('query').text
        description = topic.find('description').text
        type = topic.find('type').text
        topic_dict = {
            'id': id,
            'query': query,
            'description': description,
            'type': type}
        topic_dicts.append(topic_dict)
    return topic_dicts


def read_qrels(test=False):
    if test:
        location = data_dir + 'queries/2020_test_qrels.list.txt'
    else:
        location = data_dir + 'queries/2020_train_qrels.list.txt'
    with open(location, 'r') as f:
        data = f.readlines()
    data = [line.strip().split() for line in data]
    qrels_dicts = []
    for line in data:
        relevance = int(line[3])
        episode = line[2]
        episode = episode.split('_')
        time = float(episode[1])
        episode = episode[0].split(':')[2]
        qrels_dict = {
            'id': int(line[0]),
            'episode_filename_prefix': episode,
            'time': time,
            'relevance': relevance
        }
        qrels_dicts.append(qrels_dict)
    return qrels_dicts


def make_segments(transcript):
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


def read_segments():
    location = data_dir + "jsonl/all_segments/all_segments.json"
    data = []
    with open(location) as f:
        for line in tqdm(f, desc="Reading jsonl", leave=False):
            data.append(json.loads(line))
    return data


if __name__ == '__main__':
    make_segment_indices()
