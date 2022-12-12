import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from load_data import data_dir, read_queries, read_qrels, read_metadata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from keybert import KeyBERT


INDEX_TYPES = ['text', 'episode_info', 'show_episode_info']

def get_relevant_documents_query(input_query, k=100, use_keybert=True, use_description=True, index_type='text', n_grams=1):
    if use_keybert:
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        query_str = input_query['query']
        if use_description:
            query_str += '. ' + input_query['description']
        keywords = kw_model.extract_keywords(query_str, keyphrase_ngram_range=(1, n_grams), stop_words='english', top_n=5)

        boolean_query_builder = querybuilder.get_boolean_query_builder()
        should = querybuilder.JBooleanClauseOccur['should'].value
        for keyword in keywords:
            boost = querybuilder.get_boost_query(querybuilder.get_term_query(keyword[0]), keyword[1])
            boolean_query_builder.add(boost, should)
        query = boolean_query_builder.build()
    else:
        query = input_query['query']

    index_dir = data_dir
    if index_type == 'text':
        index_dir += 'all_text_index'
    elif index_type == 'episode_info':
        index_dir += 'text_ep_desc_title_index'
    elif index_type == 'show_episode_info':
        index_dir += 'text_show_ep_title_desc_index'
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(0.9, 0.4)
    # searcher.set_rm3(10, 10, 0.5)
    hits = searcher.search(query, k=k)
    docs = [int(hits[i].docid) for i in range(len(hits))]
    scores = [hits[i].score for i in range(len(hits))]
    return docs, scores


def get_relevant_documents(test=False, k=100, use_keybert=True, use_description=True, index_type='text', n_grams=1):
    queries = read_queries(test)
    for query in queries:
        docs, scores = get_relevant_documents_query(query, k, use_keybert, use_description, index_type, n_grams)
        query['docs'] = docs
        query['scores'] = scores
    # print(queries)
    return queries


def score_relevant_documents(query_results):
    qrles = read_qrels(False)
    metadata = read_metadata()
    recalls = [[] for i in range(5)]
    ndcgs = []
    for query in query_results:
        id = query['id']

        for threshold in range(5):
            query_qrels = [qrel for qrel in qrles if qrel['id'] == id and qrel['relevance'] >= threshold]
            qrels_docs = [qrel['episode_filename_prefix'] for qrel in query_qrels]
            qrels_docs = [metadata[metadata['episode_filename_prefix'] == doc]['id'].to_numpy()[0] for doc in
                          qrels_docs]
            if len(qrels_docs):
                intersection = set(query['docs']).intersection(set(qrels_docs))
                recall = len(intersection) / len(set(qrels_docs))
                recalls[threshold].append(recall)
                # print(f'Query {id}, threshold {threshold}: {len(intersection)}/{len(qrels_docs)}, {recall:.2f}')
        # ndcg
        relevance = query['scores']
        query_qrels = [qrel for qrel in qrles if qrel['id'] == id]
        # delete duplicates, keep the highest relevance
        query_qrels = sorted(query_qrels, key=lambda x: x['relevance'], reverse=True)
        filtered_qrels = []
        for qrel in query_qrels:
            if qrel['episode_filename_prefix'] not in [q['episode_filename_prefix'] for q in filtered_qrels]:
                filtered_qrels.append(qrel)
        query_qrels = filtered_qrels
        qrels_docs = [qrel['episode_filename_prefix'] for qrel in query_qrels]
        qrels_docs = [metadata[metadata['episode_filename_prefix'] == doc]['id'].to_numpy()[0] for doc in qrels_docs]
        found_docs = query['docs']
        true_relevance = []
        for doc in found_docs:
            if doc in qrels_docs:
                true_relevance.append(query_qrels[qrels_docs.index(doc)]['relevance'])
            else:
                true_relevance.append(0)

        ndcg = ndcg_score([true_relevance], [relevance])
        ndcgs.append(ndcg)
    recalls = [np.mean(recalls[i]) for i in range(5)]
    print(f"Average recall: {recalls}")
    print(f"Average NDCG: {np.mean(ndcgs)}")
    return recalls, np.mean(ndcgs)


def find_best_k(use_keybert=True, use_description=True, index_type='text', n_grams=1):
    ks = np.logspace(start=1, stop=5, num=15, base=10, dtype=int).tolist()
    # ks = [10, 25]
    print(ks)
    k_recalls = []
    k_ndcgs = []
    for k in ks:
        print(f'k={k}')
        recalls, ndcg = score_relevant_documents(get_relevant_documents(False, k, use_keybert, use_description, index_type, n_grams))
        k_recalls.append(recalls)
        k_ndcgs.append(ndcg)

    threshold_recalls = np.array(k_recalls).T
    for i in range(5):
        plt.plot(ks, threshold_recalls[i], label=f'{i}')
    plt.plot(ks, k_ndcgs, label='ndcg')

    title = f'Recall and nDCG vs k'
    if use_keybert:
        title += ' with KeyBERT'
        if use_description:
            title += ' using description'
        title += f' n_grams={n_grams}'
    title += f'\nindex_type={index_type}'
    plt.title(title)
    plt.xscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # find_best_k(False, False, 'episode_info', 1)
    # find_best_k(True, False, 'episode_info', 1)
    # find_best_k(True, True, 'episode_info', 1)
    # find_best_k(False, False, 'show_episode_info', 1)
    find_best_k(True, False, 'show_episode_info', 2)
    find_best_k(True, True, 'show_episode_info', 2)
