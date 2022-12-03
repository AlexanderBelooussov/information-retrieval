import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from load_data import data_dir, read_queries, read_qrels, read_metadata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
from keybert import KeyBERT


def get_relevant_documents_query(input_query, k=100, use_keybert=True, use_description=True):
    if use_keybert:
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        query_str = input_query['query']
        if use_description:
            query_str += '. ' + input_query['description']
        keywords = kw_model.extract_keywords(query_str, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=5)

        boolean_query_builder = querybuilder.get_boolean_query_builder()
        should = querybuilder.JBooleanClauseOccur['should'].value
        for keyword in keywords:
            boost = querybuilder.get_boost_query(querybuilder.get_term_query(keyword[0]), keyword[1])
            boolean_query_builder.add(boost, should)
        query = boolean_query_builder.build()
    else:
        query = input_query['query']

    searcher = LuceneSearcher(data_dir + 'podcast_collection_jsonl')
    searcher.set_bm25(0.9, 0.4)
    # searcher.set_rm3(10, 10, 0.5)
    hits = searcher.search(query, k=k)
    docs = [int(hits[i].docid) for i in range(len(hits))]
    scores = [hits[i].score for i in range(len(hits))]
    return docs, scores


def get_relevant_documents(test=False, k=100, use_keybert=True, use_description=True):
    queries = read_queries(test)
    for query in queries:
        docs, scores = get_relevant_documents_query(query, k, use_keybert, use_description)
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
        qrels_docs = [qrel['episode_filename_prefix'] for qrel in query_qrels]
        qrels_docs = [metadata[metadata['episode_filename_prefix'] == doc]['id'].to_numpy()[0] for doc in qrels_docs]
        true_relevances = [qrel['relevance'] for qrel in query_qrels]
        true_relevance = []
        for doc in query['docs']:
            if doc in qrels_docs:
                # find index of doc in qrels_docs
                index = qrels_docs.index(doc)
                true_relevance.append(true_relevances[index])
            else:
                true_relevance.append(0)
        ndcg = ndcg_score([true_relevance], [relevance])
        ndcgs.append(ndcg)
    recalls = [np.mean(recalls[i]) for i in range(5)]
    print(f"Average recall: {recalls}")
    print(f"Average NDCG: {np.mean(ndcgs)}")
    return recalls, np.mean(ndcgs)


def find_best_k(use_keybert=True, use_description=True):
    ks = np.logspace(start=1, stop=5, num=15, base=10, dtype=int).tolist()
    # ks = [10, 25]
    print(ks)
    k_recalls = []
    k_ndcgs = []
    for k in ks:
        print(f'k={k}')
        recalls, ndcg = score_relevant_documents(get_relevant_documents(False, k, use_keybert, use_description))
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
    plt.title(title)
    plt.xscale('log')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    find_best_k(False, False)
    find_best_k(True, False)
    find_best_k(True, True)
