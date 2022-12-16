from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from keybert import KeyBERT
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from sklearn.metrics import PrecisionRecallDisplay, ndcg_score

from load_data import data_dir, read_queries, read_qrels, read_metadata
from query_expansion import expand_query

INDEX_TYPES = ["all_text", "text_ep", "text_show_ep", "all_segments", "segments_ep", "segments_show_ep"]


def build_query(keywords):
    """
    Build a query using a list of (keywords, weight) pairs
    """
    boolean_query_builder = querybuilder.get_boolean_query_builder()
    should = querybuilder.JBooleanClauseOccur['should'].value
    for keyword in keywords:
        if keyword[1] > 0:
            boost = querybuilder.get_boost_query(querybuilder.get_term_query(keyword[0]), keyword[1])
            boolean_query_builder.add(boost, should)
    query = boolean_query_builder.build()
    return query


def get_relevant_documents_query(input_query, k=100, use_keybert=True, use_description=True,
                                 index_type='text_show_ep', n_grams=1, top_n=25, query_expansion=True):
    """
    Retrieve documents using Pyserini
    """
    assert index_type in INDEX_TYPES, f"{index_type} is not a valid index"
    if query_expansion:
        top_n = min(top_n, 10)  # limit keywords if using query expansion
    if use_keybert:
        kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        query_str = input_query['query']
        if use_description:
            query_str += '. ' + input_query['description']
        keywords = kw_model.extract_keywords(query_str, keyphrase_ngram_range=(1, n_grams), stop_words='english',
                                             top_n=top_n)
        if query_expansion:
            new_keywords = {}
            for keyword in keywords:
                new_keywords.update(expand_query(keyword[0], keyword[1], n_grams))
            keywords = [(k, v) for k, v in new_keywords.items()]
            keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
            keywords = keywords[:1024]
        query = build_query(keywords)
    else:
        # if query_expansion:
        #     new_keywords = {}
        #     new_keywords.update(expand_query(input_query['query'], 1, n_grams))
        #     keywords = [(k, v) for k, v in new_keywords.items()]
        #     keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        #     keywords = keywords[:1024]
        #     query = build_query(keywords)
        # else:
        query = input_query['query']
        if use_description:
            query += ". " + input_query['description']

    index_dir = data_dir
    if index_type in ["all_text", "all_segments"]:
        index_dir += index_type + "_index"
    else:
        index_dir += index_type + "_title_desc_index"
    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(0.9, 0.4)
    if not use_keybert and query_expansion:
        searcher.set_rm3(10, 10, 0.5)
    hits = searcher.search(query, k=k)
    docs = [int(hits[i].docid) for i in range(len(hits))]
    scores = [hits[i].score for i in range(len(hits))]
    return docs, scores


def get_relevant_documents(test=False, k=100, use_keybert=True, use_description=True, index_type='text_show_ep',
                           n_grams=1, query_expansion=True):
    """
    Get results for all queries using Pyserini
    """
    queries = read_queries(test)
    for query in queries:
        docs, scores = get_relevant_documents_query(query, k, use_keybert, use_description, index_type, n_grams,
                                                    query_expansion=query_expansion)
        query['docs'] = docs
        query['scores'] = scores
    # print(queries)
    return queries


def score_relevant_documents(query_results):
    """
    Score results from Pyserini
    """
    qrles = read_qrels(False)
    metadata = read_metadata()
    recalls = [[] for i in range(5)]
    fprs = [[] for i in range(5)]
    precisions = [[] for i in range(5)]
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
                false_positives = len(set(query['docs']).difference(set(qrels_docs)))
                true_positives = len(intersection)
                false_negatives = len(set(qrels_docs).difference(set(query['docs'])))
                true_negatives = len(metadata) - false_negatives - false_positives - true_positives
                fpr = false_positives / (false_positives + true_negatives)
                precision = true_positives / (
                            true_positives + false_positives) if false_positives + true_positives > 0 else 0
                recalls[threshold].append(recall)
                fprs[threshold].append(fpr)
                precisions[threshold].append(precision)
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
    fprs = [np.mean(fprs[i]) for i in range(5)]
    precisions = [np.mean(precisions[i]) for i in range(5)]
    print(f"Average recall: {recalls}")
    print(f"Average FPR: {fprs}")
    print(f"Average Precisions: {precisions}")
    print(f"Average NDCG: {np.mean(ndcgs)}")
    return recalls, precisions, np.mean(ndcgs), fprs


def precision_relevant_documents(query_results, recalls):
    precisions = [[] for _ in range(5)]
    metadata = read_metadata()
    qrles = read_qrels(False)

    # repeat for different recalls:
    for recall in recalls:
        current_precisions = [[] for _ in range(5)]
        for query in query_results:
            id = query['id']

            for threshold in range(5):
                query_qrels = [qrel for qrel in qrles if qrel['id'] == id and qrel['relevance'] >= threshold]
                qrels_docs = [qrel['episode_filename_prefix'] for qrel in query_qrels]
                qrels_docs = [metadata[metadata['episode_filename_prefix'] == doc]['id'].to_numpy()[0] for doc in
                              qrels_docs]
                if len(qrels_docs):
                    cutoff = int(recall * len(query["docs"]))
                    docs = query["docs"][:cutoff + 1]

                    intersection = set(docs).intersection(set(qrels_docs))
                    true_positives = len(intersection)
                    false_positives = cutoff + 1 - true_positives
                    precision = true_positives / (true_positives + false_positives)
                    current_precisions[threshold].append(precision)
        for i in range(5):
            precisions[i].append(np.mean(current_precisions[i]))

    return precisions


def precision_recall_plot():
    """
    From sklearn example: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-precision-recall-curve-for-each-class-and-iso-f1-curves
    """
    recall = np.linspace(0, 1, num=1001)
    queries = get_relevant_documents(k=1000)
    precision = precision_relevant_documents(queries, recall)

    # setup plot details
    colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

    _, ax = plt.subplots(figsize=(7, 8))

    # Show F1 score lines
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    for i, color in zip(range(5), colors):
        display = PrecisionRecallDisplay(
            recall=recall,
            precision=precision[i]
        )
        display.plot(ax=ax, name=f"Precision-recall for class {i}", color=color)

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title("Extension of Precision-Recall curve to multi-class")

    plt.show()


def find_best_k(use_keybert=True, use_description=True, index_type='text', n_grams=1, query_expansion=False):
    """
    Plot recall graph over different K's
    """
    ks = np.logspace(start=1, stop=5, num=15, base=10, dtype=int).tolist()

    # ks = [1000, 2000, 3000, 4000, 5000]
    print(ks)
    k_recalls = []
    k_precisions = []
    k_fprs = []
    k_ndcgs = []
    for k in ks:
        print(f'k={k}')
        recalls, precisions, ndcg, fprs = score_relevant_documents(
            get_relevant_documents(False, k, use_keybert, use_description, index_type, n_grams, query_expansion))
        k_recalls.append(recalls)
        k_precisions.append(precisions)
        k_fprs.append(fprs)
        k_ndcgs.append(ndcg)

    fig, ax2 = plt.subplots()
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    threshold_recalls = np.array(k_recalls).T
    threshold_precisions = np.array(k_fprs).T
    for i in range(5):
        # ax1.plot(threshold_precisions[i], threshold_recalls[i], label=f'threshold {i}')
        ax2.plot(ks, threshold_recalls[i], label=f'threshold {i}')
    ax2.plot(ks, k_ndcgs, label='ndcg')
    # ax1.plot([0, 1], [0, 1], linestyle='dotted')

    # title = f'FPR, Recall and nDCG vs k'
    title = f'Recall and nDCG vs k'
    if use_keybert:
        title += ' with KeyBERT'
        if use_description:
            title += ' using description'
        title += f'\nn_grams={n_grams}'
    title += f' index_type={index_type}'
    if query_expansion: title += f' with query expansion'
    fig.suptitle(title)
    # ax1.set_xlabel('Recall')
    # ax1.set_ylabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_xlabel('k')
    ax2.set_xscale('log')
    # ax1.legend()
    ax2.legend()
    fig.show()


if __name__ == '__main__':
    # precision_recall_plot()
    # find_best_k(False, False, 'all_text')
    # find_best_k(False, False, 'text_show_ep')
    # find_best_k(False, True, 'text_show_ep')
    # find_best_k(True, True, 'text_show_ep')
    # find_best_k(True, True, 'text_show_ep', n_grams=3)
    find_best_k(True, True, 'text_show_ep', n_grams=3, query_expansion=True)
