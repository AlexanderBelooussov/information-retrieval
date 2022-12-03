from datetime import datetime

from document_retrieval import get_relevant_documents_query
from load_data import read_metadata, read_queries, read_qrels, read_podcasts, read_transcripts
from segment_retrieval import split_transcript, retrieve_segments
from tqdm import tqdm


def process_query(query, transcripts):
    relevant_docs, _ = get_relevant_documents_query(query, k=100, use_keybert=True, use_description=True)
    relevant_docs = [transcripts[doc] for doc in relevant_docs]
    split_transcripts = []
    for doc in relevant_docs:
        split_transcripts.extend(split_transcript(doc))
    retrieve_segments(query, split_transcripts)
    # TODO continue here


if __name__ == '__main__':
    now = datetime.now()
    transcripts = read_transcripts().to_dict("index")
    queries = read_queries()
    for query in queries:
        print(query)
        process_query(query, transcripts)
    print(f"Finished in {datetime.now() - now}")


