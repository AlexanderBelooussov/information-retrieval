from keybert import KeyBERT
from sklearn.metrics import ndcg_score
from load_data import *
import numpy as np


transctipts = read_transcripts()
metadata = read_metadata()
metadata = metadata.to_numpy()
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
query = "trump call ukrainian president"
query_desc = "The White House released a rough transcript of President Donald Trump’s phone call with the Ukrainian President in November 2019.  What were people saying about the conversation in the call?  News items about the call are relevant."
query_full = query + " " + query_desc
query_keywords = kw_model.extract_keywords(query_full, keyphrase_ngram_range=(1, 1), stop_words='english', top_n=10, seed_keywords=query.split(" "))
query_keywords_dict = {}
for kw in query_keywords:
    query_keywords_dict[kw[0]] = kw[1]
print(query_keywords)

# true_relevance = np.asarray([[0, 0, 1, 0, 1]])
# scores = np.asarray([[.1, .2, .3, 4, 70]])
# score = ndcg_score(true_relevance, scores, k=3)
# print(score)

for i, transctipt in enumerate(tqdm(transctipts.to_numpy())):
    doc = transctipt[1]

    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words="english", top_n=50, seed_keywords=query.split(" "))
    score = sum([query_keywords_dict[kw[0]] * kw[1] for kw in keywords if kw[0] in query_keywords_dict.keys()])
    print(score)
    if score > 0.8:
        print(score)
        print(metadata[i])
        print(keywords)
        break

