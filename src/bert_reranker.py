import psutil
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from load_data import read_segments, read_queries

# nboost/pt-bert-large-msmarco ? => too big
tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
# tokenizer = AutoTokenizer.from_pretrained("nboost/pt-bert-large-msmarco")
model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")
# model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)


def rerank_segments(segments, query: str):
    """
    Rerank using BERT
    """

    # query = tokenizer.tokenize(query)
    # query = tokenizer.convert_tokens_to_ids(query)[:64]

    def normalize(l):
        m = min(l)
        s = max(l) - m
        norm = [(float(i) - m) / s for i in l]
        return norm

    scores = []
    # batch_size = 4
    batch_size = 1
    chunks = [segments[x:x + batch_size] for x in range(0, len(segments), batch_size)] if batch_size > 1 else segments
    for chunk in tqdm(chunks, desc="Reranking using BERT"):
        if batch_size > 1:
            queries = [query] * len(chunk)
            passages = [segment["contents"] for segment in chunk]
            inputs = tokenizer(queries, passages, return_tensors="pt", padding=True, truncation=True,
                               max_length=512, truncation_strategy="only_second")
        else:
            passage = chunk["contents"]
            inputs = tokenizer(query, passage, return_tensors="pt", truncation=True,
                               max_length=512, truncation_strategy="only_second")

        inputs = inputs.to(device)
        output = model(**inputs)
        output = list(output.logits[:, 1].tolist())
        scores.extend(output)
        if psutil.virtual_memory()[2] > 95: raise RuntimeError("No more please")
    return normalize(scores)


if __name__ == "__main__":
    segment = read_segments()[:999]
    query = read_queries()[0]["query"]
    rerank_segments(segment, query)
