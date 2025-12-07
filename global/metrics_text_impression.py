# metrics_text_impression.py

import nltk
from rouge_score import rouge_scorer
import bert_score
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

# -----------------------------
# ROUGE
# -----------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_rouge(reference, hypothesis):
    reference = "" if reference is None else str(reference)
    hypothesis = "" if hypothesis is None else str(hypothesis)

    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

# -----------------------------
# BERTScore
# -----------------------------
def compute_bertscore(references, hypotheses):
    references = ["" if r is None else str(r) for r in references]
    hypotheses = ["" if h is None else str(h) for h in hypotheses]

    P, R, F1 = bert_score.score(
        hypotheses,
        references,
        lang="en",
        rescale_with_baseline=True
    )
    return F1.mean().item()

# -----------------------------
# TF-IDF Cosine Similarity
# -----------------------------
def compute_cosine_similarity(reference, hypothesis):
    reference = "" if reference is None else str(reference)
    hypothesis = "" if hypothesis is None else str(hypothesis)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([reference, hypothesis])
    sim_matrix = (tfidf * tfidf.T).toarray()
    return float(sim_matrix[0, 1])

# -----------------------------
# SBERT Semantic Similarity
# -----------------------------
# Model loads once
sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def compute_sbert_similarity(reference, hypothesis):
    reference = "" if reference is None else str(reference)
    hypothesis = "" if hypothesis is None else str(hypothesis)

    emb1 = sbert_model.encode(reference, convert_to_tensor=True)
    emb2 = sbert_model.encode(hypothesis, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb1, emb2).item()
    return float(similarity)
