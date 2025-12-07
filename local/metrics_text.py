# metrics_text.py

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import numpy as np

nltk.download('punkt')

# -----------------------------
# BLEU
# -----------------------------
def compute_bleu(reference, hypothesis):
    if not isinstance(reference, str):
        reference = "" if pd.isna(reference) else str(reference)
    if not isinstance(hypothesis, str):
        hypothesis = "" if pd.isna(hypothesis) else str(hypothesis)

    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    return sentence_bleu([ref_tokens], hyp_tokens)

# -----------------------------
# ROUGE
# -----------------------------
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_rouge(reference, hypothesis):
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
    # Convert to str to avoid float errors
    references = [str(r) for r in references]
    hypotheses = [str(h) for h in hypotheses]

    P, R, F1 = bert_score.score(
        hypotheses, 
        references, 
        lang="en",
        rescale_with_baseline=True
    )
    return F1.mean().item()


# -----------------------------
# Cosine similarity (optional simple metric)
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_cosine_similarity(reference, hypothesis):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([reference, hypothesis])
    sim_matrix = (tfidf * tfidf.T).toarray()
    return sim_matrix[0, 1]
