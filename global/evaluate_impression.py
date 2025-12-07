# evaluate_impression.py

import pandas as pd
from metrics_text_impressions import (
    compute_rouge,
    compute_bertscore,
    compute_cosine_similarity,
    compute_sbert_similarity
)

def evaluate_impressions(csv_path):
    df = pd.read_csv(csv_path)

    df["ground_truth"] = df["ground_truth"].fillna("").astype(str).str.strip()
    df["generated"] = df["generated"].fillna("").astype(str).str.strip()

  # Remove empty rows for both columns
    df = df[(df["ground_truth"] != "") & (df["generated"] != "")]

    references = df["ground_truth"].tolist()
    generated  = df["generated"].tolist()

    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    all_tfidf_cosine = []
    all_sbert = []

    print("Computing BERTScore (this step is slow)…")
    bert_f1 = compute_bertscore(references, generated)

    for ref, hyp in zip(references, generated):

        # ROUGE scores
        rouge_scores = compute_rouge(ref, hyp)
        all_rouge1.append(rouge_scores["rouge1"])
        all_rouge2.append(rouge_scores["rouge2"])
        all_rougeL.append(rouge_scores["rougeL"])

        # TF-IDF cosine similarity
        tfidf_sim = compute_cosine_similarity(ref, hyp)
        all_tfidf_cosine.append(tfidf_sim)

        # SBERT semantic similarity
        sbert_sim = compute_sbert_similarity(ref, hyp)
        all_sbert.append(sbert_sim)

    results = {
        "ROUGE-1": sum(all_rouge1)/len(all_rouge1),
        "ROUGE-2": sum(all_rouge2)/len(all_rouge2),
        "ROUGE-L": sum(all_rougeL)/len(all_rougeL),
        "TFIDF-Cosine-Sim": sum(all_tfidf_cosine)/len(all_tfidf_cosine),
        "SBERT-Semantic-Sim": sum(all_sbert)/len(all_sbert),
        "BERTScore-F1": bert_f1,
    }

    return results

if __name__ == "__main__":
    csv_path = "/content/generated_impressions_p17_cleaned.csv"
    out = evaluate_impressions(csv_path)

    print("\n====== IMPRESSION EVALUATION ======")
    for k, v in out.items():
        print(f"{k}: {v:.4f}")
