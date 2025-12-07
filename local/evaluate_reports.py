# evaluate_reports.py

import pandas as pd
from metrics_text import compute_bleu, compute_rouge, compute_bertscore, compute_cosine_similarity
from clinical_labels import extract_labels, f1_score_label

def evaluate(csv_path):
    df = pd.read_csv(csv_path)

    all_bleu = []
    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    all_cosine = []
    clinical_f1 = []

    # Clean data: replace NaN with empty strings
    df["ground_truth"] = df["ground_truth"].fillna("").astype(str)
    df["generated"] = df["generated"].fillna("").astype(str)

    references = df["ground_truth"].tolist()
    generated = df["generated"].tolist()
    # Remove rows where both are empty
    df = df[(df["ground_truth"] != "") | (df["generated"] != "")]

    # ---------------- BERTScore (done once for whole dataset) ----------------
    print("Computing BERTScore (slow)…")
    bert_f1 = compute_bertscore(references, generated)

    # ---------------- Instance-wise metrics ----------------
    for ref, hyp in zip(references, generated):

        # BLEU
        bleu = compute_bleu(ref, hyp)
        all_bleu.append(bleu)

        # ROUGE
        rouge_scores = compute_rouge(ref, hyp)
        all_rouge1.append(rouge_scores["rouge1"])
        all_rouge2.append(rouge_scores["rouge2"])
        all_rougeL.append(rouge_scores["rougeL"])

        # Cosine similarity
        cos = compute_cosine_similarity(ref, hyp)
        all_cosine.append(cos)

        # Clinical F1
        gt_labels = extract_labels(ref)
        pred_labels = extract_labels(hyp)
        clinical_f1.append(f1_score_label(gt_labels, pred_labels))

    # ---------------- Summary ----------------
    results = {
        "BLEU": sum(all_bleu)/len(all_bleu),
        "ROUGE-1": sum(all_rouge1)/len(all_rouge1),
        "ROUGE-2": sum(all_rouge2)/len(all_rouge2),
        "ROUGE-L": sum(all_rougeL)/len(all_rougeL),
        "BERTScore-F1": bert_f1,
        "Cosine-Similarity": sum(all_cosine)/len(all_cosine),
        "Clinical-F1": sum(clinical_f1)/len(clinical_f1)
    }

    return results


if __name__ == "__main__":
    csv_path ="/content/generated_impressions_p17_cleaned.csv"
    out = evaluate(csv_path)
    print("\n====== FINAL EVALUATION ======")
    for k,v in out.items():
        print(f"{k}: {v:.4f}")
