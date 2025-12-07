# clinical_labels.py

import re

CHEST_FINDINGS = {
    "pneumothorax": ["pneumothorax"],
    "effusion": ["effusion", "pleural effusion"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac silhouette"],
    "consolidation": ["consolidation"],
    "pneumonia": ["pneumonia", "infectious opacity"],
    "edema": ["edema", "interstitial edema", "pulmonary edema"],
    "fracture": ["fracture"],
    "atelectasis": ["atelectasis"],
    "opacity": ["opacity"],
    "normal": ["no acute disease", "no significant findings", "normal study"],
}

def extract_labels(text):
    text = text.lower()
    labels = {key: 0 for key in CHEST_FINDINGS}

    for label, keywords in CHEST_FINDINGS.items():
        for k in keywords:
            if re.search(r"\b" + re.escape(k) + r"\b", text):
                labels[label] = 1

    return labels


def f1_score_label(gt_labels, pred_labels):
    tp = sum(gt_labels[k] == 1 and pred_labels[k] == 1 for k in gt_labels)
    fp = sum(gt_labels[k] == 0 and pred_labels[k] == 1 for k in gt_labels)
    fn = sum(gt_labels[k] == 1 and pred_labels[k] == 0 for k in gt_labels)

    if tp == 0:
        return 0
    return tp / (tp + 0.5*(fp + fn))
