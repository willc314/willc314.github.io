"""
Audits a sentiment classifier for dialect disparity.

Data: Groenwold et al. (2020), 2,019 intent-equivalent sentence pairs. The
African American English side is real tweets; the Standard American English
side was written by human annotators.

What is borrowed and what is mine:
  - The classifier is an off-the-shelf library model. I did not train it.
    That is deliberate. It is what a working developer would actually use.
  - The audit below is mine. Pairing, per-group reporting, the fixed
    metrics, and the significance test.
The model was always available. The audit was not. That gap is the point.

Run:  python dialect_analysis.py EMNLP-AAVE-files
Make sure to have python downloaded and have scipy and transformers installed via pip
"""

import statistics
import sys

from scipy import stats
from transformers import pipeline

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def load_pairs(folder):
    """Read the two parallel files and pair them line by line."""
    def lines(name):
        with open(f"{folder}/{name}", encoding="utf-8") as f:
            return [ln for ln in f.read().split("\n") if ln.strip()]

    aave, sae = lines("aave_samples.txt"), lines("sae_samples.txt")

    # If the counts differ the files are not aligned, and pairing them would
    # produce a confident, entirely fake result. Stop instead.
    if len(aave) != len(sae):
        sys.exit(f"Not aligned: {len(aave)} vs {len(sae)} lines.")

    return list(zip(aave, sae))


def score(texts, clf):
    """Return (predicted label, probability of positive) for each text."""
    out = []
    for i in range(0, len(texts), 32):
        for scores in clf(texts[i:i + 32]):
            probs = {d["label"].upper(): d["score"] for d in scores}
            out.append((max(probs, key=probs.get), probs.get("POSITIVE", 0.0)))
    return out


def main(folder):
    pairs = load_pairs(folder)
    clf = pipeline("sentiment-analysis", model=MODEL,
                   truncation=True, max_length=256, top_k=None)

    aave = score([a for a, _ in pairs], clf)
    sae = score([s for _, s in pairs], clf)

    # These metrics were chosen before the first run. Whatever comes out gets
    # reported, including no difference at all, which is a real result.
    deltas = [a_p - s_p for (_, a_p), (_, s_p) in zip(aave, sae)]
    aave_neg = sum(1 for lab, _ in aave if "NEG" in lab) / len(pairs)
    sae_neg = sum(1 for lab, _ in sae if "NEG" in lab) / len(pairs)
    disagree = sum(1 for (a_l, _), (s_l, _) in zip(aave, sae) if a_l != s_l)

    print(f"Pairs                     {len(pairs)}")
    print(f"Negative, AAE side        {aave_neg:.1%}")
    print(f"Negative, SAE side        {sae_neg:.1%}")
    print(f"Gap                       {aave_neg - sae_neg:+.1%}")
    print(f"Mean change in positivity {statistics.mean(deltas):+.4f}")
    print(f"Labels disagree           {disagree / len(pairs):.1%}")
    print(f"Wilcoxon p                {stats.wilcoxon(deltas).pvalue:.2e}")

    print("\nThe tweets date from around 2013. The SAE side was written by annotators, not the original posters.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "EMNLP-AAVE-files")
