import re
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

KAGGLE = "data/Phishing_Email.csv"
ZENODO = "data/Phishing_validation_emails.csv"

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t))
    return re.sub(r"\s+", " ", t.lower()).strip()

def load(path):
    d = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore").dropna()
    d["text"] = d["Email Text"].map(clean)
    d = d[d.text != ""].drop_duplicates(subset="text")
    d["label"] = d["Email Type"].str.contains("phish", case=False).astype(int)
    return d[["text", "label"]].reset_index(drop=True)

kag, zen = load(KAGGLE), load(ZENODO)
# remove any Zenodo email that also appears in the Kaggle data, so the test stays independent
n_before = len(zen)
zen = zen[~zen.text.isin(set(kag.text))].reset_index(drop=True)
print("Kaggle:", kag.shape, "| Zenodo:", n_before, "-> after removing", n_before - len(zen), "overlapping emails:", zen.shape)

def fit_eval(train, test, weights=None):
    tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    A = tf.fit_transform(train.text); B = tf.transform(test.text)
    m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(A, train.label, sample_weight=weights)
    return precision_recall_fscore_support(test.label, m.predict(B), average="binary", zero_division=0)[:3]

results = {"Kaggle only": [], "Kaggle + Zenodo (weight 1)": [], "Kaggle + Zenodo (weight 5)": []}
for tri, tei in StratifiedKFold(5, shuffle=True, random_state=42).split(zen, zen.label):
    ztr, zte = zen.iloc[tri], zen.iloc[tei]
    results["Kaggle only"].append(fit_eval(kag, zte))
    both = pd.concat([kag, ztr], ignore_index=True)
    results["Kaggle + Zenodo (weight 1)"].append(fit_eval(both, zte))
    w = np.r_[np.ones(len(kag)), np.full(len(ztr), 5.0)]
    results["Kaggle + Zenodo (weight 5)"].append(fit_eval(both, zte, w))

print("\nTested on held-out Zenodo folds (5-fold):")
for name, r in results.items():
    r = np.array(r)
    print(f"{name:30s} P={r[:,0].mean():.3f} R={r[:,1].mean():.3f} F1={r[:,2].mean():.3f} (+/-{r[:,2].std():.3f})")