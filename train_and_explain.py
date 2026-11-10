import os, re
import numpy as np, pandas as pd
import joblib, shap
import matplotlib
matplotlib.use("Agg")                      # save plots as files, no window needed
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

CSV = "data/Phishing_Email.csv"
OUT = "model"                              # your folder is called "model"
os.makedirs(OUT, exist_ok=True)

# ---- 1. load + clean (same as your baseline) ----
df = pd.read_csv(CSV).drop(columns=["Unnamed: 0"], errors="ignore").dropna()
def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t))
    return re.sub(r"\s+", " ", t.lower()).strip()
df["text"] = df["Email Text"].map(clean)
df = df[df.text != ""].drop_duplicates(subset="text")
df["label"] = df["Email Type"].str.contains("phish", case=False).astype(int)
print(df.shape, df.label.value_counts().to_dict())

# ---- 2. train the final text model ----
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
Xtr, Xte = tfidf.fit_transform(X_train), tfidf.transform(X_test)
lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y_train)
print(classification_report(y_test, lr.predict(Xte), target_names=["safe", "phishing"]))
joblib.dump(tfidf, f"{OUT}/tfidf.pkl"); joblib.dump(lr, f"{OUT}/model.pkl")
joblib.dump(np.asarray(Xtr.mean(axis=0)).ravel(), f"{OUT}/background_mean.pkl")

# ---- 3. SHAP ----
names = np.array(tfidf.get_feature_names_out())
explainer = shap.LinearExplainer(lr, Xtr)
sample = Xte[:300]
sv = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(sv, sample.toarray(), feature_names=names, max_display=20, show=False)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150); plt.close()

def reasons(text, top=5):
    """Plain-English reasons for one email, from its SHAP values."""
    x = tfidf.transform([clean(text)])
    s = explainer.shap_values(x)[0]
    prob = lr.predict_proba(x)[0, 1]
    order = np.argsort(-np.abs(s))[:top]
    out = []
    for i in order:
        if s[i] == 0: continue
        direction = "towards PHISHING" if s[i] > 0 else "towards SAFE"
        out.append(f'The word/phrase "{names[i]}" pushes the verdict {direction} (impact {s[i]:+.2f})')
    return prob, out

for lab, nm in [(1, "phishing"), (0, "safe")]:
    i = int(np.where(y_test.values == lab)[0][0])
    prob, rs = reasons(X_test.iloc[i])
    print(f"\nExample {nm} email -> phishing probability {prob:.2f}")
    for r in rs: print(" -", r)
    plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, explainer.shap_values(Xte[i])[0],
                                           feature_names=names, max_display=10, show=False)
    plt.tight_layout(); plt.savefig(f"shap_waterfall_{nm}.png", dpi=150); plt.close()
print("\nSaved: model/tfidf.pkl, model/model.pkl, shap_summary.png, shap_waterfall_phishing.png, shap_waterfall_safe.png")