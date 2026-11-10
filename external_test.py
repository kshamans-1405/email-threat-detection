import re
import numpy as np, pandas as pd, joblib
from sklearn.metrics import classification_report, confusion_matrix

VAL_CSV = "data/Phishing_validation_emails.csv"     # the Zenodo file
tfidf = joblib.load("model/tfidf.pkl")
lr = joblib.load("model/model.pkl")

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t))
    return re.sub(r"\s+", " ", t.lower()).strip()

df = pd.read_csv(VAL_CSV)
print("Columns:", df.columns.tolist(), "| rows:", len(df))
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
# pick the text and label columns (edit these two lines if the names differ)
TEXT_COL  = next(c for c in df.columns if "text" in c.lower() or "body" in c.lower() or "email" in c.lower() and df[c].astype(str).str.len().mean() > 50)
LABEL_COL = next(c for c in df.columns if c != TEXT_COL)
print("Using text column:", TEXT_COL, "| label column:", LABEL_COL)
print(df[LABEL_COL].value_counts())

df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df = df.drop_duplicates(subset=[TEXT_COL])
y = df[LABEL_COL].astype(str).str.contains("phish", case=False).astype(int)
X = tfidf.transform(df[TEXT_COL].map(clean))
pred = lr.predict(X)
print("\n=== External validation (never used in training) ===")
print(classification_report(y, pred, target_names=["safe", "phishing"]))
print(confusion_matrix(y, pred))

# --- a few hand-written probes ---
probes = [
 "Dear customer, your bank account has been suspended. Click here immediately to verify your password: http://secure-login.example-bank.xyz/verify",
 "URGENT: You have won a $3000 gift card! Claim now before the offer expires: http://bit.ly/claim-now",
 "Hi team, attaching the notes from today's meeting. Let me know if I missed anything. Thanks, Priya",
 "Your Amazon order #4821 has shipped and will arrive on Thursday. Track your package in your account.",
]
print("\n=== Hand-written probes (phishing probability) ===")
for t, p in zip(probes, lr.predict_proba(tfidf.transform([clean(t) for t in probes]))[:, 1]):
    print(f"{p:.2f}  {t[:80]}")