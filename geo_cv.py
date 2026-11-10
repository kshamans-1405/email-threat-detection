import os, re, email
from email import policy
from urllib.parse import urlparse
import numpy as np, pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

DATA_DIR = "data/archive (6)"                                        # same value as in geo_lookup.py
GEO_CSV  = "geo.csv"                                     # created by geo_lookup.py
FOLDERS  = {"easy_ham": 0, "hard_ham": 0, "spam_2": 1}
URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.I)
SHORT = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"}
URG = re.compile(r"\b(urgent|verify|suspend|immediately|act now|limited time|free|winner|click here)\b", re.I)

def dom(a):
    m = re.search(r'@([\w.-]+)', str(a)); return m.group(1).lower() if m else ""
def host(u):
    try: return (urlparse(u).hostname or "").lower()
    except ValueError: return ""

def feats(msg, text):
    urls = URL_RE.findall(text); hosts = [host(u) for u in urls]
    frm, rep = msg.get("From", ""), msg.get("Reply-To", "")
    return {"n_urls": len(urls),
            "n_ip_urls": sum(bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", h)) for h in hosts),
            "n_http_only": sum(u.lower().startswith("http://") for u in urls),
            "n_shortened": sum(".".join(h.split(".")[-2:]) in SHORT for h in hosts),
            "max_url_len": max([len(u) for u in urls], default=0),
            "max_subdomains": max([max(len(h.split(".")) - 2, 0) for h in hosts], default=0),
            "has_at_in_url": int(any("@" in u for u in urls)),
            "has_reply_to": int(bool(rep)),
            "reply_mismatch": int(bool(rep) and dom(rep) != dom(frm)),
            "n_received": len(msg.get_all("Received", [])),
            "missing_to": int(not str(msg.get("To", "")).strip()),
            "exclaim": text.count("!"),
            "upper_ratio": sum(c.isupper() for c in text) / max(len(text), 1),
            "urgency_words": len(URG.findall(text)),
            "from_tld": dom(frm).split(".")[-1]}

rows = []
for name, label in FOLDERS.items():
    d = os.path.join(DATA_DIR, name, name)
    if not os.path.isdir(d):
        d = os.path.join(DATA_DIR, name)
    for fn in os.listdir(d):
        if fn == "cmds":
            continue
        msg = email.message_from_bytes(open(os.path.join(d, fn), "rb").read(), policy=policy.default)
        b = msg.get_body(preferencelist=("plain", "html"))
        try: text = b.get_content() if b else ""
        except Exception: text = ""
        r = {"file": fn, "source": name, "label": label,
             "text": re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text.lower())).strip()}
        r.update(feats(msg, text)); rows.append(r)
df = pd.DataFrame(rows)

# --- add the geo results ---
geo = pd.read_csv(GEO_CSV)[["file", "source", "n_ips", "n_countries", "iso", "radius"]]
df = df.merge(geo, on=["file", "source"], how="left")
df["resolved"] = df.iso.notna().astype(int)
df["radius"] = df.radius.fillna(0)
tld = df.from_tld.fillna("").str.upper().replace({"UK": "GB"})
df["cctld_mismatch"] = ((tld.str.len() == 2) & (df.resolved == 1) & (tld != df.iso.fillna(""))).astype(int)
df = df[df.text != ""].drop_duplicates(subset="text").reset_index(drop=True)
print(df.shape, df.label.value_counts().to_dict())
print("\nShare of emails with a resolved origin location, by class:")
print(df.groupby("label")["resolved"].mean().round(2).to_string())

URLC = ["n_urls","n_ip_urls","n_http_only","n_shortened","max_url_len","max_subdomains","has_at_in_url"]
HDRC = ["has_reply_to","reply_mismatch","n_received","missing_to"]
STYC = ["exclaim","upper_ratio","urgency_words"]
GEOC = ["n_countries","cctld_mismatch","radius"]   # no "has IP" columns: that gap is a collection artifact

def cv(name, cols, data):
    res = []
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=42).split(data, data.label):
        tr, te = data.iloc[tri], data.iloc[tei]
        tf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        A, B = tf.fit_transform(tr.text), tf.transform(te.text)
        if cols:
            sc = StandardScaler().fit(tr[cols])
            A = hstack([A, csr_matrix(sc.transform(tr[cols]))]).tocsr()
            B = hstack([B, csr_matrix(sc.transform(te[cols]))]).tocsr()
        m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(A, tr.label)
        res.append(precision_recall_fscore_support(te.label, m.predict(B), average="binary")[:3])
    r = np.array(res)
    print(f"{name:34s} P={r[:,0].mean():.3f} R={r[:,1].mean():.3f} F1={r[:,2].mean():.3f} (+/-{r[:,2].std():.3f})")

# Only emails whose origin IP was located, so both classes are compared on equal terms
sub = df[df.resolved == 1].reset_index(drop=True)
print("\nEvaluating on emails with a located origin:", sub.shape, sub.label.value_counts().to_dict())
cv("text only", [], sub)
cv("text + geo", GEOC, sub)
cv("text + URL + header + style", URLC + HDRC + STYC, sub)
cv("text + everything", URLC + HDRC + STYC + GEOC, sub)
print("\nMean geo features by class (located emails only):")
print(sub.groupby("label")[GEOC].mean().round(2).T.to_string())
df.drop(columns=["text"]).to_csv("features_all.csv", index=False)
print("\nSaved features_all.csv")