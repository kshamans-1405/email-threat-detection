"""pipeline.py - analyse one email and return a verdict, risk score and evidence.
Never opens or fetches any link from the email: URLs are only analysed as text."""
import re, glob, ipaddress, email
from email import policy
from urllib.parse import urlparse
import numpy as np, joblib
from rapidfuzz.distance import Levenshtein

MODEL_DIR = "model"
tfidf = joblib.load(f"{MODEL_DIR}/tfidf.pkl")
lr = joblib.load(f"{MODEL_DIR}/model.pkl")
bg_mean = joblib.load(f"{MODEL_DIR}/background_mean.pkl")
NAMES = np.array(tfidf.get_feature_names_out())

# GeoLite database is optional: without it the geo part is simply skipped
_geo_reader = None
try:
    import geoip2.database, geoip2.errors
    _hits = glob.glob("geo/**/GeoLite2-City.mmdb", recursive=True)
    if _hits:
        _geo_reader = geoip2.database.Reader(_hits[0])
except Exception:
    _geo_reader = None

URL_RE = re.compile(r'https?://[^\s<>"\')]+', re.I)
IP_RE = re.compile(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?")
SHORT = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly"}
BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "facebook", "netflix",
          "paytm", "flipkart", "hdfc", "icici", "sbi", "instagram", "linkedin"]

def clean(t):
    t = re.sub(r"<[^>]+>", " ", str(t))
    return re.sub(r"\s+", " ", t.lower()).strip()

def defang(u):
    return u.replace("http", "hxxp", 1).replace(".", "[.]")

def _dom(a):
    m = re.search(r"@([\w.-]+)", str(a)); return m.group(1).lower() if m else ""

def _host(u):
    try: return (urlparse(u).hostname or "").lower()
    except ValueError: return ""

def _lookup(ip):
    if _geo_reader is None: return None
    try:
        r = _geo_reader.city(ip)
    except Exception:
        return None
    if r.location.latitude is None: return None
    return {"ip": ip, "country": r.country.name, "iso": r.country.iso_code, "city": r.city.name,
            "lat": r.location.latitude, "lon": r.location.longitude}

def analyze_eml(raw_bytes):
    msg = email.message_from_bytes(raw_bytes, policy=policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    try: text = body.get_content() if body else ""
    except Exception: text = ""
    evidence = []                                   # plain-English findings

    # ---- 1. text model + explanation (exact linear SHAP: coef * (x - background mean)) ----
    ctext = clean(text)
    ml_used = len(ctext) >= 20                      # an empty body gives the text model nothing to judge
    if ml_used:
        x = tfidf.transform([ctext])
        ml_prob = float(lr.predict_proba(x)[0, 1])
        contrib = lr.coef_[0] * (np.asarray(x.todense()).ravel() - bg_mean)
        top = np.argsort(-np.abs(contrib))[:5]
        text_reasons = [f'"{NAMES[i]}" pushes towards {"PHISHING" if contrib[i] > 0 else "SAFE"} ({contrib[i]:+.2f})'
                        for i in top if contrib[i] != 0]
    else:
        ml_prob, text_reasons = 0.0, []
        evidence.append("No readable body text, so the text model was not used")

    # ---- 2. URL risk (0-1) ----
    urls = URL_RE.findall(text); hosts = [_host(u) for u in urls]
    url_risk = 0.0
    n_ip = sum(bool(re.fullmatch(r"\d+\.\d+\.\d+\.\d+", h)) for h in hosts)
    if n_ip: url_risk += 0.5; evidence.append(f"{n_ip} link(s) use a raw IP address instead of a domain name")
    if any(".".join(h.split(".")[-2:]) in SHORT for h in hosts):
        url_risk += 0.2; evidence.append("Contains a shortened link that hides the real destination")
    if any("@" in u for u in urls):
        url_risk += 0.15; evidence.append("A link contains '@', which can disguise the real address")
    if any(len(h.split(".")) - 2 >= 3 for h in hosts):
        url_risk += 0.15; evidence.append("A link has an unusually long chain of subdomains")
    for h in set(hosts):
        parts = h.split(".")
        if len(parts) < 2: continue
        label = parts[-2]
        for b in BRANDS:
            if label != b and ((len(b) >= 5 and Levenshtein.distance(label, b) == 1) or
                               re.fullmatch(rf"(.*-)?{b}(-.*)?", label) and label != b):
                url_risk += 0.4; evidence.append(f"Link domain '{defang(h)}' imitates the brand '{b}'"); break
    url_risk = min(url_risk, 1.0)

    # ---- 3. header risk (0-1) ----
    frm, rep = msg.get("From", ""), msg.get("Reply-To", "")
    hdr_risk = 0.0
    if rep and _dom(rep) != _dom(frm):
        hdr_risk += 0.4; evidence.append(f"Reply-To domain ({_dom(rep)}) differs from sender domain ({_dom(frm)})")
    if frm and not str(msg.get("To", "")).strip():          # only meaningful when there is a real sender
        hdr_risk += 0.1; evidence.append("No visible recipient (To) header")
    auth = (str(msg.get("Authentication-Results", "")) + " " + str(msg.get("Received-SPF", ""))).lower()
    for key, w, label in [("spf=fail", .3, "SPF"), ("spf=softfail", .2, "SPF"), ("dkim=fail", .2, "DKIM"), ("dmarc=fail", .3, "DMARC")]:
        if key in auth:
            hdr_risk += w; evidence.append(f"{label} authentication check failed")
    hdr_risk = min(hdr_risk, 1.0)

    # ---- 4. geolocation (0-1 risk) ----
    ips = []
    for h in msg.get_all("Received", []):
        for ip in IP_RE.findall(str(h)):
            try:
                if ipaddress.ip_address(ip).is_global: ips.append(ip)
            except ValueError: pass
    ips = list(dict.fromkeys(ips))                                         # drop repeated hops
    hops = [g for g in (_lookup(ip) for ip in reversed(ips)) if g]       # oldest hop first
    geo_risk, origin = 0.0, (hops[0] if hops else None)
    if _geo_reader is None:
        geo_note = "Geolocation unavailable (GeoLite database not found)"
    elif not ips:
        geo_note = "No public IP address found in the headers"
    elif not hops:
        geo_note = "IP addresses found but could not be located"
    else:
        geo_note = f"Likely origin infrastructure: {origin['city'] or 'unknown city'}, {origin['country']} (approximate)"
        n_countries = len({g["iso"] for g in hops})
        if n_countries >= 3:
            geo_risk += 0.3; evidence.append(f"Message route crosses {n_countries} countries")
        tld = _dom(frm).split(".")[-1].upper().replace("UK", "GB")
        if len(tld) == 2 and tld != origin["iso"]:
            geo_risk += 0.4; evidence.append(f"Weak signal: sender domain suggests {tld} but the message originated in {origin['iso']}")
    geo_risk = min(geo_risk, 1.0)

    # ---- 5. combine: the ML probability is the base, other signals add to it ----
    score = 100 * min(1.0, ml_prob + 0.20 * hdr_risk + 0.25 * url_risk + 0.10 * geo_risk)
    verdict = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    if not ml_used and verdict == "LOW":
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "score": round(score, 1), "ml_probability": round(ml_prob, 3) if ml_used else None,
            "components": {"header_risk": hdr_risk, "url_risk": url_risk, "geo_risk": geo_risk},
            "text_reasons": text_reasons, "evidence": evidence, "geo_note": geo_note,
            "geo_points": hops, "urls_defanged": [defang(u) for u in urls[:10]],
            "subject": str(msg.get("Subject", ""))}

if __name__ == "__main__":
    import sys, json
    print(json.dumps(analyze_eml(open(sys.argv[1], "rb").read()), indent=2, default=str))