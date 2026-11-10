# AI-Powered Email Threat Detection, GeoLocation Platform

A mini project that classifies emails as phishing or safe with machine learning, explains the decision, and maps where the message came from using the IP addresses in its headers. The result is a single risk score with the reasons behind it, in a Streamlit app.

**Live demo:** < https://email-threat-detection-5uqudrxewzfcb399w9pwrp.streamlit.app/ >


---

## Features

- **Threat detection:** TF-IDF text model with Logistic Regression, trained on about 17,500 unique emails.
- **Link and header checks:** rule-based signals such as raw-IP links, URL shorteners, look-alike brand domains, Reply-To mismatch and failed SPF/DKIM/DMARC.
- **GeoLocation:** finds public IPs in the `Received` headers, looks them up in MaxMind GeoLite2 and plots the route on a map. Hops are numbered, oldest first, and route lines are optional.
- **Explainability:** exact linear SHAP shows which words pushed the email towards phishing.
- **Hybrid risk score (0-100)** with a plain-language verdict: LOW, MEDIUM, HIGH or INCONCLUSIVE (when there is no readable body).
- **Two ways to input an email:** upload a `.eml` file, or paste the email source or text.
- **Streamlit app** with a green theme and a downloadable JSON result.

## How it works

```
.eml file or pasted text
        |
   parse (headers, body, links)
        |
   +----+-----------------+------------------+
   |                      |                  |
 text model            link/header          GeoIP of Received
 (TF-IDF + LR)         rules                hops (GeoLite2)
   |                      |                  |
   +----------+-----------+------------------+
              |
   risk score = 100 * min(1, ml_prob + 0.20*header + 0.25*url + 0.10*geo)
              |
   verdict + reasons (SHAP words, rule evidence) + map
```

Verdict levels: **HIGH** at 70 or above, **MEDIUM** at 40 to 69, **LOW** below 40. The weights are a design choice and are not calibrated probabilities.

## Project structure

```
Email Threat Prediciton/
├── app.py                 # Streamlit interface
├── pipeline.py            # analyze_eml(): model, link/header/geo signals, SHAP reasons
├── geo_lookup.py          # builds geo.csv and the dataset/route maps
├── geo_cv.py              # cross-validation: text vs link/header/style/geo features
├── train_and_explain.py   # trains the final model, saves it, makes SHAP plots
├── external_test.py       # tests on an independent dataset
├── combined_test.py       # combined-dataset experiment
├── requirements.txt
├── .streamlit/config.toml # theme
├── model/                 # tfidf.pkl, model.pkl, background_mean.pkl
├── data/                  # datasets (not included)
├── geo/                   # GeoLite2-City.mmdb (not included)
├── samples/               # test .eml files
├── screenshot/
└── README.md
```

## Datasets

Datasets are not included in the repo. Download them yourself:

- **Kaggle "Phishing Email" dataset** (`Phishing_Email.csv`, columns `Email Text` and `Email Type`): 17,514 unique emails after removing duplicates. Used to train the text model.
- **SpamAssassin Public Corpus** (`easy_ham`, `hard_ham`, `spam_2`): full emails with headers. Used for the geo and header experiments.
- **Zenodo phishing validation set** (`Phishing_validation_emails.csv`): used as an independent test. It lists 2,000 rows but contains only **100 unique emails** (77 safe, 23 phishing), so it was de-duplicated before testing.

Place them under `data/`.

## Setup

1. Install Python 3.10 or newer (developed on 3.14).
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. For geolocation, create a free MaxMind account, download **GeoLite2-City** (`.mmdb`) and place it anywhere under `geo/`. The file is not included because the licence does not allow redistribution. Without it, the app still runs and shows no map.
5. Train the model once:

```powershell
python train_and_explain.py
```

6. Run the app from the project folder (so `.streamlit/config.toml` is found):

```powershell
streamlit run app.py
```

Open http://localhost:8501, then upload a `.eml` file or paste email text. Sample emails are in `samples/`.

## Results

**Text model, 5-fold cross-validation (internal):** F1 of about 0.98.

**Independent test (100 unique emails, 77 safe and 23 phishing):**

| Metric | Value |
|---|---|
| Accuracy | 0.74 |
| Phishing recall | 0.83 |
| Phishing precision | 0.46 |

Internal scores are much higher than external ones. That gap is the main finding: the model does well on data like its training data but weakly on different data, because it partly learns dataset-specific wording.

**Feature experiments:** adding link, header, style or geo features to the text did not raise F1 over text alone. Geo and header features are therefore used as supporting signals in the risk score, not as model inputs.

**Explainability:** SHAP plots (`shap_summary.png`, `shap_waterfall_*.png`) show that some top words are dataset artifacts rather than real phishing cues.

## Limitations

- **A location is a clue, not proof.** VPNs, proxies and compromised servers hide the real sender, and GeoIP accuracy varies (each result has an accuracy radius).
- **Old data, new database.** The email corpora are old while GeoLite2 reflects current IP allocations, so some locations may be wrong.
- **Dataset shift.** Performance drops on an independent dataset (see Results). The external test set is also small (100 unique emails), so those numbers are rough.
- **Spam is not the same as phishing.** The header and geo experiments use SpamAssassin spam as a stand-in for phishing.
- **Collection artifacts.** In the SpamAssassin data, spam emails almost always contain an IP address while legitimate ones often do not, which reflects how the data was collected and not real behaviour. Features based on this were removed.
- **Risk score weights** are heuristic and were not validated.
- **SPF, DKIM and DMARC** are read from the headers but not re-verified.
- **SHAP** explains what the model uses, not what is causally true.
- Many emails have no usable public IP, so only some can be mapped.

## Safety notes

- Links in emails are never opened or fetched. They are only analysed as text and shown defanged (`hxxp://evil[.]com`).
- Uploaded or pasted emails are not stored. Please do not submit sensitive or personal emails to a public demo.


## Tech stack

Python, pandas, scikit-learn, SHAP, geoip2, folium, streamlit-folium, rapidfuzz, Streamlit


## Acknowledgements

This product includes GeoLite data created by MaxMind, available from https://www.maxmind.com.

Kaggle phishing email dataset, SpamAssassin Public Corpus, Zenodo phishing validation dataset.
