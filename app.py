import json, re
import folium
import streamlit as st
from streamlit_folium import st_folium
from pipeline import analyze_eml

st.set_page_config(page_title="Email Threat Detection & GeoLocation", layout="wide")

# ---------- styling (static CSS only; nothing from an email is ever placed in here) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, textarea, input { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(160deg, #eaf8f0 0%, #cfeedc 42%, #a9dcba 100%); }
header[data-testid="stHeader"] { background: transparent; }
.hero { background: linear-gradient(120deg, #14532d 0%, #1f7a4d 55%, #3bb273 100%); color: #fff;
        padding: 26px 32px; border-radius: 18px; box-shadow: 0 8px 24px rgba(20,83,45,.28); margin-bottom: 18px; }
.hero h1 { color: #fff; margin: 0 0 6px 0; font-size: 1.9rem; font-weight: 700; }
.hero p { margin: 0; opacity: .92; font-size: .95rem; }
.verdict { background: rgba(255,255,255,.9); border-radius: 16px; padding: 16px 22px; margin: 6px 0 14px 0;
           box-shadow: 0 4px 14px rgba(20,83,45,.15); }
.vtitle { font-weight: 700; letter-spacing: .08em; font-size: .85rem; }
.vscore { font-size: 2.6rem; font-weight: 700; color: #10372a; line-height: 1.1; }
.vscore span { font-size: 1.1rem; font-weight: 500; opacity: .6; }
.bar { background: #dfeee5; border-radius: 999px; height: 12px; margin-top: 8px; overflow: hidden; }
.fill { height: 12px; border-radius: 999px; }
div[data-testid="stMetric"] { background: rgba(255,255,255,.88); border-left: 6px solid #2d9d5b; border-radius: 12px;
                              padding: 10px 16px; box-shadow: 0 3px 10px rgba(20,83,45,.12); }
div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255,255,255,.72); border-radius: 16px;
                              border: 1px solid rgba(31,122,77,.25); }
button[role="tab"] { font-weight: 600; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1> AI-Powered Email Threat Detection &amp; GeoLocation</h1>
  <p>Upload an email or paste its text to get a risk estimate, the reasons behind it, and where it was sent from.
     Links are never opened and nothing is stored. Please do not submit sensitive or personal emails to a public demo.</p>
</div>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def to_raw(text):
    """Pasted text -> raw email bytes. Returns (bytes, has_headers)."""
    text = text.replace("\r\n", "\n").strip()
    first = text.splitlines()[0] if text else ""
    head = "\n".join(text.splitlines()[:40])
    looks_like_header = bool(re.match(r"^[A-Za-z][\w-]*:\s", first) or re.match(r"^From \S+ ", first))   # 2nd form: mbox "From user date" line
    has_headers = looks_like_header and bool(
        re.search(r"^(received|return-path|message-id|mime-version|from|subject|date|to):", head, re.I | re.M))
    if has_headers:
        return text.encode("utf-8", "replace"), True
    return ("Subject: (pasted text)\n\n" + text).encode("utf-8", "replace"), False

def unwrap(coords):
    """Keep consecutive points on the short way round the globe (avoids lines crossing the whole map)."""
    out = [list(coords[0])]
    for lat, lon in coords[1:]:
        while lon - out[-1][1] > 180: lon -= 360
        while lon - out[-1][1] < -180: lon += 360
        out.append([lat, lon])
    return out

def build_map(pts, show_lines):
    m = folium.Map(location=[20, 0], zoom_start=2)
    groups = {}
    for i, p in enumerate(pts, 1):
        groups.setdefault((p["lat"], p["lon"]), []).append((i, p))
    for (lat, lon), items in groups.items():
        label = ",".join(str(i) for i, _ in items)
        color = "#d62828" if any(i == 1 for i, _ in items) else "#1f7a4d"
        html = (f'<div style="background:{color};color:#fff;border:2px solid #fff;border-radius:50%;min-width:26px;'
                f'height:26px;line-height:22px;text-align:center;font-weight:700;font-size:12px;padding:0 4px;'
                f'box-shadow:0 1px 5px rgba(0,0,0,.45);">{label}</div>')
        tip = "<br>".join(f"hop {i}: {p['ip']} ({p['country']})" for i, p in items)
        folium.Marker([lat, lon], icon=folium.DivIcon(html=html, icon_size=(28, 28), icon_anchor=(14, 14)),
                      tooltip=tip).add_to(m)
    coords = [[p["lat"], p["lon"]] for p in pts]
    if show_lines and len(coords) > 1:
        folium.PolyLine(unwrap(coords), color="#1f7a4d", weight=2, opacity=0.55, dash_array="6 8").add_to(m)
    m.fit_bounds(coords, max_zoom=5)
    return m

# ---------- input: file or pasted text ----------
tab_file, tab_text = st.tabs([" Upload .eml file", " Paste email text"])
with tab_file:
    f = st.file_uploader("Upload an email (.eml)", type=["eml"])
with tab_text:
    pasted = st.text_area("Paste the email here", height=200,
                          placeholder="Paste the full email source (with headers) for the best analysis, or just the message text.")
    go = st.button("Analyse pasted text", type="primary")
    st.caption("Full source works best: it includes headers, so the route map and header checks can run. "
               "In Gmail: open the email, click the three dots, then 'Show original'.")

if f is not None:
    fid = getattr(f, "file_id", f"{f.name}-{f.size}")
    if st.session_state.get("file_id") != fid:
        st.session_state["file_id"] = fid
        st.session_state["result"] = analyze_eml(f.getvalue())
        st.session_state["headers_ok"] = True
if go:
    if pasted.strip():
        raw, ok = to_raw(pasted)
        st.session_state["result"] = analyze_eml(raw)
        st.session_state["headers_ok"] = ok
    else:
        st.warning("Please paste some email text first.")

r = st.session_state.get("result")
if r is None:
    st.info("Upload an .eml file or paste email text above to begin.")
    st.stop()
if not st.session_state.get("headers_ok", True):
    st.caption("Pasted text had no headers, so only the wording and links were analysed (no route or header checks).")

# ---------- verdict ----------
color = {"HIGH": "#c1121f", "MEDIUM": "#e09f12", "LOW": "#2d9d5b"}.get(r["verdict"], "#6c757d")
st.markdown(f"""
<div class="verdict" style="border-left:8px solid {color}">
  <div class="vtitle" style="color:{color}">RISK LEVEL: {r['verdict']}</div>
  <div class="vscore">{r['score']:.0f}<span> / 100</span></div>
  <div class="bar"><div class="fill" style="width:{min(max(r['score'], 0), 100):.0f}%;background:{color}"></div></div>
</div>
""", unsafe_allow_html=True)
st.text("Subject: " + r["subject"])          # st.text shows email-derived text as plain text only

c1, c2, c3, c4 = st.columns(4)
c1.metric("Text model", "n/a" if r["ml_probability"] is None else f"{r['ml_probability']:.0%}")
c2.metric("Header risk", f"{r['components']['header_risk']:.0%}")
c3.metric("Link risk", f"{r['components']['url_risk']:.0%}")
c4.metric("Route risk", f"{r['components']['geo_risk']:.0%}")

left, right = st.columns([1, 1])
with left:
    with st.container(border=True):
        st.subheader("Why this score")
        st.markdown("**Message wording** (what the text model reacted to)")
        if r["text_reasons"]:
            for line in r["text_reasons"]:
                st.text("• " + line)
        else:
            st.text("• No readable body text.")
        st.markdown("**Other signals** (links, headers, route)")
        if r["evidence"]:
            for line in r["evidence"]:
                st.text("• " + line)
        else:
            st.text("• No rule-based warning signs found.")
        with st.expander("Links found (defanged, never opened)"):
            if r["urls_defanged"]:
                for u in r["urls_defanged"]:
                    st.text(u)
            else:
                st.text("No links found.")

with right:
    with st.container(border=True):
        st.subheader("Where it came from")
        st.caption(r["geo_note"])
        pts = r["geo_points"]
        if pts:
            show_lines = st.checkbox("Show route lines", value=False)
            st_folium(build_map(pts, show_lines), height=400, use_container_width=True, returned_objects=[])
            st.caption("Numbers show the hops, oldest first (red = most likely origin). "
                       "Locations are approximate and can be hidden by VPNs, proxies or compromised servers.")
        else:
            st.info("No map available for this email.")

st.download_button("Download analysis (JSON)", json.dumps(r, indent=2, default=str), "analysis.json", "application/json")

with st.expander("How to read this result"):
    st.markdown(
        "- The score starts from the text model's probability and adds small amounts for suspicious links, headers and route.\n"
        "- It is a **risk estimate, not proof**. On a small independent test the model caught most phishing but also flagged some safe emails.\n"
        "- Location comes from IP addresses in the headers. It is approximate and can be misleading.\n"
        "- Levels: LOW below 40, MEDIUM 40-69, HIGH 70 and above."
    )