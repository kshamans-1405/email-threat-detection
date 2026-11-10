import os, re, ipaddress, email
from email import policy
import pandas as pd
import geoip2.database, geoip2.errors
import folium
from folium.plugins import MarkerCluster

DATA_DIR = "data/archive (6)"                                        # folder holding easy_ham, hard_ham, spam_2
DB_PATH  = "geo/GeoLite2-City_20261103/GeoLite2-City.mmdb"
FOLDERS  = {"easy_ham": 0, "hard_ham": 0, "spam_2": 1}   # 0 = legit, 1 = spam
IP_RE = re.compile(r"\[?(\d{1,3}(?:\.\d{1,3}){3})\]?")

def extract_ips(msg):
    ips = []
    for h in msg.get_all("Received", []):
        for ip in IP_RE.findall(str(h)):
            try:
                if ipaddress.ip_address(ip).is_global:
                    ips.append(ip)
            except ValueError:
                pass
    return ips            # newest hop first, so the sender side is the LAST one

reader = geoip2.database.Reader(DB_PATH)
def lookup(ip):
    try:
        r = reader.city(ip)
    except (geoip2.errors.AddressNotFoundError, ValueError):
        return None
    if r.location.latitude is None:
        return None
    return {"country": r.country.name, "iso": r.country.iso_code, "city": r.city.name,
            "lat": r.location.latitude, "lon": r.location.longitude,
            "radius": r.location.accuracy_radius}

rows = []
for name, label in FOLDERS.items():
    d = os.path.join(DATA_DIR, name, name)
    if not os.path.isdir(d):
        d = os.path.join(DATA_DIR, name)
    for fn in os.listdir(d):
        if fn == "cmds":
            continue
        msg = email.message_from_bytes(open(os.path.join(d, fn), "rb").read(), policy=policy.default)
        ips = extract_ips(msg)
        geos = [lookup(ip) for ip in ips]
        row = {"file": fn, "source": name, "label": label, "n_ips": len(ips),
               "n_countries": len({g["iso"] for g in geos if g}),
               "all_ips": ";".join(ips), "origin_ip": None, "country": None, "iso": None,
               "city": None, "lat": None, "lon": None, "radius": None}
        if ips and geos[-1]:
            row.update({"origin_ip": ips[-1], **geos[-1]})
        rows.append(row)

geo = pd.DataFrame(rows)
geo.to_csv("geo.csv", index=False)

# --- how much worked? (report these numbers) ---
geo["has_ip"] = geo.n_ips > 0
geo["resolved"] = geo.country.notna()
print("Emails:", len(geo))
print(geo.groupby("label")[["has_ip", "resolved"]].agg(["sum", "mean"]).round(2))
for lab, nm in [(0, "legit"), (1, "spam")]:
    print(f"\nTop origin countries ({nm}):")
    print(geo[(geo.label == lab) & geo.resolved].country.value_counts().head(8).to_string())

# --- map 1: all origins, spam red / legit blue ---
m = folium.Map(location=[20, 0], zoom_start=2)
cluster = MarkerCluster().add_to(m)
for _, r in geo[geo.resolved].sample(frac=1, random_state=1).head(1500).iterrows():
    folium.CircleMarker([r.lat, r.lon], radius=4, color="red" if r.label else "blue",
                        fill=True, popup=f"{r.origin_ip} | {r.city}, {r.country} | {'spam' if r.label else 'legit'}"
                        ).add_to(cluster)
m.save("dataset_map.html")

# --- map 2: route of one spam email through its hops (oldest hop first) ---
cand = geo[(geo.label == 1) & (geo.n_countries >= 2)]
if len(cand):
    pick = cand.iloc[0]
    d = os.path.join(DATA_DIR, pick.source, pick.source)
    if not os.path.isdir(d):
        d = os.path.join(DATA_DIR, pick.source)
    msg = email.message_from_bytes(open(os.path.join(d, pick.file), "rb").read(), policy=policy.default)
    pts = [(ip, lookup(ip)) for ip in reversed(extract_ips(msg))]
    pts = [(ip, g) for ip, g in pts if g]
    m2 = folium.Map(location=[20, 0], zoom_start=2)
    for i, (ip, g) in enumerate(pts, 1):
        folium.Marker([g["lat"], g["lon"]], popup=f"hop {i}: {ip} ({g['country']})").add_to(m2)
    folium.PolyLine([[g["lat"], g["lon"]] for _, g in pts], color="red").add_to(m2)
    m2.save("route_map.html")
    print("\nRoute map saved for file", pick.file)
print("\nSaved geo.csv, dataset_map.html, route_map.html")