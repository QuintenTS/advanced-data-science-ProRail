import joblib
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product
from difflib import get_close_matches
import datetime as dt
import openpyxl

# ---------------- CONFIG ----------------
ARTIFACT_PATH = "logit_calibrated.joblib"
GROUPS_XLSX   = "Grouped_locations.xlsx"
DEFAULT_GROUP = "Nederland"

st.set_page_config(page_title="ProRail Request Approval Predictor", page_icon="🚆", layout="centered")

# ---------------- THEME ----------------
st.markdown("""
<style>
body { background-color: #f7f7f7; font-family: "Inter", sans-serif; }
h1 { color: #E60000; font-weight: 700; }
.stButton>button {
    background-color: #E60000; color: white;
    border-radius: 6px; border: none;
    padding: 0.6em 1.2em; font-weight: 600;
}
.kpi-card {
    background-color: white; padding: 20px 25px; margin-bottom: 12px;
    border-radius: 10px; border-left: 6px solid #E60000;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.kpi-value { font-size: 2.4rem; font-weight: 700; color: #E60000; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL COEFFICIENTS ----------------
COEFS = {
    "Requested_weight": -0.064936,
    "Requested_speed": -0.036086,
    "Requested_length": -0.214400,
    "Stabling": -0.012399,
    "Tolerance": -0.129264,
    "Day_sin": -0.035388,
    "Day_cos": -0.005701,
    "Hour_sin": -0.007461,
    "Hour_cos": -0.040901,
    "Rotterdam - Rotterdam": 0.277915,
    "Nederland - Rotterdam": 0.042613,
    "Rotterdam - Nederland": 0.024650,
    "Nederland - België grens": 0.005760,
    "Nederland - Nederland": 0.000000,
    "Duitsland grens - Rotterdam": -0.001326,
    "België grens - Rotterdam": -0.002233,
    "België grens - Nederland": -0.006480,
    "Rotterdam - België grens": -0.015127,
    "Duitsland grens - België grens": -0.020880,
    "Nederland - Duitsland grens": -0.026128,
    "België grens - Duitsland grens": -0.034736,
    "Duitsland grens - Nederland": -0.038650,
    "Duitsland grens - Duitsland grens": -0.049321,
    "België grens - België grens": -0.057192,
    "Rotterdam - Duitsland grens": -0.060293,
}

# ---------------- LOAD MODEL ----------------
artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

# ---------------- LOAD LOCATION GROUPS ----------------
@st.cache_resource
def load_groups():
    wide = pd.read_excel(GROUPS_XLSX).applymap(lambda x: str(x).strip() if pd.notna(x) else x)
    long = wide.melt(value_name="Location", var_name="Group").dropna()
    long["Location_norm"] = long["Location"].str.lower().str.strip()
    loc2grp = dict(zip(long["Location_norm"], long["Group"].str.strip()))
    group_labels = [str(c).strip() for c in wide.columns]
    if DEFAULT_GROUP not in group_labels: group_labels.append(DEFAULT_GROUP)
    norm2original = {loc.lower(): loc for loc in long["Location"]}
    return loc2grp, group_labels, norm2original

loc2grp, group_labels, norm2original = load_groups()

# ---------------- HELPERS ----------------
def cyc(v,p): return np.sin(2*np.pi*v/p), np.cos(2*np.pi*v/p)

def resolve_loc(s):
    s = (s or "").strip().lower()
    if s in norm2original:
        return norm2original[s]
    hits = get_close_matches(s, norm2original.keys(), n=1, cutoff=0.6)
    return norm2original[hits[0]] if hits else None

def map_group(loc): return loc2grp.get(loc.lower(), DEFAULT_GROUP) if loc else DEFAULT_GROUP

def od_dummies(og,dg):
    pairs = [f"{o} - {d}" for o,d in product(group_labels, group_labels)]
    row = {p:0 for p in pairs}; key = f"{og} - {dg}"
    if key in row: row[key] = 1
    return pd.DataFrame([row])

def align(df):
    for c in feature_cols:
        if c not in df: df[c] = 0
    return df[feature_cols]

def predict(df): return float(model.predict_proba(df)[0,1])

def logodds_impacts(x): return {f: float(x[f])*COEFS.get(f,0) for f in feature_cols}

def relative_importance(imp):
    abs_vals = {k: abs(v) for k,v in imp.items()}
    tot = sum(abs_vals.values())
    return {k:(abs_vals[k]/tot*100) if tot>0 else 0 for k in abs_vals}

def pretty(f):
    return f.replace("_"," ").replace("Requested","Requested ").title()

# ---------------- UI ----------------
st.title("Train Path Request Approval Predictor")

with st.form("inputs"):
    origin = st.text_input("Origin station")
    dest   = st.text_input("Destination station")
    date   = st.date_input("Operation date", value=dt.date.today())
    time   = st.time_input("Operation time", value=dt.time(9,0))

    w   = st.number_input("Requested weight [ton]", min_value=0.0, step=0.1)
    s   = st.number_input("Requested speed [km/h]", min_value=0.0, step=0.1)
    l   = st.number_input("Requested length [m]", min_value=0.0, step=0.1)
    tol = st.number_input("Time tolerance [min]", min_value=0, step=1)
    stabling = st.checkbox("Stabling required", value=False)

    submitted = st.form_submit_button("Predict")

# ---------------- RUN PREDICTION ----------------
if submitted:
    o = resolve_loc(origin); d = resolve_loc(dest)
    og = map_group(o); dg = map_group(d)

    dow = date.weekday(); hour = time.hour
    d_s,d_c = cyc(dow,7); h_s,h_c = cyc(hour,24)

    Xnum = pd.DataFrame([{
        "Requested_weight": w,
        "Requested_speed": s,
        "Requested_length": l,
        "Stabling": 1.0 if stabling else 0.0,
        "Tolerance": float(tol),
        "Day_sin": d_s, "Day_cos": d_c,
        "Hour_sin": h_s, "Hour_cos": h_c
    }])

    X = align(pd.concat([Xnum, od_dummies(og,dg)], axis=1))

    proba = predict(X)

    st.markdown(f"""
    <div class="kpi-card">
        <div>Estimated Approval Probability</div>
        <div class="kpi-value">{proba:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    # baseline (model-neutral 0-vector)
    zero_df = pd.DataFrame([{c:0 for c in feature_cols}])
    p_base = predict(zero_df)
    delta_p = proba - p_base

    # compute weighted contributions
    impacts = logodds_impacts(X.iloc[0])
    rel = relative_importance(impacts)
    final_contrib = {f: delta_p*(rel[f]/100) for f in rel}

    # filter out <0.01% impact
    filtered = {f:v for f,v in final_contrib.items() if abs(v) >= 0.0001}
    top = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # table
    st.subheader("Feature influence on final approval probability")
    df = pd.DataFrame({
        "Feature": [pretty(f) for f,_ in top],
        "Impact on Probability": [f"{v*100:.2f}%" for _,v in top]
    })
    st.dataframe(df, hide_index=True)
    st.caption(f"Total change vs model baseline: {delta_p*100:.2f}%")

    # explanation
    st.subheader("Explanation")
    for feat,val in top:
        pct = val*100
        name = pretty(feat)

        if pct < -3: msg = f"• **{name}** strongly reduced approval (**{pct:.2f}%**)."
        elif pct < -1: msg = f"• **{name}** moderately reduced approval (**{pct:.2f}%**)."
        elif pct < -0.3: msg = f"• **{name}** had a small negative effect (**{pct:.2f}%**)."
        elif pct > 1: msg = f"• **{name}** increased approval (**{pct:.2f}%**)."
        elif pct > 0.3: msg = f"• **{name}** had a minor positive effect (**{pct:.2f}%**)."
        else:
            continue

        st.markdown(msg)


