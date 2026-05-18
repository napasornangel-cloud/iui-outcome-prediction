import io
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# =============================
# Paths
# =============================
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

PW_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "final_model" / "XGBoost_Baseline_calibration_base_model.joblib"
PW_CALIBRATOR_PATH = PROJECT_ROOT / "models" / "saved_models" / "final_model" / "isotonic_calibrator_final_xgb.joblib"
PW_SHAP_IMG        = PROJECT_ROOT / "reports" / "figures" / "shap_final_xgb" / "SHAP_Beeswarm_Final_XGBoost_Baseline.png"

FV_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "first_visit_model" / "xgb_first_visit_base_model.joblib"
FV_CALIBRATOR_PATH = PROJECT_ROOT / "models" / "saved_models" / "first_visit_model" / "isotonic_calibrator_first_visit.joblib"
FV_SHAP_IMG        = PROJECT_ROOT / "reports" / "figures" / "first_visit_model" / "SHAP_Beeswarm_FirstVisit.png"

# =============================
# Feature lists
# =============================
PW_FEATURES = [
    "Uterine_Factors", "Total_Female_Pathology", "Ovulatory_Factors",
    "Cycle_Day", "First_Count", "Pre_Count", "Post_TPMSC",
    "Gynecological_Surgical_History", "Delta_Motile", "Age_Female",
    "First_Volume", "Post_Count", "Menstrual_Interval_Days",
    "First_Progressive_Motile", "First_TPMSC", "BMI_InfertilityType_Interaction",
]

FV_FEATURES = [
    "Age_Female", "Body_Mass_Index", "Infertility_Type",
    "Total_infertile_duration", "Menstrual_Interval_Days",
    "Uterine_Factors", "Ovulatory_Factors", "Tubal_Factors",
    "Endometriosis_Factors", "Gynecological_Surgical_History",
    "Total_Female_Pathology", "First_Volume", "First_Count",
    "First_Progressive_Motile", "First_TPMSC", "BMI_InfertilityType_Interaction",
]

PW_LOW_CUTOFF  = 0.045802
PW_HIGH_CUTOFF = 0.100962
FV_LOW_CUTOFF  = 0.036885
FV_HIGH_CUTOFF = 0.098507

PW_DISPLAY_MAP = {
    "Uterine_Factors":                 "Uterine factor",
    "Total_Female_Pathology":          "Number of female pathology factors",
    "Ovulatory_Factors":               "Ovulatory factor",
    "Cycle_Day":                       "IUI cycle day",
    "Post_TPMSC":                      "Postwash TPMSC",
    "First_Count":                     "Initial sperm concentration",
    "Pre_Count":                       "Prewash sperm concentration",
    "Gynecological_Surgical_History":  "Prior gynecologic surgery",
    "Post_Count":                      "Postwash sperm count",
    "Delta_Motile":                    "Change in sperm motility after wash",
    "Age_Female":                      "Female age",
    "First_Progressive_Motile":        "Initial progressive motility",
    "First_Volume":                    "Initial semen volume",
    "Menstrual_Interval_Days":         "Menstrual cycle length",
    "First_Motile":                    "Initial total motility",
    "BMI_InfertilityType_Interaction": "BMI and infertility type combined",
    "First_TPMSC":                     "Initial TPMSC",
}

FV_DISPLAY_MAP = {
    "Age_Female":                    "Female age",
    "Body_Mass_Index":               "BMI",
    "Infertility_Type":              "Infertility type",
    "Total_infertile_duration":      "Duration of infertility",
    "Menstrual_Interval_Days":       "Menstrual cycle length",
    "Uterine_Factors":               "Uterine factor",
    "Ovulatory_Factors":             "Ovulatory factor",
    "Tubal_Factors":                 "Tubal factor",
    "Endometriosis_Factors":         "Endometriosis",
    "Gynecological_Surgical_History":"Prior gynecologic surgery",
    "Total_Female_Pathology":        "Number of female pathology factors",
    "First_Volume":                  "Initial semen volume",
    "First_Count":                   "Initial sperm concentration",
    "First_Progressive_Motile":      "Initial progressive motility",
    "First_TPMSC":                   "Initial TPMSC",
    "BMI_InfertilityType_Interaction":"BMI and infertility type combined",
}

PW_REQUIRED_RAW = [
    "Uterine_Factors", "Tubal_Factors", "Ovarian_Factors",
    "Ovulatory_Factors", "Cervical_Factors", "Endometriosis_Factors",
    "Multisystem_Factors", "Cycle_Day", "Post_TPMSC", "First_Count",
    "Pre_Count", "Gynecological_Surgical_History", "Post_Count",
    "Post_Motile", "Pre_Motile", "Age_Female", "First_Progressive_Motile",
    "First_Volume", "Menstrual_Interval_Days", "First_Motile",
    "Body_Mass_Index", "Infertility_Type",
]

FV_REQUIRED_RAW = [
    "Age_Female", "Body_Mass_Index", "Infertility_Type",
    "Total_infertile_duration", "Menstrual_Interval_Days",
    "Uterine_Factors", "Tubal_Factors", "Ovarian_Factors",
    "Ovulatory_Factors", "Cervical_Factors", "Endometriosis_Factors",
    "Multisystem_Factors", "Gynecological_Surgical_History",
    "First_Volume", "First_Count", "First_Progressive_Motile",
]

PW_MEDIANS = {
    "Uterine_Factors": 0.0, "Tubal_Factors": 0.0, "Ovarian_Factors": 0.0,
    "Ovulatory_Factors": 0.0, "Cervical_Factors": 0.0,
    "Endometriosis_Factors": 0.0, "Multisystem_Factors": 0.0,
    "Cycle_Day": 14.0, "Post_TPMSC": 10.641124, "First_Count": 41.31,
    "Pre_Count": 42.6, "Gynecological_Surgical_History": 0.0,
    "Post_Count": 22.2, "Post_Motile": 96.93, "Pre_Motile": 57.6,
    "Age_Female": 35.0, "First_Progressive_Motile": 52.56,
    "First_Volume": 3.0, "Menstrual_Interval_Days": 29.0,
    "First_Motile": 54.7, "Body_Mass_Index": 21.718066,
    "Infertility_Type": 0.0,
}

FV_MEDIANS = {
    "Age_Female": 35.0, "Body_Mass_Index": 21.718066,
    "Infertility_Type": 0.0, "Total_infertile_duration": 36.0,
    "Menstrual_Interval_Days": 29.0,
    "Uterine_Factors": 0.0, "Tubal_Factors": 0.0, "Ovarian_Factors": 0.0,
    "Ovulatory_Factors": 0.0, "Cervical_Factors": 0.0,
    "Endometriosis_Factors": 0.0, "Multisystem_Factors": 0.0,
    "Gynecological_Surgical_History": 0.0,
    "First_Volume": 3.0, "First_Count": 41.31,
    "First_Progressive_Motile": 52.56,
}

VALIDATION_RULES = {
    "Age_Female":               (18, 55),
    "Body_Mass_Index":          (10, 60),
    "Menstrual_Interval_Days":  (15, 180),
    "Cycle_Day":                (1, 40),
    "First_Volume":             (0, 20),
    "First_Count":              (0, 500),
    "First_Motile":             (0, 100),
    "First_Progressive_Motile": (0, 100),
    "Pre_Count":                (0, 500),
    "Pre_Motile":               (0, 100),
    "Post_Count":               (0, 500),
    "Post_TPMSC":               (0, 500),
    "Post_Motile":              (0, 100),
    "Total_infertile_duration": (0, 360),
}

# =============================
# Page config + CSS
# =============================
st.set_page_config(
    page_title="IUI Pregnancy Probability Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0f2b4a; padding-top: 2rem; }
section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
.main { background: #f5f7fa; }

.risk-low  { background:#fff5f5; border:2px solid #fc8181; border-radius:20px; padding:2rem 2.5rem; text-align:center; margin-bottom:1rem; }
.risk-mid  { background:#fffbeb; border:2px solid #f6ad55; border-radius:20px; padding:2rem 2.5rem; text-align:center; margin-bottom:1rem; }
.risk-high { background:#f0fff4; border:2px solid #68d391; border-radius:20px; padding:2rem 2.5rem; text-align:center; margin-bottom:1rem; }
.risk-group-label { font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.4rem; }
.risk-group-label-low  { color:#c53030; }
.risk-group-label-mid  { color:#c05621; }
.risk-group-label-high { color:#276749; }
.risk-prob { font-family:'DM Serif Display',serif; font-size:4rem; line-height:1; margin:0.4rem 0; }
.risk-prob-low  { color:#c53030; }
.risk-prob-mid  { color:#c05621; }
.risk-prob-high { color:#276749; }
.risk-sub { font-size:0.85rem; color:#718096; }

.prob-bar-wrap { margin:1rem 0 1.8rem; position:relative; height:12px; background:#e2e8f0; border-radius:8px; }
.prob-bar-fill { height:12px; border-radius:8px; }
.prob-bar-fill-low  { background:linear-gradient(90deg,#fed7d7,#fc8181); }
.prob-bar-fill-mid  { background:linear-gradient(90deg,#feebc8,#f6ad55); }
.prob-bar-fill-high { background:linear-gradient(90deg,#c6f6d5,#68d391); }
.tier-marker { position:absolute; top:-6px; width:2px; height:24px; background:#94a3b8; }
.tier-marker-label { position:absolute; top:22px; font-size:0.62rem; color:#94a3b8; transform:translateX(-50%); white-space:nowrap; }

.dot-grid { display:flex; flex-wrap:wrap; gap:3px; margin:0.5rem 0; max-width:340px; }
.dot { width:10px; height:10px; border-radius:50%; }
.dot-active-low  { background:#fc8181; }
.dot-active-mid  { background:#f6ad55; }
.dot-active-high { background:#68d391; }
.dot-inactive { background:#e2e8f0; }

.badge-pw { display:inline-block; background:#e3f2fd; color:#0d47a1; border-radius:8px; padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:600; }
.badge-fv { display:inline-block; background:#e8f5e9; color:#1b5e20; border-radius:8px; padding:0.2rem 0.8rem; font-size:0.8rem; font-weight:600; }
.info-card { background:white; border-radius:16px; padding:1.5rem 2rem; box-shadow:0 2px 12px rgba(15,43,74,0.07); margin-bottom:1rem; }
.info-card h3 { font-family:'DM Serif Display',serif; color:#0f2b4a; font-size:1.05rem; margin-bottom:0.5rem; }
.section-header { font-family:'DM Serif Display',serif; color:#0f2b4a; font-size:1.3rem; border-bottom:2px solid #e3eafc; padding-bottom:0.4rem; margin:1.5rem 0 1rem; }
.form-group-label { font-size:0.8rem; font-weight:600; color:#1565c0; text-transform:uppercase; letter-spacing:0.08em; margin:1.2rem 0 0.4rem; }

.factors-card { background:white; border-radius:16px; padding:1.5rem 2rem; box-shadow:0 2px 12px rgba(15,43,74,0.07); margin-bottom:1rem; }
.factor-group-title { font-size:0.82rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:1rem; padding-bottom:0.4rem; border-bottom:1px solid #f1f5f9; }
.factor-group-against { color:#c53030; }
.factor-group-favor   { color:#276749; }
.factor-row { display:flex; align-items:center; margin-bottom:0.9rem; gap:1rem; }
.factor-name { flex:0 0 220px; font-size:0.88rem; color:#2d3748; }
.factor-bar-bg { flex:1; background:#f1f5f9; border-radius:6px; height:8px; overflow:hidden; }
.factor-bar-against { height:8px; border-radius:6px; background:#fc8181; }
.factor-bar-favor   { height:8px; border-radius:6px; background:#68d391; }
.factor-label { flex:0 0 65px; font-size:0.78rem; color:#94a3b8; text-align:right; }

.cycle-card { background:white; border-radius:12px; padding:1rem 1.5rem; box-shadow:0 2px 8px rgba(15,43,74,0.07); margin-bottom:0.5rem; border-left:4px solid #e2e8f0; }
.cycle-card-low  { border-left-color:#fc8181; }
.cycle-card-mid  { border-left-color:#f6ad55; }
.cycle-card-high { border-left-color:#68d391; }

.val-warn { background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:0.6rem 1rem; font-size:0.84rem; color:#856404; margin-bottom:0.5rem; }
.disclaimer { background:#fff8e1; border-radius:12px; padding:0.8rem 1.2rem; color:#744210; font-size:0.82rem; line-height:1.6; margin-top:1rem; }
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# =============================
# Session state
# =============================
if "cycle_history" not in st.session_state:
    st.session_state.cycle_history = []

# =============================
# Core helpers
# =============================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def assign_tier(p_cal, model_type="postwash"):
    if model_type == "postwash":
        lo, hi = PW_LOW_CUTOFF, PW_HIGH_CUTOFF
        obs    = {"low": "about 4 in 100", "mid": "about 10 in 100", "high": "about 18 in 100"}
        obs_n  = {"low": 4, "mid": 10, "high": 18}
    else:
        lo, hi = FV_LOW_CUTOFF, FV_HIGH_CUTOFF
        obs    = {"low": "about 4 in 100", "mid": "about 9 in 100", "high": "about 13 in 100"}
        obs_n  = {"low": 4, "mid": 9, "high": 13}
    if p_cal < lo:
        return "Low", "low", obs["low"], obs_n["low"]
    if p_cal < hi:
        return "Intermediate", "mid", obs["mid"], obs_n["mid"]
    return "High", "high", obs["high"], obs_n["high"]

def get_display_name(raw_name, model_type="postwash"):
    dm = PW_DISPLAY_MAP if model_type == "postwash" else FV_DISPLAY_MAP
    return dm.get(str(raw_name), str(raw_name).replace("_", " "))

@st.cache_resource
def load_model(model_type="postwash"):
    if model_type == "postwash":
        return joblib.load(PW_BASE_MODEL_PATH), joblib.load(PW_CALIBRATOR_PATH)
    return joblib.load(FV_BASE_MODEL_PATH), joblib.load(FV_CALIBRATOR_PATH)

def validate_inputs(raw_inputs):
    warnings = []
    for field, (lo, hi) in VALIDATION_RULES.items():
        val = raw_inputs.get(field)
        if val is not None and val != 0.0 and not (lo <= val <= hi):
            warnings.append(f"{field.replace('_', ' ')}: {val} (expected {lo}–{hi})")
    return warnings

def compute_pw_features(df_raw):
    df = df_raw.copy()
    missing = [c for c in PW_REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n- " + "\n- ".join(missing))
    for c in PW_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in PW_REQUIRED_RAW:
        if df[c].isna().any():
            df[c] = df[c].fillna(PW_MEDIANS.get(c, 0.0))
            imputed.append(c)
    if imputed:
        st.info(f"ℹ️ Missing values imputed with training median for: {', '.join(imputed)}")
    df["Total_Female_Pathology"]          = (df["Uterine_Factors"] + df["Tubal_Factors"] + df["Ovarian_Factors"] +
                                              df["Ovulatory_Factors"] + df["Cervical_Factors"] +
                                              df["Endometriosis_Factors"] + df["Multisystem_Factors"])
    df["Delta_Motile"]                    = df["Post_Motile"] - df["Pre_Motile"]
    df["BMI_InfertilityType_Interaction"] = df["Body_Mass_Index"] * df["Infertility_Type"]
    df["First_TPMSC"]                     = (df["First_Volume"] * df["First_Count"] * df["First_Progressive_Motile"] / 100).clip(upper=200)
    return df[PW_FEATURES].copy()

def compute_fv_features(df_raw):
    df = df_raw.copy()
    missing = [c for c in FV_REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns:\n- " + "\n- ".join(missing))
    for c in FV_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in FV_REQUIRED_RAW:
        if df[c].isna().any():
            df[c] = df[c].fillna(FV_MEDIANS.get(c, 0.0))
            imputed.append(c)
    if imputed:
        st.info(f"ℹ️ Missing values imputed with training median for: {', '.join(imputed)}")
    df["Total_Female_Pathology"]          = (df["Uterine_Factors"] + df["Tubal_Factors"] + df["Ovarian_Factors"] +
                                              df["Ovulatory_Factors"] + df["Cervical_Factors"] +
                                              df["Endometriosis_Factors"] + df["Multisystem_Factors"])
    df["BMI_InfertilityType_Interaction"] = df["Body_Mass_Index"] * df["Infertility_Type"]
    df["First_TPMSC"]                     = (df["First_Volume"] * df["First_Count"] * df["First_Progressive_Motile"] / 100).clip(upper=200)
    return df[FV_FEATURES].copy()

def predict(X, model_type="postwash"):
    model, calibrator = load_model(model_type)
    return np.clip(calibrator.predict(model.predict_proba(X)[:, 1]), 0, 1)

def get_factors(X_row, model_type="postwash", top_k=5):
    model, _ = load_model(model_type)
    xgb = model.named_steps["model"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:         shap_vals = shap_vals[:, :, 1]
    sv   = shap_vals.reshape(-1)
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(np.ravel(base)) >= 2 else float(np.ravel(base)[0])
    z, items = float(base), []
    for j in np.argsort(np.abs(sv))[::-1]:
        dz       = float(sv[j])
        p_before = sigmoid(z); z += dz; p_after = sigmoid(z)
        items.append({"name": get_display_name(X_row.columns[j], model_type), "delta": (p_after - p_before) * 100.0})
    against = [x for x in items if x["delta"] < 0][:top_k]
    favor   = [x for x in items if x["delta"] > 0][:top_k]
    def normalize(lst):
        if not lst: return lst
        mx = max(abs(x["delta"]) for x in lst)
        for x in lst:
            x["strength"] = abs(x["delta"]) / mx if mx > 0 else 0
            x["label"] = "Strong" if x["strength"] > 0.66 else "Moderate" if x["strength"] > 0.33 else "Mild"
        return lst
    return normalize(against), normalize(favor)

# =============================
# Render helpers
# =============================
def render_prob_bar(p_cal, tier_key, model_type):
    lo = PW_LOW_CUTOFF if model_type == "postwash" else FV_LOW_CUTOFF
    hi = PW_HIGH_CUTOFF if model_type == "postwash" else FV_HIGH_CUTOFF
    cap = 0.30
    fill_pct = min(p_cal / cap, 1.0) * 100
    lo_pct   = min(lo / cap, 1.0) * 100
    hi_pct   = min(hi / cap, 1.0) * 100
    st.markdown(f"""
    <div style="margin:1rem 0 1.8rem;">
        <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-bottom:4px;">
            <span>0%</span><span style="color:#94a3b8;">30%+</span>
        </div>
        <div class="prob-bar-wrap">
            <div class="prob-bar-fill prob-bar-fill-{tier_key}" style="width:{fill_pct:.1f}%"></div>
            <div class="tier-marker" style="left:{lo_pct:.1f}%"><div class="tier-marker-label">Low / Mid</div></div>
            <div class="tier-marker" style="left:{hi_pct:.1f}%"><div class="tier-marker-label">Mid / High</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

def render_dot_grid(obs_n, tier_key):
    color_hex = {"low": "#fc8181", "mid": "#f6ad55", "high": "#68d391"}[tier_key]
    dots = "".join(
        f'<div class="dot dot-active-{tier_key}"></div>' if i < obs_n else '<div class="dot dot-inactive"></div>'
        for i in range(100)
    )
    st.markdown(f"""
    <div style="margin:0.5rem 0 1rem;">
        <div style="font-size:0.78rem; color:#718096; margin-bottom:0.5rem;">Out of 100 similar patients in our cohort:</div>
        <div class="dot-grid">{dots}</div>
        <div style="font-size:0.78rem; color:#4a5568; margin-top:0.5rem;">
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{color_hex};margin-right:4px;vertical-align:middle;"></span>
            {obs_n} became pregnant per cycle
        </div>
    </div>""", unsafe_allow_html=True)

def render_result(p_cal, model_type="postwash"):
    tier_label, tier_key, obs, obs_n = assign_tier(p_cal, model_type)
    badge_class = "badge-pw" if model_type == "postwash" else "badge-fv"
    badge_text  = "🔬 Procedure-Day Model" if model_type == "postwash" else "🏥 Pre-treatment Model"
    st.markdown(f"""
    <div class="risk-{tier_key}">
        <div style="margin-bottom:0.8rem;"><span class="{badge_class}">{badge_text}</span></div>
        <div class="risk-group-label risk-group-label-{tier_key}">{tier_label} Probability</div>
        <div class="risk-prob risk-prob-{tier_key}">{p_cal:.1%}</div>
        <div class="risk-sub">estimated pregnancy probability per cycle</div>
    </div>
    <div class="disclaimer">
        This estimate is intended to support clinical counseling — not to replace it.
        Always consider the full clinical picture when discussing outcomes with your patient.
    </div>""", unsafe_allow_html=True)
    render_prob_bar(p_cal, tier_key, model_type)
    render_dot_grid(obs_n, tier_key)

def render_factors(against, favor):
    col1, col2 = st.columns(2)
    with col1:
        if against:
            html = '<div class="factors-card"><div class="factor-group-title factor-group-against">↓ Working against pregnancy</div>'
            for f in against:
                html += f'<div class="factor-row"><div class="factor-name">{f["name"]}</div><div class="factor-bar-bg"><div class="factor-bar-against" style="width:{f["strength"]*100:.0f}%"></div></div><div class="factor-label">{f["label"]}</div></div>'
            st.markdown(html + "</div>", unsafe_allow_html=True)
    with col2:
        if favor:
            html = '<div class="factors-card"><div class="factor-group-title factor-group-favor">↑ Working in favor of pregnancy</div>'
            for f in favor:
                html += f'<div class="factor-row"><div class="factor-name">{f["name"]}</div><div class="factor-bar-bg"><div class="factor-bar-favor" style="width:{f["strength"]*100:.0f}%"></div></div><div class="factor-label">{f["label"]}</div></div>'
            st.markdown(html + "</div>", unsafe_allow_html=True)
    st.caption("Factors are specific to this patient and ranked by how strongly they influence the result.")

def plot_shap_waterfall(X_row, model_type="postwash"):
    model, _ = load_model(model_type)
    xgb = model.named_steps["model"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb)
    exp = explainer(X_row)
    dm  = PW_DISPLAY_MAP if model_type == "postwash" else FV_DISPLAY_MAP
    exp.feature_names = [dm.get(c, c) for c in X_row.columns]
    fig, _ = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(exp[0], max_display=10, show=False)
    plt.tight_layout()
    return fig

# =============================
# PDF generation
# =============================
def generate_pdf_report(p_cal, model_type, against, favor, patient_id=""):
    tier_label, tier_key, obs, obs_n = assign_tier(p_cal, model_type)
    model_name = "Procedure-Day Model" if model_type == "postwash" else "Pre-treatment Model"
    tier_color  = {"low": colors.HexColor("#c53030"), "mid": colors.HexColor("#c05621"), "high": colors.HexColor("#276749")}[tier_key]
    tier_bg     = {"low": colors.HexColor("#fff5f5"), "mid": colors.HexColor("#fffbeb"), "high": colors.HexColor("#f0fff4")}[tier_key]
    tier_border = {"low": colors.HexColor("#fc8181"), "mid": colors.HexColor("#f6ad55"), "high": colors.HexColor("#68d391")}[tier_key]

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

    title_s   = ParagraphStyle("t",  fontName="Helvetica-Bold",    fontSize=16, textColor=colors.HexColor("#0f2b4a"), spaceAfter=4)
    sub_s     = ParagraphStyle("s",  fontName="Helvetica",         fontSize=9,  textColor=colors.HexColor("#718096"), spaceAfter=12)
    heading_s = ParagraphStyle("h",  fontName="Helvetica-Bold",    fontSize=10, textColor=colors.HexColor("#0f2b4a"), spaceBefore=12, spaceAfter=4)
    body_s    = ParagraphStyle("b",  fontName="Helvetica",         fontSize=9,  textColor=colors.HexColor("#4a5568"), spaceAfter=3)
    disc_s    = ParagraphStyle("d",  fontName="Helvetica-Oblique", fontSize=7.5, textColor=colors.HexColor("#92400e"))
    tier_s    = ParagraphStyle("tl", fontName="Helvetica-Bold",    fontSize=9,  textColor=tier_color)
    prob_s    = ParagraphStyle("pp", fontName="Helvetica-Bold",    fontSize=28, textColor=tier_color, alignment=TA_RIGHT)
    co_s      = ParagraphStyle("co", fontName="Helvetica",         fontSize=8,  textColor=colors.HexColor("#4a5568"), alignment=TA_RIGHT)
    ps_s      = ParagraphStyle("ps", fontName="Helvetica",         fontSize=8,  textColor=colors.HexColor("#718096"))

    story = [
        Paragraph("IUI Pregnancy Probability Report", title_s),
        Paragraph(f"Generated {datetime.now().strftime('%d %B %Y, %H:%M')}  |  {model_name}  |  Research prototype", sub_s),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e3eafc"), spaceAfter=12),
    ]
    if patient_id:
        story += [Paragraph(f"Patient / Cycle reference: {patient_id}", body_s), Spacer(1, 6)]

    result_table = Table([
        [Paragraph(f"<b>{tier_label} Probability</b>", tier_s), Paragraph(f"<b>{p_cal:.1%}</b>", prob_s)],
        [Paragraph("estimated pregnancy probability per cycle", ps_s),
         Paragraph(f"Among similar patients: <b>{obs} became pregnant</b>", co_s)],
    ], colWidths=[9*cm, 8*cm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), tier_bg),
        ("BOX",           (0,0),(-1,-1), 1.5, tier_border),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("RIGHTPADDING",  (0,0),(-1,-1), 14),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [result_table, Spacer(1, 14)]

    if against or favor:
        story += [Paragraph("Influencing Factors", heading_s),
                  HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8)]

    def factor_table(items, label, color):
        rows = [[Paragraph(f"<b>{label}</b>", ParagraphStyle("fl", fontName="Helvetica-Bold", fontSize=8, textColor=color)), "", ""]]
        for f in items:
            rows.append([Paragraph(f["name"], body_s), Paragraph(f["label"], ParagraphStyle("ll", fontName="Helvetica", fontSize=8, textColor=colors.HexColor("#94a3b8"))), ""])
        t = Table(rows, colWidths=[10*cm, 2.5*cm, 4.5*cm])
        t.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LEFTPADDING",(0,0),(-1,-1),0)]))
        return t

    if against:
        story += [factor_table(against, "Working against pregnancy", colors.HexColor("#c53030")), Spacer(1, 6)]
    if favor:
        story += [factor_table(favor, "Working in favor of pregnancy", colors.HexColor("#276749"))]

    story += [
        Spacer(1, 16),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0"), spaceAfter=8),
        Paragraph("This report is generated by a research prototype for academic purposes only. "
                  "It is intended to support clinical counseling and should not be used as the sole basis for clinical decision-making. "
                  "Outputs are statistical estimates derived from a single-center retrospective cohort at a Thai fertility center. "
                  "External validation has not yet been performed.", disc_s),
    ]
    doc.build(story)
    buf.seek(0)
    return buf

# =============================
# Cycle history
# =============================
def add_to_history(p_cal, model_type, against, favor, label=""):
    tier_label, tier_key, obs, obs_n = assign_tier(p_cal, model_type)
    entry = {
        "label":      label or f"Cycle {len(st.session_state.cycle_history) + 1}",
        "p_cal":      p_cal, "tier_label": tier_label, "tier_key": tier_key,
        "model_type": model_type, "against": against, "favor": favor,
        "timestamp":  datetime.now().strftime("%H:%M"),
    }
    if len(st.session_state.cycle_history) >= 3:
        st.session_state.cycle_history.pop(0)
    st.session_state.cycle_history.append(entry)

def render_cycle_history():
    if not st.session_state.cycle_history:
        return
    st.markdown('<div class="section-header">📈 Cycle History (this session)</div>', unsafe_allow_html=True)
    cols = st.columns(len(st.session_state.cycle_history))
    for col, entry in zip(cols, st.session_state.cycle_history):
        with col:
            badge = "🔬" if entry["model_type"] == "postwash" else "🏥"
            prob_color = {"low": "#c53030", "mid": "#c05621", "high": "#276749"}[entry["tier_key"]]
            st.markdown(f"""
            <div class="cycle-card cycle-card-{entry['tier_key']}">
                <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:4px;">{badge} {entry['label']} · {entry['timestamp']}</div>
                <div style="font-family:'DM Serif Display',serif; font-size:2rem; color:{prob_color};">{entry['p_cal']:.1%}</div>
                <div style="font-size:0.78rem; color:#718096;">{entry['tier_label']} probability</div>
            </div>""", unsafe_allow_html=True)

    if len(st.session_state.cycle_history) > 1:
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="white")
        ax.set_facecolor("white")
        probs  = [e["p_cal"] * 100 for e in st.session_state.cycle_history]
        labels = [e["label"] for e in st.session_state.cycle_history]
        bcolors = {"low": "#fc8181", "mid": "#f6ad55", "high": "#68d391"}
        ax.bar(labels, probs, color=[bcolors[e["tier_key"]] for e in st.session_state.cycle_history], width=0.5, zorder=3)
        ax.set_ylabel("Probability (%)", fontsize=9)
        ax.set_ylim(0, max(probs) * 1.6 + 1)
        ax.axhline(PW_LOW_CUTOFF * 100, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(PW_HIGH_CUTOFF * 100, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.5)
        for spine in ["top", "right"]: ax.spines[spine].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if st.button("Clear history", key="clear_history"):
        st.session_state.cycle_history = []
        st.rerun()

def build_pw_example():
    return pd.DataFrame([{"Uterine_Factors":0,"Tubal_Factors":0,"Ovarian_Factors":0,"Ovulatory_Factors":0,
        "Cervical_Factors":0,"Endometriosis_Factors":0,"Multisystem_Factors":0,"Cycle_Day":14,
        "Post_TPMSC":10.0,"First_Count":40.0,"Pre_Count":35.0,"Gynecological_Surgical_History":0,
        "Post_Count":12.0,"Post_Motile":80.0,"Pre_Motile":60.0,"Age_Female":32.0,
        "First_Progressive_Motile":40.0,"First_Volume":2.5,"Menstrual_Interval_Days":28.0,
        "First_Motile":60.0,"Body_Mass_Index":22.0,"Infertility_Type":0}])

def build_fv_example():
    return pd.DataFrame([{"Age_Female":32.0,"Body_Mass_Index":22.0,"Infertility_Type":0,
        "Total_infertile_duration":24.0,"Menstrual_Interval_Days":28.0,
        "Uterine_Factors":0,"Tubal_Factors":0,"Ovarian_Factors":0,"Ovulatory_Factors":0,
        "Cervical_Factors":0,"Endometriosis_Factors":0,"Multisystem_Factors":0,
        "Gynecological_Surgical_History":0,"First_Volume":2.5,"First_Count":40.0,"First_Progressive_Motile":40.0}])

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding-bottom:1.5rem;">
        <div style="font-size:2rem;">🔬</div>
        <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:white; line-height:1.3;">
            IUI Pregnancy<br>Probability Tool
        </div>
        <div style="font-size:0.75rem; color:#90caf9; margin-top:0.4rem;">Research prototype</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigation", ["✏️  Single Patient","📂  Multiple Patients","🔍  Detailed Analysis","ℹ️  About"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem; color:#90caf9; margin-bottom:0.4rem;'>📥 Download CSV Templates</div>", unsafe_allow_html=True)
    st.download_button("Procedure-Day Template", build_pw_example().to_csv(index=False).encode("utf-8"),
                       "iui_procedure_day_template.csv","text/csv", use_container_width=True)
    st.download_button("Pre-treatment Template", build_fv_example().to_csv(index=False).encode("utf-8"),
                       "iui_pretreatment_template.csv","text/csv", use_container_width=True)
    st.markdown("<div style='font-size:0.72rem; color:#90caf9; margin-top:1.5rem; line-height:1.6;'>For research use only.<br>Not a clinical decision system.</div>", unsafe_allow_html=True)

# =============================
# Pages
# =============================
if "Single" in page:
    st.markdown('<div class="section-header">✏️ Single Patient Prediction</div>', unsafe_allow_html=True)
    model_choice = st.radio("Which model would you like to use?",
        ["🔬 Procedure-Day Model — uses sperm wash results (day of IUI)",
         "🏥 Pre-treatment Model — uses baseline data only (before IUI begins)"])
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"
    st.caption("Any field left at 0 will be automatically filled with the population median from the training cohort.")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="form-group-label">Female Factors</div>', unsafe_allow_html=True)
            age_female              = st.number_input("Age (years)", 18.0, 55.0, 35.0, 1.0)
            bmi                     = st.number_input("BMI (kg/m2)", 10.0, 60.0, 21.7, 0.1)
            menstrual_interval_days = st.number_input("Menstrual cycle length (days)", 15.0, 180.0, 29.0, 1.0)
            infertility_type        = st.selectbox("Infertility type", [0,1],
                                        format_func=lambda x: "Primary — no prior pregnancy" if x==0 else "Secondary — prior pregnancy")
            infertile_duration      = st.number_input("Duration of infertility (months)", 0.0, 360.0, 36.0, 1.0)
            cycle_day = st.number_input("IUI cycle day", 1.0, 40.0, 14.0, 1.0) if model_type == "postwash" else 14.0
            st.markdown('<div class="form-group-label">Female Pathology (0 = absent, 1 = present)</div>', unsafe_allow_html=True)
            c1a, c1b = st.columns(2)
            with c1a:
                uterine_factors     = st.selectbox("Uterine",   [0,1])
                ovarian_factors     = st.selectbox("Ovarian",   [0,1])
                cervical_factors    = st.selectbox("Cervical",  [0,1])
                multisystem_factors = st.selectbox("Multisystem",[0,1])
            with c1b:
                tubal_factors         = st.selectbox("Tubal",          [0,1])
                ovulatory_factors     = st.selectbox("Ovulatory",      [0,1])
                endometriosis_factors = st.selectbox("Endometriosis",  [0,1])
                gyn_surgery           = st.selectbox("Prior gyn. surgery",[0,1])
        with col2:
            st.markdown('<div class="form-group-label">Initial Semen Analysis</div>', unsafe_allow_html=True)
            first_volume      = st.number_input("Volume (mL)",            0.0, 20.0,  3.0,  0.1)
            first_count       = st.number_input("Sperm count (x10^6/mL)", 0.0,500.0, 41.3, 0.1)
            first_motile      = st.number_input("Total motility (%)",      0.0,100.0, 54.7, 0.1)
            first_prog_motile = st.number_input("Progressive motility (%)",0.0,100.0, 52.6, 0.1)
            if model_type == "postwash":
                st.markdown('<div class="form-group-label">Prewash Semen</div>', unsafe_allow_html=True)
                pre_count  = st.number_input("Sperm count (x10^6/mL) ", 0.0,500.0, 42.6, 0.1)
                pre_motile = st.number_input("Total motility (%) ",      0.0,100.0, 57.6, 0.1)
                st.markdown('<div class="form-group-label">Postwash Semen</div>', unsafe_allow_html=True)
                post_count  = st.number_input("Sperm count (x10^6/mL)  ",0.0,500.0, 22.2,  0.1)
                post_tpmsc  = st.number_input("TPMSC (x10^6)",            0.0,500.0, 10.6,  0.1)
                post_motile = st.number_input("Total motility (%)  ",     0.0,100.0, 96.93, 0.1)
            else:
                pre_count = pre_motile = post_count = post_tpmsc = post_motile = 0.0
        cycle_label = st.text_input("Cycle label (optional)", placeholder="e.g. Cycle 1, Jan 2025")
        submitted   = st.form_submit_button("Calculate Pregnancy Probability", use_container_width=True, type="primary")

    if submitted:
        raw_inputs = {
            "Age_Female": age_female, "Body_Mass_Index": bmi,
            "Menstrual_Interval_Days": menstrual_interval_days,
            "Cycle_Day": cycle_day if model_type == "postwash" else None,
            "First_Volume": first_volume, "First_Count": first_count,
            "First_Motile": first_motile, "First_Progressive_Motile": first_prog_motile,
            "Pre_Count": pre_count if model_type == "postwash" else None,
            "Pre_Motile": pre_motile if model_type == "postwash" else None,
            "Post_Count": post_count if model_type == "postwash" else None,
            "Post_TPMSC": post_tpmsc if model_type == "postwash" else None,
            "Post_Motile": post_motile if model_type == "postwash" else None,
            "Total_infertile_duration": infertile_duration,
        }
        for w in validate_inputs({k: v for k, v in raw_inputs.items() if v is not None}):
            st.markdown(f'<div class="val-warn">⚠️ Value out of expected range — {w}</div>', unsafe_allow_html=True)

        try:
            with st.spinner("Calculating..."):
                if model_type == "postwash":
                    raw_df = pd.DataFrame([{
                        "Uterine_Factors":uterine_factors,"Tubal_Factors":tubal_factors,
                        "Ovarian_Factors":ovarian_factors,"Ovulatory_Factors":ovulatory_factors,
                        "Cervical_Factors":cervical_factors,"Endometriosis_Factors":endometriosis_factors,
                        "Multisystem_Factors":multisystem_factors,"Cycle_Day":cycle_day,
                        "Post_TPMSC":post_tpmsc,"First_Count":first_count,"Pre_Count":pre_count,
                        "Gynecological_Surgical_History":gyn_surgery,"Post_Count":post_count,
                        "Post_Motile":post_motile,"Pre_Motile":pre_motile,"Age_Female":age_female,
                        "First_Progressive_Motile":first_prog_motile,"First_Volume":first_volume,
                        "Menstrual_Interval_Days":menstrual_interval_days,"First_Motile":first_motile,
                        "Body_Mass_Index":bmi,"Infertility_Type":infertility_type}])
                    X = compute_pw_features(raw_df)
                else:
                    raw_df = pd.DataFrame([{
                        "Age_Female":age_female,"Body_Mass_Index":bmi,"Infertility_Type":infertility_type,
                        "Total_infertile_duration":infertile_duration,"Menstrual_Interval_Days":menstrual_interval_days,
                        "Uterine_Factors":uterine_factors,"Tubal_Factors":tubal_factors,"Ovarian_Factors":ovarian_factors,
                        "Ovulatory_Factors":ovulatory_factors,"Cervical_Factors":cervical_factors,
                        "Endometriosis_Factors":endometriosis_factors,"Multisystem_Factors":multisystem_factors,
                        "Gynecological_Surgical_History":gyn_surgery,"First_Volume":first_volume,
                        "First_Count":first_count,"First_Progressive_Motile":first_prog_motile}])
                    X = compute_fv_features(raw_df)

                p_val = float(predict(X, model_type)[0])
                against, favor = get_factors(X, model_type, top_k=5)

            render_result(p_val, model_type)
            st.markdown('<div class="section-header">What is influencing this result?</div>', unsafe_allow_html=True)
            render_factors(against, favor)
            add_to_history(p_val, model_type, against, favor, label=cycle_label or "")

            st.markdown('<div class="section-header">Download Report</div>', unsafe_allow_html=True)
            pdf_buf = generate_pdf_report(p_val, model_type, against, favor, patient_id=cycle_label)
            st.download_button("📄 Download PDF Summary", data=pdf_buf,
                               file_name=f"IUI_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf", use_container_width=True)
        except Exception as e:
            st.error(str(e))

    render_cycle_history()

elif "Multiple" in page:
    st.markdown('<div class="section-header">📂 Multiple Patients — Batch Prediction</div>', unsafe_allow_html=True)
    model_choice = st.radio("Which model?",["🔬 Procedure-Day Model","🏥 Pre-treatment Model"],horizontal=True)
    model_type   = "postwash" if "Procedure-Day" in model_choice else "first_visit"
    st.write("Upload a CSV file with one row per patient cycle. Download the template from the sidebar.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upl_calc")
    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write(f"**{len(df_raw)} records loaded** — preview:")
        st.dataframe(df_raw.head(), use_container_width=True)
        if st.button("Calculate Probabilities", use_container_width=True, type="primary"):
            try:
                with st.spinner("Processing..."):
                    X      = compute_pw_features(df_raw) if model_type == "postwash" else compute_fv_features(df_raw)
                    p_cals = predict(X, model_type)
                    tiers  = [assign_tier(float(p), model_type) for p in p_cals]
                    out    = df_raw.copy()
                    out["Pregnancy probability"] = [f"{p:.1%}" for p in p_cals]
                    out["Risk group"]            = [t[0] for t in tiers]
                st.success(f"Done — {len(out)} records processed")
                from collections import Counter
                counts = Counter([t[0] for t in tiers])
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Low", counts.get("Low",0))
                c2.metric("🟡 Intermediate", counts.get("Intermediate",0))
                c3.metric("🟢 High", counts.get("High",0))
                st.dataframe(out[["Pregnancy probability","Risk group"]], use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Full Results", out.to_csv(index=False).encode("utf-8"),
                                   "iui_predictions.csv","text/csv", use_container_width=True)
            except Exception as e:
                st.error(str(e))

elif "Detailed" in page:
    st.markdown('<div class="section-header">🔍 Detailed Analysis — Single Patient</div>', unsafe_allow_html=True)
    st.write("Upload a CSV, select a row, and see a full breakdown of what is driving the prediction for that patient.")
    model_choice = st.radio("Which model?",["🔬 Procedure-Day Model","🏥 Pre-treatment Model"],horizontal=True)
    model_type   = "postwash" if "Procedure-Day" in model_choice else "first_visit"
    uploaded2 = st.file_uploader("Upload CSV", type=["csv"], key="upl_exp")
    if uploaded2 is not None:
        df_raw2 = pd.read_csv(uploaded2)
        st.dataframe(df_raw2.head(), use_container_width=True)
        row_idx = st.number_input("Select row to analyse", 0, max(0, len(df_raw2)-1), 0, 1)
        if st.button("Analyse This Patient", type="primary"):
            try:
                with st.spinner("Analysing..."):
                    X2    = compute_pw_features(df_raw2) if model_type == "postwash" else compute_fv_features(df_raw2)
                    x_row = X2.iloc[[int(row_idx)]].copy()
                    p_val = float(predict(x_row, model_type)[0])
                    against, favor = get_factors(x_row, model_type, top_k=5)
                render_result(p_val, model_type)
                st.markdown('<div class="section-header">What is influencing this result?</div>', unsafe_allow_html=True)
                render_factors(against, favor)
                pdf_buf = generate_pdf_report(p_val, model_type, against, favor)
                st.download_button("📄 Download PDF Summary", data=pdf_buf,
                                   file_name=f"IUI_report_row{row_idx}.pdf", mime="application/pdf",
                                   use_container_width=True)
                with st.expander("Show technical SHAP chart"):
                    try:
                        fig = plot_shap_waterfall(x_row, model_type)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception:
                        st.info("SHAP chart could not be generated.")
            except Exception as e:
                st.error(str(e))

elif "About" in page:
    st.markdown('<div class="section-header">ℹ️ About This Tool</div>', unsafe_allow_html=True)
    st.markdown("This tool estimates the probability of clinical pregnancy per IUI cycle, based on a machine learning model developed from a retrospective cohort at a Thai fertility center. It is intended to support patient counseling — not to replace clinical judgment.")

    st.markdown('<div class="section-header">Which model should I use?</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="info-card"><span class="badge-pw">🔬 Procedure-Day Model</span>
            <h3>Use on the day of IUI</h3>
            <p style="color:#475569;font-size:0.92rem;line-height:1.7;">Requires sperm wash results. Use when postwash semen parameters are available — it gives a more complete picture of the cycle's chances.</p>
            <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">Sensitivity 61.1% &nbsp;·&nbsp; NPV 95.6%</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="info-card"><span class="badge-fv">🏥 Pre-treatment Model</span>
            <h3>Use at the initial consultation</h3>
            <p style="color:#475569;font-size:0.92rem;line-height:1.7;">Requires only baseline clinical and semen parameters. Use to counsel patients before IUI begins — no sperm wash results needed.</p>
            <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem;">Sensitivity 65.9% &nbsp;·&nbsp; NPV 95.8%</p></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">How to read the result</div>', unsafe_allow_html=True)
    st.markdown("""
    **The probability (%)** is the model's estimate of how likely this patient is to achieve clinical pregnancy in this cycle.

    **The risk group** shows where this patient falls compared to similar patients in our cohort:

    | Risk group | What it means |
    |---|---|
    | 🔴 Low | About 4 in 100 similar patients became pregnant per cycle |
    | 🟡 Intermediate | About 9–10 in 100 similar patients became pregnant per cycle |
    | 🟢 High | About 13–18 in 100 similar patients became pregnant per cycle |

    **The factors chart** shows which parameters are working for or against this patient. Bars show how strongly each factor influences the result — Strong, Moderate, or Mild.
    """)

    with st.expander("Technical details — for researchers"):
        st.markdown("""
        **Algorithm:** XGBoost with isotonic regression calibration  
        **Cohort:** 2,945 cycles, 1,761 patients (single-center, Thailand)  
        **Validation:** Patient-level holdout (80/20), bootstrap 95% CI (n=1,000)

        | Metric | Procedure-Day | Pre-treatment |
        |---|---|---|
        | ROC-AUC | 0.663 (0.570–0.742) | 0.625 (0.528–0.721) |
        | Sensitivity | 61.1% (46.2–76.9%) | 65.9% (51.1–80.4%) |
        | NPV | 95.6% (93.6–97.7%) | 95.8% (93.7–97.9%) |
        """)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.caption("Procedure-Day Model — SHAP")
            if PW_SHAP_IMG.exists():
                st.image(str(PW_SHAP_IMG), use_container_width=True)
            else:
                st.info("SHAP image not found.")
        with col_s2:
            st.caption("Pre-treatment Model — SHAP")
            if FV_SHAP_IMG.exists():
                st.image(str(FV_SHAP_IMG), use_container_width=True)
            else:
                st.info("SHAP image not found.")

    st.markdown("""<div class="disclaimer">
    This tool is a research prototype for academic purposes only. It is intended to support clinical judgment and should not be used as the sole basis for clinical decision-making. Outputs are statistical estimates derived from a single-center retrospective cohort of IUI cycles performed at a Thai fertility center. External validation has not yet been performed.
    </div>""", unsafe_allow_html=True)