import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# =============================
# Paths
# =============================
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Postwash model
PW_BASE_MODEL_PATH = PROJECT_ROOT / "models" / "saved_models" / "final_model" / "XGBoost_Baseline_calibration_base_model.joblib"
PW_CALIBRATOR_PATH = PROJECT_ROOT / "models" / "saved_models" / "final_model" / "isotonic_calibrator_final_xgb.joblib"
PW_SHAP_IMG        = PROJECT_ROOT / "reports" / "figures" / "shap_final_xgb" / "SHAP_Beeswarm_Final_XGBoost_Baseline.png"

# First visit model
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

# =============================
# Tier cutoffs
# =============================
PW_LOW_CUTOFF  = 0.038760
PW_HIGH_CUTOFF = 0.099379
FV_LOW_CUTOFF  = 0.039
FV_HIGH_CUTOFF = 0.080
VERY_LOW_CUTOFF = 0.01

# =============================
# Display maps
# =============================
PW_DISPLAY_MAP = {
    "Uterine_Factors":                 "Uterine factor",
    "Total_Female_Pathology":          "Total female pathology factors",
    "Ovulatory_Factors":               "Ovulatory factor",
    "Cycle_Day":                       "IUI cycle day",
    "Post_TPMSC":                      "Postwash TPMSC (million)",
    "First_Count":                     "Initial sperm concentration (×10⁶/mL)",
    "Pre_Count":                       "Prewash sperm concentration (×10⁶/mL)",
    "Gynecological_Surgical_History":  "Prior gynecologic surgery",
    "Post_Count":                      "Postwash sperm concentration (×10⁶/mL)",
    "Delta_Motile":                    "Change in total motility after wash (%)",
    "Age_Female":                      "Female age (years)",
    "First_Progressive_Motile":        "Initial progressive motility (%)",
    "First_Volume":                    "Initial semen volume (mL)",
    "Menstrual_Interval_Days":         "Menstrual cycle length (days)",
    "First_Motile":                    "Initial total motility (%)",
    "BMI_InfertilityType_Interaction": "BMI × infertility type",
    "First_TPMSC":                     "Initial TPMSC (million)",
}

FV_DISPLAY_MAP = {
    "Age_Female":                    "Female age (years)",
    "Body_Mass_Index":               "BMI (kg/m²)",
    "Infertility_Type":              "Infertility type",
    "Total_infertile_duration":      "Infertility duration (months)",
    "Menstrual_Interval_Days":       "Menstrual cycle length (days)",
    "Uterine_Factors":               "Uterine factor",
    "Ovulatory_Factors":             "Ovulatory factor",
    "Tubal_Factors":                 "Tubal factor",
    "Endometriosis_Factors":         "Endometriosis",
    "Gynecological_Surgical_History":"Prior gynecologic surgery",
    "Total_Female_Pathology":        "Total female pathology score",
    "First_Volume":                  "Initial semen volume (mL)",
    "First_Count":                   "Initial sperm concentration (×10⁶/mL)",
    "First_Progressive_Motile":      "Initial progressive motility (%)",
    "First_TPMSC":                   "Initial TPMSC (million)",
    "BMI_InfertilityType_Interaction":"BMI × infertility type",
}

# =============================
# Required raw columns
# =============================
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

# =============================
# Training medians
# =============================
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

section[data-testid="stSidebar"] {
    background: #0f2b4a;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] * { color: #e8f0fe !important; }
.main { background: #f5f7fa; }

.result-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    box-shadow: 0 2px 12px rgba(15,43,74,0.08);
    margin-bottom: 1.2rem;
    border-left: 4px solid #1565c0;
}
.result-card h3 {
    font-family: 'DM Serif Display', serif;
    color: #0f2b4a;
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}
.tier-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.03em;
}
.tier-low  { background: #fdecea; color: #c62828; }
.tier-mid  { background: #fff8e1; color: #e65100; }
.tier-high { background: #e8f5e9; color: #2e7d32; }
.metric-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.metric-box {
    flex: 1;
    min-width: 130px;
    background: #f0f4ff;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box .label {
    font-size: 0.75rem;
    color: #5c6bc0;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-box .value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #0f2b4a;
    line-height: 1;
}
.section-header {
    font-family: 'DM Serif Display', serif;
    color: #0f2b4a;
    font-size: 1.3rem;
    border-bottom: 2px solid #e3eafc;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}
.form-group-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #1565c0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.2rem 0 0.4rem;
}
.interp-box {
    background: #e8f0fe;
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: #1a237e;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-top: 0.8rem;
}
.model-badge-pw {
    display: inline-block;
    background: #e3f2fd;
    color: #0d47a1;
    border-radius: 8px;
    padding: 0.2rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.model-badge-fv {
    display: inline-block;
    background: #e8f5e9;
    color: #1b5e20;
    border-radius: 8px;
    padding: 0.2rem 0.8rem;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.disclaimer-box {
    background: #fff8e1;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    color: #e65100;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 0.6rem;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================
# Helpers
# =============================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def assign_tier(p_cal, model_type="postwash"):
    if model_type == "postwash":
        lo, hi = PW_LOW_CUTOFF, PW_HIGH_CUTOFF
        rates  = {"low": "4.6%", "mid": "10.0%", "high": "18.8%"}
    else:
        lo, hi = FV_LOW_CUTOFF, FV_HIGH_CUTOFF
        rates  = {"low": "2.2%", "mid": "7.5%", "high": "11.2%"}

    if p_cal < lo:
        return "Tier 1 (Low Probability)", "tier-low", rates["low"]
    if p_cal < hi:
        return "Tier 2 (Intermediate Probability)", "tier-mid", rates["mid"]
    return "Tier 3 (High Probability)", "tier-high", rates["high"]

def very_low_flag(p_cal):
    return "Yes" if p_cal < VERY_LOW_CUTOFF else "No"

def get_display_name(raw_name, model_type="postwash"):
    dm = PW_DISPLAY_MAP if model_type == "postwash" else FV_DISPLAY_MAP
    return dm.get(str(raw_name), str(raw_name).replace("_", " "))

def validate_inputs(row: dict) -> list:
    errors = []
    for col, (lo, hi) in VALIDATION_RULES.items():
        val = row.get(col)
        if val is not None and not (lo <= float(val) <= hi):
            errors.append(f"{col}: value {val} outside expected range [{lo}, {hi}]")
    return errors

@st.cache_resource
def load_model(model_type="postwash"):
    if model_type == "postwash":
        m = joblib.load(PW_BASE_MODEL_PATH)
        c = joblib.load(PW_CALIBRATOR_PATH)
    else:
        m = joblib.load(FV_BASE_MODEL_PATH)
        c = joblib.load(FV_CALIBRATOR_PATH)
    return m, c

def compute_pw_features(df_raw):
    df = df_raw.copy()
    missing = [c for c in PW_REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError("Missing columns:\n- " + "\n- ".join(missing))
    for c in PW_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in PW_REQUIRED_RAW:
        if df[c].isna().any():
            med = PW_MEDIANS.get(c, 0.0)
            df[c] = df[c].fillna(med)
            imputed.append(f"{c} → {med}")
    if imputed:
        st.warning("⚠️ Missing values imputed:\n" + "\n".join(f"- {x}" for x in imputed))
    df["Total_Female_Pathology"]          = (df["Uterine_Factors"] + df["Tubal_Factors"] +
                                              df["Ovarian_Factors"] + df["Ovulatory_Factors"] +
                                              df["Cervical_Factors"] + df["Endometriosis_Factors"] +
                                              df["Multisystem_Factors"])
    df["Delta_Motile"]                    = df["Post_Motile"] - df["Pre_Motile"]
    df["BMI_InfertilityType_Interaction"] = df["Body_Mass_Index"] * df["Infertility_Type"]
    df["First_TPMSC"]                     = (df["First_Volume"] * df["First_Count"] *
                                              df["First_Progressive_Motile"] / 100).clip(upper=200)
    return df[PW_FEATURES].copy()

def compute_fv_features(df_raw):
    df = df_raw.copy()
    missing = [c for c in FV_REQUIRED_RAW if c not in df.columns]
    if missing:
        raise ValueError("Missing columns:\n- " + "\n- ".join(missing))
    for c in FV_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in FV_REQUIRED_RAW:
        if df[c].isna().any():
            med = FV_MEDIANS.get(c, 0.0)
            df[c] = df[c].fillna(med)
            imputed.append(f"{c} → {med}")
    if imputed:
        st.warning("⚠️ Missing values imputed:\n" + "\n".join(f"- {x}" for x in imputed))
    df["Total_Female_Pathology"]          = (df["Uterine_Factors"] + df["Tubal_Factors"] +
                                              df["Ovarian_Factors"] + df["Ovulatory_Factors"] +
                                              df["Cervical_Factors"] + df["Endometriosis_Factors"] +
                                              df["Multisystem_Factors"])
    df["BMI_InfertilityType_Interaction"] = df["Body_Mass_Index"] * df["Infertility_Type"]
    df["First_TPMSC"]                     = (df["First_Volume"] * df["First_Count"] *
                                              df["First_Progressive_Motile"] / 100).clip(upper=200)
    return df[FV_FEATURES].copy()

def predict(X, model_type="postwash"):
    model, calibrator = load_model(model_type)
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = np.clip(calibrator.predict(p_raw), 0, 1)
    return p_raw, p_cal

def local_explain(X_row, model_type="postwash", top_k=8):
    model, _ = load_model(model_type)
    xgb = model.named_steps["model"] if hasattr(model, "named_steps") else model
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(X_row)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    if shap_vals.ndim == 3:         shap_vals = shap_vals[:, :, 1]
    sv    = shap_vals.reshape(-1)
    base  = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1] if len(np.ravel(base)) >= 2 else float(np.ravel(base)[0])
    base  = float(base)
    order = np.argsort(np.abs(sv))[::-1][:min(top_k, len(sv))]
    z, rows = base, []
    for j in order:
        label = get_display_name(X_row.columns[j], model_type)
        dz    = float(sv[j])
        p_before = sigmoid(z); z += dz; p_after = sigmoid(z)
        delta_pp = (p_after - p_before) * 100.0
        arrow    = "↑" if delta_pp >= 0 else "↓"
        rows.append((label, round(delta_pp, 2),
                     f"{arrow} increases probability" if delta_pp >= 0
                     else f"{arrow} decreases probability"))
    return pd.DataFrame(rows, columns=["Factor", "Approx. change (pp)", "Direction"])

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

def plot_gauge(p_cal, model_type="postwash"):
    tier_label, _, _ = assign_tier(p_cal, model_type)
    color = "#c62828" if "Tier 1" in tier_label else ("#e65100" if "Tier 2" in tier_label else "#2e7d32")
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("white")
    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta), color="#e8edf5", linewidth=20, solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - p_cal * np.pi, 300)
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=20, solid_capstyle="round")
    ax.text(0, -0.05, f"{p_cal:.1%}", ha="center", va="center", fontsize=22,
            fontweight="bold", color="#0f2b4a", fontfamily="serif")
    ax.text(0, -0.42, "Calibrated probability", ha="center", va="center", fontsize=9, color="#78909c")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.6, 1.2); ax.axis("off")
    return fig

def render_result_card(p_raw, p_cal, model_type="postwash"):
    tier_label, tier_css, obs_rate = assign_tier(p_cal, model_type)
    vlow = very_low_flag(p_cal)
    badge_class = "model-badge-pw" if model_type == "postwash" else "model-badge-fv"
    badge_text  = "🔬 Procedure-Day Model" if model_type == "postwash" else "🏥 Pre-treatment Model"

    col_gauge, col_detail = st.columns([1, 2])
    with col_gauge:
        st.pyplot(plot_gauge(p_cal, model_type), use_container_width=True)
    with col_detail:
        st.markdown(f"""
        <div class="result-card">
            <span class="{badge_class}">{badge_text}</span>
            <h3>Prediction Result</h3>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="label">Raw per-cycle</div>
                    <div class="value">{p_raw:.1%}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Calibrated per-cycle</div>
                    <div class="value">{p_cal:.1%}</div>
                </div>
            </div>
            <span class="tier-badge {tier_css}">{tier_label}</span>
            &nbsp;&nbsp;<small style="color:#78909c">Very low flag: {vlow}</small>
        </div>
        """, unsafe_allow_html=True)

    tier_key = "Tier 1" if "Tier 1" in tier_label else ("Tier 2" if "Tier 2" in tier_label else "Tier 3")

    if model_type == "postwash":
        interp = {
            "Tier 1": f"This result falls in the <b>low-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>.",
            "Tier 2": f"This result falls in the <b>intermediate-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>.",
            "Tier 3": f"This result falls in the <b>high-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>.",
        }
    else:
        interp = {
            "Tier 1": f"Based on pre-procedural parameters, this result falls in the <b>low-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>. This estimate is available before sperm washing.",
            "Tier 2": f"Based on pre-procedural parameters, this result falls in the <b>intermediate-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>. This estimate is available before sperm washing.",
            "Tier 3": f"Based on pre-procedural parameters, this result falls in the <b>high-probability</b> group. In the study cohort, patients in this tier had an observed pregnancy rate of <b>{obs_rate} per cycle</b>. This estimate is available before sperm washing.",
        }

    st.markdown(f'<div class="interp-box">💬 {interp[tier_key]}</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer-box">
    ‡ SHAP explanations reflect model output before probability calibration.
    </div>
    """, unsafe_allow_html=True)

def build_pw_example():
    return pd.DataFrame([{
        "Uterine_Factors": 0, "Tubal_Factors": 0, "Ovarian_Factors": 0,
        "Ovulatory_Factors": 0, "Cervical_Factors": 0, "Endometriosis_Factors": 0,
        "Multisystem_Factors": 0, "Cycle_Day": 14, "Post_TPMSC": 10.0,
        "First_Count": 40.0, "Pre_Count": 35.0, "Gynecological_Surgical_History": 0,
        "Post_Count": 12.0, "Post_Motile": 80.0, "Pre_Motile": 60.0,
        "Age_Female": 32.0, "First_Progressive_Motile": 40.0, "First_Volume": 2.5,
        "Menstrual_Interval_Days": 28.0, "First_Motile": 60.0,
        "Body_Mass_Index": 22.0, "Infertility_Type": 0,
    }])

def build_fv_example():
    return pd.DataFrame([{
        "Age_Female": 32.0, "Body_Mass_Index": 22.0, "Infertility_Type": 0,
        "Total_infertile_duration": 24.0, "Menstrual_Interval_Days": 28.0,
        "Uterine_Factors": 0, "Tubal_Factors": 0, "Ovarian_Factors": 0,
        "Ovulatory_Factors": 0, "Cervical_Factors": 0,
        "Endometriosis_Factors": 0, "Multisystem_Factors": 0,
        "Gynecological_Surgical_History": 0,
        "First_Volume": 2.5, "First_Count": 40.0, "First_Progressive_Motile": 40.0,
    }])

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
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "✏️  Manual Entry",
        "📂  Batch CSV",
        "🔍  Explanation",
        "ℹ️  Model Info"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem; color:#90caf9; margin-bottom:0.4rem;'>📥 Download CSV Templates</div>", unsafe_allow_html=True)
    st.download_button(
        "Procedure-Day Model Template",
        build_pw_example().to_csv(index=False).encode("utf-8"),
        "iui_procedure_day_template.csv", "text/csv",
        use_container_width=True
    )
    st.download_button(
        "Pre-treatment Model Template",
        build_fv_example().to_csv(index=False).encode("utf-8"),
        "iui_pretreatment_template.csv", "text/csv",
        use_container_width=True
    )
    st.markdown("""
    <div style="font-size:0.72rem; color:#90caf9; margin-top:1.5rem; line-height:1.6;">
    ⚠️ For research use only.<br>
    Not a clinical decision system.<br>
    Always use with clinical judgment.
    </div>
    """, unsafe_allow_html=True)

# =============================
# Pages
# =============================

if "Manual" in page:
    st.markdown('<div class="section-header">✏️ Manual Entry — Single Patient Cycle</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select prediction model:",
        ["🔬 Procedure-Day Model (day of IUI, after sperm preparation)",
         "🏥 Pre-treatment Model (before IUI initiation, no postwash data required)"],
        horizontal=True
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"

    st.caption("Leave fields as 0 if value is unavailable — missing values will be imputed automatically.")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="form-group-label">🩺 Female & Cycle Factors</div>', unsafe_allow_html=True)
            age_female              = st.number_input("Female age (years)", 18.0, 55.0, 35.0, 1.0)
            bmi                     = st.number_input("BMI (kg/m²)", 10.0, 60.0, 21.7, 0.1)
            menstrual_interval_days = st.number_input("Menstrual cycle interval (days)", 15.0, 180.0, 29.0, 1.0)
            infertility_type        = st.selectbox(
                "Infertility type",
                options=[0, 1],
                format_func=lambda x: "Primary (no prior pregnancy)" if x == 0 else "Secondary (prior pregnancy)"
            )
            infertile_duration = st.number_input("Infertility duration (months)", 0.0, 360.0, 36.0, 1.0)

            if model_type == "postwash":
                cycle_day = st.number_input("IUI cycle day", 1.0, 40.0, 14.0, 1.0)
            else:
                cycle_day = 14.0

            st.markdown('<div class="form-group-label">🔬 Female Pathology (0=absent, 1=present)</div>', unsafe_allow_html=True)
            c1a, c1b = st.columns(2)
            with c1a:
                uterine_factors       = st.selectbox("Uterine",       [0, 1])
                ovarian_factors       = st.selectbox("Ovarian",       [0, 1])
                cervical_factors      = st.selectbox("Cervical",      [0, 1])
                multisystem_factors   = st.selectbox("Multisystem",   [0, 1])
            with c1b:
                tubal_factors         = st.selectbox("Tubal",         [0, 1])
                ovulatory_factors     = st.selectbox("Ovulatory",     [0, 1])
                endometriosis_factors = st.selectbox("Endometriosis", [0, 1])
                gyn_surgery           = st.selectbox("Gyn. surgery",  [0, 1])

        with col2:
            st.markdown('<div class="form-group-label">💉 Initial Semen Sample</div>', unsafe_allow_html=True)
            first_volume      = st.number_input("Volume (mL)",             0.0, 20.0,  3.0,  0.1)
            first_count       = st.number_input("Sperm count (×10⁶/mL)",  0.0, 500.0, 41.3, 0.1)
            first_motile      = st.number_input("Total motility (%)",      0.0, 100.0, 54.7, 0.1)
            first_prog_motile = st.number_input("Progressive motility (%)",0.0, 100.0, 52.6, 0.1)

            if model_type == "postwash":
                st.markdown('<div class="form-group-label">🧫 Prewash Semen</div>', unsafe_allow_html=True)
                pre_count  = st.number_input("Sperm count (×10⁶/mL) ", 0.0, 500.0, 42.6, 0.1)
                pre_motile = st.number_input("Motility (%) ",           0.0, 100.0, 57.6, 0.1)

                st.markdown('<div class="form-group-label">✅ Postwash Semen</div>', unsafe_allow_html=True)
                post_count  = st.number_input("Sperm count (×10⁶/mL)  ", 0.0, 500.0, 22.2,  0.1)
                post_tpmsc  = st.number_input("TPMSC (×10⁶)",             0.0, 500.0, 10.6,  0.1)
                post_motile = st.number_input("Motility (%)  ",           0.0, 100.0, 96.93, 0.1)
            else:
                pre_count = pre_motile = post_count = post_tpmsc = post_motile = 0.0

        submitted = st.form_submit_button("🚀 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        try:
            with st.spinner("Running prediction..."):
                if model_type == "postwash":
                    raw_df = pd.DataFrame([{
                        "Uterine_Factors": uterine_factors, "Tubal_Factors": tubal_factors,
                        "Ovarian_Factors": ovarian_factors, "Ovulatory_Factors": ovulatory_factors,
                        "Cervical_Factors": cervical_factors, "Endometriosis_Factors": endometriosis_factors,
                        "Multisystem_Factors": multisystem_factors, "Cycle_Day": cycle_day,
                        "Post_TPMSC": post_tpmsc, "First_Count": first_count,
                        "Pre_Count": pre_count, "Gynecological_Surgical_History": gyn_surgery,
                        "Post_Count": post_count, "Post_Motile": post_motile, "Pre_Motile": pre_motile,
                        "Age_Female": age_female, "First_Progressive_Motile": first_prog_motile,
                        "First_Volume": first_volume, "Menstrual_Interval_Days": menstrual_interval_days,
                        "First_Motile": first_motile, "Body_Mass_Index": bmi,
                        "Infertility_Type": infertility_type,
                    }])
                    X = compute_pw_features(raw_df)
                else:
                    raw_df = pd.DataFrame([{
                        "Age_Female": age_female, "Body_Mass_Index": bmi,
                        "Infertility_Type": infertility_type,
                        "Total_infertile_duration": infertile_duration,
                        "Menstrual_Interval_Days": menstrual_interval_days,
                        "Uterine_Factors": uterine_factors, "Tubal_Factors": tubal_factors,
                        "Ovarian_Factors": ovarian_factors, "Ovulatory_Factors": ovulatory_factors,
                        "Cervical_Factors": cervical_factors, "Endometriosis_Factors": endometriosis_factors,
                        "Multisystem_Factors": multisystem_factors,
                        "Gynecological_Surgical_History": gyn_surgery,
                        "First_Volume": first_volume, "First_Count": first_count,
                        "First_Progressive_Motile": first_prog_motile,
                    }])
                    X = compute_fv_features(raw_df)

                p_raw, p_cal = predict(X, model_type)

            st.success("✅ Prediction complete")
            render_result_card(float(p_raw[0]), float(p_cal[0]), model_type)
            st.markdown('<div class="section-header">🧠 Top Factors Influencing This Prediction</div>', unsafe_allow_html=True)
            st.caption("† SHAP values reflect model output before probability calibration.")
            expl = local_explain(X, model_type, top_k=8)
            st.dataframe(expl, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(str(e))

elif "Batch" in page:
    st.markdown('<div class="section-header">📂 Batch Prediction from CSV</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select prediction model:",
        ["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"],
        horizontal=True
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"

    st.write("Upload a CSV with the required input columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="upl_calc")

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write("**Preview**")
        st.dataframe(df_raw.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            try:
                with st.spinner("Processing..."):
                    X = compute_pw_features(df_raw) if model_type == "postwash" else compute_fv_features(df_raw)
                    p_raw, p_cal = predict(X, model_type)
                    out = df_raw.copy()
                    out["raw_per_cycle_probability"]        = p_raw
                    out["calibrated_per_cycle_probability"] = p_cal
                    tiers = [assign_tier(float(x), model_type) for x in p_cal]
                    out["risk_tier"]     = [t[0] for t in tiers]
                    out["very_low_flag"] = [very_low_flag(float(x)) for x in p_cal]

                st.success(f"✅ {len(out)} rows processed")
                show_cols = ["raw_per_cycle_probability", "calibrated_per_cycle_probability",
                             "risk_tier", "very_low_flag"]
                st.dataframe(out[show_cols], use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download Results (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "iui_predictions_out.csv", "text/csv")
            except Exception as e:
                st.error(str(e))

elif "Explanation" in page:
    st.markdown('<div class="section-header">🔍 Explain One Row from CSV</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select prediction model:",
        ["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"],
        horizontal=True
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"

    uploaded2 = st.file_uploader("Upload CSV", type=["csv"], key="upl_exp")

    if uploaded2 is not None:
        df_raw2 = pd.read_csv(uploaded2)
        st.dataframe(df_raw2.head(), use_container_width=True)
        row_idx = st.number_input("Row index to explain", 0, max(0, len(df_raw2)-1), 0, 1)

        if st.button("🔍 Explain This Row", type="primary"):
            try:
                with st.spinner("Generating explanation..."):
                    X2    = compute_pw_features(df_raw2) if model_type == "postwash" else compute_fv_features(df_raw2)
                    x_row = X2.iloc[[int(row_idx)]].copy()
                    p_raw, p_cal = predict(x_row, model_type)

                render_result_card(float(p_raw[0]), float(p_cal[0]), model_type)

                st.markdown('<div class="section-header">📊 SHAP Waterfall Plot</div>', unsafe_allow_html=True)
                st.caption("† SHAP values reflect model output before probability calibration.")
                try:
                    fig = plot_shap_waterfall(x_row, model_type)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception:
                    pass

                st.markdown('<div class="section-header">📋 Top Factors (Table)</div>', unsafe_allow_html=True)
                expl = local_explain(x_row, model_type, top_k=8)
                st.dataframe(expl, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(str(e))

elif "Model" in page:
    st.markdown('<div class="section-header">ℹ️ Model Information</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"])

    with tab1:
        st.markdown("""
        <div class="result-card">
            <span class="model-badge-pw">🔬 Procedure-Day Model</span>
            <h3>Model Details</h3>
            <ul style="color:#37474f; line-height:2;">
                <li>Algorithm: <b>XGBoost</b> with 16 selected predictors</li>
                <li>Timing: <b>Day of IUI procedure</b> (after sperm preparation)</li>
                <li>Imbalance handling: No resampling (scale_pos_weight)</li>
                <li>Calibration: Post-hoc isotonic regression</li>
                <li>Feature selection: Gain-based importance with 1-SE rule</li>
                <li>Training: 2,348 cycles | Test: 597 cycles | Event rate: ~6.2%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Validation Performance</div>', unsafe_allow_html=True)
        pw_metrics = pd.DataFrame({
            "Metric":  ["ROC-AUC", "PR-AUC", "Brier (uncalibrated)", "Brier (calibrated)",
                        "Sensitivity", "Specificity", "NPV", "PPV"],
            "Value":   [0.664, 0.130, 0.219, 0.064, 0.901, 0.329, 0.978, 0.090],
            "95% CI":  ["0.579–0.745", "0.078–0.202", "0.212–0.228", "0.047–0.082",
                        "0.804–0.978", "0.291–0.369", "0.955–0.995", "0.063–0.118"],
            "Note":    ["Bootstrap (n=1,000)"] * 4 + ["Calibrated threshold = 0.039"] * 4,
        })
        st.dataframe(pw_metrics, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">📊 Risk Tier Performance</div>', unsafe_allow_html=True)
        pw_tiers = pd.DataFrame({
            "Risk Tier":            ["Tier 1 (Low)", "Tier 2 (Intermediate)", "Tier 3 (High)"],
            "Probability cutoff":   ["< 3.9%", "3.9–9.9%", "≥ 9.9%"],
            "N cycles":             [370, 211, 16],
            "Observed pregnancies": [17, 21, 3],
            "Pregnancy rate":       ["4.6%", "10.0%", "18.8%"],
        })
        st.dataframe(pw_tiers, use_container_width=True, hide_index=True)
        st.caption("Cutoffs derived from P50 and P75 of calibrated probability distribution. Monotonic increase confirmed.")

        st.markdown('<div class="section-header">🧬 SHAP Feature Importance</div>', unsafe_allow_html=True)
        st.caption("† SHAP values reflect model output before calibration.")
        if PW_SHAP_IMG.exists():
            st.image(str(PW_SHAP_IMG), use_container_width=True)
        else:
            st.info(f"SHAP image not found: {PW_SHAP_IMG}")

    with tab2:
        st.markdown("""
        <div class="result-card">
            <span class="model-badge-fv">🏥 Pre-treatment Model</span>
            <h3>Model Details</h3>
            <ul style="color:#37474f; line-height:2;">
                <li>Algorithm: <b>XGBoost</b> with 16 pre-specified predictors</li>
                <li>Timing: <b>Before IUI initiation</b> (first clinic visit, no postwash data required)</li>
                <li>Imbalance handling: No resampling (scale_pos_weight)</li>
                <li>Calibration: Post-hoc isotonic regression</li>
                <li>Training: 2,348 cycles | Test: 597 cycles | Event rate: ~6.2%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">📊 Validation Performance</div>', unsafe_allow_html=True)
        fv_metrics = pd.DataFrame({
            "Metric":  ["ROC-AUC", "PR-AUC", "Brier (uncalibrated)", "Brier (calibrated)",
                        "Sensitivity", "Specificity", "NPV", "PPV"],
            "Value":   [0.624, 0.096, 0.214, 0.065, 0.902, 0.317, 0.978, 0.089],
            "95% CI":  ["0.542–0.707", "0.064–0.145", "—", "0.048–0.082",
                        "0.804–0.978", "0.282–0.356", "0.952–0.995", "0.062–0.116"],
            "Note":    ["Bootstrap (n=1,000)"] * 4 + ["Calibrated threshold = 0.044"] * 4,
        })
        st.dataframe(fv_metrics, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-header">📊 Risk Tier Performance</div>', unsafe_allow_html=True)
        fv_tiers = pd.DataFrame({
            "Risk Tier":            ["Tier 1 (Low)", "Tier 2 (Intermediate)", "Tier 3 (High)"],
            "Probability cutoff":   ["< 3.9%", "3.9–8.0%", "≥ 8.0%"],
            "N cycles":             [180, 265, 152],
            "Observed pregnancies": [4, 20, 17],
            "Pregnancy rate":       ["2.2%", "7.5%", "11.2%"],
        })
        st.dataframe(fv_tiers, use_container_width=True, hide_index=True)
        st.caption("Cutoffs optimized for first visit probability distribution. Monotonic increase confirmed.")

        st.markdown('<div class="section-header">📊 Comparison: Procedure-Day vs Pre-treatment Model</div>', unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Metric":            ["ROC-AUC", "PR-AUC", "Brier (calibrated)", "Sensitivity", "Specificity", "NPV", "PPV"],
            "Procedure-Day":     ["0.664 (0.579–0.745)", "0.130 (0.078–0.202)", "0.064 (0.047–0.082)",
                                  "90.1% (80.4–97.8%)", "32.9% (29.1–36.9%)", "97.8% (95.5–99.5%)", "9.0% (6.3–11.8%)"],
            "Pre-treatment":     ["0.624 (0.542–0.707)", "0.096 (0.064–0.145)", "0.065 (0.048–0.082)",
                                  "90.2% (80.4–97.8%)", "31.7% (28.2–35.6%)", "97.8% (95.2–99.5%)", "8.9% (6.2–11.6%)"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("Both models tested on identical held-out test set (597 cycles, 41 pregnancies). Bootstrap 95% CI from 1,000 resamples.")

        st.markdown('<div class="section-header">🧬 SHAP Feature Importance</div>', unsafe_allow_html=True)
        st.caption("† SHAP values reflect model output before calibration.")
        if FV_SHAP_IMG.exists():
            st.image(str(FV_SHAP_IMG), use_container_width=True)
        else:
            st.info(f"SHAP image not found: {FV_SHAP_IMG}")

    st.markdown("""
    <div style="background:#fff8e1; border-radius:12px; padding:1rem 1.4rem;
                color:#e65100; font-size:0.88rem; line-height:1.7; margin-top:1.5rem;">
    ⚠️ Disclaimer<br>
    This tool is a research prototype for academic purposes only. It is intended to support clinical judgment
    and should not be used as the sole basis for clinical decision-making. Outputs are statistical estimates
    derived from a single-center retrospective cohort of IUI cycles performed at a Thai fertility center.
    External validation in independent cohorts has not yet been performed.
    </div>
    """, unsafe_allow_html=True)