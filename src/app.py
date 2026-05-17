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

# =============================
# Tier cutoffs
# =============================
PW_LOW_CUTOFF  = 0.038760
PW_HIGH_CUTOFF = 0.099379
FV_LOW_CUTOFF  = 0.039
FV_HIGH_CUTOFF = 0.080

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
.risk-low {
    background: #fff5f5;
    border: 2px solid #fc8181;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-mid {
    background: #fffbeb;
    border: 2px solid #f6ad55;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-high {
    background: #f0fff4;
    border: 2px solid #68d391;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.risk-label { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.risk-label-low  { color: #c53030; }
.risk-label-mid  { color: #c05621; }
.risk-label-high { color: #276749; }
.risk-prob { font-family: 'DM Serif Display', serif; font-size: 3.5rem; line-height: 1; margin: 0.3rem 0; }
.risk-prob-low  { color: #c53030; }
.risk-prob-mid  { color: #c05621; }
.risk-prob-high { color: #276749; }
.risk-sublabel { font-size: 0.85rem; color: #718096; margin-top: 0.2rem; }
.risk-obs { font-size: 0.9rem; margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid #e2e8f0; color: #4a5568; line-height: 1.6; }
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
    color: #744210;
    font-size: 0.82rem;
    line-height: 1.6;
    margin-top: 1rem;
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
        obs    = {"low": "4.6%", "mid": "10.0%", "high": "18.8%"}
    else:
        lo, hi = FV_LOW_CUTOFF, FV_HIGH_CUTOFF
        obs    = {"low": "2.2%", "mid": "7.5%", "high": "11.2%"}
    if p_cal < lo:
        return "Low", "low", obs["low"]
    if p_cal < hi:
        return "Intermediate", "mid", obs["mid"]
    return "High", "high", obs["high"]

def get_display_name(raw_name, model_type="postwash"):
    dm = PW_DISPLAY_MAP if model_type == "postwash" else FV_DISPLAY_MAP
    return dm.get(str(raw_name), str(raw_name).replace("_", " "))

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
        raise ValueError("Missing required columns:\n- " + "\n- ".join(missing))
    for c in PW_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in PW_REQUIRED_RAW:
        if df[c].isna().any():
            med = PW_MEDIANS.get(c, 0.0)
            df[c] = df[c].fillna(med)
            imputed.append(f"{c} → {med}")
    if imputed:
        st.warning("Some missing values were filled with training set medians:\n" +
                   "\n".join(f"- {x}" for x in imputed))
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
        raise ValueError("Missing required columns:\n- " + "\n- ".join(missing))
    for c in FV_REQUIRED_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    imputed = []
    for c in FV_REQUIRED_RAW:
        if df[c].isna().any():
            med = FV_MEDIANS.get(c, 0.0)
            df[c] = df[c].fillna(med)
            imputed.append(f"{c} → {med}")
    if imputed:
        st.warning("Some missing values were filled with training set medians:\n" +
                   "\n".join(f"- {x}" for x in imputed))
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
    return p_cal

def local_explain(X_row, model_type="postwash", top_k=6):
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
        label    = get_display_name(X_row.columns[j], model_type)
        dz       = float(sv[j])
        p_before = sigmoid(z); z += dz; p_after = sigmoid(z)
        delta_pp = (p_after - p_before) * 100.0
        arrow    = "↑" if delta_pp >= 0 else "↓"
        direction = f"{arrow} Increases chance" if delta_pp >= 0 else f"{arrow} Decreases chance"
        rows.append({
            "Factor":    label,
            "Effect":    direction,
            "Impact":    f"{abs(delta_pp):.1f} percentage points",
        })
    return pd.DataFrame(rows)

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

def render_result(p_cal, model_type="postwash"):
    tier_label, tier_key, obs_rate = assign_tier(p_cal, model_type)
    badge_class = "model-badge-pw" if model_type == "postwash" else "model-badge-fv"
    badge_text  = "🔬 Procedure-Day Model" if model_type == "postwash" else "🏥 Pre-treatment Model"

    st.markdown(f'<span class="{badge_class}">{badge_text}</span>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="risk-{tier_key}">
        <div class="risk-label risk-label-{tier_key}">{tier_label} Probability</div>
        <div class="risk-prob risk-prob-{tier_key}">{p_cal:.1%}</div>
        <div class="risk-sublabel">Estimated pregnancy probability per cycle</div>
        <div class="risk-obs">
            In the study cohort, patients in this group had an observed pregnancy rate of
            <b>{obs_rate} per cycle</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    This is a statistical estimate to support clinical counseling, not a guarantee of outcome.
    Always interpret results in the context of the full clinical picture.
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
        "✏️  Single Patient",
        "📂  Multiple Patients",
        "🔍  Detailed Analysis",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<div style='font-size:0.78rem; color:#90caf9; margin-bottom:0.4rem;'>📥 Download CSV Templates</div>", unsafe_allow_html=True)
    st.download_button(
        "Procedure-Day Template",
        build_pw_example().to_csv(index=False).encode("utf-8"),
        "iui_procedure_day_template.csv", "text/csv",
        use_container_width=True
    )
    st.download_button(
        "Pre-treatment Template",
        build_fv_example().to_csv(index=False).encode("utf-8"),
        "iui_pretreatment_template.csv", "text/csv",
        use_container_width=True
    )
    st.markdown("""
    <div style="font-size:0.72rem; color:#90caf9; margin-top:1.5rem; line-height:1.6;">
    For research use only.<br>
    Not a clinical decision system.
    </div>
    """, unsafe_allow_html=True)

# =============================
# Pages
# =============================

if "Single" in page:
    st.markdown('<div class="section-header">✏️ Single Patient Prediction</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Which model would you like to use?",
        ["🔬 Procedure-Day Model — uses sperm wash results (day of IUI)",
         "🏥 Pre-treatment Model — uses initial assessment only (before IUI)"],
        horizontal=False
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"
    st.caption("Fields left blank or set to 0 will be filled with population medians from the training cohort.")

    with st.form("manual_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="form-group-label">Female Factors</div>', unsafe_allow_html=True)
            age_female              = st.number_input("Age (years)", 18.0, 55.0, 35.0, 1.0)
            bmi                     = st.number_input("BMI (kg/m²)", 10.0, 60.0, 21.7, 0.1)
            menstrual_interval_days = st.number_input("Menstrual cycle length (days)", 15.0, 180.0, 29.0, 1.0)
            infertility_type        = st.selectbox(
                "Infertility type",
                options=[0, 1],
                format_func=lambda x: "Primary — no prior pregnancy" if x == 0 else "Secondary — prior pregnancy"
            )
            infertile_duration = st.number_input("Duration of infertility (months)", 0.0, 360.0, 36.0, 1.0)

            if model_type == "postwash":
                cycle_day = st.number_input("IUI cycle day", 1.0, 40.0, 14.0, 1.0)
            else:
                cycle_day = 14.0

            st.markdown('<div class="form-group-label">Female Pathology (0 = absent, 1 = present)</div>', unsafe_allow_html=True)
            c1a, c1b = st.columns(2)
            with c1a:
                uterine_factors       = st.selectbox("Uterine",            [0, 1])
                ovarian_factors       = st.selectbox("Ovarian",            [0, 1])
                cervical_factors      = st.selectbox("Cervical",           [0, 1])
                multisystem_factors   = st.selectbox("Multisystem",        [0, 1])
            with c1b:
                tubal_factors         = st.selectbox("Tubal",              [0, 1])
                ovulatory_factors     = st.selectbox("Ovulatory",          [0, 1])
                endometriosis_factors = st.selectbox("Endometriosis",      [0, 1])
                gyn_surgery           = st.selectbox("Prior gyn. surgery", [0, 1])

        with col2:
            st.markdown('<div class="form-group-label">Initial Semen Analysis</div>', unsafe_allow_html=True)
            first_volume      = st.number_input("Volume (mL)",              0.0, 20.0,  3.0,  0.1)
            first_count       = st.number_input("Sperm count (×10⁶/mL)",   0.0, 500.0, 41.3, 0.1)
            first_motile      = st.number_input("Total motility (%)",       0.0, 100.0, 54.7, 0.1)
            first_prog_motile = st.number_input("Progressive motility (%)", 0.0, 100.0, 52.6, 0.1)

            if model_type == "postwash":
                st.markdown('<div class="form-group-label">Prewash Semen</div>', unsafe_allow_html=True)
                pre_count  = st.number_input("Sperm count (×10⁶/mL) ", 0.0, 500.0, 42.6, 0.1)
                pre_motile = st.number_input("Total motility (%) ",     0.0, 100.0, 57.6, 0.1)

                st.markdown('<div class="form-group-label">Postwash Semen</div>', unsafe_allow_html=True)
                post_count  = st.number_input("Sperm count (×10⁶/mL)  ", 0.0, 500.0, 22.2,  0.1)
                post_tpmsc  = st.number_input("TPMSC (×10⁶)",             0.0, 500.0, 10.6,  0.1)
                post_motile = st.number_input("Total motility (%)  ",     0.0, 100.0, 96.93, 0.1)
            else:
                pre_count = pre_motile = post_count = post_tpmsc = post_motile = 0.0

        submitted = st.form_submit_button("Calculate Pregnancy Probability", use_container_width=True, type="primary")

    if submitted:
        try:
            with st.spinner("Calculating..."):
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

                p_cal = predict(X, model_type)

            render_result(float(p_cal[0]), model_type)

            st.markdown('<div class="section-header">Key Factors Influencing This Result</div>', unsafe_allow_html=True)
            expl = local_explain(X, model_type, top_k=6)
            st.dataframe(expl, use_container_width=True, hide_index=True)
            st.caption("Factors are ranked by how much they affect the predicted probability for this patient.")

        except Exception as e:
            st.error(str(e))

elif "Multiple" in page:
    st.markdown('<div class="section-header">📂 Multiple Patients — Batch Prediction</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Which model would you like to use?",
        ["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"],
        horizontal=True
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"

    st.write("Upload a CSV file with one row per patient cycle. Download the template from the sidebar to see the required columns.")
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

                    out   = df_raw.copy()
                    tiers = [assign_tier(float(p), model_type) for p in p_cals]
                    out["Pregnancy probability (%)"]  = [f"{p:.1%}" for p in p_cals]
                    out["Risk group"]                 = [t[0] for t in tiers]
                    out["Observed rate (cohort)"]     = [t[2] for t in tiers]

                st.success(f"Done — {len(out)} records processed")

                from collections import Counter
                counts = Counter([t[0] for t in tiers])
                c1, c2, c3 = st.columns(3)
                c1.metric("Low risk", counts.get("Low", 0))
                c2.metric("Intermediate risk", counts.get("Intermediate", 0))
                c3.metric("High risk", counts.get("High", 0))

                show_cols = ["Pregnancy probability (%)", "Risk group", "Observed rate (cohort)"]
                st.dataframe(out[show_cols], use_container_width=True, hide_index=True)

                st.download_button(
                    "⬇️ Download Results",
                    out.to_csv(index=False).encode("utf-8"),
                    "iui_predictions.csv", "text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(str(e))

elif "Detailed" in page:
    st.markdown('<div class="section-header">🔍 Detailed Analysis — Single Row</div>', unsafe_allow_html=True)
    st.write("Upload a CSV, select a row, and see a detailed breakdown of what factors drive the prediction.")

    model_choice = st.radio(
        "Which model?",
        ["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"],
        horizontal=True
    )
    model_type = "postwash" if "Procedure-Day" in model_choice else "first_visit"

    uploaded2 = st.file_uploader("Upload CSV", type=["csv"], key="upl_exp")

    if uploaded2 is not None:
        df_raw2 = pd.read_csv(uploaded2)
        st.dataframe(df_raw2.head(), use_container_width=True)
        row_idx = st.number_input("Select row to analyse", 0, max(0, len(df_raw2)-1), 0, 1)

        if st.button("Analyse This Row", type="primary"):
            try:
                with st.spinner("Analysing..."):
                    X2    = compute_pw_features(df_raw2) if model_type == "postwash" else compute_fv_features(df_raw2)
                    x_row = X2.iloc[[int(row_idx)]].copy()
                    p_cal = predict(x_row, model_type)

                render_result(float(p_cal[0]), model_type)

                st.markdown('<div class="section-header">What is driving this result?</div>', unsafe_allow_html=True)
                expl = local_explain(x_row, model_type, top_k=8)
                st.dataframe(expl, use_container_width=True, hide_index=True)
                st.caption("Factors ranked by their influence on this patient's predicted probability.")

                with st.expander("Show detailed SHAP chart"):
                    st.caption("SHAP values reflect the model output before probability calibration.")
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

    st.markdown("""
    This tool provides per-cycle pregnancy probability estimates for patients undergoing
    intrauterine insemination (IUI), based on a machine learning model trained on a
    retrospective cohort from a Thai fertility center.

    **Two models are available:**
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="result-card">
            <span class="model-badge-pw">🔬 Procedure-Day Model</span>
            <h3>When to use</h3>
            <p style="color:#37474f; font-size:0.92rem; line-height:1.7;">
            Use on the day of IUI, after sperm preparation results are available.
            Incorporates postwash semen parameters for a more complete picture.
            </p>
            <h3>Key performance</h3>
            <p style="color:#37474f; font-size:0.92rem; line-height:1.7;">
            ROC-AUC 0.664 &nbsp;|&nbsp; NPV 97.8% &nbsp;|&nbsp; Sensitivity 90.1%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card">
            <span class="model-badge-fv">🏥 Pre-treatment Model</span>
            <h3>When to use</h3>
            <p style="color:#37474f; font-size:0.92rem; line-height:1.7;">
            Use at the initial consultation before IUI begins.
            Requires only baseline clinical and semen parameters — no sperm wash results needed.
            </p>
            <h3>Key performance</h3>
            <p style="color:#37474f; font-size:0.92rem; line-height:1.7;">
            ROC-AUC 0.624 &nbsp;|&nbsp; NPV 97.8% &nbsp;|&nbsp; Sensitivity 90.2%
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Groups</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔬 Procedure-Day Model", "🏥 Pre-treatment Model"])
    with tab1:
        st.dataframe(pd.DataFrame({
            "Risk group":           ["Low", "Intermediate", "High"],
            "Probability range":    ["< 3.9%", "3.9–9.9%", "≥ 9.9%"],
            "Cycles in test set":   [370, 211, 16],
            "Observed pregnancies": [17, 21, 3],
            "Observed rate":        ["4.6%", "10.0%", "18.8%"],
        }), use_container_width=True, hide_index=True)

    with tab2:
        st.dataframe(pd.DataFrame({
            "Risk group":           ["Low", "Intermediate", "High"],
            "Probability range":    ["< 3.9%", "3.9–8.0%", "≥ 8.0%"],
            "Cycles in test set":   [180, 265, 152],
            "Observed pregnancies": [4, 20, 17],
            "Observed rate":        ["2.2%", "7.5%", "11.2%"],
        }), use_container_width=True, hide_index=True)

    with st.expander("Technical details — for researchers"):
        st.markdown("""
        **Algorithm:** XGBoost with isotonic regression calibration

        **Training cohort:** 2,348 IUI cycles | **Test cohort:** 597 cycles

        **Split method:** Patient-level GroupShuffleSplit (80/20, seed 42)

        **Event rate:** ~6.2% clinical pregnancy per cycle

        **Validation:** Bootstrap 95% CI from 1,000 resamples

        | Metric | Procedure-Day | Pre-treatment |
        |---|---|---|
        | ROC-AUC | 0.664 (0.579–0.745) | 0.624 (0.542–0.707) |
        | PR-AUC | 0.130 (0.078–0.202) | 0.096 (0.064–0.145) |
        | Brier (calibrated) | 0.064 (0.047–0.082) | 0.065 (0.048–0.082) |
        | Sensitivity | 90.1% (80.4–97.8%) | 90.2% (80.4–97.8%) |
        | Specificity | 32.9% (29.1–36.9%) | 31.7% (28.2–35.6%) |
        | NPV | 97.8% (95.5–99.5%) | 97.8% (95.2–99.5%) |
        | PPV | 9.0% (6.3–11.8%) | 8.9% (6.2–11.6%) |
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

    st.markdown("""
    <div class="disclaimer-box">
    This tool is a research prototype for academic purposes only. It is intended to support clinical judgment
    and should not be used as the sole basis for clinical decision-making. Outputs are statistical estimates
    derived from a single-center retrospective cohort of IUI cycles performed at a Thai fertility center.
    External validation in independent cohorts has not yet been performed.
    </div>
    """, unsafe_allow_html=True)