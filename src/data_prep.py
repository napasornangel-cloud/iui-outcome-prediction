import pandas as pd
import numpy as np
from pathlib import Path

from src.feature_engineering import run_feature_engineering

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data/raw/final_coding.xlsx"
CLEAN_PATH = BASE_DIR / "data/processed/cycle_level_ready_for_ml.csv"
FEATURES_PATH = BASE_DIR / "data/processed/cycle_level_features.csv"


def clean_basic_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # NOTE: "none" is intentionally NOT included here. Verified against the
    # raw dataset that "none" appears only in Gynecological_Surgical_History,
    # where it carries clinical meaning ("no prior surgery") rather than
    # representing a missing value. Converting it to NaN here would prevent
    # clean_gynecological_surgical_history() from ever mapping it to 0, since
    # that function's own false_like set (which includes "none") would never
    # see the raw string. If "none" appears in other free-text columns in the
    # future, re-check whether it should be excluded per-column instead of
    # globally.
    weird_na = {"", " ", "na", "n/a", "nan", "null", "-"}

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].apply(
                lambda x: x.strip() if isinstance(x, str) else x
            )
            data[col] = data[col].apply(
                lambda x: np.nan if isinstance(x, str) and x.lower() in weird_na else x
            )

    return data


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    numeric_cols = [
        "Age_Female", "Age_Male", "Total_infertile_duration",
        "Pregnancy_History", "Number_Of_Alive_Children", "Number_Of_Miscarriages",
        "Infertility_Type", "Body_Mass_Index", "Menstrual",
        "Menstrual_Interval_Days", "Menstrual_Duration_Days", "Dysmenorrhea",
        "FSH_Baseline", "LH_Baseline", "E2_Baseline", "PRL_Baseline",
        "Uterine_Factors", "Tubal_Factors", "Ovarian_Factors", "Ovulatory_Factors",
        "Cervical_Factors", "Endometriosis_Factors", "Multisystem_Factors",
        "Alcohol", "Smoke",
        "First_Volume", "First_Count", "First_Motile", "First_Progressive_Motile",
        "First_Normal_Morpho",
        "Pre_Volume", "Pre_Count", "Pre_Motile", "Pre_Progressive_Motile", "Pre_TPMSC",
        "Post_Count", "Post_Motile", "Post_Progressive_Motile", "Post_TPMSC",
        "Cycle_Type", "OI_Clomiphene", "OI_Letrozole", "OI_Gonadotropins",
        "Cycle_Day", "Cycle_Number", "Ovary_Stimulation_Round", "hCG_Dose",
        "Mature_Follicle_Count", "Endometrium_Thickness",
        "Endo_Type_Triple", "Endo_Type_Intermediate", "Endo_Type_Mixed",
        "Result",
    ]

    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def add_hcg_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "hCG_Dose" in data.columns:
        x = pd.to_numeric(data["hCG_Dose"], errors="coerce")

        data["hCG_Used"] = np.where(
            x.isna(), np.nan,
            np.where(x == 0, 0, 1)
        )

        data["hCG_Type"] = np.where(
            x.isna(), np.nan,
            np.where(x.isin([0, 1, 2]), x, np.nan)
        )

        unexpected_values = x[(x.notna()) & (~x.isin([0, 1, 2]))]
        if len(unexpected_values) > 0:
            raise ValueError(
                f"Unexpected values in hCG_Dose: {sorted(unexpected_values.unique())}"
            )

        data = data.drop(columns=["hCG_Dose"])

    return data


def clean_gynecological_surgical_history(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "Gynecological_Surgical_History" in data.columns:
        false_like = {"0", "0.0", "none", "false", "no", ""}
        missing_like = {"nan", "na", "n/a", "null"}

        s = data["Gynecological_Surgical_History"].apply(
            lambda x: x.strip().lower() if isinstance(x, str) else x
        )

        data["Gynecological_Surgical_History"] = s.apply(
            lambda x: 0 if (x in false_like or x == 0)
            else (np.nan if pd.isna(x) or x in missing_like else 1)
        )

    return data


def filter_cycle_1_to_3(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "Cycle_Number" in data.columns:
        data["Cycle_Number"] = pd.to_numeric(data["Cycle_Number"], errors="coerce")
        data = data[data["Cycle_Number"].isin([1, 2, 3])]

    return data


def final_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "Result" in data.columns:
        data["Result"] = pd.to_numeric(data["Result"], errors="coerce")
        data = data.dropna(subset=["Result"])

        valid_vals = {0, 1, 0.0, 1.0}
        invalid_vals = set(data["Result"].unique()) - valid_vals
        if invalid_vals:
            raise ValueError(f"Invalid values in Result: {sorted(invalid_vals)}")

    if "HN" in data.columns:
        data = data.dropna(subset=["HN"])
        data["HN"] = data["HN"].astype(str).str.strip()
        data = data[data["HN"] != ""]

    if {"HN", "Cycle_Number"}.issubset(data.columns):
        duplicated = data.duplicated(subset=["HN", "Cycle_Number"])
        if duplicated.any():
            raise ValueError("Duplicate records found for (HN, Cycle_Number)")

    return data


def main():
    df = pd.read_excel(RAW_PATH, sheet_name="final")

    df = clean_basic_values(df)
    df = coerce_numeric_columns(df)
    df = add_hcg_features(df)
    df = clean_gynecological_surgical_history(df)
    df = filter_cycle_1_to_3(df)
    df = final_sanity_checks(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)

    # IMPORTANT: cast Path -> str เพื่อ compatibility กับของเดิม
    run_feature_engineering(str(CLEAN_PATH), str(FEATURES_PATH))


if __name__ == "__main__":
    main()