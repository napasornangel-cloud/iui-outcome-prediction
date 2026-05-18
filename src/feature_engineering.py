from pathlib import Path
import numpy as np
import pandas as pd


def add_sperm_wash_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Ratio TPMSC (with clipping)
    if {"Post_TPMSC", "Pre_TPMSC"}.issubset(data.columns):
        data["Ratio_TPMSC"] = np.where(
            data["Pre_TPMSC"] > 0,
            data["Post_TPMSC"] / data["Pre_TPMSC"],
            np.nan,
        )
        data["Ratio_TPMSC"] = data["Ratio_TPMSC"].clip(0, 10)

    if {"Post_Progressive_Motile", "Pre_Progressive_Motile"}.issubset(data.columns):
        data["Delta_Progressive_Motile"] = (
            data["Post_Progressive_Motile"] - data["Pre_Progressive_Motile"]
        )

    if {"Post_Motile", "Pre_Motile"}.issubset(data.columns):
        data["Delta_Motile"] = data["Post_Motile"] - data["Pre_Motile"]

    return data


def add_cycle_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if {"Mature_Follicle_Count", "Endometrium_Thickness"}.issubset(data.columns):
        data["Follicle_Endo_Product"] = (
            data["Mature_Follicle_Count"] * data["Endometrium_Thickness"]
        )

    if {"Cycle_Number", "Ovary_Stimulation_Round"}.issubset(data.columns):
        data["Cumulative_Treatment"] = (
            data["Cycle_Number"] * data["Ovary_Stimulation_Round"]
        )

    return data


def add_female_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if {"Age_Female", "FSH_Baseline"}.issubset(data.columns):
        data["Age_FSH_Interaction"] = (
            data["Age_Female"] * data["FSH_Baseline"]
        )

    if {"Body_Mass_Index", "Infertility_Type"}.issubset(data.columns):
        data["BMI_InfertilityType_Interaction"] = (
            data["Body_Mass_Index"] * data["Infertility_Type"]
        )

    factor_cols = [
        "Uterine_Factors",
        "Tubal_Factors",
        "Ovarian_Factors",
        "Ovulatory_Factors",
        "Cervical_Factors",
        "Endometriosis_Factors",
        "Multisystem_Factors",
    ]
    available_factor_cols = [col for col in factor_cols if col in data.columns]

    if available_factor_cols:
        data["Total_Female_Pathology"] = data[available_factor_cols].sum(
            axis=1,
            min_count=1,
        )

    return data


def add_sperm_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if {"First_Volume", "First_Count", "First_Progressive_Motile"}.issubset(data.columns):
        data["First_TPMSC"] = (
            data["First_Volume"]
            * data["First_Count"]
            * data["First_Progressive_Motile"] / 100
        ).clip(upper=200)

    return data


def add_binary_clinical_flags(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if "Post_TPMSC" in data.columns:
        data["Low_TPMSC"] = np.where(
            data["Post_TPMSC"].isna(),
            np.nan,
            (data["Post_TPMSC"] < 5).astype(float),
        )

    if "Endometrium_Thickness" in data.columns:
        data["Thin_Endometrium"] = np.where(
            data["Endometrium_Thickness"].isna(),
            np.nan,
            (data["Endometrium_Thickness"] < 7).astype(float),
        )

    if "Age_Female" in data.columns:
        data["Advanced_Age"] = (data["Age_Female"] >= 38).astype(int)

    return data


def summarize_added_features(df: pd.DataFrame, candidate_features: list[str]) -> None:
    added_features = [feature for feature in candidate_features if feature in df.columns]

    print(f"\nEngineered features created: {len(added_features)}")
    for feature in added_features:
        missing_pct = df[feature].isna().mean() * 100
        print(f"  - {feature}: missing {missing_pct:.1f}%")


def run_feature_engineering(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)
    print(f"Input dataset shape: {df.shape}")

    steps = [
        ("Sperm wash features", add_sperm_wash_features),
        ("Cycle quality features", add_cycle_quality_features),
        ("Female interaction features", add_female_interaction_features),
        ("Sperm quality features", add_sperm_quality_features),
        ("Binary clinical flags", add_binary_clinical_flags),
    ]

    for step_name, step_func in steps:
        print(f"[Running] {step_name}")
        df = step_func(df)
        print(f"          Current shape: {df.shape}")

    engineered_features = [
        "Ratio_TPMSC",
        "Delta_Progressive_Motile",
        "Delta_Motile",
        "Follicle_Endo_Product",
        "Cumulative_Treatment",
        "Age_FSH_Interaction",
        "BMI_InfertilityType_Interaction",
        "Total_Female_Pathology",
        "First_TPMSC",
        "Low_TPMSC",
        "Thin_Endometrium",
        "Advanced_Age",
    ]

    summarize_added_features(df, engineered_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nSaved engineered dataset to: {output_path}")
    print(f"Final dataset shape: {df.shape}")

    return df


if __name__ == "__main__":
    INPUT_PATH = Path("data/processed/cycle_level_ready_for_ml.csv")
    OUTPUT_PATH = Path("data/processed/cycle_level_features.csv")

    run_feature_engineering(INPUT_PATH, OUTPUT_PATH)