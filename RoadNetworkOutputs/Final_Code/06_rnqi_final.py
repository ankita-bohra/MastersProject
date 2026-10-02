from pathlib import Path
import pandas as pd

BASE = Path("outputs") / "metadata"
FILES = {
    "connectivity": BASE / "city_wise_connectivity_scores.csv",
    "efficiency": BASE / "city_wise_efficiency_scores.csv",
    "hierarchy": BASE / "city_wise_hierarchy_scores.csv",
}
CITIES = ["Chandigarh", "Jaipur", "Pune", "Kolkata", "Ranipet"]
OUT = BASE / "final_rnqi_scores.csv"
VAL = BASE / "final_rnqi_validation.csv"
REF = BASE / "final_rnqi_reference_range.csv"

def load(path, pillar):
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    df = pd.read_csv(path)
    city = next((c for c in ["city","City","CITY"] if c in df.columns), None)

    score_candidates = [
        f"{pillar}_score",
        f"{pillar}_score_corrected",
        f"{pillar.title()} Score",
        f"{pillar.title()} Score Corrected",
        pillar,
    ]

    score = next((c for c in score_candidates if c in df.columns), None)

    if city is None or score is None:
        raise ValueError(f"Cannot identify columns in {path}. Columns: {list(df.columns)}")
    x = df[[city, score]].copy()
    x.columns = ["city", pillar]
    x["city"] = x["city"].astype(str).str.strip()
    x[pillar] = pd.to_numeric(x[pillar], errors="coerce")
    if x["city"].duplicated().any() or x[pillar].isna().any():
        raise ValueError(f"Invalid {pillar} data in {path}")
    if ((x[pillar] < 0) | (x[pillar] > 100)).any():
        raise ValueError(f"{pillar} contains a score outside 0–100")
    return x

def main():
    print("=" * 72)
    print("RNQI — FINAL COMPOSITE CALCULATION")
    print("=" * 72)
    print("Formula: RNQI = (Connectivity + Efficiency + Hierarchy) / 3")
    print()

    df = load(FILES["connectivity"], "connectivity")
    for pillar in ["efficiency", "hierarchy"]:
        df = df.merge(load(FILES[pillar], pillar), on="city", how="outer", validate="one_to_one")

    if set(df.city) != set(CITIES):
        raise ValueError(f"City mismatch. Found: {sorted(df.city)}")
    df["city"] = pd.Categorical(df["city"], CITIES, ordered=True)
    df = df.sort_values("city").reset_index(drop=True)

    # Equal-weight arithmetic mean of the three normalized pillar scores.
    df["rnqi_score"] = df[["connectivity","efficiency","hierarchy"]].mean(axis=1)
    df["formula_check"] = (
        (df["connectivity"] + df["efficiency"] + df["hierarchy"]) / 3
        - df["rnqi_score"]
    ).abs()

    if not (df["formula_check"] <= 1e-12).all():
        raise ValueError("RNQI arithmetic validation failed.")

    output = df[["city","connectivity","efficiency","hierarchy","rnqi_score"]].copy()
    output.columns = ["city","connectivity_score","efficiency_score","hierarchy_score","rnqi_score"]

    # Store simple validation flags alongside the final scores.
    validation = output.copy()
    validation["all_pillars_0_100"] = validation[
        ["connectivity_score","efficiency_score","hierarchy_score"]
    ].apply(lambda r: ((r >= 0) & (r <= 100)).all(), axis=1)
    validation["formula_valid"] = df["formula_check"] <= 1e-12
    validation["calculation_error"] = df["formula_check"]

    # Reference RNQI range for the five benchmark cities: mean ± 1 SD.
    # The three pillar means are retained alongside the RNQI reference values
    # so the benchmark profile can be compared with individual city results.
    
    rnqi_mean = output["rnqi_score"].mean()
    # The five benchmark areas are treated as the complete reference population.
    # Therefore, population SD is used for the classification reference range.

    rnqi_sd = output["rnqi_score"].std(ddof=0)
    reference = pd.DataFrame([{
    "rnqi_mean": rnqi_mean,
    "rnqi_sd": rnqi_sd,
    "rnqi_reference_lower": rnqi_mean - rnqi_sd,
    "rnqi_reference_upper": rnqi_mean + rnqi_sd,
    "reference_sd_basis": (
        "Population SD (ddof=0); five benchmark areas treated "
        "as complete reference population"
    ),
    "connectivity_mean": output["connectivity_score"].mean(),
    "efficiency_mean": output["efficiency_score"].mean(),
    "hierarchy_mean": output["hierarchy_score"].mean(),
}])

    BASE.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False)
    validation.to_csv(VAL, index=False)
    reference.to_csv(REF, index=False)

    print("FINAL RNQI SCORES")
    print("-" * 72)
    for _, r in output.iterrows():
        print(f"{str(r.city):<12} C={r.connectivity_score:8.4f}  E={r.efficiency_score:8.4f}  H={r.hierarchy_score:8.4f}  RNQI={r.rnqi_score:8.4f}")

    print()
    print("VALIDATION")
    print("-" * 72)
    print(f"Cities validated        : {len(output)}")
    print(f"All pillar scores 0–100 : {'YES' if validation.all_pillars_0_100.all() else 'NO'}")
    print(f"RNQI formula validated  : {'YES' if validation.formula_valid.all() else 'NO'}")
    print(f"Maximum arithmetic error: {validation.calculation_error.max():.3e}")
    print()
    print(f"Saved: {OUT}")
    print(f"Saved: {VAL}")
    print(f"Saved: {REF}")
    print()
    print("REFERENCE / STANDARD RNQI RANGE (MEAN ± 1 SD)")
    print("-" * 72)
    r = reference.iloc[0]
    print(f"RNQI mean               : {r.rnqi_mean:.4f}")
    print(f"RNQI SD                 : {r.rnqi_sd:.4f}")
    print(f"Reference RNQI range    : {r.rnqi_reference_lower:.4f} to {r.rnqi_reference_upper:.4f}")
    print(f"Reference Connectivity  : {r.connectivity_mean:.4f}")
    print(f"Reference Efficiency    : {r.efficiency_mean:.4f}")
    print(f"Reference Hierarchy     : {r.hierarchy_mean:.4f}")

if __name__ == "__main__":
    main()
