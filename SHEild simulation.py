import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# CALCULATE RISK SCORE
# ==========================================================

def calculate_risk(df):
    """
    Calculates a composite risk score and assigns risk levels (Low/Medium/High).
    Incorporates women crime volume, crime rate, trend, and children crime if available.
    """
    df = df.copy()

    # Crime Trend (women)
    df["trend"] = df["cases_2023"] - df["cases_2021"]

    # Base features
    features = ["cases_2023", "crime_rate_2023", "trend"]
    norm_cols = ["cases_norm", "crime_norm", "trend_norm"]

    scaler = MinMaxScaler()
    df[norm_cols] = scaler.fit_transform(df[features])

    # Weighted Risk Score (women-focused)
    df["risk_score"] = (
        0.5 * df["cases_norm"]
        + 0.3 * df["crime_norm"]
        + 0.2 * df["trend_norm"]
    )

    # Boost risk score if children crime data is present
    if "children_cases_2023" in df.columns:
        child_scaler = MinMaxScaler()
        df["children_norm"] = child_scaler.fit_transform(
            df[["children_cases_2023"]]
        )
        # Blend: 85% women risk + 15% children risk
        df["risk_score"] = 0.85 * df["risk_score"] + 0.15 * df["children_norm"]

    # Convert to Safety Score (0–100)
    df["safety_score"] = ((1 - df["risk_score"]) * 100).round(1)

    # Percentile-based thresholds (relative classification)
    low = df["risk_score"].quantile(0.33)
    high = df["risk_score"].quantile(0.66)

    def classify(score):
        if score <= low:
            return "Low"
        elif score <= high:
            return "Medium"
        return "High"

    df["risk_level"] = df["risk_score"].apply(classify)

    return df


# ==========================================================
# SIMULATE CRIME CHANGE
# ==========================================================

def simulate_crime_change(df, percentage):
    """
    Simulates increase/decrease in crime cases.
    Example:
        20 -> increase by 20%
       -10 -> decrease by 10%
    """

    simulated = df.copy()

    simulated["cases_2023"] = (
        simulated["cases_2023"]
        * (1 + percentage / 100)
    ).round()

    simulated = calculate_risk(simulated)

    return simulated


# ==========================================================
# CITY SUMMARY
# ==========================================================

def get_city_summary(df, city):
    """
    Returns a comprehensive summary dict for a given city/location.
    Includes women crime, children crime, and territory metadata.
    """
    row = df[df["city"] == city].iloc[0]

    summary = {
        "city": row["city"],
        "state_ut": row.get("state_ut", "N/A"),
        "territory_type": row.get("territory_type", "N/A"),
        "district": row.get("district", "N/A"),
        "cases_2021": int(row["cases_2021"]),
        "cases_2022": int(row["cases_2022"]),
        "cases_2023": int(row["cases_2023"]),
        "crime_rate": float(row["crime_rate_2023"]),
        "chargesheet_rate": float(row["chargesheet_rate"]),
        "risk_score": round(float(row["risk_score"]), 2),
        "risk_level": row["risk_level"],
        "safety_score": round(float(row["safety_score"]), 1),
    }

    # Include children data if available
    for col in ["children_cases_2023", "pocso_2023", "kidnapping_children_2023"]:
        if col in row.index:
            summary[col] = int(row[col])

    return summary


# ==========================================================
# AI RECOMMENDATIONS
# ==========================================================

def generate_recommendation(risk_level):

    if risk_level == "High":

        return [
            "Avoid isolated roads after dark.",
            "Prefer verified cab services.",
            "Share live location with trusted contacts.",
            "Keep emergency numbers easily accessible.",
            "Travel in groups whenever possible."
        ]

    elif risk_level == "Medium":

        return [
            "Remain aware of surroundings.",
            "Prefer well-lit public routes.",
            "Use trusted transportation.",
            "Inform family while travelling."
        ]

    else:

        return [
            "Continue following basic safety practices.",
            "Stay aware of surroundings.",
            "Use verified transport during late hours."
        ]


# ==========================================================
# DASHBOARD METRICS
# ==========================================================

def dashboard_metrics(df):
    """
    Returns key dashboard KPIs for the given dataframe slice.
    """
    if df.empty:
        metrics = {
            "total_cities": 0,
            "highest_risk_city": "N/A",
            "average_risk": 0.0,
            "average_safety": 0.0,
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "total_women_cases_2023": 0,
            "total_children_cases_2023": 0
        }
        return metrics

    metrics = {
        "total_cities": len(df),
        "highest_risk_city": df.loc[df["risk_score"].idxmax(), "city"],
        "average_risk": round(df["risk_score"].mean(), 2),
        "average_safety": round(df["safety_score"].mean(), 1),
        "high_risk": int((df["risk_level"] == "High").sum()),
        "medium_risk": int((df["risk_level"] == "Medium").sum()),
        "low_risk": int((df["risk_level"] == "Low").sum()),
        "total_women_cases_2023": int(df["cases_2023"].sum()),
    }

    if "children_cases_2023" in df.columns:
        metrics["total_children_cases_2023"] = int(df["children_cases_2023"].sum())

    return metrics


# ==========================================================
# TOP RISKY CITIES
# ==========================================================

def top_risky_cities(df, n=5):

    return (
        df.sort_values(
            "risk_score",
            ascending=False
        )
        .head(n)
    )


# ==========================================================
# SAFEST CITIES
# ==========================================================

def safest_cities(df, n=5):

    return (
        df.sort_values(
            "risk_score"
        )
        .head(n)
    )


# ==========================================================
# RISK DISTRIBUTION
# ==========================================================

def risk_distribution(df):

    return (
        df["risk_level"]
        .value_counts()
        .reset_index()
        .rename(
            columns={
                "index": "Risk Level",
                "risk_level": "Count"
            }
        )
    )


# ==========================================================
# FUTURE PREDICTION (Simple Projection)
# =======================================================

def predict_next_year(city_row):
    """
    Projects next year's cases using a simple linear trend.
    """

    change = (
        city_row["cases_2023"]
        - city_row["cases_2022"]
    )

    prediction = city_row["cases_2023"] + change

    return max(0, int(prediction))
