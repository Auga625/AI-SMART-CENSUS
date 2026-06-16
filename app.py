import streamlit as st
import pandas as pd
import numpy as np
import folium

from folium.plugins import HeatMap
from streamlit_folium import st_folium

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI SMART CENSUS",
    page_icon="census_profile.jpg",
    layout="wide"
)

# =========================
# CUSTOM CSS (UNCHANGED — ORIGINAL)
# =========================
st.markdown("""
<style>

.stApp {
    background-color: #F4F7FB;
}

header[data-testid="stHeader"] {
    background-color: #0B1F3A;
}

h1, h2, h3 {
    color: #0B1F3A;
    font-weight: 600;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.highlight {
    background: #F2C94C;
    padding: 10px;
    border-radius: 8px;
    color: #0B1F3A;
    font-weight: 600;
}

.stButton>button {
    background-color: #1F4E79;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton>button:hover {
    background-color: #163A5F;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px;
    border-radius: 8px;
    transition: 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #A9A9A9;
    color: white;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER (UNCHANGED STYLE)
# =========================
st.markdown("""
<div style='background-color:#0B1F3A;
padding:20px;
border-radius:10px'>
    <h1 style='color:white;margin:0;'>
    AI SMART CENSUS PROJECT
    </h1>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.image("census_profile.jpg", width=250)

menu = st.sidebar.radio(
    "Select Module",
    [
        "POPULATION FORECAST",
        "MIGRATION TRACKING",
        "YOUTH RISK ANALYSIS",
        "POPULATION HEATMAP",
        "RAW DATA"
    ]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Census CSV",
    type=["csv"]
)

# =========================
# LOAD DATA
# =========================
if uploaded_file is None:
    st.info("⬅️ Upload a CSV file from the sidebar to begin")
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.lower()

# Encode district
df["district_code"] = df["district"].astype("category").cat.codes

# =========================
# POPULATION FORECAST
# =========================
if menu == "POPULATION FORECAST":

    st.markdown("""
    <div class='card'>
    <h3>AI HYBRID POPULATION FORECAST</h3>
    </div>
    """, unsafe_allow_html=True)

    df_model = df.sort_values(["district", "year"]).copy()

    df_model["prev_population"] = df_model.groupby("district")["population"].shift(1)
    df_model["arith_growth"] = df_model["population"] - df_model["prev_population"]
    df_model["pct_growth"] = df_model["arith_growth"] / (df_model["prev_population"] + 1)
    df_model["exp_growth"] = np.log1p(df_model["population"])
    df_model["incremental_increase"] = df_model.groupby("district")["arith_growth"].diff()

    df_model["year_squared"] = df_model["year"] ** 2
    df_model["year_cubed"] = df_model["year"] ** 3

    df_model = df_model.fillna(0)

    features = [
        "year",
        "district_code",
        "prev_population",
        "arith_growth",
        "pct_growth",
        "exp_growth",
        "incremental_increase",
        "year_squared",
        "year_cubed"
    ]

    X = df_model[features]
    y = df_model["population"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.03)
    }

    results = []
    best_model = None
    best_r2 = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "R2": round(r2, 4)
        })

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results))

    future_year = st.slider("Select Forecast Year", 2021, 2050, 2030)

    latest = df_model.groupby("district").tail(1).copy()

    latest["year"] = future_year
    latest["year_squared"] = future_year ** 2
    latest["year_cubed"] = future_year ** 3
    latest["exp_growth"] = np.log1p(latest["population"])

    latest = latest.fillna(0)

    latest["predicted_population"] = best_model.predict(latest[features]).astype(int)

    st.markdown(f"""
    <div class='highlight'>
    Best Model R² Score: {round(best_r2,4)}<br>
    Forecast Year: {future_year}
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(latest[["district", "predicted_population"]])
    st.bar_chart(latest.set_index("district")["predicted_population"])

# =========================
# MIGRATION TRACKING
# =========================
elif menu == "MIGRATION TRACKING":

    st.markdown("""
    <div class='card'>
    <h3>MIGRATION TRACKING</h3>
    </div>
    """, unsafe_allow_html=True)

    df["pop_change"] = df.groupby("district")["population"].diff()

    df["migration"] = np.where(
        df["pop_change"] > 50000, "Inflow",
        np.where(df["pop_change"] < -50000, "Outflow", "Stable")
    )

    st.dataframe(df[["district", "year", "pop_change", "migration"]])

# =========================
# YOUTH RISK ANALYSIS
# =========================
elif menu == "YOUTH RISK ANALYSIS":

    st.markdown("""
    <div class='card'>
    <h3>YOUTH RISK ANALYSIS</h3>
    </div>
    """, unsafe_allow_html=True)

    df["unemployment_rate"] = df["unemployed"] / (df["employed"] + df["unemployed"] + 1)

    features = ["youth_pop", "population", "district_code"]

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(df[features], df["unemployment_rate"])

    df["risk_score"] = rf.predict(df[features])

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[0, 0.2, 0.4, 1],
        labels=["Low", "Medium", "High"]
    )

    st.dataframe(df[["district", "year", "risk_level"]])

# =========================
# POPULATION HEATMAP
# =========================
elif menu == "POPULATION HEATMAP":

    st.markdown("""
    <div class='card'>
    <h3>POPULATION HEATMAP</h3>
    </div>
    """, unsafe_allow_html=True)

    latest = df[df["year"] == df["year"].max()]

    m = folium.Map(location=[1.37, 32.29], zoom_start=6, tiles="CartoDB positron")

    heat_data = list(zip(latest["latitude"], latest["longitude"], latest["population"]))

    HeatMap(heat_data).add_to(m)

    st_folium(m, width=900, height=500)

# =========================
# RAW DATA
# =========================
elif menu == "RAW DATA":

    st.markdown("""
    <div class='card'>
    <h3>RAW DATA</h3>
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df)