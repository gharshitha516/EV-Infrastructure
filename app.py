import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# -------------------------
# Helper loaders & utils
# -------------------------
@st.cache_data
def load_data(data_path="ev_clustered.csv", centers_path="cluster_centers.csv"):
    # Load CSV files
    df = pd.read_csv(data_path)

    # fix latitude column name
    if "lattitude" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"lattitude": "latitude"})
    if "lon" in df.columns and "longitude" not in df.columns:
        df = df.rename(columns={"lon": "longitude"})

    # ensure numeric
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # load cluster centers CSV
    centers = pd.read_csv(centers_path)

    # normalize center columns
    if "lat_center" in centers.columns and "latitude" not in centers.columns:
        centers = centers.rename(columns={"lat_center": "latitude", "lon_center": "longitude"})
    if "lattitude" in centers.columns and "latitude" not in centers.columns:
        centers = centers.rename(columns={"lattitude": "latitude"})

    return df, centers


@st.cache_resource
def load_models(kmeans_path="kmeans_model.pkl", scaler_path="scaler.pkl"):
    kmeans, scaler = None, None
    try:
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
    except:
        kmeans = None

    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except:
        scaler = None

    return kmeans, scaler


def safe_head(df, n=5):
    return df.head(n)


def kpi_cards(df):
    total = len(df)
    states = df["state"].nunique()
    cities = df["city"].nunique()
    top_type = df["type"].mode().iloc[0] if "type" in df.columns else "N/A"
    return total, states, cities, top_type


def recommend_for_cluster(cluster_id, df, centers):
    c_df = df[df["cluster"] == cluster_id]
    n = len(c_df)
    top_cities = c_df["city"].value_counts().head(5).to_dict()
    center = centers.loc[cluster_id] if cluster_id in centers.index else None

    return {
        "count": n,
        "top_cities": top_cities,
        "center": center.to_dict() if center is not None else None,
    }


# -------------------------
# App Layout and Theme
# -------------------------
st.set_page_config(
    page_title="⚡ EV Charging Stations — India",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .big-title { font-size:34px; font-weight:700; color:#0b3d91; }
    .subtle { color:#555; }
    .metric-card {
        border-radius:10px;
        padding:12px;
        background:rgba(255,255,255,0.9);
        box-shadow:0 1px 6px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load Data
# -------------------------
try:
    df, centers = load_data()
except FileNotFoundError:
    st.error("CSV files not found. Upload `ev_clustered.csv` and `cluster_centers.csv`.")
    st.stop()

kmeans, scaler = load_models()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.title("Filters & Tools")

state_list = ["All"] + sorted(df["state"].dropna().unique())
sel_state = st.sidebar.selectbox("State", state_list)

df_filtered = df[df["state"] == sel_state] if sel_state != "All" else df.copy()

city_list = ["All"] + sorted(df_filtered["city"].dropna().unique())
sel_city = st.sidebar.selectbox("City", city_list)

if sel_city != "All":
    df_filtered = df_filtered[df_filtered["city"] == sel_city]

cluster_list = ["All"] + sorted(df["cluster"].unique())
sel_cluster = st.sidebar.selectbox("Cluster", cluster_list)

if sel_cluster != "All":
    df_filtered = df_filtered[df_filtered["cluster"] == int(sel_cluster)]

# -------------------------
# Header & KPIs
# -------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="big-title">EV Charging Stations — India Dashboard ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Interactive geospatial dashboard + clustering + insights.</div>', unsafe_allow_html=True)
with col2:
    st.image("https://static.streamlit.io/examples/cat.jpg", width=90)

st.markdown("___")

total, num_states, num_cities, top_type = kpi_cards(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Stations", f"{total:,}")
c2.metric("States Covered", num_states)
c3.metric("Cities Covered", num_cities)
c4.metric("Most Common Charger", f"{top_type} kW")

# -------------------------
# Tabs
# -------------------------
tabs = st.tabs(["Overview", "Map", "Clusters", "Insights", "Predict"])

# -------- OVERVIEW TAB -------- #
with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(safe_head(df_filtered, 15))

    st.markdown("### Top States & Cities")
    colA, colB = st.columns(2)

    with colA:
        state_counts = df["state"].value_counts().head(15)
        fig = px.bar(
            x=state_counts.index,
            y=state_counts.values,
            title="Top 15 States",
            labels={"x": "State", "y": "Stations"},
            color=state_counts.values,
            color_continuous_scale="Blues",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        city_counts = df["city"].value_counts().head(20)
        fig2 = px.bar(
            x=city_counts.values,
            y=city_counts.index,
            orientation="h",
            title="Top 20 Cities",
            labels={"x": "Stations", "y": "City"},
            color=city_counts.values,
            color_continuous_scale="Oranges",
        )
        st.plotly_chart(fig2, use_container_width=True)

# -------- MAP TAB -------- #
with tabs[1]:
    st.subheader("Interactive Map — EV Charging Stations")

    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="latitude",
        lon="longitude",
        color="cluster",
        hover_name="name",
        zoom=4,
        height=700,
        mapbox_style="open-street-map",
        color_continuous_scale=px.colors.qualitative.T10,
    )
    st.plotly_chart(fig_map, use_container_width=True)

# -------- CLUSTERS TAB -------- #
with tabs[2]:
    st.subheader("Cluster Overview")
    st.table(df.groupby("cluster").size().reset_index(name="count"))

    st.markdown("### Cluster Centers")
    fig_cent = px.scatter_mapbox(
        centers,
        lat="latitude",
        lon="longitude",
        zoom=4,
        height=500,
        mapbox_style="open-street-map",
    )
    fig_cent.update_traces(marker=dict(size=15, color="black", symbol="x"))
    st.plotly_chart(fig_cent, use_container_width=True)

# -------- INSIGHTS TAB -------- #
with tabs[3]:
    st.subheader("Insights & Recommendations")
    st.write("- High concentration: Bangalore, Delhi, New Delhi, Chennai, Mumbai.")
    st.write("- Under-served: North-East, Rajasthan interiors, Himalayan belt.")
    st.write("- Most common chargers: 7 kW, 6 kW, 12 kW.")
    st.write("### Recommendations")
    st.write("1. Add stations in under-served regions.")
    st.write("2. Deploy fast-chargers for EV adoption.")
    st.write("3. Strengthen highway EV corridors.")

# -------- PREDICT TAB -------- #
with tabs[4]:
    st.subheader("Predict Cluster for New Location")

    lat_in = st.number_input("Latitude", value=20.0, step=0.01)
    lon_in = st.number_input("Longitude", value=77.0, step=0.01)

    if kmeans is None or scaler is None:
        st.warning("Prediction model not loaded. Add kmeans_model.pkl & scaler.pkl.")
    else:
        if st.button("Predict Cluster"):
            inp = np.array([[lat_in, lon_in]])
            scaled = scaler.transform(inp)
            pred = int(kmeans.predict(scaled)[0])
            st.success(f"Predicted Cluster: {pred}")

# -------------------------
# Footer Downloads
# -------------------------
st.markdown("---")
st.subheader("Download Cleaned CSV")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download Dataset", csv, "ev_clustered.csv", "text/csv")

st.markdown("Made with ❤️  by Your Name")

