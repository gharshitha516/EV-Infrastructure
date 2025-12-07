import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

# -----------------------------
# Load Data and Models
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ev_clustered.csv")
    centers = pd.read_csv("cluster_centers.csv")
    return df, centers

df, centers = load_data()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="EV Infrastructure",
    page_icon="🗺️",
    layout="wide"
)

st.title("⚡EV Infrastructure Analytics & Optimization Platform")
st.write("An dashboard that analyzes the distribution of EV charging stations across India using data visualization and clustering.")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

state_filter = st.sidebar.multiselect(
    "Select State(s)", 
    sorted(df["state"].unique()),
)

city_filter = st.sidebar.multiselect(
    "Select City(s)", 
    sorted(df["city"].unique()),
)

cluster_filter = st.sidebar.multiselect(
    "Select Cluster (0–4)", 
    sorted(df["cluster"].unique()),
)

filtered_df = df.copy()

if state_filter:
    filtered_df = filtered_df[filtered_df["state"].isin(state_filter)]

if city_filter:
    filtered_df = filtered_df[filtered_df["city"].isin(city_filter)]

if cluster_filter:
    filtered_df = filtered_df[filtered_df["cluster"].isin(cluster_filter)]

st.subheader("📄 Filtered Dataset Preview")
st.dataframe(filtered_df.head(20))

# -----------------------------
# Metrics Section
# -----------------------------
st.subheader("📌 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Stations", len(df))
col2.metric("Unique States", df["state"].nunique())
col3.metric("Unique Cities", df["city"].nunique())
col4.metric("Clusters", df["cluster"].nunique())

# -----------------------------
# Map Visualization
# -----------------------------
st.subheader("🔸EV Station Map")

fig = px.scatter_mapbox(
    filtered_df,
    lat="latitude",
    lon="longitude",
    color="cluster",
    zoom=4,
    mapbox_style="open-street-map",
    hover_name="name",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bar Charts
# -----------------------------
st.subheader("📍 Top States by EV Charging Station Count")

state_counts = df["state"].value_counts().head(10)
fig_state = px.bar(
    state_counts,
    x=state_counts.index,
    y=state_counts.values,
    labels={"x": "State", "y": "Stations"},
    color=state_counts.values,
    color_continuous_scale="Oranges"
)
st.plotly_chart(fig_state, use_container_width=True)

st.subheader("📍 Top Cities by EV Charging Station Count")

city_counts = df["city"].value_counts().head(10)
fig_city = px.bar(
    city_counts,
    x=city_counts.index,
    y=city_counts.values,
    labels={"x": "City", "y": "Stations"},
    color=city_counts.values,
    color_continuous_scale="Teal"
)
st.plotly_chart(fig_city, use_container_width=True)

# -----------------------------
# Charging Type Distribution
# -----------------------------
st.subheader("🔌 EV Charger Power Distribution")

type_counts = df["type"].value_counts().sort_index()
fig_type = px.bar(
    type_counts,
    x=type_counts.index.astype(str),
    y=type_counts.values,
    labels={"x": "Type (kW)", "y": "Count"},
    color=type_counts.values,
    color_continuous_scale="Blues"
)
st.plotly_chart(fig_type, use_container_width=True)

# -----------------------------
# Cluster Centers
# -----------------------------
st.subheader("🎯 Cluster Center Coordinates")

st.dataframe(centers)

# -----------------------------
# Insights Section
# -----------------------------
st.subheader("💡Project Insights")

st.write("""
### Observations:
1.Bangalore, Delhi, New Delhi, Chennai, and Mumbai have the highest EV charging density.
2.North-East, Rajasthan interior, and hilly states show significant EV infrastructure gaps.
3.7 kW chargers dominate the market, followed by 6 kW and 12 kW.
4.K-Means clustering (k=5) reveals clear regional EV adoption zones.

### Recommendations:
1.Increase EV stations in under-served regions.
2.Add more high kW fast chargers.
3.Strengthen highway EV corridors between major metro cities.
""")
