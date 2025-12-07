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
def load_data(data_path="ev_clustered.xlsx", centers_path="cluster_centers.xlsx"):
    # robust column handling: try both 'latitude' and 'lattitude'
    df = pd.read_excel(data_path)
    if "lattitude" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"lattitude": "latitude"})
    if "lon" in df.columns and "longitude" not in df.columns:
        df = df.rename(columns={"lon":"longitude"})
    # ensure dtype
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    centers = pd.read_excel(centers_path)
    # normalize center column names
    if "lat_center" in centers.columns and "latitude" not in centers.columns:
        centers = centers.rename(columns={"lat_center":"latitude", "lon_center":"longitude"})
    if "latitude" not in centers.columns and "lattitude" in centers.columns:
        centers = centers.rename(columns={"lattitude":"latitude"})
    return df, centers

@st.cache_resource
def load_models(kmeans_path="kmeans_model.pkl", scaler_path="scaler.pkl"):
    kmeans, scaler = None, None
    try:
        with open(kmeans_path, "rb") as f:
            kmeans = pickle.load(f)
    except Exception:
        kmeans = None
    try:
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
    return kmeans, scaler

def safe_head(df, n=5):
    return df.head(n)

def kpi_cards(df):
    total = len(df)
    states = df['state'].nunique()
    cities = df['city'].nunique()
    top_type = df['type'].mode().iloc[0] if 'type' in df.columns and not df['type'].isna().all() else "N/A"
    return total, states, cities, top_type

def recommend_for_cluster(cluster_id, df, centers):
    # compute cluster bounding box and example recommendation
    c_df = df[df['cluster'] == cluster_id]
    n = len(c_df)
    top_cities = c_df['city'].value_counts().head(5).to_dict()
    center = centers.loc[cluster_id] if cluster_id in centers.index else None
    return {"count": n, "top_cities": top_cities, "center": center.to_dict() if center is not None else None}

# -------------------------
# App Layout and Theme
# -------------------------
st.set_page_config(
    page_title="⚡ EV Charging Stations — India",
    layout="wide",
    initial_sidebar_state="expanded",
)

# small CSS for nicer header
st.markdown(
    """
    <style>
    .big-title {
        font-size:34px;
        font-weight:700;
        color: #0b3d91;
    }
    .subtle {
        color: #555;
    }
    .metric-card {
        border-radius:10px;
        padding:12px;
        background: linear-gradient(90deg, rgba(255,255,255,0.9), rgba(250,250,252,0.9));
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Load
# -------------------------
try:
    df, centers = load_data()
except FileNotFoundError:
    st.error("Data files not found. Please upload `ev_clustered.xlsx` and `cluster_centers.xlsx` to the app folder.")
    st.stop()

kmeans, scaler = load_models()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Filters & Tools")
st.sidebar.markdown("Use filters to focus on a state or city. Use Predict to check cluster for coordinates.")

state_list = ["All"] + sorted(df['state'].dropna().unique().tolist())
sel_state = st.sidebar.selectbox("State", state_list)

if sel_state != "All":
    df_filtered = df[df['state'] == sel_state].copy()
else:
    df_filtered = df.copy()

city_list = ["All"] + sorted(df_filtered['city'].dropna().unique().tolist())
sel_city = st.sidebar.selectbox("City", city_list)

if sel_city != "All":
    df_filtered = df_filtered[df_filtered['city'] == sel_city].copy()

# cluster selector for focused view
cluster_options = ["All"] + sorted(df['cluster'].unique().tolist(), key=int)
sel_cluster = st.sidebar.selectbox("Show cluster", cluster_options)

if sel_cluster != "All":
    df_filtered = df_filtered[df_filtered['cluster'] == int(sel_cluster)].copy()

st.sidebar.markdown("---")
if kmeans is None or scaler is None:
    st.sidebar.warning("KMeans model or scaler not found. Prediction tool will be disabled.")
else:
    st.sidebar.success("KMeans model loaded.")

# -------------------------
# Main: Header & KPIs
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="big-title">EV Charging Stations — India Dashboard ⚡</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Interactive project showing distribution, clusters, recommendations, and a prediction tool.</div>', unsafe_allow_html=True)
with col2:
    st.image("https://static.streamlit.io/examples/cat.jpg", width=90)  # replace with your logo if you want

st.markdown("___")

total, num_states, num_cities, top_type = kpi_cards(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Stations", f"{total:,}")
k2.metric("States Covered", f"{num_states}")
k3.metric("Unique Cities", f"{num_cities}")
k4.metric("Most Common Charger", f"{top_type} kW")

st.markdown("---")

# -------------------------
# Tabs: Overview, Map, Clusters, Insights, Predict
# -------------------------
tabs = st.tabs(["Overview", "Map", "Clusters", "Insights", "Predict"])

# -------------------------
# Tab 1: Overview
# -------------------------
with tabs[0]:
    st.subheader("Overview & Dataset")
    st.write("Quick look at cleaned data and summary charts.")
    st.dataframe(safe_head(df_filtered, 15))

    st.markdown("### Top states and cities")
    c1, c2 = st.columns(2)
    with c1:
        state_counts = df['state'].value_counts().head(15)
        fig = px.bar(x=state_counts.index, y=state_counts.values, labels={'x':'State','y':'Stations'},
                     title="Top 15 States by Number of Stations", color=state_counts.values, color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, xaxis_tickangle= -45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        city_counts = df['city'].value_counts().head(20)
        fig2 = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
                      labels={'x':'Stations','y':'City'}, title="Top 20 Cities (Cleaned)", color=city_counts.values, color_continuous_scale="Oranges")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Tab 2: Map
# -------------------------
with tabs[1]:
    st.subheader("Interactive Map — All Stations")
    st.write("Zoom and hover to inspect stations. Use filters (sidebar) to focus.")
    map_df = df_filtered.copy()
    # ensure lat/lon columns
    if 'latitude' not in map_df.columns and 'lattitude' in map_df.columns:
        map_df = map_df.rename(columns={'lattitude': 'latitude'})
    # Plotly Mapbox scatter (open-street-map style)
    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="cluster",
        hover_name="name",
        hover_data={"state":True, "city":True, "type":True},
        zoom=4,
        height=700,
        color_continuous_scale=px.colors.qualitative.T10
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------
# Tab 3: Clusters
# -------------------------
with tabs[2]:
    st.subheader("Clusters & Region Insights")
    st.write("View cluster centers and quick recommendations per cluster.")
    # cluster summary table
    cluster_counts = df.groupby('cluster').size().reset_index(name='count').sort_values('cluster')
    st.table(cluster_counts)

    # plot cluster centers on map
    if not centers.empty:
        centers_cols = centers.columns.tolist()
        lat_col = [c for c in centers_cols if 'lat' in c.lower()][0]
        lon_col = [c for c in centers_cols if 'lon' in c.lower() or 'long' in c.lower()][0]
        centers = centers.rename(columns={lat_col:'latitude', lon_col:'longitude'})
        fig_cent = px.scatter_mapbox(centers, lat='latitude', lon='longitude', zoom=4, height=500)
        fig_cent.update_traces(marker=dict(size=15, color='black', symbol='x'))
        fig_cent.update_layout(mapbox_style='open-street-map', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_cent, use_container_width=True)

    st.markdown("### Cluster Details")
    cluster_sel = st.selectbox("Choose cluster", sorted(df['cluster'].unique().tolist()))
    info = recommend_for_cluster(cluster_sel, df, centers)
    st.write(f"Stations in cluster: {info['count']}")
    st.write("Top cities in this cluster:")
    st.write(info['top_cities'])
    if info['center']:
        st.write("Cluster center coordinates:", info['center'])

# -------------------------
# Tab 4: Insights
# -------------------------
with tabs[3]:
    st.subheader("Insights & Recommendations")
    st.write("- **High concentration**: Bangalore, Delhi, New Delhi, Chennai, Mumbai.")
    st.write("- **Under-served areas**: North-East, interior Rajasthan, Himalayan states.")
    st.write("- **Most common chargers**: 7 kW, 6 kW, 12 kW.")
    st.write("**Recommendations:**")
    st.write("1) Add stations in under-served regions.")
    st.write("2) Increase fast-charging options (higher kW).")
    st.write("3) Deploy chargers along highways connecting major clusters.")

# -------------------------
# Tab 5: Predict
# -------------------------
with tabs[4]:
    st.subheader("Predict cluster for a new location")
    st.write("Enter coordinates to predict cluster using trained KMeans model (if available).")
    colA, colB = st.columns(2)
    lat_in = float(colA.number_input("Latitude", value=20.0, step=0.01, format="%.5f"))
    lon_in = float(colB.number_input("Longitude", value=77.0, step=0.01, format="%.5f"))

    if kmeans is None or scaler is None:
        st.warning("Model or scaler missing. Upload kmeans_model.pkl and scaler.pkl in the app folder to enable predictions.")
    else:
        if st.button("Predict cluster"):
            coords = np.array([[lat_in, lon_in]])
            coords_scaled = scaler.transform(coords)
            predicted = int(kmeans.predict(coords_scaled)[0])
            st.success(f"Predicted Cluster: {predicted}")
            # show distance to cluster center
            try:
                center = centers.loc[predicted]
                st.write("Cluster center (approx):", center.to_dict())
            except Exception:
                pass

# -------------------------
# Footer (download)
# -------------------------
st.markdown("---")
st.subheader("Download Data & Resources")
cold, colu = st.columns(2)
with cold:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download full cleaned CSV", data=csv, file_name="ev_clustered.csv", mime="text/csv")
with colu:
    st.write("Model files (kmeans_model.pkl, scaler.pkl) should be placed in the app folder for prediction features.")

st.markdown("Made with ❤️  •  EV Charging Stations — India")
