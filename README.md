# ⚡ EV Infrastructure Analytics & Optimization Platform

A geospatial analytics and machine learning project designed to evaluate and optimize India's EV charging infrastructure.  
The platform analyzes 1,500+ EV charging stations using data cleaning, exploratory analysis, visual insights, and K-Means clustering to identify high-density zones, underserved regions, and opportunities for network expansion.

---

## 📌 Project Overview
- Cleaned, standardized, and validated EV charging station data (coordinates, duplicates, city names).
- Conducted Exploratory Data Analysis (EDA) to uncover geographic and charger-type patterns.
- Applied **K-Means clustering (k = 5)** to segment India's EV infrastructure into meaningful regions.
- Mapped hotspots, sparse coverage zones, and regional demand clusters.
- Exported processed datasets, models, and cluster outputs for deployment and further analysis.

---

## 🔧 Technologies Used
- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-Learn (StandardScaler, KMeans)**

---

## 📊 Key Insights
- **Top States:** Maharashtra, Delhi, Karnataka, Tamil Nadu  
- **Top Cities:** Bangalore, Delhi, New Delhi, Chennai, Mumbai  
- **Most Common Charger Ratings:** 7 kW, 6 kW, 8 kW, 12 kW  
- Clustering reveals strong EV presence in major metros and southern corridors, with clear gaps in the North-East, central India, and hill states.

---

## 🌍 Visual Highlights
- State-wise and city-wise station distribution  
- Charging type distribution  
- Nationwide scatter plot of EV station locations  
- K-Means cluster segmentation of EV regions (k = 5)

---

## 📁 Model & Data Outputs
- `ev_clustered.csv` — cleaned dataset with cluster labels  
- `scaler.pkl` — fitted StandardScaler  
- `kmeans_model.pkl` — trained K-Means model  
- `cluster_centers.csv` — unscaled cluster centroids

---

## 🚀 Workflow Summary
1. Data ingestion & preprocessing  
2. Cleaning missing values and inconsistencies  
3. Standardizing geographic fields  
4. Exploratory Data Analysis  
5. Geospatial visualization  
6. Feature scaling  
7. Elbow Method to determine optimal clusters  
8. K-Means clustering  
9. Insight generation & recommendations  
10. Saving outputs for deployment

---

## 📌 Recommendations
- Expand EV infrastructure in low-density regions (NE India, interior Rajasthan, hill states).  
- Increase deployment of higher kW fast chargers.  
- Strengthen EV corridors between metro clusters  
  (e.g., Mumbai–Pune–Bangalore, Delhi–Jaipur–Ahmedabad, Chennai–Bangalore–Hyderabad).

---

## 📘 Notebook
The full analysis and model pipeline are available in the project’s Colab/Jupyter notebook.

---
---

