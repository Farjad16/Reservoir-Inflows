import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
import os
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.let_it_rain import rain
from scipy.interpolate import griddata

# Load the trained model
import os
model = joblib.load(os.path.join("models", "XGBoost.pkcls"))

# Set page config and title
st.set_page_config(page_title="Reservoir Inflow Predictor", layout="wide")
st.markdown("""
    <style>
    html, body, [class*='css']  { font-size: 18px !important; }
    section.main > div { padding: 2rem; }
    .collapsible { background-color: #f0f2f6; border-radius: 10px; padding: 1rem; margin-bottom: 1.5rem; }
    .stApp {
        background-image: url('https://cdn-blog.zameen.com/blog/wp-content/uploads/2020/10/Cover-1440x625-3.jpg');
        background-size: cover;
        background-position: center;
    }
    .expander > div[role='button'] p {
        font-size: 20px;
        font-weight: bold;
        color: #003366;
    }
    @keyframes bounce {
      0%, 20%, 50%, 80%, 100% {transform: translateY(0);} 
      40% {transform: translateY(-15px);} 
      60% {transform: translateY(-10px);} 
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='font-size: 45px; color: #0E76A8; text-align: center; animation: bounce 2s infinite;'>💧 Reservoir Inflow Prediction App</h1>
<p style='text-align: center;'>Forecast water inflows using advanced Machine Learning Model</p>
""", unsafe_allow_html=True)

st.markdown("---")

stations = [
    "DEOSAI", "HUSHEY", "RAMA", "RATTU", "USHKORE", "SHIGER", "YASIN",
    "GILGIT", "SHENDURE", "NALTAR", "BESHAM", "PHULRA", "DAGGAR", "SKARDU"
]
parameters = ["Precipitation", "Tmax", "Tmin", "RH", "Solar Rad"]

station_coords = {
    "DEOSAI": (35.033, 75.417), "HUSHEY": (35.450, 76.350), "RAMA": (35.282, 74.800), "RATTU": (35.180, 74.983),
    "USHKORE": (36.033, 73.400), "SHIGER": (35.422, 75.633), "YASIN": (36.167, 73.200), "GILGIT": (35.920, 74.300),
    "SHENDURE": (35.633, 74.950), "NALTAR": (36.033, 74.200), "BESHAM": (35.926, 72.864), "PHULRA": (35.283, 73.033),
    "DAGGAR": (34.511, 72.491), "SKARDU": (35.300, 75.600)
}

main_tab, info_tab = st.tabs(["🔍 Prediction Tool", "📘 Model & Insights"])

with main_tab:
    tab1, tab2 = st.tabs(["📄 Upload Excel File", "✍️ Manual Entry"])

    with tab1:
        st.header("Upload Excel File for Batch Prediction")
        uploaded_file = st.file_uploader("Upload your Excel file with climate data:", type=[".xlsx", ".xls"])
        param_for_map = st.selectbox("Select Parameter to Display on Map", options=parameters, index=0)

        if uploaded_file is not None:
            try:
                with st.spinner("🔍 Processing your file and predicting inflow..."):
                    time.sleep(1.5)
                    df = pd.read_excel(uploaded_file)
                    feature_columns = df.columns.drop(['Date', 'Reservoir Inflows (m3/s)'], errors='ignore')
                    X = df[feature_columns]
                    prediction = model.predict(X)
                    df['Predicted Reservoir Inflow (m3/s)'] = prediction

                st.success("✅ Prediction successful!")
                st.dataframe(df[['Date'] + list(feature_columns) + ['Predicted Reservoir Inflow (m3/s)']], use_container_width=True)

                if 'Date' in df.columns:
                    fig = px.line(df, x='Date', y='Predicted Reservoir Inflow (m3/s)', title='📈 Predicted Inflow Over Time')
                    fig.update_traces(line=dict(color='#0077b6', width=3))
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("🗺️ Spatial Distribution Animation")
                frame_slider = st.slider("Select frame (row) to animate", min_value=0, max_value=len(df)-1, value=0)

                map_values = []
                for station in stations:
                    col_name = f"{station} {param_for_map}"
                    if col_name in df.columns:
                        lat, lon = station_coords[station]
                        map_values.append({"Station": station, "Latitude": lat, "Longitude": lon, "Value": df[col_name].iloc[frame_slider]})

                if map_values:
                    map_df = pd.DataFrame(map_values)

                    grid_lat = np.linspace(34.0, 36.5, 100)
                    grid_lon = np.linspace(72.0, 76.5, 100)
                    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
                    points = map_df[['Longitude', 'Latitude']].values
                    values = map_df['Value'].values
                    grid_values = griddata(points, values, (grid_lon, grid_lat), method='cubic')

                    fig = go.Figure(data=
                        go.Contour(
                            z=grid_values,
                            x=np.linspace(72.0, 76.5, 100),
                            y=np.linspace(34.0, 36.5, 100),
                            colorscale='Viridis',
                            colorbar=dict(title=param_for_map),
                            contours=dict(showlabels=True)
                        )
                    )
                    fig.update_layout(
                        title=f"{param_for_map} Spatial Distribution",
                        xaxis_title="Longitude",
                        yaxis_title="Latitude",
                        width=800,
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📅 Download Predictions", csv, "predicted_inflows.csv", "text/csv")

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    with tab2:
        st.header("Enter Meteorological Data Manually")
        st.markdown("Use the collapsible sections below to enter data for a single day. Hover over info icons for guidance.")

        input_data = {}
        for param in parameters:
            with st.expander(f"📌 {param} (All Stations)", expanded=False):
                cols = st.columns(4)
                for i, station in enumerate(stations):
                    col = cols[i % 4]
                    key = f"{station} {param}"
                    tooltip = f"Enter {param} value for {station}"
                    input_data[key] = col.number_input(f"{station}", key=key, help=tooltip)

        if st.button("🔮 Predict Reservoir Inflow"):
            try:
                with st.spinner("🎀 Running AI model for prediction..."):
                    X_input = pd.DataFrame([input_data])
                    prediction = model.predict(X_input)[0]
                    rain(emoji="💧", font_size=28, falling_speed=5, animation_length=1.5)
                st.success(f"🌊 Predicted Reservoir Inflow: {prediction:.2f} m³/s")

                style_metric_cards()
                st.metric("Predicted Inflow", f"{prediction:.2f} m³/s", delta=None)

                fig = px.bar(x=["Predicted Inflow"], y=[prediction], text=[f"{prediction:.2f} m³/s"],
                             labels={'x': '', 'y': 'm³/s'}, height=400, title="📊 Inflow Level")
                fig.update_traces(marker_color='#00b4d8', textposition='outside')
                fig.update_layout(yaxis_range=[0, max(10000, prediction * 1.5)])
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")

with info_tab:
    st.header("📘 Model Information & Insights")
    st.markdown("""
    This tab provides helpful insights and visual summaries for better interpretation:

    - **Input Parameters**: 70 input features from 14 stations (5 parameters each)
    - **Model Used**: XGBoost ( Gradient Boosting)
    - **Prediction Target**: Daily reservoir inflow at Tarbela Dam (m³/s)

    ### 📊 Summary Statistics
    - Max/Min/Avg inflow, latest prediction trends

    ### 🤔 Model Insights
    - Feature importance ranking *(coming soon)*
    - Inflow patterns and date-based trends

    ### 🧹 How to Use
    1. Upload a valid Excel sheet or use manual input.
    2. Make sure all values are realistic and no cell is missing.
    3. Hit "Predict" and wait for the results.

    💡 **Need Help?** Hover over input fields or upload examples.
    """)

st.markdown("---")
st.markdown("Made by Group No 10 Batch 2021-2025 | Model: XGBoost")
