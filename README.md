# 💧 Reservoir Inflow Prediction App

This Streamlit app predicts daily reservoir inflow at Tarbela Dam using meteorological inputs from 14 different stations.

---

## 📦 Features

- Upload Excel files for batch prediction
- Enter data manually for individual-day prediction
- Interactive map to visualize meteorological parameters
- AI-based prediction using trained XGBoost model
- Download predicted results as CSV

---

## 🗂 Input Data Format

The input Excel file must include:
- `Date` column (optional but recommended for plots)
- 70 meteorological features: 5 parameters (Precipitation, Tmax, Tmin, RH, Solar Radiation) × 14 stations
- (Optional) `Reservoir Inflows (m3/s)` for comparing actual inflow vs predicted inflow

### 📍 Example Column Names:
```
DEOSAI Precipitation
DEOSAI Tmax
DEOSAI Tmin
DEOSAI RH
DEOSAI Solar Rad
HUSHEY Precipitation
...
SKARDU Solar Rad
```

---

## 📂 Project Structure

```
reservoir-inflow-app/
├── Reservoir Inflow App.py      # Main Streamlit app file
├── XGBoost.pkcls                # Pre-trained XGBoost model
├── requirements.txt             # Python dependencies
└── README.md                    # App description and instructions
```

---

## 🚀 Deployment on Streamlit Cloud

1. Upload the project to a **GitHub** repository.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and log in.
3. Click on **"New App"**, select your repo and `Reservoir Inflow App.py`.
4. Click **Deploy** and your app will be live!

---

## 👥 Developed By

**Group 10 | Batch 2021–2025**

**Model Used:** XGBoost

---

## 📬 Contact

For feedback or support, reach out to the developers via your university contact channels.

---

Enjoy forecasting! 🌊
