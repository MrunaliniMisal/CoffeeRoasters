import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import datetime

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Afficionado Coffee Dashboard ", layout="wide", page_icon="☕")

# Theme Colors
ESPRESSO = "#4B230E" 
ROAST = "#D96A13"    

# --- 2. DATA LOADING ---
DATA_PATH = "index.csv" 
MODEL_PATH = "forecast_model.pkl"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        # Calculate revenue if not present
        df["revenue"] = df["transaction_qty"] * df["unit_price"]
        # Standardize dates
        df['time_seconds'] = pd.to_timedelta(df['transaction_time']).dt.total_seconds()
        df['day_offset'] = (df['time_seconds'].diff() < 0).cumsum()
        df['date'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df['day_offset'], unit='D')
        df['hour'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S').dt.hour
        return df
    return None

df = load_data()
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")
if df is not None:
    store_list = sorted(df["store_location"].unique().tolist())
    store = st.sidebar.selectbox("Select Store", store_list)
    forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 1, 30, 10)
    metric_choice = st.sidebar.radio("Select Metric", ["Revenue", "Quantity"])
    y_col = "revenue" if metric_choice == "Revenue" else "transaction_qty"
    
    # Crucial: Filter the data for the selected store
    filtered_df = df[df["store_location"] == store].copy()
else:
    st.sidebar.error("Data file not found!")
    st.stop()

# --- 4. MAIN DASHBOARD ---
st.title("☕ Afficionado Coffee Roasters : Data-Driven Forecasting & Peak Demand Prediction")

# KPI Metrics
rev, trans, qty = filtered_df["revenue"].sum(), filtered_df["transaction_id"].nunique(), filtered_df["transaction_qty"].sum()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Revenue", f"${rev:,.0f}")
m2.metric("Transactions", f"{trans:,}")
m3.metric("Quantity", f"{qty:,}")
m4.metric("Avg Order", f"${(rev/trans):.2f}" if trans > 0 else 0)

st.divider()

# --- SECTION 2 & 3: STORE & PRODUCT ANALYTICS ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"Total {metric_choice} by Store")
    store_data = df.groupby("store_location")[y_col].sum().reset_index()
    # Highlight the selected store in the bar chart
    fig1 = px.bar(store_data, x="store_location", y=y_col, template="plotly_dark", text=y_col,
                  color="store_location", color_discrete_map={store: ROAST}, color_discrete_sequence=[ESPRESSO]) 
    fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig1.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader(f"Hourly Demand Pattern: {store}")
    hourly_data = filtered_df.groupby("hour")[y_col].sum().reset_index()
    fig2 = px.area(hourly_data, x="hour", y=y_col, color_discrete_sequence=[ROAST], template="plotly_dark")
    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

with col_right:
    st.subheader(f"Revenue by Product Category: {store}")
    cat_data = filtered_df.groupby("product_category")[y_col].sum().sort_values(ascending=True).reset_index()
    fig3 = px.bar(cat_data, y="product_category", x=y_col, orientation='h', text=y_col,
                  color_discrete_sequence=[ROAST], template="plotly_dark")
    fig3.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Store × Hour Heatmap (Global)")
    heatmap_data = df.pivot_table(values=y_col, index="store_location", columns="hour", aggfunc="sum")
    fig4 = px.imshow(heatmap_data, color_continuous_scale="YlOrBr", template="plotly_dark")
    fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)

    # --- SECTION 1: HISTORICAL DAILY TREND ---
st.header(f" Daily {metric_choice} Trend ({store})")
daily_trend = filtered_df.groupby('date')[y_col].sum().reset_index()
fig_trend = px.line(daily_trend, x='date', y=y_col, template="plotly_dark", color_discrete_sequence=[ROAST])
fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Date", yaxis_title=metric_choice)
st.plotly_chart(fig_trend, use_container_width=True)


# --- SECTION 4: STORE-AWARE FORECASTING ---
st.divider()
st.header(f" {forecast_days}-Day Forecasting for {store}")

if model:
    # 1. Setup future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + datetime.timedelta(days=1), periods=forecast_days)
    
    # 2. Get Historical Daily Totals for Scaling
    historical_daily = filtered_df.groupby('date')[y_col].sum().tail(14).reset_index()
    last_actual_value = historical_daily[y_col].iloc[-1]
    store_avg = historical_daily[y_col].mean()

    # 3. Generate Predictions with Store Multiplier
    daily_predictions = []
    for d in future_dates:
        # Get base model prediction
        base_pred = model.predict([[10, d.weekday(), 1 if d.weekday() >= 5 else 0]])[0]
        
        # Scaling logic: ensures Hell's Kitchen > Astoria > Manhattan based on actuals
        store_multiplier = store_avg / 2000 
        day_map = {0: 0.95, 1: 0.92, 2: 0.94, 3: 0.98, 4: 1.05, 5: 1.15, 6: 1.10}
        
        scaled_val = base_pred * store_multiplier * day_map.get(d.weekday(), 1.0)
        daily_predictions.append(scaled_val)
    
    # 4. Smoothing Logic (Gap Closing)
    diff = last_actual_value - daily_predictions[0]
    smoothed_preds = [p + (diff * (0.7**i)) for i, p in enumerate(daily_predictions)]
    forecast_df = pd.DataFrame({'date': future_dates, y_col: smoothed_preds})

    # 5. Visual Connector (Connect actual line to forecast line)
    connector = pd.DataFrame({'date': [historical_daily['date'].iloc[-1]], y_col: [last_actual_value]})
    forecast_df_plot = pd.concat([connector, forecast_df]).reset_index(drop=True)

    # 6. Build Forecast Chart
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=historical_daily['date'], y=historical_daily[y_col], 
                                name="Actual Daily Sales", line=dict(color=ESPRESSO, width=3), mode='lines+markers'))
    fig_fc.add_trace(go.Scatter(x=forecast_df_plot['date'], y=forecast_df_plot[y_col], 
                                name="Predicted Demand", line=dict(color=ROAST, width=3, dash='dash'), mode='lines+markers'))

    # Auto-adjust Y-axis range for "Zoomed" effect
    all_vals = pd.concat([historical_daily[y_col], forecast_df[y_col]])
    y_min, y_max = all_vals.min() * 0.95, all_vals.max() * 1.05

    fig_fc.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=metric_choice,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(range=[y_min, y_max], gridcolor="#333", tickformat=",.0f")
    )

    st.plotly_chart(fig_fc, use_container_width=True)
else:
    st.warning("Forecast model not found. Please check your model path.")

    # --- SECTION 6: FOOTER ---
st.markdown("""
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1E1E1E;
        color: #7F8C8D;
        text-align: center;
        padding: 20px;
        font-family: sans-serif;
        border-top: 1px solid #333;
        margin-top: 50px;
    }
    .footer-highlight { color: #D96A13; font-weight: bold; }
    </style>
    <div class="footer">
        <p>Afficionado Coffee Roasters | <span class="footer-highlight">Data-Driven Operational Intelligence</span></p>
        <p style="font-size: 0.8em;">© Mrunalini Misal | www.linkedin.com/in/mrunalini-misal-263735192 | Mentor : linkedin.com/in/saiprasad-kagne</p>
    </div>
    """, unsafe_allow_html=True)
