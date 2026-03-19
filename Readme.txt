☕ Afficionado Coffee Roasters: Demand Forecasting
1. Overview

This project builds a data-driven forecasting system to predict sales and identify peak demand periods across coffee store locations. It helps shift operations from intuition-based decisions to proactive, data-driven planning.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2. Objectives

Forecast daily sales revenue

Identify peak hours (rush periods)

Enable store-specific inventory & staffing decisions
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
3. Approach

Time-series data preparation from transaction logs

Feature engineering (lags, day-of-week, hourly patterns)

Model: Gradient Boosting Regressor

Time-based train-test validation
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
4. Results

MAPE: 9.41% (~90% accuracy)

Successfully captures morning demand spikes

Provides reliable short-term forecasts
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
5. Key Insights

Morning (8 AM–12 PM) drives majority of revenue

Demand varies significantly by store location

Weekend sales increase for premium products
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
6. Deployment

Built an interactive Streamlit dashboard for:

Store-wise forecasts

Demand visualization

Decision support
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
7. Tech Stack

Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Author

Mrunalini Misal
This project was completed as part of the Business Analyst Internship at Unified Mentor.
