from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# --- PAGE CONFIG ---
st.set_page_config(page_title="Telco Churn ROI Dashboard", layout="wide")

# --- PATHS ---
# Adjust these if the structure is different
MODEL_PATH = Path("models/telco_churn_xgboost_v1.pkl")
DATA_PATH = Path("data/processed/test_df.parquet")


# --- LOAD RESOURCES ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)


# --- SIDEBAR: BUSINESS PARAMETERS ---
st.sidebar.header("💰 Business Strategy Settings")

LTV = st.sidebar.number_input("Customer Lifetime Value (LKR)", value=9000, step=100)
CAMPAIGN_COST = st.sidebar.number_input("Cost of Contact (SMS)", value=3, step=1)
OFFER_COST = st.sidebar.number_input("Cost of Incentive (Discount)", value=225, step=5)
ACCEPTANCE_RATE = st.sidebar.slider("Offer Acceptance Rate", 0.1, 1.0, 0.3, 0.05)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Model Logic:**
    The model predicts the probability of churn for each customer.
    We target customers with high risk, but targeting costs money.
    Adjust the inputs to see how the **Net Profit** changes.
    """
)

# --- MAIN APP ---
st.title("📡 Telco Churn Prediction & ROI Optimizer")
st.markdown(
    "Use this dashboard to determine the **Optimal Decision Threshold** for the retention campaign."
)

try:
    model = load_model()
    df = load_data()

    # Define features (Same as training)
    features = [
        "trend_data_w4_vs_w1",
        "data_gini_coefficient",
        "trend_spend_w4_vs_w1",
        "trend_data_w3_vs_w1",
        "data_volatility_shift",
        "peak_spend_week",
        "spend_volatility_shift",
        "lowest_data_week",
        "trend_spend_w2_vs_w1",
        "peak_data_week",
        "trend_spend_w3_vs_w1",
        "trend_data_w2_vs_w1",
        "ratio_min_daily_data_to_avg",
        "pct_video_w4",
        "spend_consistency_score",
        "pct_messaging_w4",
        "pct_messaging_w3",
        "pct_video_w3",
        "pct_messaging_w2",
        "ratio_min_daily_spend_to_avg",
        "pct_messaging_w1",
        "pct_video_w2",
        "pct_video_w1",
    ]

    X = df[features]
    y_true = df["churn"]

    # Run Inference
    if "y_prob" not in st.session_state:
        st.session_state["y_prob"] = model.predict_proba(X)[:, 1]

    y_prob = st.session_state["y_prob"]

    # --- FINANCIAL CALCULATION ---
    # Create a DataFrame for calculation
    calc_df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    calc_df = calc_df.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # Vectorized Profit Calculation
    # Revenue Saved = (Is Churner?) * (Accepts Offer?) * LTV
    calc_df["revenue"] = calc_df["y_true"] * ACCEPTANCE_RATE * LTV
    # Cost = Contact Cost + (Accepts Offer? * Offer Cost)
    calc_df["cost"] = CAMPAIGN_COST + (ACCEPTANCE_RATE * OFFER_COST)

    # Net Profit per customer targeted
    calc_df["net_profit"] = calc_df["revenue"] - calc_df["cost"]

    # Cumulative Profit (The Profit Curve)
    calc_df["cum_profit"] = calc_df["net_profit"].cumsum()

    # Find Optimal Point
    max_profit_idx = calc_df["cum_profit"].idxmax()
    max_profit = calc_df.loc[max_profit_idx, "cum_profit"]
    optimal_threshold = calc_df.loc[max_profit_idx, "y_prob"]
    optimal_customers = max_profit_idx + 1

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Max Potential Profit", f"LKR {max_profit:,.0f}")
    col2.metric("Optimal Threshold", f"{optimal_threshold:.4f}")
    col3.metric("Targeted Customers", f"{optimal_customers:,}")

    # --- PLOTS ---
    st.subheader("Profit Curve Analysis")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(calc_df.index, calc_df["cum_profit"], color="green", label="Cumulative Profit")

    # Mark the peak
    ax.scatter(optimal_customers, max_profit, color="red", s=100, zorder=5)
    ax.axvline(optimal_customers, color="red", linestyle="--", alpha=0.5, label="Optimal Cutoff")
    ax.axhline(0, color="black", linewidth=1)

    ax.set_xlabel("Number of Customers Contacted (Sorted by Risk)")
    ax.set_ylabel("Net Profit (LKR)")
    ax.set_title("Dynamic Profit Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.markdown(f"""
    ### 💡 Strategic Insight
    To maximize profit with these costs, we should target the top **{optimal_customers}** high-risk customers.
    This corresponds to a threshold of **{optimal_threshold:.4f}**.
    
    If we target more people, the cost of incentives outweighs the saved revenue.
    If we target fewer, we miss too many churners (opportunity cost).
    """)

except Exception as e:
    st.error(f"Error loading model or data. Did we run 'make pipeline' first? Error: {e}")
