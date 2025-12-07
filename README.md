# 📡 Telco Churn Prediction: Maximizing ROI via Behavioral Trend Analysis

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blueviolet)
![Optuna](https://img.shields.io/badge/Tuning-Optuna-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Business Impact:** Achieved a projected **981% ROI** and **LKR 5.04M Net Profit** by optimizing the classification threshold for profitability rather than accuracy.

## 📋 Table of Contents
- [Project Summary](#-project-summary)
- [Business Problem & Assumptions](#-business-problem--assumptions)
- [Data Overview](#-data-overview)
- [Data Engineering & Strategy](#-data-engineering--strategy)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Performance](#-model-performance)
- [Financial Impact Analysis](#-financial-impact-analysis)
- [Project Structure](#-project-structure)

---

## 🚀 Project Summary
This project aims to build a profit-driven churn prediction system for a Sri Lankan Telecommunications provider, using a customer's behavior from their first 4 weeks.

By engineering **robust behavioral features** (Trend Deltas, Gini Coefficients, "Cliff" drops, Volatility Shifts) and utilizing **XGBoost**, the model identifies high-risk customers with **92.4% AUC-ROC** and **99% Recall** at **the Optimal Threshold (0.0201)** to capture the "Walking Dead"—customers who have mentally churned but haven't cancelled yet—maximizing the financial recovery of the retention campaign.

---

## 💼 Business Problem & Assumptions
**The Challenge:** A Sri Lankan telco company suffers from customers churning their service. Identifying customers *before* they leave is critical, but traditional metrics (Accuracy) fail to account for the financial asymmetry between a False Negative (Lost LTV) and a False Positive (Cost of Retention Offer). Using a customer's behavior from their first 4 weeks (Days 1-28), can we predict if they will become **at-risk** of churning?

**Market Context (Sri Lanka / Dialog Axiata Proxies):**
* **Average Revenue Per User (ARPU):** LKR 750.00
* **Customer Lifetime Value (LTV):** LKR 9,000.00 (12-month retention cap)
* **Retention Campaign Cost:** LKR 3.00 (SMS) + LKR 225.00 (30% Discount Offer)
* **Goal:** Maximize **Net Profit** (Revenue Saved - Campaign Cost).

---

## 💾 Data Overview
**Source:** [Link to Datasets](https://www.kaggle.com/datasets/lasaljaywardena/real-world-churn)

**Snapshot Period:** January 1, 2023, to March 31, 2023

**Dataset Overview:**
The dataset consists of **14 CSV files** tracking daily activity for **65,005 customers** over 3 months.
* **Core Labels:** `train_cxid.csv` and `test_cxid.csv` containing Customer IDs and Churn status.
* **App Usage (MB):** Daily data consumption for specific apps (Facebook, YouTube, TikTok, WhatsApp, Helakuru) and generic background usage.
* **Financial Spend (LKR):** Daily spend on Main Data Packs vs. Value-Added Services (VAS).
* **Voice Logs (Minutes):** Duration of Incoming/Outgoing calls, split by Domestic vs. International/Roaming.

---

## 🛠 Data Engineering & Strategy
### 1. Handling Data Leakage
Initial models showed 99% accuracy due to **target leakage** (usage dropping *before* churn). 
* **Solution:** Implemented a strict **Observation Window** (Days 1-28) vs. **Prediction Window** (Day 60-90) split. The model only sees the first 4 weeks of behavior to predict churn happening 2 months later.

### 2. Feature Engineering: The "Robust" Approach
Instead of relying on raw volume (which varies by user), I engineered **39 unit-invariant features** based on a 28-day observation window. These features isolate behavioral *changes* that signal churn:
* **Trend Deltas:** Measuring the velocity of disengagement (e.g., *Week 4 usage vs. Week 1 usage*).
* **The "Cliff":** Quantifying the percentage drop from a user's "personal peak" week to their current week.
* **Stability Metrics:** Using **Z-Scores** and **Volatility Shifts** to detect statistically significant crashes in usage consistency.
* **Gini Coefficient:** Adapted from economics to measure "Habit Strength" (0 = steady daily usage, 1 = sporadic spikes).
* **Wallet Share:** Tracking shifts in usage composition (e.g., ratio of Video vs. Messaging apps).

---

## 🔍 Exploratory Data Analysis
**Key Insights:**
* **Data Profiling:** High-integrity dataset with no missing values; class imbalance (23.1% churn) addressed via weighting/SMOTE.
* **Key Predictor:** The strongest signal is **"Gradual Fade"**—users whose data consumption drops significantly from Week 1 to Week 4.
* **Feature Engineering:** Reduced dimensionality by removing 19 collinear features, retaining 23 robust predictors via **IV & WoE analysis** for modeling.
* **User Segmentation:** Identified a critical **"Early Burnout"** segment (Cluster 1) via **UMAP** and **k-Means** that spends heavily in Week 1 before usage crashes by ~53%.

---

## 🤖 Model Performance
This is optimized by hyperparameters-tuning via **Optuna** and tracking experiments with **MLflow**.

* **AUC-ROC Score:** **0.924** (Excellent predictive power)
* **Lift:** **3.98x** (Top 10% of targeted customers are ~4x more likely to churn than average)
* **Trade-off:**
    * **Standard View (Threshold 0.5):** High Precision (0.78), Moderate Recall (0.67)"
    * **Business View (Threshold 0.02):** Low Precision (0.32), **Massive Recall (0.99)**. The strategy is to "catch everyone" because missing a churner is expensive in LKR, while contacting a loyalist is relatively cheap.

---

## 💰 Financial Impact Analysis
**The Strategic Shift:**
A default decision threshold (0.5) yielded good precision but missed too many churners (low recall). By analyzing the **Profit Curve**, the threshold is moved to **0.0201** to gain profit.

### Results at Optimal Threshold (0.0201):
* **ROI:** **981.7%** (For every LKR 1 spent, LKR 10.82 is generated)
* **Projected Net Profit:** **LKR 5,040,042** (on the test set).
* **Recall:** **99%** (We catch almost every potential churner).

### Sensitivity Analysis:

I simulated three offer strategies. The **Gold Tier (30% off)** was selected as the optimal balance between acceptance rate and margin preservation.
* *Platinum (50% off):* Higher revenue saved, but lower ROI (630%).
* *Bronze (20% off):* Highest ROI (1335%), but lower total profit due to lower acceptance.

---

## 📂 Project Structure
```bash
├── data/
│   ├── raw/                 # Original 12 raw usage files
│   ├── interim/             # Combined wide-format data
│   └── processed/           # Final feature-engineered parquet files
├── notebooks/
│   ├── 1.0-yusuftejakusumah-data-preparation.ipynb       # Leakage fix & Feature Engineering
│   ├── 2.0-yusuftejakusumah-exploratory-data-analysis.ipynb     # Sweetviz & Data Profiling, IV &WoE Analysis, Redundancy Analysis, UMAP & Hierarcical Clustering
│   └── 3.0-yusuftejakusumah-modeling-and-evaluation.ipynb # XGBoost, Optuna, Profit Curves
├── models/
│   └── telco_churn_xgboost_v0.pkl       # Final serialized model
├── reports/
│   ├── mlruns/              # MLflow experiment tracking logs
│   └── figures/           # Generated graphics and figures to be used in reporting
└── README.md