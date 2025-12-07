import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, classification_report, 
                             ConfusionMatrixDisplay)

def comprehensive_churn_evaluation(
    model,
    X,
    y,
    model_name="XGBoost Classifier",
    run_id="A1",
    # --- Business Parameters ---
    ltv=500,                # Lifetime Value ($)
    cost_offer=20,          # Cost of incentive ($) - Only paid if accepted
    cost_contact=1,         # Cost of contact ($) - Paid for everyone targeted
    currency="USD",
    
    # --- Dynamic Acceptance Logic ---
    acceptance_rate_base=0.5,       # Normal acceptance rate
    acceptance_rate_high_risk=0.1,  # "Lost Cause" acceptance rate
    high_risk_threshold=0.9         # Probability threshold for "Lost Cause"
):
    # --- 1. Preparation & Probabilities ---
    y = np.array(y)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Model must support predict_proba")

    # Create Analysis DataFrame
    df_res = pd.DataFrame({"y_true": y, "y_prob": y_prob})
    df_res = df_res.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # --- 2. Microscopic Financial Calculation ---
    # Define Acceptance Probability per row
    df_res['prob_accept'] = np.where(
        df_res['y_prob'] > high_risk_threshold, 
        acceptance_rate_high_risk, 
        acceptance_rate_base
    )
    
    # Expected Revenue & Cost (Per Row)
    df_res['expected_revenue'] = df_res['y_true'] * df_res['prob_accept'] * ltv
    df_res['expected_cost'] = cost_contact + (df_res['prob_accept'] * cost_offer)
    
    # Net Profit per Row & Cumulative
    df_res['row_profit'] = df_res['expected_revenue'] - df_res['expected_cost']
    df_res['cum_profit'] = df_res['row_profit'].cumsum()
    df_res['cum_revenue'] = df_res['expected_revenue'].cumsum()
    df_res['cum_cost'] = df_res['expected_cost'].cumsum()
    
    # Cumulative stats for Gains Curve
    df_res['cum_tp'] = df_res["y_true"].cumsum() 
    df_res['cum_total'] = df_res.index + 1

    # --- 3. Optimization & Detailed Metrics ---
    max_profit_idx = df_res['cum_profit'].idxmax()
    max_profit = df_res.loc[max_profit_idx, 'cum_profit']
    optimal_threshold = df_res.loc[max_profit_idx, 'y_prob']
    optimal_customers = max_profit_idx + 1
    total_customers = len(df_res)
    
    # Extract the "Optimal Row" for snapshot metrics
    row = df_res.loc[max_profit_idx]
    
    # --- RE-INSERTED: Detailed Waste & Efficiency Logic ---
    total_targeted = row["cum_total"]
    tp_count = row["cum_tp"]
    fp_count = total_targeted - tp_count
    
    # Slice the dataframe to analyze ONLY the targeted group
    target_group = df_res.iloc[:optimal_customers]
    
    # Averages for the specific targeted segment
    avg_risk = target_group['y_prob'].mean()
    avg_acceptance = target_group['prob_accept'].mean()
    
    # Calculate specific 'Cost of False Positives' (Waste)
    # Logic: Cost of contact (for all FPs) + Cost of Offer (for FPs who accepted)
    # Note: FPs are where y_true == 0
    fp_mask = target_group['y_true'] == 0
    waste_contact = fp_mask.sum() * cost_contact
    waste_incentive = (target_group.loc[fp_mask, 'prob_accept'] * cost_offer).sum()
    cost_of_fp = waste_contact + waste_incentive
    
    # ROI
    optimal_roi = (max_profit / row['cum_cost']) * 100

    # --- 4. Technical Performance Metrics ---
    # Lift Analysis
    df_res["decile"] = pd.qcut(df_res.index, 10, labels=False) + 1
    lift_data = (
        df_res.groupby("decile")
        .agg(total=("y_true", "count"), actual_churners=("y_true", "sum"))
        .sort_index()
    )
    global_churn_rate = y.mean()
    lift_data["churn_rate"] = lift_data["actual_churners"] / lift_data["total"]
    lift_data["lift"] = lift_data["churn_rate"] / global_churn_rate
    top_decile_lift = lift_data.loc[1, "lift"]

    # Predictions for Classification Reports
    y_pred_default = (y_prob >= 0.5).astype(int)            
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int) 

    # --- 5. Executive Report (Merged & Complete) ---
    print(f"Run ID: {run_id}\n")
    print("=========================================================")
    print(f"  EVALUATION REPORT: {model_name.upper()}")
    print("=========================================================\n")
    
    # --- SECTION A: The Strategic & Financial Deep Dive (Restored) ---
    print(f" STRATEGIC DECISION (Optimal Threshold: {optimal_threshold:.4f})")
    print("---------------------------------------------------------")
    print(f" Target Volume:           {int(total_targeted):,} customers")
    print(f" Coverage:                {total_targeted / total_customers:.1%} of total customer base")
    print(f" Avg. Risk of Target:     {avg_risk:.1%}")
    print(f" Est. Acceptance Rate:    {avg_acceptance:.1%} (weighted average)")
    print("---------------------------------------------------------\n")

    print(f" FINANCIAL IMPACT ({currency})")
    print("---------------------------------------------------------")
    print(f" Total Revenue Saved:     {currency} {row['cum_revenue']:,.0f}")
    print(f" Total Campaign Spend:    {currency} {row['cum_cost']:,.0f}")
    print(f" NET PROFIT:              {currency} {max_profit:,.0f}")
    print(f" ROI:                     {optimal_roi:.1f}%")
    print("---------------------------------------------------------\n")
    
    print(" EFFICIENCY & WASTE ANALYSIS")
    print("---------------------------------------------------------")
    print(f" Correctly Targeted:      {int(tp_count):,} (Churners)")
    print(f" Incorrectly Targeted:    {int(fp_count):,} (Loyalists)")
    print(f" Cost of False Positives: {currency} {cost_of_fp:,.0f} (Cannibalization/Waste)")
    print(f" Spend Efficiency:        For every {currency} 1 spent, we generate {currency} {row['cum_revenue']/row['cum_cost']:.2f}")
    print("---------------------------------------------------------\n")

    # --- SECTION B: Model Performance (Technical Context) ---
    print(" MODEL PERFORMANCE METRICS")
    print("---------------------------------------------------------")
    print(f" AUC-ROC Score:           {roc_auc_score(y, y_prob):.3f}")
    print(f" Top Decile Lift:         {top_decile_lift:.2f}x")
    print("---------------------------------------------------------")

    print("\n CLASSIFICATION COMPARISON")
    print("---------------------------------------------------------")
    print(">>> 1. Standard Threshold (0.5) - Raw Model Power:")
    print(classification_report(y, y_pred_default, target_names=["Stay", "Churn"]))
    print("\n>>> 2. Optimal Threshold (Profit Max) - Business Reality:")
    print(classification_report(y, y_pred_optimal, target_names=["Stay", "Churn"]))

    # --- 6. Visualizations (The Complete Dashboard) ---
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3)

    # Plot 1: Profit Curve (Dynamic)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_res["cum_total"], df_res["cum_profit"], color="green", lw=2)
    ax1.scatter(optimal_customers, max_profit, color="red", s=100, zorder=5)
    ax1.axvline(optimal_customers, color="red", ls="--", alpha=0.5)
    ax1.set_title("Profit Curve (Dynamic Expected Value)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Customers Contacted")
    ax1.set_ylabel(f"Profit ({currency})")
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cumulative Gains (Standard)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_res.index/len(df_res), df_res["cum_tp"]/df_res["y_true"].sum(), color="blue", lw=2)
    ax2.plot([0, 1], [0, 1], "k--", label="Random")
    ax2.set_title("Cumulative Gains (Recall)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("% of Base Contacted")
    ax2.set_ylabel("% of Churners Captured")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Lift Chart
    ax3 = fig.add_subplot(gs[0, 2])
    sns.barplot(x=lift_data.index, y=lift_data["lift"], ax=ax3, palette="viridis", hue=lift_data.index, legend=False)
    ax3.axhline(1.0, color="red", ls="--")
    ax3.set_title("Lift by Decile", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Lift Multiplier")

    # Plot 4: Confusion Matrix (Standard 0.5)
    ax4 = fig.add_subplot(gs[1, 0])
    ConfusionMatrixDisplay.from_predictions(y, y_pred_default, ax=ax4, cmap="Blues", colorbar=False)
    ax4.set_title("Confusion Matrix (Thresh=0.5)", fontsize=12, fontweight='bold')

    # Plot 5: Confusion Matrix (Optimal)
    ax5 = fig.add_subplot(gs[1, 1])
    ConfusionMatrixDisplay.from_predictions(y, y_pred_optimal, ax=ax5, cmap="Greens", colorbar=False)
    ax5.set_title(f"Confusion Matrix (Thresh={optimal_threshold:.2f})", fontsize=12, fontweight='bold')

    # Plot 6: Acceptance Probability Dist (The "Microscopic" Check)
    ax6 = fig.add_subplot(gs[1, 2])
    # Show distribution of acceptance ONLY for the targeted group
    sns.histplot(target_group['prob_accept'], bins=5, ax=ax6, color='orange')
    ax6.set_title("Acceptance Odds of Targeted Users", fontsize=12, fontweight='bold')
    ax6.set_xlabel("Probability of Accepting Offer")

    plt.tight_layout()
    plt.show()

    return df_res


def run_sensitivity_analysis(
    model, 
    X, 
    y, 
    scenarios, 
    ltv, 
    cost_contact, 
    currency="USD",
    high_risk_threshold=0.9, 
    high_risk_decay=0.2  # High risk customers are only 20% as likely to accept as base
):
    """
    Fast, memory-efficient scenario comparison for Churn strategies.
    
    Args:
        scenarios: List of dicts, e.g. [{'rate': 0.18, 'cost': 150, 'label': 'Bronze'}, ...]
        high_risk_decay: Multiplier for high-risk customers. 
                         If scenario rate is 20% (0.20), high risk folks accept at 0.20 * 0.2 = 4%.
    """
    
    # --- 1. Pre-Computation ---
    print("Pre-computing model probabilities...")
    y = np.array(y)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Model must support predict_proba")

    # Create a lightweight structured array or DataFrame for sorting
    data = pd.DataFrame({"y_true": y, "y_prob": y_prob})
    data = data.sort_values("y_prob", ascending=False).reset_index(drop=True)
    
    # Convert to numpy arrays for raw speed in the loop
    y_true_sorted = data["y_true"].values
    y_prob_sorted = data["y_prob"].values
    n_customers = len(y_true_sorted)
    
    # Pre-calculate the High Risk Mask (Boolean array)
    # True if customer is "Lost Cause" (>90% risk)
    is_high_risk = y_prob_sorted > high_risk_threshold
    
    results = []

    print(f"Running analysis on {len(scenarios)} scenarios...")
    
    for sc in scenarios:
        base_rate = sc['rate']
        offer_cost = sc['cost']
        label = sc['label']
        
        # A. Dynamic Acceptance Vector
        # If high risk, rate = base_rate * decay. Else, rate = base_rate.
        # This is the "Microscopic" view applied efficiently
        prob_accept = np.where(
            is_high_risk, 
            base_rate * high_risk_decay, 
            base_rate
        )
        
        # B. Vectorized Financial Math
        # Exp Revenue = y_true * prob_accept * LTV
        exp_revenue = y_true_sorted * prob_accept * ltv
        
        # Exp Cost = Contact + (prob_accept * Offer Cost)
        exp_cost = cost_contact + (prob_accept * offer_cost)
        
        # Profit per customer
        row_profit = exp_revenue - exp_cost
        
        # Cumulative Sum to find the "Peak" of the mountain
        cum_profit = np.cumsum(row_profit)
        
        # C. Capture Metrics at Max Profit
        max_profit_idx = np.argmax(cum_profit)
        max_profit = cum_profit[max_profit_idx]
        optimal_customers = max_profit_idx + 1
        
        # Calculate ROI at that specific optimal point
        total_spend_at_peak = np.sum(exp_cost[:optimal_customers])
        roi = (max_profit / total_spend_at_peak) * 100
        
        results.append({
            "Scenario": label,
            "Base Acceptance": f"{base_rate:.1%}",
            "Offer Cost": f"{offer_cost}",
            "Max Profit": max_profit,
            "Optimal Volume": optimal_customers,
            "ROI": roi,
            "cum_profit_curve": cum_profit # Store curve for plotting
        })

    # --- 3. Outputs ---
    df_results = pd.DataFrame(results).drop(columns=["cum_profit_curve"])
    
    # Print Table
    print("\n=========================================================")
    print(f" SENSITIVITY ANALYSIS: STRATEGY COMPARISON ({currency})")
    print("=========================================================")
    print(df_results.to_string(index=False))
    print("=========================================================\n")
    
    # --- 4. Visualization ---
    plt.figure(figsize=(12, 7))
    
    colors = sns.color_palette("viridis", len(scenarios))
    
    # X-axis range (Customers Contacted)
    x_axis = np.arange(1, n_customers + 1)
    
    for i, res in enumerate(results):
        # Plot the profit curve stored in the results
        plt.plot(x_axis, res['cum_profit_curve'], label=f"{res['Scenario']} (ROI: {res['ROI']:.1f}%)", 
                 color=colors[i], linewidth=2.5)
        
        # Mark the peak
        peak_x = res['Optimal Volume']
        peak_y = res['Max Profit']
        plt.scatter(peak_x, peak_y, color=colors[i], s=100, zorder=5, edgecolors='white')

    plt.title(f"Profit Impact of Different Offer Strategies ({currency})", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Customers Contacted (Sorted by Risk)")
    plt.ylabel(f"Net Profit ({currency})")
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    
def analyze_prediction_errors(model, X, y, threshold=0.5, top_features=5):
    """
    Deep dive into False Positives and False Negatives.
    
    Args:
        model: Trained model.
        X: Feature set (Test set).
        y: True labels (Test set).
        threshold: The decision threshold (the optimal one found in the !).
        top_features: Number of features to analyze in the deviation report.
    """
    print(f"--- ERROR ANALYSIS REPORT (Threshold: {threshold}) ---")
    
    # 1. Setup Data
    X_analysis = X.copy()
    y = np.array(y)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Model must support predict_proba")
        
    y_pred = (y_prob >= threshold).astype(int)
    
    # 2. Categorize Predictions
    # TP: Predicted 1, Actual 1
    # FP: Predicted 1, Actual 0
    # FN: Predicted 0, Actual 1
    # TN: Predicted 0, Actual 0
    
    conditions = [
        (y == 1) & (y_pred == 1), # TP
        (y == 0) & (y_pred == 1), # FP
        (y == 1) & (y_pred == 0), # FN
        (y == 0) & (y_pred == 0)  # TN
    ]
    choices = ['TP (Correct Catch)', 'FP (False Alarm)', 'FN (Missed Churn)', 'TN (Safe)']
    
    X_analysis['category'] = np.select(conditions, choices, default='Unknown')
    X_analysis['y_true'] = y
    X_analysis['y_prob'] = y_prob
    
    # 3. Probability Distribution of Errors
    plt.figure(figsize=(14, 6))
    
    # Plot A: The "Risk Landscape"
    plt.subplot(1, 2, 1)
    sns.histplot(data=X_analysis, x='y_prob', hue='category', element="step", stat="density", common_norm=False)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Cutoff ({threshold})')
    plt.title("Where do the Errors Live? (Probability Dist)")
    plt.xlabel("Predicted Probability")
    
    # Plot B: Calibration Check (Reliability Curve)
    plt.subplot(1, 2, 2)
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Your Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title("Calibration Plot (Reliability)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 4. Feature Profiling (Why did we mess up?)
    # We compare the Mean value of features for Errors vs Correct predictions
    
    # Get numeric features only for aggregation
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Group by Category
    grouped = X_analysis.groupby('category')[numeric_cols].mean()
    
    print("\n>>> PROFILING THE MISTAKES")
    print("Compare the average feature values of Errors vs. Correct predictions.\n")
    
    # A. Analyzing False Positives (Who did we annoy?)
    # Compare FP (Wrongly targeted) vs TP (Correctly targeted)
    # This tells us: "What makes a False Positive look like a True Positive?"
    if 'FP (False Alarm)' in grouped.index and 'TP (Correct Catch)' in grouped.index:
        print(f"--- FALSE POSITIVE ANALYSIS (Loyalists we scared) ---")
        fp_profile = grouped.loc['FP (False Alarm)']
        tp_profile = grouped.loc['TP (Correct Catch)']
        tn_profile = grouped.loc['TN (Safe)']
        
        # Calculate % difference from the "Safe" customers (TN)
        # If FPs have much higher values than TNs, that's why the model got confused.
        diff = ((fp_profile - tn_profile) / tn_profile).replace([np.inf, -np.inf], 0)
        top_confusers = diff.abs().sort_values(ascending=False).head(top_features)
        
        print("Why did the model think they would churn?")
        for feature in top_confusers.index:
            val_fp = fp_profile[feature]
            val_tn = tn_profile[feature]
            pct_diff = diff[feature] * 100
            print(f"- {feature}: FP Avg ({val_fp:.2f}) is {pct_diff:+.1f}% vs Loyal Avg ({val_tn:.2f})")
            
    # B. Analyzing False Negatives (Who escaped?)
    # Compare FN (Missed) vs TN (Correctly ignored)
    # This tells us: "What makes a False Negative look Safe?"
    print(f"\n--- FALSE NEGATIVE ANALYSIS (Churners we missed) ---")
    if 'FN (Missed Churn)' in grouped.index and 'TN (Safe)' in grouped.index:
        fn_profile = grouped.loc['FN (Missed Churn)']
        tp_profile = grouped.loc['TP (Correct Catch)']
        
        # Compare Missed Churners (FN) to Caught Churners (TP)
        # Why did these specific churners look different?
        diff_missed = ((fn_profile - tp_profile) / tp_profile).replace([np.inf, -np.inf], 0)
        top_hiders = diff_missed.abs().sort_values(ascending=False).head(top_features)
        
        print("Why did the model think they were safe?")
        for feature in top_hiders.index:
            val_fn = fn_profile[feature]
            val_tp = tp_profile[feature]
            pct_diff = diff_missed[feature] * 100
            print(f"- {feature}: FN Avg ({val_fn:.2f}) is {pct_diff:+.1f}% vs Caught Churners ({val_tp:.2f})")

    return X_analysis # Returns the dataframe with categories for further manual inspection

    
    