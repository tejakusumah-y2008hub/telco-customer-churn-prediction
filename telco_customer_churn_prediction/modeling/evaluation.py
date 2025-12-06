import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
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

    # --- 2. Microscopic Financial Calculation (Code 2 Logic) ---
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

    # --- 4. Technical Performance Metrics (Code 1 Features) ---
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