import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay


def advanced_churn_evaluation(
    model,
    X,
    y,
    model_name="XGBoost Classifier",
    # --- Business Parameters ---
    ltv=500,  # Lifetime Value ($)
    cost_offer=20,  # Cost of incentive ($)
    cost_contact=1,  # Cost of contact ($)
    acceptance_rate=0.5,  # Probability churner accepts offer
    currency="USD",
):
    # --- 1. Preparation ---
    # Ensure inputs are standard formatting
    y = np.array(y)

    # Get Probabilities (Critical for Lift & Thresholding)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        raise ValueError("Model must support predict_proba for business calculations.")

    # Create a DataFrame for analysis
    df_res = pd.DataFrame({"y_true": y, "y_prob": y_prob})
    df_res = df_res.sort_values("y_prob", ascending=False).reset_index(drop=True)

    # --- 2. The Profit Curve & Optimal Threshold ---
    # We calculate the business outcome at EVERY possible threshold (row by row)

    # Cumulative stats
    df_res["cum_tp"] = df_res["y_true"].cumsum()
    df_res["cum_fp"] = (1 - df_res["y_true"]).cumsum()
    df_res["cum_total"] = df_res.index + 1

    # Business Formula
    # Profit = (Revenue from Saved Churners) - (Cost of Campaign)
    # Revenue = (True Positives * Acceptance_Rate * LTV)
    # Cost = (Contacted_Count * (Cost_Contact + Cost_Offer))

    df_res["campaign_cost"] = df_res["cum_total"] * (cost_contact + cost_offer)
    df_res["revenue_saved"] = df_res["cum_tp"] * acceptance_rate * ltv
    df_res["net_profit"] = df_res["revenue_saved"] - df_res["campaign_cost"]

    # Find the Sweet Spot (Max Profit)
    max_profit_idx = df_res["net_profit"].idxmax()
    max_profit = df_res.loc[max_profit_idx, "net_profit"]
    optimal_threshold = df_res.loc[max_profit_idx, "y_prob"]
    optimal_customers = max_profit_idx + 1
    total_customers = len(df_res)

    # ROI at optimal point
    optimal_roi = (max_profit / df_res.loc[max_profit_idx, "campaign_cost"]) * 100

    # --- 3. Decile Analysis (The "Lift" Report) ---
    df_res["decile"] = pd.qcut(df_res.index, 10, labels=False) + 1
    lift_data = (
        df_res.groupby("decile")
        .agg(total_customers=("y_true", "count"), actual_churners=("y_true", "sum"))
        .sort_index()
    )

    # Calculate Lift
    global_churn_rate = y.mean()
    lift_data["churn_rate"] = lift_data["actual_churners"] / lift_data["total_customers"]
    lift_data["lift"] = lift_data["churn_rate"] / global_churn_rate

    top_decile_lift = lift_data.loc[1, "lift"]  # Decile 1 (Top 10%)  # noqa: F841

    # --- 4. Printing the Executive Report ---
    print("=========================================================")
    print(f"  EXECUTIVE SUMMARY: {model_name.upper()}")
    print("=========================================================\n")

    print(f" FINANCIAL IMPACT ANALYSIS (At Optimal Threshold: {optimal_threshold:.2f})")
    print("---------------------------------------------------------")
    print(f"Max Potential Profit:     {currency} {max_profit:,.2f}")
    print(f"Return on Investment:     {optimal_roi:.1f}%")
    print(
        f"Target Volume:            {optimal_customers} customers ({optimal_customers / total_customers:.1%}% of base)"
    )
    print("---------------------------------------------------------")

    print("\n>>> LIFT & PERFORMANCE METRICS")
    print("---------------------------------------------------------")
    print(f"Top Decile Lift:          {top_decile_lift:.2f}x (Industry Target: >3.0x)")
    print(f"AUC-ROC Score:            {roc_auc_score(y, y_prob):.3f}")

    # Recalculate Confusion Matrix at OPTIMAL threshold (not 0.5)
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    print("\nClassification Report (Optimized for Profit):")
    print(classification_report(y, y_pred_opt, target_names=["Stay", "Churn"]))

    # --- 5. Visualizations ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.3)

    # A. Profit Curve
    axes[0, 0].plot(df_res["cum_total"], df_res["net_profit"], color="green", linewidth=2)
    axes[0, 0].scatter(
        optimal_customers,
        max_profit,
        color="red",
        s=100,
        zorder=5,
        label=f"Max Profit: {currency} {max_profit:,.0f}",
    )
    axes[0, 0].axvline(optimal_customers, color="red", linestyle="--", alpha=0.5)
    axes[0, 0].set_title("Profit Curve (Business Impact)", fontsize=14)
    axes[0, 0].set_xlabel("Number of Customers Contacted (Sorted by Risk)")
    axes[0, 0].set_ylabel(f"Net Profit ({currency})")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # B. Cumulative Gains Curve (Industry Standard)
    # % of total churners captured by contacting top % of base
    axes[0, 1].plot(
        df_res.index / total_customers,
        df_res["cum_tp"] / df_res["y_true"].sum(),
        color="blue",
        linewidth=2,
        label="Model",
    )
    axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random Guessing")
    axes[0, 1].set_title("Cumulative Gains Curve (Recall by Volume)", fontsize=14)
    axes[0, 1].set_xlabel("% of Customer Base Contacted")
    axes[0, 1].set_ylabel("% of All Churners Captured")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # C. Lift per Decile (Bar Chart)
    sns.barplot(
        x=lift_data.index,
        y=lift_data["lift"],
        ax=axes[1, 0],
        palette="viridis",
        hue=lift_data.index,
        legend=False,
    )
    axes[1, 0].axhline(1.0, color="red", linestyle="--", label="Baseline (Random)")
    axes[1, 0].set_title("Lift by Decile (Top 10% vs Average)", fontsize=14)
    axes[1, 0].set_ylabel("Lift Multiplier (x times better)")
    axes[1, 0].set_xlabel("Decile (1=Highest Risk)")
    axes[1, 0].legend()

    # D. Confusion Matrix (At Optimal Threshold)
    ConfusionMatrixDisplay.from_predictions(
        y,
        y_pred_opt,
        ax=axes[1, 1],
        cmap="Blues",
        normalize=None,
        display_labels=["Stay", "Churn"],
    )
    axes[1, 1].set_title(f"Confusion Matrix\n(Threshold = {optimal_threshold:.2f})", fontsize=14)

    plt.show()


def run_sensitivity_analysis(
    model, X, y, scenarios=None, ltv=500, cost_offer=20, cost_contact=5, currency="USD"
):
    print("--- SENSITIVITY ANALYSIS: Impact of Campaign Success Rate ---")

    # Define three scenarios: Pessimistic, Base, Optimistic if scenarios are not provided
    if scenarios is None:
        scenarios = [
            {"rate": 0.2, "label": "Pessimistic (20% Accept)"},
            {"rate": 0.5, "label": "Base Case (50% Accept)"},
            {"rate": 0.8, "label": "Optimistic (80% Accept)"},
        ]

    results = []

    for scen in scenarios:
        # We reuse logic from the main function, but simplified for comparison
        y_prob = model.predict_proba(X)[:, 1]
        df = pd.DataFrame({"y_true": y, "y_prob": y_prob}).sort_values("y_prob", ascending=False)

        df["cum_tp"] = df["y_true"].cumsum()
        df["total_contacted"] = df.index + 1

        # Calculate profit curve for this specific scenario
        revenue = df["cum_tp"] * scen["rate"] * ltv
        cost = df["total_contacted"] * (cost_offer + cost_contact)
        profit_ = revenue - cost

        max_profit = profit_.max()
        results.append(
            {
                "Scenario": scen["label"],
                "Max_Profit": max_profit,
                "Optimal_Customers": profit_.idxmax() + 1,
            }
        )

        plt.plot(df["total_contacted"], profit_, label=scen["label"])

    # Plot formatting
    plt.title("Profit Sensitivity: What if the Campaign Performs Differently?")
    plt.xlabel("Number of Customers Contacted")
    plt.ylabel(f"Projected Net Profit ({currency})")
    plt.axhline(0, color="black", linestyle="--")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Output the comparison table
    print(pd.DataFrame(results))


# Define a Python function that calculates profit from Truth/Prediction
def profit_calculator(y_true, y_pred, ltv, cost, acceptance):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    revenue = tp * acceptance * ltv
    marketing_spend = (tp + fp) * cost
    return revenue - marketing_spend
