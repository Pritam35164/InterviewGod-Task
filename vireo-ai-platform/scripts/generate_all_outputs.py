"""
Generate all outputs for Vireo Audio Refund Analysis.
Creates board_pack.json, board_pack.md, monthly_summary.csv, reason_summary.csv,
agent_summary.csv, avoidable_summary.csv, evaluation_results.json, cost_analysis.json.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from src.data_loader import load_raw_datasets, clean_and_deduplicate
from src.policy import compute_dataset_avoidability
from src.analysis import (
    calculate_overall_kpis,
    calculate_monthly_summary,
    calculate_reason_summary,
    calculate_agent_summary
)
from src.evaluation import run_evaluation

def generate_all():
    print("1. Loading raw datasets & cleaning...")
    raw = load_raw_datasets()
    df = clean_and_deduplicate(raw)

    print("2. Assessing support policy avoidability...")
    df_avoid = compute_dataset_avoidability(df)

    print("3. Computing deterministic financial analysis & KPIs...")
    kpis = calculate_overall_kpis(df_avoid)
    monthly = calculate_monthly_summary(df_avoid)
    reasons = calculate_reason_summary(df_avoid)
    agents = calculate_agent_summary(df_avoid)

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save CSVs
    print("4. Exporting summary CSV files...")
    df_monthly = pd.DataFrame(monthly["rows"])
    df_monthly.to_csv(output_dir / "monthly_summary.csv", index=False)

    df_reasons = pd.DataFrame(reasons["rows"])
    df_reasons.to_csv(output_dir / "reason_summary.csv", index=False)

    df_agents = pd.DataFrame(agents["rows"])
    df_agents.to_csv(output_dir / "agent_summary.csv", index=False)

    df_avoidable = df_avoid[df_avoid["is_potentially_avoidable"]][
        ["ticket_id", "created_at", "month", "agent_id", "clean_refund_inr", "avoidable_amount_inr", "avoidability_drivers", "refund_reason_code"]
    ]
    df_avoidable.to_csv(output_dir / "avoidable_summary.csv", index=False)

    # Cost Analysis
    print("5. Computing AI cost arithmetic...")
    monthly_tickets_volume = 2817  # 650 tickets/week * 52 / 12
    prompt_tokens_per_ticket = 450
    completion_tokens_per_ticket = 120
    usd_to_inr = 84.0

    cost_per_prompt_token_usd = 0.50 / 1_000_000
    cost_per_completion_token_usd = 1.50 / 1_000_000

    cost_per_ticket_inr = (
        (prompt_tokens_per_ticket * cost_per_prompt_token_usd) +
        (completion_tokens_per_ticket * cost_per_completion_token_usd)
    ) * usd_to_inr

    one_full_run_cost_inr = len(df) * cost_per_ticket_inr
    monthly_cost_inr = monthly_tickets_volume * cost_per_ticket_inr

    cost_analysis = {
        "monthly_volume_tickets": monthly_tickets_volume,
        "assumed_tokens_per_ticket": {
            "prompt_tokens": prompt_tokens_per_ticket,
            "completion_tokens": completion_tokens_per_ticket,
            "total_tokens": prompt_tokens_per_ticket + completion_tokens_per_ticket
        },
        "openrouter_pricing_usd": {
            "prompt_per_1m": 0.50,
            "completion_per_1m": 1.50
        },
        "usd_to_inr_exchange_rate": usd_to_inr,
        "calculated_cost_per_ticket_inr": round(cost_per_ticket_inr, 4),
        "one_full_dataset_run_cost_inr": round(one_full_run_cost_inr, 2),
        "estimated_monthly_operational_cost_inr": round(monthly_cost_inr, 2),
        "development_cost_under_free_credits_inr": 0.0,
        "arithmetic_explanation": [
            f"Monthly Volume = 650 tickets/week × 52 weeks / 12 months = {monthly_tickets_volume} tickets/month.",
            f"Cost per ticket = (450 prompt tokens × $0.50/M + 120 completion tokens × $1.50/M) × ₹84 = ₹{cost_per_ticket_inr:.4f} per ticket.",
            f"One full run (11,600 tickets) cost = 11,600 × ₹{cost_per_ticket_inr:.4f} = ₹{one_full_run_cost_inr:,.2f}.",
            f"Estimated monthly cost = 2,817 × ₹{cost_per_ticket_inr:.4f} = ₹{monthly_cost_inr:,.2f} per month."
        ]
    }
    with open(output_dir / "cost_analysis.json", "w", encoding="utf-8") as f:
        json.dump(cost_analysis, f, indent=2)

    # Evaluation
    print("6. Running ground-truth evaluation...")
    eval_res = run_evaluation(df_avoid)

    # Board Pack JSON
    board_pack = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": kpis,
        "monthly": monthly,
        "reasons": reasons,
        "agents": agents,
        "cost": cost_analysis,
        "evaluation": eval_res
    }

    with open(output_dir / "board_pack.json", "w", encoding="utf-8") as f:
        json.dump(board_pack, f, indent=2)

    # Board Pack Markdown
    board_md = f"""# Vireo Audio — Refund Intelligence Board Pack

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Policy v3.2 · Internal

## Executive Summary

| Metric | Value |
|---|---|
| Period | {kpis['period_start']} to {kpis['period_end']} ({kpis['months_observed']} months) |
| Total Unique Tickets | {kpis['total_tickets']:,} |
| Refund Tickets | {kpis['refund_tickets']:,} |
| Refund Rate | {kpis['refund_rate_pct']:.1f}% |
| **Reconciled Refund Value** | **₹{kpis['refund_value_inr']:,.2f}** |
| Average Refund per Ticket | ₹{kpis['average_refund_inr']:,.2f} |
| Raw Export Uncleaned Sum | ₹{kpis['raw_export_value_inr']:,.2f} |
| **Potentially Avoidable Count** | **{kpis['potentially_avoidable_count']:,} tickets** |
| **Potentially Avoidable Value** | **₹{kpis['potentially_avoidable_value_inr']:,.2f}** ({kpis['avoidable_share_pct']:.1f}% of total refunds) |
| **Quarterlyized Opportunity** | **₹{kpis['quarterlyized_opportunity_inr']:,.2f} / quarter** |
| **Annualized Opportunity** | **₹{kpis['annualized_opportunity_inr']:,.2f} / year** |

### Financial Reconciliation

| Step | Explanation | Refund Value | Count | Per Quarter |
|---|---|---|---|---|
| 1 | Raw export, summed as delivered | ₹{kpis['reconciliation'][0]['refund_value_inr']:,.2f} | 2,465 | ₹{kpis['reconciliation'][0]['per_quarter_inr']:,.2f} |
| 2 | Legacy FD amounts converted from paise to rupees | ₹{kpis['reconciliation'][1]['refund_value_inr']:,.2f} | 2,465 | ₹{kpis['reconciliation'][1]['per_quarter_inr']:,.2f} |
| 3 | Re-imported duplicate tickets removed | ₹{kpis['reconciliation'][2]['refund_value_inr']:,.2f} | 2,340 | ₹{kpis['reconciliation'][2]['per_quarter_inr']:,.2f} |

**Key Finding:** Sameer's helpdesk figure (~₹11 Lakhs/quarter) matches the reconciled total (₹11.18 Lakhs/quarter). Arjun's "well over a crore a quarter" resulted from raw export mixing legacy Freshdesk paise with rupees and double-counting re-imported tickets.

## Monthly Summary

| Month | Tickets | Refunds | Value (INR) | Rate % | Avg Refund | Avoidable Count | Avoidable Value (INR) |
|---|---|---|---|---|---|---|---|
"""
    for r in monthly["rows"]:
        board_md += f"| {r['month']} | {r['tickets']:,} | {r['refund_count']:,} | ₹{r['refund_value_inr']:,.2f} | {r['refund_rate_pct']:.1f}% | ₹{r['average_refund_inr']:,.2f} | {r['potentially_avoidable_count']} | ₹{r['potentially_avoidable_value_inr']:,.2f} |\n"

    board_md += "\n## Reason Code Breakdown\n\n| Reason Code | Refund Count | Refund Value (INR) | Value Share % | Avoidable Value (INR) |\n|---|---|---|---|---|\n"
    for r in reasons["rows"]:
        board_md += f"| {r['reason_code']} | {r['refund_count']:,} | ₹{r['refund_value_inr']:,.2f} | {r['value_share_pct']:.1f}% | ₹{r['potentially_avoidable_value_inr']:,.2f} |\n"

    board_md += "\n## Top Agents Summary (Workload Normalized)\n\n| Agent ID | Agent Name | Team | Tier | Tickets | Refunds | Value (INR) | Refunds / 100 Tix | Value / 100 Tix | Avoidable Value |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for r in agents["rows"][:15]:
        board_md += f"| {r['agent_id']} | {r['agent_name']} | {r['team']} | T{r['tier']} | {r['tickets_handled']:,} | {r['refund_count']:,} | ₹{r['refund_value_inr']:,.2f} | {r['refunds_per_100_tickets']:.1f} | ₹{r['refund_value_per_100_tickets_inr']:,.2f} | ₹{r['potentially_avoidable_value_inr']:,.2f} |\n"

    board_md += f"""
## AI Accuracy & Evaluation Methodology

- Evaluated Sample: {eval_res['total_evaluated']} human-reviewed tickets
- Reason Classification Accuracy: {eval_res['reason_classification_accuracy_pct']:.1f}%
- Avoidable Classification Accuracy: {eval_res['avoidable_classification_accuracy_pct']:.1f}%
- Malformed AI Responses: {eval_res['malformed_responses_count']} ({eval_res['malformed_responses_pct']:.1f}%)
- Unresolved Cases: {eval_res['unresolved_cases_count']} ({eval_res['unresolved_cases_pct']:.1f}%)

## Operational AI Cost Arithmetic

- Cost per ticket: ₹{cost_analysis['calculated_cost_per_ticket_inr']:.4f}
- Estimated monthly cost (2,817 tickets/mo): ₹{cost_analysis['estimated_monthly_operational_cost_inr']:,.2f} / month (~$1.14 USD/month)
- One full dataset run (11,600 tickets): ₹{cost_analysis['one_full_dataset_run_cost_inr']:,.2f}
"""

    with open(output_dir / "board_pack.md", "w", encoding="utf-8") as f:
        f.write(board_md)

    print("Outputs generated successfully!")

if __name__ == "__main__":
    generate_all()
