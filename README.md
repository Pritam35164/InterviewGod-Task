# 🎧 Vireo Audio — Support Refund Intelligence Platform

A streamlined, deterministic Python and AI-assisted data analysis platform built to answer Arjun Mehta's (Finance Controller) core business question:

> *"Refunds have gone up and I can't see why. I need a monthly summary of refunds by reason code and by agent - who is giving away money and for what. Something I can drop straight into the board pack. The helpdesk export is a mess, so good luck."*

---

## 🌟 Executive Summary & Financial Findings

Across the complete 18-month dataset (January 2025 – June 2026; 11,600 unique tickets):

* **Total Reconciled Refund Payout:** **₹6,709,932** across 2,340 refund tickets.
* **Reconciled Quarterly Average:** **₹1,118,322 per quarter (~₹11.18 Lakhs/quarter)**.
* **Resolution of Export Discrepancy:** Reconciles the difference between Finance's uncleaned export (> ₹1 Crore/quarter) and IT Helpdesk Admin Sameer's report (~₹11 Lakhs/quarter). The raw export contained legacy Freshdesk values stored in **PAISE** (cents) and 638 re-imported duplicate tickets.
* **Volume Growth Decomposition:** Spend grew 1.64x between H1 2025 and H1 2026. Python deterministic analysis proves that the **refund rate per ticket remained virtually flat at ~20%**. Refund growth was 100% driven by customer contact volume growth (1.71x), not agent leniency.
* **Potentially Avoidable Refund Value:** **₹911,509** (13.6% of total refunds) identified as potentially avoidable under Support Operating Policy v3.2.
* **Financial Opportunity:** Represents **₹151,918 per quarter** (annualized to **₹607,673 per year**).

---

## 📦 The 5 Core Deliverables

This repository prioritizes the five required outcomes:

1. **Working Python AI-Assisted Tool:** `src/data_loader.py`, `src/analysis.py`, `src/classifier.py`, `src/policy.py`, `src/llm_client.py`.
2. **Business KPI with Actual ₹ Impact:** ₹911,509 avoidable refund value | ₹151,918/quarter opportunity.
3. **Evaluation / Accuracy Methodology:** `src/evaluation.py` benchmarking against 110 human-reviewed ground truth tickets.
4. **One-Page Non-Technical Memo to Arjun Mehta:** [`memo_to_arjun.md`](file:///d:/Activity/vireo-ai-platform/memo_to_arjun.md).
5. **Maximum 3-Minute Video Recording Script:** [`recording_script.md`](file:///d:/Activity/vireo-ai-platform/recording_script.md).

---

## 📊 Financial Reconciliation Table

| Step | Data Cleaning Action | Refund Value (INR) | Count | Quarterly Average |
|---|---|---|---|---|
| **Step 1** | Raw Export (Uncleaned sum) | ₹230,124,081 | 2,465 | ₹38,354,014 |
| **Step 2** | Legacy Freshdesk currency fix (Paise to INR) | ₹7,070,943 | 2,465 | ₹1,178,490 |
| **Step 3** | Ticket Deduplication (Helpdesk preferred) | **₹6,709,932** | **2,340** | **₹1,118,322** |

---

## 📁 Repository Structure

```
vireo-refund-analysis/
├── data/
│   ├── raw/                 # tickets.csv, agents.csv, orders.csv, customers.csv, products.csv, support-policy.pdf
│   └── evaluation_set.json  # 110 human-reviewed ground truth evaluation tickets
├── src/
│   ├── data_loader.py       # Deterministic parsing, legacy currency conversion, deduplication, joins
│   ├── analysis.py          # Python/Pandas owned math, aggregations, ratios, per-100 metrics
│   ├── policy.py            # Support policy rules (SLA credits, Goodwill cap, Dual remedy check)
│   ├── classifier.py        # OpenRouter Nemotron text interpretation & structured JSON output
│   ├── llm_client.py        # OpenRouter API client with retries and thinking-tag removal
│   ├── evaluation.py        # Accuracy evaluation against ground-truth dataset
│   └── laya_router.py       # Lightweight intent & complexity decision router
├── outputs/
│   ├── board_pack.json      # Structured board pack metrics
│   ├── board_pack.md        # Board pack formatted report
│   ├── monthly_summary.csv  # Monthly aggregated numbers
│   ├── reason_summary.csv   # Reason code breakdown
│   ├── agent_summary.csv    # Workload-normalized agent metrics
│   ├── avoidable_summary.csv# Avoidable refund ticket details
│   ├── evaluation_results.json # Accuracy & error analysis results
│   └── cost_analysis.json   # AI API cost arithmetic
├── app.py                   # Interactive Streamlit Dashboard UI
├── memo_to_arjun.md         # 1-page non-technical executive memo
├── recording_script.md      # 3-minute video presentation script
├── requirements.txt         # Minimal Python dependencies
├── .env.example             # Environment configuration template
├── Dockerfile               # Reproducible container deployment
├── docker-compose.yml       # Docker Compose setup
└── README.md                # Documentation
```

---

## 🚀 Quick Start & How to Run

### 1. Local Setup

```bash
# Clone or navigate to project directory
cd vireo-ai-platform

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Add your OPENROUTER_API_KEY in .env if using live LLM calls
```

### 2. Run Data Pipeline & Generate All Outputs

```bash
python scripts/generate_all_outputs.py
```

This runs the deterministic data cleaning, policy assessment, financial analysis, AI evaluation, and cost arithmetic, populating the `outputs/` folder.

### 3. Launch Interactive Streamlit Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser to explore:
- Overall KPIs & Financial Reconciliation
- Monthly Refund Trend & Volume Growth Decomposition
- Reason Code Breakdown
- Agent Metrics (Workload Normalized per 100 tickets)
- Potentially Avoidable Refund Value & Policy Clauses
- AI Evaluation & Accuracy Results
- Live Ticket AI Classifier Sandbox

### 4. Run via Docker

```bash
docker compose up --build
```

---

## 🎯 Evaluation Methodology & Results

Evaluated against **110 human-reviewed tickets** in `data/evaluation_set.json`:

* **Reason Classification Accuracy:** **100.0%**
* **Avoidable Classification Accuracy:** **67.7%**
* **Malformed Responses:** **0.0%**
* **Unresolved Cases:** **0.0%**

---

## 💰 Operational AI Cost Arithmetic

* **Monthly Support Volume:** ~2,817 tickets/month (650 tickets/week × 52 / 12).
* **Token Consumption per Ticket:** ~450 prompt tokens + ~120 completion tokens = 570 tokens.
* **OpenRouter NVIDIA Nemotron Pricing:** $0.50 / 1M prompt tokens, $1.50 / 1M completion tokens ($1 USD = ₹84 INR).
* **Calculated Cost per Ticket:** **₹0.034 INR** (~$0.0004 USD).
* **One Full Run Cost (11,600 tickets):** **₹394.40 INR** (~$4.70 USD).
* **Estimated Monthly Operational Cost:** **₹95.78 INR / month** (~$1.14 USD / month).
* **Development Cost under Free Credits:** **₹0 INR**.
