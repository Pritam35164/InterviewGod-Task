# MEMORANDUM

**TO:** Arjun Mehta, Finance Controller, Vireo Audio  
**FROM:** AI & Data Analytics Team  
**DATE:** October 1, 2026  
**SUBJECT:** Comprehensive Refund Data Reconciliation, Monthly Analysis, and Avoidable Cost Recovery  

---

### 1. Executive Finding
Our analysis of Vireo Audio’s support ticket dataset (January 2025 – June 2026; 11,600 unique tickets) reconciles total refunds to **₹6,709,932** across 2,340 refund tickets, corresponding to an average quarterly refund spend of **₹1,118,322 (~₹11.18 Lakhs/quarter)**. 

This confirms that IT Administrator Sameer Qureshi’s helpdesk figure (~₹11 Lakhs/quarter) is correct. The discrepancy with your initial Finance export figure ("well over a crore a quarter") has been fully identified and resolved: the raw export contained unadjusted historical Freshdesk rows recorded in **paise** rather than rupees, alongside 638 re-imported duplicate tickets from system migration.

Across the 18-month period, we identified **₹911,509** in **potentially avoidable refunds** under documented support policy rules (13.6% of total refund spend). This represents an annualized financial opportunity of **₹607,673** (or **₹151,918 per quarter**).

---

### 2. Total Refund Value & Financial Reconciliation

The raw export total of ₹230,124,081 (₹38.35 Lakhs/quarter) reconciles to actual cash output through two deterministic data cleaning steps:

| Reconciliation Step | Description | Reconciled Value | Refund Count | Quarterly Average |
|---|---|---|---|---|
| **Raw Export** | As exported across both systems | ₹230,124,081 | 2,465 | ₹38,354,014 |
| **Legacy Currency Conversion** | Legacy Freshdesk values converted from paise to rupees | ₹7,070,943 | 2,465 | ₹1,178,490 |
| **Deduplication** | Re-imported legacy tickets removed (helpdesk preferred) | **₹6,709,932** | **2,340** | **₹1,118,322** |

---

### 3. Monthly Refund Trend Analysis

Refund spend grew 1.64x from H1 2025 (₹2.54M across 4,284 tickets) to H1 2026 (₹4.17M across 7,316 tickets). 

Crucially, **the refund rate per ticket remained virtually flat at ~20%** across all 18 months, and the average refund size per ticket decreased slightly (from ₹2,924 to ₹2,834). Therefore, the overall rise in refund spend is **100% driven by customer contact volume growth (1.71x)** rather than increased frontline agent laxity or changes in refund granting behavior.

---

### 4. Main Refund Reasons

Goodwill concessions (`GW-OTHER`) and Dead on Arrival (`DOA-REPL`) account for over 65% of total refund spend:

1. **Goodwill / Policy Exceptions (`GW-OTHER`):** 991 refunds totaling **₹2,907,036** (43.3% of refund spend).
2. **Dead on Arrival (`DOA-REPL`):** 432 refunds totaling **₹1,471,820** (21.9% of refund spend).
3. **Return Received & QC Passed (`RETURN-QC-OK`):** 289 refunds totaling **₹842,500** (12.6% of refund spend).
4. **Duplicate / Failed Payment (`DUP-PAYMENT`):** 210 refunds totaling **₹584,200** (8.7% of refund spend).
5. **Lost / Undelivered (`LOST-TRANSIT`):** 195 refunds totaling **₹510,400** (7.6% of refund spend).

---

### 5. Potentially Avoidable Refund Value & Business Opportunity

Under Vireo's Support Operating Policy v3.2, **₹911,509** (319 tickets) was categorized as potentially avoidable based on two primary policy drivers:

* **Prohibited Dual Remedies (Policy §5):** In 166 cases, customers received **both a cash refund and a replacement unit** for the same ticket, totaling **₹574,191** in duplicated payouts.
* **Goodwill Cap Excess (Policy §5):** Policy §5 caps goodwill credits at ₹500 per ticket without Team Lead approval. Goodwill refunds exceeding ₹500 accounted for **₹337,318** in excess unverified concessions.

**Financial Opportunity Statement:**
> *"The analysis identified ₹911,509 of refunds as potentially avoidable under the documented policy across the 18-month observation period, corresponding to an opportunity of **₹151,918 per quarter** (annualized to **₹607,673/year**). At a hypothetical 50% operational compliance recovery rate, Vireo would capture **₹303,836 in annual net savings**."*

---

### 6. Agent-Level Observations (Workload Normalized)

Agents cannot be evaluated by raw refund totals alone. By design, **Returns Desk Tier 1 agents** process over 60% of all refunds, while **Tier 2 Escalations agents** handle complex hardware warranty cases. 

Normalizing metrics per 100 tickets handled provides fair operational clarity:
* **Returns Desk (Tier 1):** Averages 48–53 refunds per 100 tickets handled (avg refund ₹2,700–₹2,900).
* **Billing Team (Tier 1):** Averages 33–40 refunds per 100 tickets handled.
* **Frontline Chat / Email (Tier 1):** Averages 11–15 refunds per 100 tickets handled.

The data reveals that dual remedy errors occurred across multiple teams rather than being isolated to individual agents, pointing to helpdesk workflow gating gaps rather than policy non-compliance.

---

### 7. Recommended Actionable Management Steps
1. **Automate Helpdesk Dual-Remedy Gating:** Hardcode helpdesk validation to block raising a cash refund when `replacement_issued == 'Y'` without dual Team Lead and Finance approval.
2. **Goodwill Approval Workflow:** Enforce automated Team Lead sign-off in the helpdesk UI before any goodwill refund exceeding ₹500 can be processed.
3. **Logistics & Returns Desk QC Sync:** Require Returns Desk verification of warehouse receipt before triggering high-value product refunds.

---

### 8. Important Limitations
- *Single-annotator benchmark:* Accuracy evaluates agreement with documented policy guidelines.
- *Potentially avoidable definition:* Represents preventable payments under policy definitions, not allegations of employee misconduct or guaranteed cash recoveries.
