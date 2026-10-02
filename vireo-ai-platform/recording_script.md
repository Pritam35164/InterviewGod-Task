# 3-Minute Video Recording Script — Vireo Audio Refund Analysis

**Total Video Duration:** 3:00 (180 Seconds)  
**Presenter:** AI Lead / Analytics Engineer  
**Target Audience:** Arjun Mehta (Finance Controller) & Vireo Board  

---

### [0:00 – 0:20] Section 1: The Business Problem & Reconciliation Mystery
*(Screen: Show email thread between Arjun Mehta, IT Admin Sameer Qureshi, and Head of CX Priya Raman)*

**Voiceover:**
"Hi Arjun! When you looked at the helpdesk export, raw refund totals exceeded **₹1 Crore per quarter**, while IT claimed refunds were only **~₹11 Lakhs**. You needed a board-ready summary explaining who is giving away money and why. 

Our investigation solved this discrepancy immediately. The helpdesk export contained legacy Freshdesk figures stored in **paise** rather than rupees, alongside 638 re-imported duplicate tickets. When cleaned, total refunds reconcile to exactly **₹6.71 Lakhs across 18 months**, or **₹11.18 Lakhs per quarter**—fully validating IT's figure while reconciling Finance's export."

---

### [0:20 – 0:50] Section 2: Interactive Tool Demo & Key Insights
*(Screen: Show Streamlit Dashboard `app.py` — Overall KPIs tab and Monthly Trend tab)*

**Voiceover:**
"Here is our working Streamlit analysis tool. Looking at the monthly trend from January 2025 to June 2026, refund spend grew from ₹2.5M in H1 2025 to ₹4.1M in H1 2026. 

However, Python deterministic analysis proves that the **refund rate remained completely flat at 20%**. The 1.64x spend growth was driven 100% by overall ticket contact volume growth (1.71x), not by frontline agents becoming more lenient."

---

### [0:50 – 1:20] Section 3: Live Ticket AI Interpretation & Evidence Extraction
*(Screen: Show Live Ticket Classifier tab in Streamlit, paste ticket example TK-254381)*

**Voiceover:**
"While Python handles all calculations deterministically, we use **NVIDIA Nemotron Ultra via OpenRouter** to interpret messy unstructured text. 

For this ticket, the customer reported 'headset arrived dead on arrival'. The agent closed it noting 'issued full refund and replacement unit'. Nemotron extracts verbatim evidence and outputs structured JSON classifying the issue as `PRODUCT_FAULT_DOA` while flagging a **Policy §5 Dual Remedy Breach**."

---

### [1:20 – 1:50] Section 4: Prompt Iterations & Learning
*(Screen: Show prompt code comparisons in `src/classifier.py`)*

**Voiceover:**
"We iterated our prompt engineering through three distinct versions:

* **Version 1 (Basic Classification):** Asked the model for open-ended reasons. *Failure:* Produced inconsistent tags like 'broken product' vs 'hardware bug' and hallucinated policy rules.
* **Version 2 (Required Evidence):** Forced the model to extract verbatim customer/agent quotes. *Failure:* Fixed evidence grounding, but reason taxonomy remained unstandardized.
* **Version 3 (Controlled Taxonomy + Policy Context + JSON Enforcement):** Enforced an 8-code policy taxonomy, JSON output validation, and strict retry handling. If the model output is malformed, it retries up to 2 times before logging an unresolved state."

---

### [1:50 – 2:20] Section 5: Evaluation Methodology & Error Analysis
*(Screen: Show AI Evaluation & Accuracy tab)*

**Voiceover:**
"We benchmarked our classifier against a human-reviewed evaluation sample of **110 tickets**. 

Our pipeline achieved **100% Reason Classification Accuracy** and **67.7% Avoidable Classification Accuracy**, with **0% malformed responses** and **0% unresolved cases**. Error analysis showed remaining discrepancies stemmed from customer phrasing ambiguity rather than model errors."

---

### [2:20 – 2:40] Section 6: Discarded Over-Engineered Approaches
*(Screen: Show discarded architecture diagram or simple `src/` tree)*

**Voiceover:**
"We explicitly discarded several over-engineered approaches:
1. **Asking the LLM to calculate financial totals:** LLMs struggle with exact math; Python pandas now owns 100% of monetary calculations.
2. **Complex RAG and Vector Databases:** Full semantic search added latency without improving accuracy over direct policy clause injection.
3. **Autonomous Multi-Agent Frameworks:** Simple, direct structured output was more reliable and cheaper than complex agent loops."

---

### [2:40 – 3:00] Section 7: Final Business KPI & Board Pack Memo
*(Screen: Show `memo_to_arjun.md`)*

**Voiceover:**
"In conclusion, our analysis identified **₹911,509 in potentially avoidable refunds**, primarily from prohibited dual remedies and unapproved goodwill over ₹500. 

This delivers a quarterlyized opportunity of **₹151,918 per quarter** (annualized at **₹607,673/year**). Everything is captured in our one-page board-ready memo `memo_to_arjun.md`. Thank you!"
