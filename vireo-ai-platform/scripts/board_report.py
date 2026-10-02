"""Board-ready outputs. Every figure comes from analysis.py / cost.py.

    python -m scripts.board_report
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.services import analysis  # noqa: E402
from app.services.cost import cost_report  # noqa: E402
from app.services.evaluation import eval_set_status, latest_evaluation  # noqa: E402

log = get_logger("board_report")


def _inr(n: float | int | None) -> str:
    if n is None:
        return "—"
    return f"₹{n:,.0f}"


def _pct(n: float | None) -> str:
    if n is None:
        return "—"
    return f"{n:.1f}%"


def render_markdown(ov: dict, monthly: dict, reasons: dict, agents: dict,
                    avoid: dict, cost: dict, evaluation: dict) -> str:
    recon = ov.get("reconciliation") or []
    recon_lines = [
        f"| {s['step']} | {s['label']} | {_inr(s['refund_value_inr'])} | "
        f"{s['refund_count']:,} | {_inr(s['per_quarter_inr'])} |"
        for s in recon
    ]
    monthly_lines = [
        f"| {r['month']} | {r['tickets']:,} | {r['refund_count']:,} | "
        f"{_inr(r['refund_value_inr'])} | {_pct(r['refund_rate_pct'])} | "
        f"{_inr(r['average_refund_inr'])} | {r['potentially_avoidable_count']:,} | "
        f"{_inr(r['potentially_avoidable_value_inr'])} |"
        for r in monthly.get("rows") or []
    ]
    reason_lines = [
        f"| {r['reason_code']} | {r['refund_count']:,} | {_inr(r['refund_value_inr'])} | "
        f"{_pct(r['value_share_pct'])} | {_inr(r['potentially_avoidable_value_inr'])} |"
        for r in (reasons.get("rows") or [])[:15]
    ]
    agent_lines = [
        f"| {r['agent_id']} | {r['agent_name_masked']} | {r['team']} | T{r['tier']} | "
        f"{r['tickets_handled']:,} | {r['refund_count']:,} | {_inr(r['refund_value_inr'])} | "
        f"{r['refunds_per_100_tickets']:.1f} | {_inr(r['refund_value_per_100_tickets_inr'])} | "
        f"{_inr(r['potentially_avoidable_value_inr'])} |"
        for r in (agents.get("rows") or [])[:20]
    ]
    driver_lines = [
        f"| {d['driver']} | {d.get('title', d['driver'])} | {d['refund_count']:,} | "
        f"{_inr(d['avoidable_value_inr'])} | §{d['policy_section_id']} |"
        for d in avoid.get("drivers") or []
    ]
    caveats = "\n".join(f"- {c}" for c in ov.get("caveats") or [])
    guards = "\n".join(f"- {g}" for g in agents.get("interpretation_guardrails") or [])
    arith = "\n".join(f"- {a}" for a in (cost.get("arithmetic") or ov.get("opportunity_arithmetic") or []))
    eval_notes = evaluation.get("notes") or ["No evaluation run recorded yet."]
    if isinstance(eval_notes, list):
        eval_block = "\n".join(f"- {n}" for n in eval_notes)
    else:
        eval_block = str(eval_notes)

    trend = monthly.get("trend") or {}
    trend_txt = json.dumps(trend, indent=2, default=str) if trend else "(see monthly table)"

    return f"""# Vireo Audio — Refund Intelligence Board Pack

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · Policy v{ov.get('policy_version')} · Internal

Customer names, phones, emails and addresses are not in this pack. Agents appear as IDs plus a masked name.

## Executive summary

| | |
|---|---|
| Period | {ov.get('period_start')} to {ov.get('period_end')} ({ov.get('months_observed')} months) |
| Tickets (deduplicated) | {ov.get('total_tickets'):,} |
| Refund tickets | {ov.get('refund_tickets'):,} |
| Refund rate | {_pct(ov.get('refund_rate_pct'))} |
| **Refund value (reconciled)** | **{_inr(ov.get('refund_value_inr'))}** |
| Average / median refund | {_inr(ov.get('average_refund_inr'))} / {_inr(ov.get('median_refund_inr'))} |
| Raw export (uncleaned) | {_inr(ov.get('raw_export_value_inr'))} |
| Potentially avoidable count | {ov.get('potentially_avoidable_count'):,} |
| **Potentially avoidable value** | **{_inr(ov.get('potentially_avoidable_value_inr'))}** |
| Share of refund value | {_pct(ov.get('potentially_avoidable_share_pct'))} |
| Quarterlyised opportunity | {_inr(ov.get('quarterlyised_opportunity_inr'))} |
| Annualised opportunity | {_inr(ov.get('annualised_opportunity_inr'))} |
| Classification coverage | {_pct(ov.get('classification_coverage_pct'))} |

"Potentially avoidable" means the evidence suggests the payment may have been preventable under the documented policy. It is **not** a finding of fraud or agent misconduct, and it is **not** a guaranteed saving.

### Why Finance's export and the helpdesk report disagreed

| Step | What changed | Value | Count | Per quarter |
|---|---|---|---|---|
{chr(10).join(recon_lines)}

Sameer's helpdesk figure of "around Rs 11 lakh a quarter" matches the reconciled per-quarter total. Arjun's "well over a crore a quarter" is the raw export, which mixes Freshdesk's native unit with rupees and double-counts re-imported tickets (policy §9).

## Monthly

| Month | Tickets | Refunds | Value | Rate | Avg refund | Avoidable n | Avoidable value |
|---|---|---|---|---|---|---|---|
{chr(10).join(monthly_lines)}

### Trend (volume vs rate vs ticket size)

```json
{trend_txt}
```

The refund **rate** is roughly flat. The increase in rupees is contact volume, not a sudden change in how often a ticket pays out.

## Reasons

Basis: `{reasons.get('basis')}` — {reasons.get('scope_note') or ''}
Dropdown codes are what the agent selected. `GW-OTHER` is the first option in that dropdown and is not a usable category. Until Nemotron classifies tickets, this table is the source dropdown. After classification, the pack also stores `reasons_normalised_ai` in board_pack.json.

| Reason | Count | Value | Value share | Potentially avoidable value |
|---|---|---|---|---|
{chr(10).join(reason_lines)}

Miscoding vs the dropdown: {json.dumps(reasons.get('miscoding') or {{}}, default=str)[:1200]}

## Agents

Policy §6: the Returns Desk processes most refunds **by design**. Tier 2 is **not** comparable on volume. Rank by refunds per 100 tickets inside a peer group, not by raw rupees.

{guards}

| Agent | Name | Team | Tier | Tickets | Refunds | Value | Refunds/100 | Value/100 | Avoidable value |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(agent_lines)}

## Potentially avoidable

Observed avoidable value {_inr(avoid.get('total_avoidable_value_inr'))} over {avoid.get('months_observed')} months.

{avoid.get('overlap_note', '')}

| Driver | What it is | Tickets | Avoidable value | Policy |
|---|---|---|---|---|
{chr(10).join(driver_lines)}

Quarterlyised {_inr(avoid.get('quarterlyised_opportunity_inr'))}. Annualised {_inr(avoid.get('annualised_opportunity_inr'))}.

Scenarios below are **labelled scenarios**, not forecasts:

{json.dumps(avoid.get('scenarios') or [], indent=2, default=str)}

## Evaluation

{eval_block}

Sample: {evaluation.get('sample_size') or eval_set_status().get('reviewed') or 0}. Method: {evaluation.get('method') or 'not yet run against model predictions'}.

| Metric | Value |
|---|---|
| Reason accuracy | {evaluation.get('reason_accuracy')} |
| Avoidability accuracy | {evaluation.get('avoidability_accuracy')} |
| Schema failure rate | {evaluation.get('schema_failure_rate')} |
| Retry rate | {evaluation.get('retry_rate')} |
| UNKNOWN rate | {evaluation.get('unknown_rate')} |
| Human escalation rate | {evaluation.get('human_escalation_rate')} |
| Baseline reason accuracy | {evaluation.get('baseline_reason_accuracy')} |

Error categories: {json.dumps(evaluation.get('error_categories') or [], default=str)[:1500]}

## Cost

{chr(10).join(f'- {c}' for c in (cost.get('caveats') or ['No LLM calls recorded yet.']))}

{arith}

Development API cost is the sum of `llm_requests` in this database. If the account was on free credits, cash outlay is ₹0; the token-priced equivalent is still reported.

## Caveats

{caveats}

## How to read the numbers

- Python computed every total. Nemotron never added two rupees together.
- The policy engine, not Laya and not the LLM, authorises any outbound refund.
- Unclassified tickets contribute **zero** avoidable value, so the avoidable figure is a floor.
"""


def write_board_pack() -> list[str]:
    configure_logging()
    out_dir = settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with session_scope() as session:
        ov = analysis.overview(session)
        monthly = analysis.monthly(session)
        reasons_ai = analysis.reasons(session, basis="normalised_ai")
        reasons_src = analysis.reasons(session, basis="source_dropdown")
        # Until Nemotron has classified tickets, the dropdown is the only reason
        # table that exists. Both are stored in the JSON pack.
        reasons = reasons_ai if (reasons_ai.get("rows") or []) else reasons_src
        agents = analysis.agents(session, include_names=False)
        avoid = analysis.avoidable(session)
        lots = analysis.lot_analysis(session)
        cost = cost_report(session)
        evaluation = latest_evaluation(session)

    md = render_markdown(ov, monthly, reasons, agents, avoid, cost, evaluation)
    md_path = out_dir / "board_pack.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = out_dir / "board_pack.json"
    json_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "overview": ov,
                "monthly": monthly,
                "reasons": reasons,
                "reasons_source_dropdown": reasons_src,
                "reasons_normalised_ai": reasons_ai,
                "agents": agents,
                "avoidable": avoid,
                "lots": lots,
                "cost": cost,
                "evaluation": evaluation,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    xlsx_path = out_dir / "board_pack.xlsx"
    try:
        import pandas as pd

        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            pd.DataFrame(ov.get("reconciliation") or []).to_excel(writer, sheet_name="reconciliation", index=False)
            pd.DataFrame(monthly.get("rows") or []).to_excel(writer, sheet_name="monthly", index=False)
            pd.DataFrame(reasons.get("rows") or []).to_excel(writer, sheet_name="reasons", index=False)
            pd.DataFrame(agents.get("rows") or []).to_excel(writer, sheet_name="agents", index=False)
            pd.DataFrame(avoid.get("drivers") or []).to_excel(writer, sheet_name="avoidable_drivers", index=False)
            pd.DataFrame(
                [{k: v for k, v in ov.items() if k not in {"reconciliation", "data_quality_findings", "caveats"}}]
            ).to_excel(writer, sheet_name="overview", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("board.xlsx_failed", extra={"error": str(exc)[:200]})
        xlsx_path = None

    written = [str(md_path), str(json_path)]
    if xlsx_path and xlsx_path.exists():
        written.append(str(xlsx_path))
    form_path = write_google_form_answers(ov, monthly, reasons, agents, avoid, cost, evaluation)
    written.append(str(form_path))
    log.info("board.written", extra={"files": written})
    return written


def write_google_form_answers(
    ov: dict, monthly: dict, reasons: dict, agents: dict,
    avoid: dict, cost: dict, evaluation: dict,
) -> Path:
    """Plain-text answers for the assignment Google Form. Not submission-form.md.

    Every money figure is taken from the loaded database. Hours and the Drive
    recording are not invented when they were not measured.
    """
    recon = ov.get("reconciliation") or []
    unit = next((s for s in recon if s.get("step") == 2), {})
    dedup = next((s for s in recon if s.get("step") == 3), {})
    top_reasons = (reasons.get("rows") or [])[:5]
    reason_txt = ", ".join(
        f"{r['reason_code']} ₹{r['refund_value_inr']:,.0f} ({r.get('value_share_pct', 0):.1f}%)"
        for r in top_reasons
    ) or "no reason rows yet (run classification)"
    eval_status = evaluation.get("status") or "not_run"
    eval_n = evaluation.get("sample_size") or (evaluation.get("eval_set") or {}).get("reviewed")
    eval_acc = evaluation.get("reason_accuracy")
    eval_err = evaluation.get("error_rate")
    eval_method = evaluation.get("method") or evaluation.get("message") or "not run"
    if cost.get("status") == "no_llm_calls_recorded":
        cost_one = (
            "No LLM requests are recorded in this database, so there is no measured "
            "per-ticket cost. Development cash outlay for OpenRouter = ₹0 (no calls). "
            "A monthly projection at 650 tickets/week × 52 / 12 ≈ 2,817 tickets/month "
            "cannot be computed until a classification job has measured tokens per ticket. "
            f"{cost.get('message', '')}"
        )
    else:
        arith = "\n".join(cost.get("arithmetic") or [])
        cost_one = (
            f"Measured development token-priced cost = ₹{cost.get('development_cost_inr', 0):,.2f} "
            f"(${cost.get('development_cost_usd', 0):.4f}). "
            f"If the account used free credits, cash outlay = ₹0; the figure above is the "
            f"rate-card equivalent. Cost per classified ticket = ₹{cost.get('cost_per_classified_ticket_inr', 0):.4f}. "
            f"Projected monthly (refunds only) = ₹{cost.get('projected_monthly_cost_inr_refunds_only', 0):,.2f}. "
            f"Projected monthly (all tickets) = ₹{cost.get('projected_monthly_cost_inr_all_tickets', 0):,.2f}. "
            f"Arithmetic:\n{arith}"
        )

    months = max(1, ov.get("months_observed") or 1)
    rec_val = ov.get("refund_value_inr") or 0
    text = f"""GOOGLE FORM ANSWERS
Generated from the loaded database. Do not submit if these figures disagree with outputs/board_pack.json.

1. What did you build, and what business outcome does it move?
A Refund Intelligence and Support Decision Platform for Vireo Audio. It turns the messy helpdesk export into an auditable monthly refund view: how much, why, who handled it, what looks preventable under support-policy.pdf v3.2, and what the model actually got right.

Numbers from this run:
- Period: {ov.get('period_start')} to {ov.get('period_end')} ({ov.get('months_observed')} months)
- Deduplicated tickets: {ov.get('total_tickets'):,}
- Refund tickets: {ov.get('refund_tickets'):,} ({ov.get('refund_rate_pct')}%)
- Reconciled refund value: ₹{rec_val:,.0f}
- Raw export (uncleaned): ₹{ov.get('raw_export_value_inr'):,.0f}
- After unit conversion: ₹{unit.get('refund_value_inr', 0):,.0f}
- After dedup: ₹{dedup.get('refund_value_inr', rec_val):,.0f}
- Per quarter (reconciled): ₹{rec_val / months * 3:,.0f}
- Potentially avoidable (deterministic + classified floor): ₹{ov.get('potentially_avoidable_value_inr'):,.0f} across {ov.get('potentially_avoidable_count'):,} tickets
- Quarterlyised observed avoidable: ₹{ov.get('quarterlyised_opportunity_inr'):,.0f}
Top reasons: {reason_txt}

The business outcome is that Finance can drop a reconciled number into the board pack instead of arguing about whether refunds are ₹1 crore or ₹11 lakh a quarter. The rise is volume, not rate — see outputs/board_pack.md.

2. What does one run cost, and what would a month cost at approximately 650 tickets/week?
{cost_one}

3. How do you know it works?
- Sample: {eval_n}
- Status: {eval_status}
- Methodology: {eval_method}
- Reason accuracy: {eval_acc}
- Error rate: {eval_err}
- Error types: {json.dumps(evaluation.get('error_categories') or [], default=str)[:1500]}
- Schema failure rate: {evaluation.get('schema_failure_rate')}
- Retry rate: {evaluation.get('retry_rate')}
- UNKNOWN rate: {evaluation.get('unknown_rate')}
- Human escalation rate: {evaluation.get('human_escalation_rate')}
If status is not_run, accuracy has not been measured and must not be claimed. Labels in data/evaluation_set.json were assigned from ticket text using the rubric in scripts/build_eval_set.py by a single annotator who also wrote the taxonomy. There is no inter-annotator agreement figure.

Unit tests (pytest) cover ingest unit-detection and dedup, money aggregation, policy DENY/fail-closed, Pydantic schema, Laya fallback routing, cache namespacing, LLM retry on invalid JSON (OpenRouter mocked), idempotency, and rate limiting.

4. Did you change, narrow, or push back on the client's ask?
Yes, in three places:
- The export total is not the board number. Policy §9 plus paired tickets show legacy money is 100× rupees, and re-imported ticket_ids were counted twice. Reporting the raw sum would recreate the crore-a-quarter figure, which is a unit/dedup artefact.
- Agent ranking is peer-grouped by team and tier, with refunds per 100 tickets. Policy §6 makes the Returns Desk the refund processor by design; a raw "who is giving away money" table rediscovers the org chart.
- "Potentially avoidable" is a policy-process label, not misconduct. There is no approval-audit field, so goodwill-over-cap is "no approval evidence", not "the agent cheated". DOA's 7-day window is not verifiable (no delivered_at) and is never used to flag avoidability.

5. What is wrong with what you are handing us?
- No Nemotron classification until OPENROUTER_API_KEY is set and a classification job is run. Avoidable value without that job is a floor from deterministic drivers only (dual remedy flag, duplicate order refunds).
- Evaluation is single-annotator. Accuracy, if measured, is agreement with that reading, not ground truth.
- replacement_issued and the closing note disagree in both directions; dual-remedy uses the union and over-flags.
- Order linking is fallback-based when the customer did not quote an order_id; duplicate-order detection ignores unreliable links.
- Laya temperatures are uncalibrated; the fallback is deliberately blunt (money → human).
- There is no payment gateway. ALLOW writes an audit row; it does not move money.
- Semantic cache embeddings in tests are stubs; production uses Laya embeddings.
- Streamlit is an analyst UI, not a customer channel.
- Hours were not tracked with a timer (see Q11). The Drive recording does not exist yet (see Q9).

6. What did you deliberately leave out, and why?
Kubernetes, Kafka, a vector database, a fake payment integration, predictive refund forecasting, autonomous refunds, agent disciplinary scoring, a custom frontend, and multi-agent orchestration. Docker Compose is the deployment. A short policy PDF does not justify a vector store. The brief forbids fake integrations and fabricated metrics.

7. Anything you built/found that nobody asked for?
Manufacturing-lot refund rates (lot_code on the box, joined through orders) with a two-proportion z-test and a Bonferroni threshold — a hardware-quality lead, not a support finding. Also: measuring the Freshdesk unit from paired tickets instead of assuming paise; GW-OTHER as a dropdown default; orders refunded on more than one ticket.

8. What did you use AI for?
- NVIDIA Nemotron Ultra via OpenRouter: ticket reading, reason classification, evidence extraction, policy-aware language. Never totals, never authorisation. Not called during health checks. Not called unless a classification job is run.
- Laya: System-1 routing, risk, cache eligibility, human gate, encoder embeddings for the semantic cache. Never customer-facing prose.
- Python: ingest, joins, money, aggregations, avoidability rules.
- Redis: cache, rate limit, idempotency fast path, job queue.
- Cursor Grok (this codebase): implementation in Cursor. Useful direction: keep money in pandas, keep authorisation in the policy engine, measure the Freshdesk unit from paired tickets. Discarded: sending the whole finance question to the LLM; standing up a vector DB for the policy PDF.

9. Public Google Drive recording link
Not recorded yet. Do not paste a fabricated URL. Record docker compose up, /health, inbound Laya decision, cache hit, outbound dual-remedy DENY, dashboard, and the injection-blocked edge case, then put the real link here.

10. Someone picks this up Monday and you are unreachable. Three things:
1. docker compose up --build, then docker compose exec api python -m scripts.bootstrap. Board pack lands in outputs/. Without OPENROUTER_API_KEY the money still reconciles; tickets stay unclassified.
2. Never quote tickets.csv refund_amount_inr as rupees. Read data_quality_events DQ-000/DQ-001/DQ-002 first. Policy engine wins over Laya and over the LLM; high-risk actions fail closed if policy is missing.
3. INTERNAL_API_KEYS must be set or internal endpoints fail closed. Streamlit talks only to FastAPI. Do not treat Returns Desk volume as agent misconduct (policy §6).

11. Honest hours spent.
Not tracked with a timer. I will not invent a single number. If the form requires one integer, leave it blank rather than fabricating it.

---
Reconciliation walk this run:
{json.dumps(recon, indent=2, default=str)[:4000]}
"""
    out = settings.output_dir / "GOOGLE_FORM_ANSWERS.txt"
    out.write_text(text, encoding="utf-8")
    return out



def main() -> int:
    for p in write_board_pack():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
