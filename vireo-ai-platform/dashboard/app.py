"""Internal analyst dashboard. Talks only to FastAPI.

No database driver and no LLM client are imported here. That is the enforcement
of the architecture, not a comment: this process physically cannot write a
PostgreSQL row or call OpenRouter.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DASHBOARD_API_KEY") or os.getenv("INTERNAL_API_KEYS", "dev-internal-key-change-me")
if "," in API_KEY:
    API_KEY = API_KEY.split(",", 1)[0].strip()

st.set_page_config(page_title="Vireo Refund Intelligence", layout="wide", page_icon="V")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=API_BASE,
        headers={"X-API-Key": API_KEY, "Accept": "application/json"},
        timeout=60.0,
    )


@st.cache_data(ttl=30, show_spinner=False)
def api_get(path: str, params: dict | None = None) -> dict:
    with _client() as c:
        r = c.get(path, params=params or {})
        r.raise_for_status()
        return r.json()


@st.cache_data(ttl=15, show_spinner=False)
def api_get_public(path: str) -> dict:
    with httpx.Client(base_url=API_BASE, timeout=10.0) as c:
        r = c.get(path)
        r.raise_for_status()
        return r.json()


def inr(n: Any) -> str:
    try:
        return f"₹{float(n):,.0f}"
    except (TypeError, ValueError):
        return "—"


st.sidebar.title("Vireo Refund Intelligence")
st.sidebar.caption(f"API `{API_BASE}`")
page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Monthly Refunds",
        "Refund Reasons",
        "Agent Analysis",
        "Potentially Avoidable",
        "Ticket Evidence",
        "Evaluation",
        "AI / Cost Metrics",
    ],
)

try:
    health = api_get_public("/health")
    st.sidebar.success(f"API {health.get('status')} · policy v{health.get('policy_version')}")
except Exception as exc:
    st.sidebar.error(f"API unreachable: {exc}")
    st.stop()


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Board overview")
    ov = api_get("/v1/analytics/overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reconciled refunds", inr(ov["refund_value_inr"]), f"{ov['refund_tickets']:,} tickets")
    c2.metric("Refund rate", f"{ov['refund_rate_pct']:.1f}%")
    c3.metric("Potentially avoidable", inr(ov["potentially_avoidable_value_inr"]))
    c4.metric("Quarterlyised opportunity", inr(ov["quarterlyised_opportunity_inr"]))

    st.caption(
        f"{ov['period_start']} – {ov['period_end']} · {ov['total_tickets']:,} tickets · "
        f"raw export {inr(ov['raw_export_value_inr'])} · classification coverage "
        f"{ov['classification_coverage_pct']:.1f}%"
    )

    st.subheader("Why the export and the helpdesk report disagreed")
    recon = pd.DataFrame(ov.get("reconciliation") or [])
    if not recon.empty:
        st.dataframe(recon, use_container_width=True, hide_index=True)

    st.warning(
        "Potentially avoidable means the evidence suggests the payment may have been "
        "preventable under the documented policy. It is not a finding of fraud or misconduct."
    )
    for c in ov.get("caveats") or []:
        st.markdown(f"- {c}")

# ---------------------------------------------------------------------------
# Monthly
# ---------------------------------------------------------------------------
elif page == "Monthly Refunds":
    st.title("Monthly refunds")
    data = api_get("/v1/analytics/monthly")
    df = pd.DataFrame(data.get("rows") or [])
    if df.empty:
        st.info("No monthly data. Run ingest first.")
    else:
        fig = px.bar(df, x="month", y="refund_value_inr", title="Refund value by month")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.line(df, x="month", y="refund_rate_pct", title="Refund rate (%)")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if data.get("trend"):
            st.subheader("Trend decomposition")
            st.json(data["trend"])

# ---------------------------------------------------------------------------
# Reasons
# ---------------------------------------------------------------------------
elif page == "Refund Reasons":
    st.title("Refund reasons")
    basis = st.radio("Basis", ["normalised_ai", "source_dropdown"], horizontal=True)
    data = api_get("/v1/analytics/reasons", {"basis": basis})
    df = pd.DataFrame(data.get("rows") or [])
    if df.empty:
        st.info("No reason rows yet.")
    else:
        fig = px.bar(df, x="reason_code", y="refund_value_inr", title="Value by reason")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
        if data.get("miscoding"):
            st.subheader("Dropdown vs normalised")
            st.json(data["miscoding"])

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
elif page == "Agent Analysis":
    st.title("Agent analysis")
    for g in api_get("/v1/analytics/agents").get("interpretation_guardrails") or []:
        st.info(g)
    data = api_get("/v1/analytics/agents")
    df = pd.DataFrame(data.get("rows") or [])
    if df.empty:
        st.info("No agent rows.")
    else:
        view = df.sort_values("refunds_per_100_tickets", ascending=False)
        st.dataframe(
            view[
                [
                    "agent_id", "agent_name_masked", "team", "tier", "tickets_handled",
                    "refund_count", "refund_value_inr", "refunds_per_100_tickets",
                    "refund_value_per_100_tickets_inr", "potentially_avoidable_value_inr",
                    "peer_group", "comparable_on_volume",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# Avoidable
# ---------------------------------------------------------------------------
elif page == "Potentially Avoidable":
    st.title("Potentially avoidable refunds")
    data = api_get("/v1/analytics/avoidable")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observed", inr(data.get("total_avoidable_value_inr")))
    c2.metric("Quarterlyised", inr(data.get("quarterlyised_opportunity_inr")))
    c3.metric("Tickets", f"{data.get('total_avoidable_count', 0):,}")
    st.caption(data.get("overlap_note", ""))
    df = pd.DataFrame(data.get("drivers") or [])
    if not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Scenarios (not forecasts)")
    for s in data.get("scenarios") or []:
        st.markdown(
            f"- **{s.get('label')}**: quarterly {inr(s.get('quarterly_inr'))} · "
            f"{s.get('caveat', '')}"
        )
    for c in data.get("caveats") or []:
        st.markdown(f"- {c}")

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
elif page == "Ticket Evidence":
    st.title("Ticket evidence")
    avoidable_only = st.checkbox("Potentially avoidable only", value=True)
    data = api_get("/v1/analytics/evidence", {"avoidable_only": avoidable_only, "limit": 50})
    st.caption(data.get("privacy_note", ""))
    rows = data.get("rows") or []
    if not rows:
        st.info("No evidence rows. Classify refund tickets first, or uncheck the filter to see structure.")
    for row in rows:
        with st.expander(
            f"{row['ticket_id']} · {inr(row['refund_value_inr'])} · {row['reason_code']} · {row['potentially_avoidable']}"
        ):
            st.write(
                {
                    "month": row.get("month"),
                    "agent_id": row.get("agent_id"),
                    "team": row.get("team"),
                    "source_reason_code": row.get("source_reason_code"),
                    "confidence": row.get("confidence"),
                    "policy": row.get("policy_citations") or row.get("policy_section_ids"),
                    "drivers": row.get("avoidability_drivers"),
                    "model": row.get("model_name"),
                    "prompt": row.get("prompt_version"),
                }
            )
            st.json(row.get("evidence") or [])

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Evaluation")
    data = api_get("/v1/analytics/evaluation")
    if data.get("status") in {"not_run", "no_evaluation_run"} or not data.get("sample_size"):
        st.warning(data.get("message") or "No evaluation run stored yet.")
        st.json(data)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample", data.get("sample_size") or "—")
    c2.metric("Reason accuracy", data.get("reason_accuracy") or "—")
    c3.metric("Avoidability accuracy", data.get("avoidability_accuracy") or "—")
    c4.metric("Schema failures", data.get("schema_failure_rate") or "—")
    st.write(data.get("method"))
    if data.get("error_categories"):
        st.subheader("Error categories")
        st.dataframe(pd.DataFrame(data["error_categories"]), use_container_width=True, hide_index=True)
    if data.get("notes"):
        st.subheader("Notes")
        for n in data["notes"]:
            st.markdown(f"- {n}")

# ---------------------------------------------------------------------------
# Cost / AI
# ---------------------------------------------------------------------------
else:
    st.title("AI / cost metrics")
    metrics = api_get_public("/metrics")
    cost = api_get("/v1/analytics/cost")
    st.subheader("Cost")
    st.json(cost)
    st.subheader("Live counters and latency")
    st.json(metrics.get("derived") or {})
    st.json(metrics.get("latency") or {})
    st.subheader("Cache")
    try:
        st.json(api_get("/v1/admin/cache"))
    except Exception as exc:
        st.caption(str(exc))
    st.caption(
        "Health checks never call Nemotron. Cache is rebuildable. Laya decisions are "
        "advisory; the policy engine authorises money movement."
    )
