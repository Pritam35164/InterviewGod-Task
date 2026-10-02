"""
Data Loader & Cleaner for Vireo Audio Refund Analysis.
Deterministic parsing, currency conversion, deduplication, and relational joins.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

def get_data_path(data_dir: str = None) -> Path:
    if data_dir:
        return Path(data_dir)
    
    # Check default locations
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / "data" / "raw",
        base_dir / "data",
        base_dir.parent / "data",
        Path("d:/Activity"),
        Path("d:/Activity/vireo-ai-platform/data/raw")
    ]
    for c in candidates:
        if (c / "tickets.csv").exists():
            return c
    raise FileNotFoundError("Could not locate tickets.csv in any standard data directory.")

def load_raw_datasets(data_dir: str = None) -> dict[str, pd.DataFrame]:
    path = get_data_path(data_dir)
    return {
        "tickets": pd.read_csv(path / "tickets.csv"),
        "agents": pd.read_csv(path / "agents.csv"),
        "orders": pd.read_csv(path / "orders.csv"),
        "customers": pd.read_csv(path / "customers.csv"),
        "products": pd.read_csv(path / "products.csv")
    }

def clean_and_deduplicate(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    tickets = raw_data["tickets"].copy()
    agents = raw_data["agents"].copy()
    orders = raw_data["orders"].copy()
    customers = raw_data["customers"].copy()
    products = raw_data["products"].copy()

    # 1. Clean money: legacy_fd is stored in PAISE (cents), helpdesk is stored in INR (rupees)
    def clean_refund(row):
        val = row["refund_amount_inr"]
        if pd.isna(val) or val <= 0:
            return 0.0
        if str(row["source_system"]).lower() == "legacy_fd":
            return float(val) / 100.0
        return float(val)

    tickets["clean_refund_inr"] = tickets.apply(clean_refund, axis=1)

    # 2. Deduplicate tickets: prefer helpdesk over legacy_fd re-imports
    tickets_dedup = (
        tickets.sort_values(by=["ticket_id", "source_system"], ascending=[True, True])
        .drop_duplicates(subset=["ticket_id"], keep="first")
        .copy()
    )

    # 3. Parse timestamps
    for col in ["created_at", "first_response_at", "resolved_at"]:
        tickets_dedup[col] = pd.to_datetime(tickets_dedup[col], errors="coerce")

    # Fix legacy_fd resolved_at (+5:30 timezone offset from reconstructed UTC logs)
    is_legacy = tickets_dedup["source_system"] == "legacy_fd"
    tickets_dedup.loc[is_legacy, "resolved_at_fixed"] = tickets_dedup.loc[is_legacy, "resolved_at"] + pd.Timedelta(hours=5, minutes=30)
    tickets_dedup.loc[~is_legacy, "resolved_at_fixed"] = tickets_dedup.loc[~is_legacy, "resolved_at"]

    # Month string YYYY-MM
    tickets_dedup["month"] = tickets_dedup["created_at"].dt.strftime("%Y-%m")

    # Handle time (hours) & First response time (minutes)
    tickets_dedup["handle_time_hours"] = (tickets_dedup["resolved_at_fixed"] - tickets_dedup["created_at"]).dt.total_seconds() / 3600.0
    tickets_dedup["first_response_minutes"] = (tickets_dedup["first_response_at"] - tickets_dedup["created_at"]).dt.total_seconds() / 60.0

    # SLA targets in minutes: chat 15m, voice 120m, social 240m, email 480m
    sla_targets = {"chat": 15, "voice": 120, "social": 240, "email": 480}
    tickets_dedup["sla_target_minutes"] = tickets_dedup["channel"].map(sla_targets).fillna(480)
    tickets_dedup["is_sla_breach"] = tickets_dedup["first_response_minutes"] > tickets_dedup["sla_target_minutes"]
    tickets_dedup["sla_credit_inr"] = np.where(tickets_dedup["is_sla_breach"], 350.0, 0.0)

    # 4. Joins
    # Merge order info
    orders_clean = orders.drop_duplicates(subset=["order_id"])
    merged = tickets_dedup.merge(orders_clean, on="order_id", how="left", suffixes=("", "_order"))

    # Merge customer info
    customers_clean = customers.drop_duplicates(subset=["customer_id"])
    merged = merged.merge(customers_clean, on="customer_id", how="left", suffixes=("", "_cust"))

    # Merge product info (on product_sku or sku)
    products_clean = products.drop_duplicates(subset=["sku"])
    merged = merged.merge(products_clean, left_on="product_sku", right_on="sku", how="left", suffixes=("", "_prod"))

    # Merge agent roster (most recent row per agent_id)
    agents_latest = agents.sort_values(by=["agent_id", "from_date"]).groupby("agent_id").last().reset_index()
    merged = merged.merge(agents_latest, on="agent_id", how="left", suffixes=("", "_agent"))

    # Flag refund & dual remedy (refund + replacement)
    merged["is_refund"] = merged["clean_refund_inr"] > 0
    merged["is_dual_remedy"] = merged["is_refund"] & (merged["replacement_issued"].astype(str).str.upper() == "Y")

    return merged

if __name__ == "__main__":
    raw = load_raw_datasets()
    df = clean_and_deduplicate(raw)
    print(f"Loaded and cleaned {len(df)} unique tickets.")
    print(f"Total refund tickets: {df['is_refund'].sum()}")
    print(f"Total refund amount: ₹{df['clean_refund_inr'].sum():,.2f}")
