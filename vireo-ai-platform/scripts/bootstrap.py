"""Load CSVs, apply deterministic rules, write the board pack.

    python -m scripts.bootstrap

Inside compose:

    docker compose exec api python -m scripts.bootstrap
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging, get_logger  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.services.classification import apply_deterministic_avoidability  # noqa: E402
from app.services.ingest import run_ingest  # noqa: E402

log = get_logger("bootstrap")


def main() -> int:
    configure_logging()
    from app.core.config import settings
    from app.db.base import Base
    from app.db.session import get_engine, session_scope
    import app.models  # noqa: F401

    if settings.DATABASE_URL.startswith("sqlite"):
        Base.metadata.create_all(get_engine())

    log.info("bootstrap.start", extra={"database": settings.DATABASE_URL.split("@")[-1]})
    with session_scope() as session:
        ingest = run_ingest(session, truncate=True)
        avoid = apply_deterministic_avoidability(session)
        session.commit()

    from scripts.board_report import write_board_pack

    pack = write_board_pack()
    print("INGEST")
    print(f"  tickets     {ingest.counts.get('tickets_loaded')}")
    print(f"  refunds     {ingest.counts.get('refunds_loaded')}")
    raw = ingest.money.get("raw_export_total_inr") or 0
    rec = ingest.money.get("after_dedup_inr") or 0
    q = ingest.money.get("reconciled_per_quarter_inr") or 0
    print(f"  raw export  Rs {raw:,.0f}")
    print(f"  reconciled  Rs {rec:,.0f}")
    print(f"  per quarter Rs {q:,.0f}")
    print("AVOIDABILITY (deterministic pass, no LLM)")
    print(f"  flagged     {avoid['flagged']}")
    print(f"  value       Rs {avoid['avoidable_value_inr']:,.0f}")
    print("BOARD PACK")
    for path in pack:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
