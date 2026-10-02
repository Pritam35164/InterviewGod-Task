"""Score the reviewed evaluation set against stored classifications.

    python -m scripts.run_evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import configure_logging  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.services.evaluation import run_evaluation  # noqa: E402


def main() -> int:
    configure_logging()
    with session_scope() as session:
        result = run_evaluation(session)
        session.commit()
    print(json.dumps(result, indent=2, default=str)[:8000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
