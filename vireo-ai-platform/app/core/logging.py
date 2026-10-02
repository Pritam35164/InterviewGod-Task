"""Structured JSON logging with request/trace correlation.

Two rules enforced here rather than left to reviewers:
  * API keys never reach a log record (`_REDACT_KEYS` + `scrub`).
  * Customer PII is not logged. `scrub` drops the fields that carry it and
    ticket/customer identifiers are logged as IDs only.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from app.core.config import settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}

# Anything whose key looks like a secret is replaced wholesale.
_REDACT_KEYS = re.compile(
    r"(api[_-]?key|authorization|secret|token|password|bearer|openrouter)", re.I
)
# Fields that carry customer PII and must not be written to logs.
_PII_KEYS = {
    "name", "customer_name", "email", "phone", "address", "city_full",
    "customer_message", "agent_notes", "message", "text", "prompt",
}
_LONG_TEXT_MAX = 0  # free text is never logged, only its length


def scrub(value: Any, _depth: int = 0) -> Any:
    """Recursively redact secrets and strip PII/free text from a log payload."""
    if _depth > 6:
        return "<truncated>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if _REDACT_KEYS.search(key):
                out[key] = "<redacted>"
            elif key.lower() in _PII_KEYS:
                out[key] = f"<omitted:{len(v)} chars>" if isinstance(v, str) else "<omitted>"
            else:
                out[key] = scrub(v, _depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(v, _depth + 1) for v in value][:50]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
            "trace_id": trace_id_var.get(),
            "service": settings.APP_NAME,
            "env": settings.APP_ENV,
        }
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if extras:
            payload.update(scrub(extras))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)[:4000]
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        tail = ""
        if extras:
            tail = " " + " ".join(f"{k}={v}" for k, v in scrub(extras).items())
        rid = request_id_var.get()
        return f"{record.levelname:<7} [{rid[:8]}] {record.name}: {record.getMessage()}{tail}"


_configured = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.LOG_JSON else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.LOG_LEVEL)
    for noisy in ("httpx", "httpcore", "urllib3", "sqlalchemy.engine.Engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def new_request_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def correlation(request_id: str | None = None, trace_id: str | None = None) -> Iterator[str]:
    rid = request_id or new_request_id()
    tid = trace_id or rid
    t1 = request_id_var.set(rid)
    t2 = trace_id_var.set(tid)
    try:
        yield rid
    finally:
        request_id_var.reset(t1)
        trace_id_var.reset(t2)


def current_request_id() -> str:
    return request_id_var.get()


def current_trace_id() -> str:
    return trace_id_var.get()
