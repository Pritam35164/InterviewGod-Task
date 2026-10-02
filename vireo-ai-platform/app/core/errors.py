"""Application error taxonomy mapped onto HTTP status codes."""

from __future__ import annotations

from typing import Any


class VireoError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_payload(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigurationError(VireoError):
    status_code = 500
    code = "configuration_error"


class RateLimitExceeded(VireoError):
    status_code = 429
    code = "rate_limit_exceeded"

    def __init__(self, message: str, retry_after: int = 60, **details: Any) -> None:
        super().__init__(message, retry_after=retry_after, **details)
        self.retry_after = retry_after


class Unauthorized(VireoError):
    status_code = 401
    code = "unauthorized"


class Forbidden(VireoError):
    status_code = 403
    code = "forbidden"


class NotFound(VireoError):
    status_code = 404
    code = "not_found"


class ValidationFailure(VireoError):
    status_code = 422
    code = "validation_failure"


class IdempotencyConflict(VireoError):
    status_code = 409
    code = "idempotency_conflict"


class DependencyUnavailable(VireoError):
    status_code = 503
    code = "dependency_unavailable"


class LLMError(VireoError):
    """OpenRouter/Nemotron transport or protocol failure."""

    status_code = 502
    code = "llm_error"


class LLMValidationError(LLMError):
    """Nemotron replied, but the payload never satisfied the Pydantic schema."""

    status_code = 502
    code = "llm_schema_invalid"


class PolicyUnavailable(VireoError):
    """Policy could not be loaded. High-risk actions must not proceed."""

    status_code = 503
    code = "policy_unavailable"
