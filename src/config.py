"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass
class Settings:
    """Settings for the agent, sandbox, LLM and monitoring components.

    Every value can be overridden with an environment variable so the same
    image can run locally and on ECS without code changes.
    """

    # Browser sandbox
    sandbox_url: str = ""                       # empty -> local Playwright fallback
    sandbox_timeout_seconds: int = 20
    screenshot_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "screenshots")

    # Reasoning layer
    llm_provider: str = "rule_based"            # "rule_based" | "bedrock"
    bedrock_model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    aws_region: str = "us-east-1"

    # Model registry
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    params_path: Path = field(default_factory=lambda: PROJECT_ROOT / "params.yaml")

    # Verdict thresholds
    phishing_threshold: float = 0.65
    suspicious_threshold: float = 0.35

    # Monitoring
    monitoring_enabled: bool = True
    events_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "monitoring" / "events.jsonl"
    )
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    psi_threshold: float = 0.2

    # Safety
    block_private_networks: bool = True
    allow_downloads: bool = False
    headless: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.suspicious_threshold < self.phishing_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= suspicious_threshold < "
                "phishing_threshold <= 1"
            )
        if self.sandbox_timeout_seconds <= 0:
            raise ValueError("sandbox_timeout_seconds must be positive")

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        return cls(
            sandbox_url=os.getenv("SANDBOX_URL", defaults.sandbox_url).strip(),
            sandbox_timeout_seconds=_env_int(
                "SANDBOX_TIMEOUT_SECONDS", defaults.sandbox_timeout_seconds
            ),
            screenshot_dir=Path(
                os.getenv("SCREENSHOT_DIR", str(defaults.screenshot_dir))
            ),
            llm_provider=os.getenv("LLM_PROVIDER", defaults.llm_provider).strip().lower(),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", defaults.bedrock_model_id),
            aws_region=os.getenv("AWS_REGION", defaults.aws_region),
            model_dir=Path(os.getenv("MODEL_DIR", str(defaults.model_dir))),
            params_path=Path(os.getenv("PARAMS_PATH", str(defaults.params_path))),
            phishing_threshold=_env_float("PHISHING_THRESHOLD", defaults.phishing_threshold),
            suspicious_threshold=_env_float(
                "SUSPICIOUS_THRESHOLD", defaults.suspicious_threshold
            ),
            monitoring_enabled=_env_bool("MONITORING_ENABLED", defaults.monitoring_enabled),
            events_path=Path(os.getenv("EVENTS_PATH", str(defaults.events_path))),
            reports_dir=Path(os.getenv("REPORTS_DIR", str(defaults.reports_dir))),
            psi_threshold=_env_float("PSI_THRESHOLD", defaults.psi_threshold),
            block_private_networks=_env_bool("BLOCK_PRIVATE_NETWORKS", True),
            allow_downloads=_env_bool("ALLOW_DOWNLOADS", False),
            headless=_env_bool("BROWSER_HEADLESS", True),
        )


def load_params(path: Path | None = None) -> dict:
    """Load the DVC ``params.yaml`` file (``PARAMS_PATH`` overrides the path)."""
    import yaml

    path = path or Path(os.getenv("PARAMS_PATH", str(Settings().params_path)))
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
