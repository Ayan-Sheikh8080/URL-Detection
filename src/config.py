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


@dataclass
class Settings:
    """Settings for the agent, sandbox, LLM and monitoring components."""

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
    events_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "monitoring" / "events.jsonl"
    )
    reports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "reports")
    psi_threshold: float = 0.2

    # Safety
    block_private_networks: bool = True
    allow_downloads: bool = False
    headless: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        settings = cls()
        return cls(
            sandbox_url=os.getenv("SANDBOX_URL", settings.sandbox_url).strip(),
            sandbox_timeout_seconds=int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "20")),
            llm_provider=os.getenv("LLM_PROVIDER", settings.llm_provider).strip().lower(),
            bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", settings.bedrock_model_id),
            aws_region=os.getenv("AWS_REGION", settings.aws_region),
            model_dir=Path(os.getenv("MODEL_DIR", str(settings.model_dir))),
            phishing_threshold=float(os.getenv("PHISHING_THRESHOLD", "0.65")),
            suspicious_threshold=float(os.getenv("SUSPICIOUS_THRESHOLD", "0.35")),
            block_private_networks=_env_bool("BLOCK_PRIVATE_NETWORKS", True),
            headless=_env_bool("BROWSER_HEADLESS", True),
        )


def load_params(path: Path | None = None) -> dict:
    """Load the DVC `params.yaml` file."""
    import yaml

    path = path or Settings().params_path
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
