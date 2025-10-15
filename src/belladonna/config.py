from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel


class AppConfig(BaseModel):
    log_level: str = "INFO"
    data_dir: str = "data"
    pubmed_email: Optional[str] = None  # optional contact for NCBI API
    ncbi_api_key: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r") as f:
            obj = yaml.safe_load(f) or {}
        return cls(**obj)
