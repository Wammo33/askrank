"""Load and validate the YAML config (business, competitors, queries, providers).

Pydantic does the validation, so a typo in the YAML fails loudly with a clear
message instead of silently producing wrong results later.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Location(BaseModel):
    city: str | None = None
    region: str | None = None
    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2, e.g. ZA")
    timezone: str | None = None

    @field_validator("country")
    @classmethod
    def _country_code(cls, v: str | None) -> str | None:
        if v is not None and (len(v) != 2 or not v.isalpha()):
            raise ValueError("country must be a 2-letter ISO code, e.g. ZA")
        return v.upper() if v else v


class Business(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    website: str | None = None

    @property
    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


class Query(BaseModel):
    id: str
    text: str


class ProviderConfig(BaseModel):
    name: str  # "openai" or "fake"
    model: str
    web_search: bool = True


class ExtractorConfig(BaseModel):
    provider: str = "openai"  # "openai" (LLM) or "rules" (alias matching only)
    model: str = "gpt-5.4-nano"


class Config(BaseModel):
    project: str
    location: Location = Field(default_factory=Location)
    target: Business
    competitors: list[Business] = Field(default_factory=list)
    queries: list[Query]
    runs_per_query: int = Field(default=5, ge=1, le=100)
    providers: list[ProviderConfig]
    extractor: ExtractorConfig = Field(default_factory=ExtractorConfig)

    @model_validator(mode="after")
    def _checks(self) -> Config:
        ids = [q.id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("query ids must be unique")
        names = [b.name for b in self.businesses]
        if len(names) != len(set(names)):
            raise ValueError("business names must be unique (target + competitors)")
        if not self.queries:
            raise ValueError("at least one query is required")
        if not self.providers:
            raise ValueError("at least one provider is required")
        return self

    @property
    def businesses(self) -> list[Business]:
        """Target first, then competitors."""
        return [self.target, *self.competitors]


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return Config.model_validate(data)
