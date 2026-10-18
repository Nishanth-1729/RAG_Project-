"""Pydantic schemas shared by the agents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Item(BaseModel):
    text: str = Field(description="concise statement in your own words (max 40 words)")
    passage: int = Field(description="number of the evidence passage that supports it")
    quote: str = Field(description="short VERBATIM quote (<= 30 words) copied from that passage")


class AnalysisCore(BaseModel):
    main_idea: Item
    research_problem: Item
    motivation: Item | None = None
    method: Item
    key_equations: list[Item] = Field(description="central equations/formulations, described in words with the equation if legible")
    complexity: Item | None = Field(default=None, description="computational / theoretical complexity claims, if stated")


class AnalysisExperiments(BaseModel):
    datasets: list[Item]
    baselines: list[Item]
    metrics: list[Item]
    main_results: list[Item] = Field(description="quantitative results with numbers when available")


class AnalysisLimits(BaseModel):
    future_work: list[Item]
    claims: list[Item] = Field(description="the paper's main contribution claims")


class Limitations(BaseModel):
    limitations: list[Item] = Field(description="2-5 limitations / shortcomings / restrictions of the paper's own approach")


FIELDS_CORE = ["main_idea", "research_problem", "motivation", "method", "key_equations", "complexity"]
FIELDS_EXP = ["datasets", "baselines", "metrics", "main_results"]
FIELDS_LIM = ["limitations", "future_work", "claims"]
ALL_FIELDS = FIELDS_CORE + FIELDS_EXP + FIELDS_LIM


class ClaimVerdict(BaseModel):
    label: Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    best_passage: int | None = Field(default=None, description="number of the passage that best decides the label, if any")
    rationale: str = Field(description="one or two sentences, referring to specific content of the passage")
