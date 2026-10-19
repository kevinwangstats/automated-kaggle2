"""
State Schema and Data Contracts for the Actor-Critic State Machine.

Defines:
- HypothesisSpec: Strictly validated Pydantic model for statistical proposals.
- AgentState: TypedDict maintaining state variables across all graph nodes.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class HypothesisSpec(BaseModel):
    """
    Formal, falsifiable statistical specification produced by the Methodologist.
    The Methodologist never writes Python code; it produces this specification.
    """
    hypothesis_id: str = Field(description="Unique identifier, e.g., hyp_iter_1_interaction_age_pclass")
    statistical_diagnosis: str = Field(
        description="The specific flaw or error cluster in diagnostics_report.json being addressed"
    )
    intervention_category: Literal["feature_engineering", "target_transform", "hyperparameter", "loss_function"] = Field(
        description="Category of the proposed intervention"
    )
    proposed_intervention: str = Field(
        description="Mathematical and programmatic specification of the feature transformation or parameter adjustment"
    )
    expected_diagnostic_shift: str = Field(
        description="Falsifiable expectation: how residual quantiles, calibration, or fold std should shift if true"
    )


class AgentState(TypedDict):
    """
    State tracking dictionary passed between graph nodes:
    Methodologist -> Coder -> Sandbox Execution -> Auditor -> DevOps Gate
    """
    competition_slug: str
    iteration: int
    max_iterations: int
    active_branch: str
    model_family: str                          # e.g., "lightgbm", "catboost", "xgboost"
    best_cv_score: float
    current_cv_score: Optional[float]
    metric_name: str
    greater_is_better: bool
    current_hypothesis: Optional[HypothesisSpec]
    last_diagnostics: Optional[Dict[str, Any]]
    execution_error_log: Optional[str]
    retry_count: int
    max_retries: int
    status: Literal["INITIALIZING", "RUNNING", "IMPROVED", "REJECTED", "FAILED", "COMPLETED"]
    history: List[Dict[str, Any]]              # Audit log of past iterations
