"""Temporary internal contracts, NOT the LLM wire protocol or official ToolEnv API.

LLM owns prompt construction and parsing. Adapters translate into these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Protocol, Union


class StopReason(str, Enum):
    FINISH = "finish"
    ASK = "ask"
    REPAIR_BUDGET = "repair_budget"
    MODEL_ERROR = "model_error"
    ENVIRONMENT_ERROR = "environment_error"
    EXECUTION_UNKNOWN = "execution_unknown"
    DEADLINE = "deadline"
    CALL_BUDGET = "call_budget"
    STEP_BUDGET = "step_budget"


@dataclass(frozen=True)
class ProposedCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Calls:
    """Successfully parsed response: zero calls ends, one call executes."""
    calls: tuple[ProposedCall, ...]


@dataclass(frozen=True)
class ModelError:
    code: Literal["model_error", "parse_error"] = "model_error"


ModelResponse = Union[Calls, ModelError]


@dataclass(frozen=True)
class Validation:
    valid: bool
    is_ask: bool = False
    code: str = "accepted"


@dataclass(frozen=True)
class Observation:
    ok: bool
    done: bool
    data: dict[str, Any] = field(default_factory=dict)
    # Advisory metadata only; a tool error still returns to the model.
    terminal: bool = False


@dataclass(frozen=True)
class Execution:
    # None means execution status is unknown: never retry automatically.
    executed: bool | None
    observation: Observation


@dataclass(frozen=True)
class CallRecord:
    proposal: ProposedCall
    execution: Execution


@dataclass(frozen=True)
class Sample:
    sample_id: str
    # Prepared input from the LLM/data boundary, never official labels/state.
    model_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RepairFeedback:
    kind: Literal["parse_error", "validation_error", "protocol_error"]
    proposal: ProposedCall | None = None
    validation: Validation | None = None


@dataclass(frozen=True)
class ModelRequest:
    sample: Sample
    step: int
    observations: tuple[CallRecord, ...]
    remaining_seconds: float
    output_tokens: int
    repair: RepairFeedback | None = None
    repairs_remaining: int = 0


class ModelBackend(Protocol):
    def respond(self, request: ModelRequest) -> ModelResponse: ...


class Environment(Protocol):
    """validate is side-effect free; execute is called at most once per proposal."""
    def validate(self, call: ProposedCall) -> Validation: ...
    def execute(self, call: ProposedCall) -> Execution: ...


EnvironmentFactory = Callable[[Sample], Environment]
TraceSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class RuntimeConfig:
    sample_deadline_seconds: float = 30.0
    max_executed_calls: int = 8
    max_steps: int = 16
    max_repairs: int = 2
    output_tokens_per_step: int = 512

    def __post_init__(self) -> None:
        import math
        if (isinstance(self.sample_deadline_seconds, bool)
                or not isinstance(self.sample_deadline_seconds, (int, float))
                or not math.isfinite(self.sample_deadline_seconds)
                or self.sample_deadline_seconds <= 0):
            raise ValueError("sample_deadline_seconds must be finite and positive")
        if type(self.max_repairs) is not int or self.max_repairs < 0:
            raise ValueError("max_repairs must be a nonnegative integer")
        for name in ("max_executed_calls", "max_steps", "output_tokens_per_step"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SampleResult:
    sample_id: str
    stop_reason: StopReason
    executed_calls: tuple[CallRecord, ...]
    uncertain_calls: tuple[ProposedCall, ...]
    last_observation: Observation | None
    elapsed_seconds: float
    failure: bool
    repairs_used: int
    # Deliberately no task_success or state_hash field in a mock result.
