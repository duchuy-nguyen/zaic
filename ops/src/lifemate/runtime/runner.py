"""Sequential orchestration only: no prompt builder, parser, HTTP or ToolEnv guesses."""
from __future__ import annotations

from copy import deepcopy
import json
import time
from typing import Callable

from .contracts import (
    CallRecord, Calls, EnvironmentFactory, Execution, ModelBackend,
    ModelError, ModelRequest, Observation, ProposedCall, RuntimeConfig, Sample,
    SampleResult, StopReason, TraceSink, Validation, RepairFeedback,
)
from .deadlines import Deadline


def _call_shape(call: object) -> bool:
    if not isinstance(call, ProposedCall):
        return False
    if not isinstance(call.name, str) or not call.name or type(call.arguments) is not dict:
        return False
    try:
        json.dumps(call.arguments, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return all(isinstance(key, str) for key in call.arguments)


class Runner:
    def __init__(self, environment_factory: EnvironmentFactory,
                 config: RuntimeConfig = RuntimeConfig(), *,
                 clock: Callable[[], float] = time.monotonic,
                 trace: TraceSink = lambda event: None):
        self.environment_factory = environment_factory
        self.config = config
        self.clock = clock
        self.trace = trace

    def run(self, sample: Sample, backend: ModelBackend) -> SampleResult:
        start = self.clock()
        deadline = Deadline(start + self.config.sample_deadline_seconds, self.clock)
        executed: list[CallRecord] = []
        uncertain: list[ProposedCall] = []
        observations: list[CallRecord] = []
        last = None
        step = 0
        repairs_used = 0
        pending_repair = None

        def emit(event: str, *, proposal=None, validation=None, executed=False,
                 observation=None, latency=0.0, stop_reason=None, **extra):
            self.trace(deepcopy(dict(
                event=event, sample_id=sample.sample_id, step=step,
                proposal=proposal, validation=validation, executed=executed,
                observation=observation, latency_seconds=max(0.0, latency),
                stop_reason=stop_reason, **extra)))

        def finish(reason: StopReason) -> SampleResult:
            emit("stop", stop_reason=reason.value)
            return SampleResult(
                sample.sample_id, reason, tuple(executed), tuple(uncertain), last,
                max(0.0, self.clock() - start),
                reason not in (StopReason.FINISH, StopReason.ASK), repairs_used)

        def repair(feedback: RepairFeedback) -> bool:
            nonlocal repairs_used, pending_repair
            allowed = repairs_used < self.config.max_repairs
            emit("repair", proposal=feedback.proposal, validation=feedback.validation,
                 error=feedback.kind, repair_scheduled=allowed,
                 repairs_used=repairs_used, max_repairs=self.config.max_repairs)
            if allowed:
                repairs_used += 1
                pending_repair = deepcopy(feedback)
            return allowed

        try:
            env = self.environment_factory(deepcopy(sample))
        except Exception:
            return finish(StopReason.ENVIRONMENT_ERROR)

        for step in range(1, self.config.max_steps + 1):
            if deadline.expired():
                return finish(StopReason.DEADLINE)
            before = self.clock()
            try:
                response = backend.respond(ModelRequest(
                    deepcopy(sample), step, deepcopy(tuple(observations)),
                    deadline.remaining(), self.config.output_tokens_per_step,
                    deepcopy(pending_repair), self.config.max_repairs - repairs_used))
            except Exception:
                emit("model", latency=self.clock() - before, error="backend_exception")
                return finish(StopReason.DEADLINE if deadline.expired() else StopReason.MODEL_ERROR)
            # Never log backend exception text or raw model output.
            emit("model", latency=self.clock() - before,
                 proposal=response.calls if isinstance(response, Calls) else None,
                 response_kind=type(response).__name__,
                 error=("parse_error" if response.code == "parse_error" else "model_error")
                 if isinstance(response, ModelError) else None)
            if deadline.expired():
                return finish(StopReason.DEADLINE)
            pending_repair = None
            if isinstance(response, ModelError):
                if response.code != "parse_error":
                    return finish(StopReason.MODEL_ERROR)
                if not repair(RepairFeedback("parse_error")):
                    return finish(StopReason.REPAIR_BUDGET)
                continue
            if (not isinstance(response, Calls)
                    or not isinstance(response.calls, tuple) or len(response.calls) > 1):
                if not repair(RepairFeedback("protocol_error")):
                    return finish(StopReason.REPAIR_BUDGET)
                continue
            # Only a successfully parsed, contract-valid empty calls list ends.
            if not response.calls:
                return finish(StopReason.FINISH)
            if len(executed) >= self.config.max_executed_calls:
                return finish(StopReason.CALL_BUDGET)
            call = response.calls[0]
            if not _call_shape(call):
                validation = Validation(False, code="invalid_call_shape")
                emit("validation", validation=validation)
                if not repair(RepairFeedback("validation_error", validation=validation)):
                    return finish(StopReason.REPAIR_BUDGET)
                continue
            call = deepcopy(call)
            before = self.clock()
            try:
                validation = env.validate(deepcopy(call))
                if (not isinstance(validation, Validation)
                        or type(validation.valid) is not bool
                        or type(validation.is_ask) is not bool):
                    raise TypeError("invalid validation contract")
            except Exception:
                emit("validation", proposal=call, latency=self.clock() - before,
                     error="validation_exception")
                return finish(StopReason.ENVIRONMENT_ERROR)
            emit("validation", proposal=call, validation=validation,
                 latency=self.clock() - before)
            if deadline.expired():
                return finish(StopReason.DEADLINE)
            if not validation.valid:
                if not repair(RepairFeedback("validation_error", call, validation)):
                    return finish(StopReason.REPAIR_BUDGET)
                continue
            # Check immediately before execute, including time spent logging.
            if deadline.expired():
                return finish(StopReason.DEADLINE)
            before = self.clock()
            try:
                execution = env.execute(deepcopy(call))
                if (not isinstance(execution, Execution)
                        or (execution.executed is not None and type(execution.executed) is not bool)
                        or not isinstance(execution.observation, Observation)
                        or type(execution.observation.ok) is not bool
                        or type(execution.observation.done) is not bool
                        or type(execution.observation.terminal) is not bool):
                    raise TypeError("invalid execution contract")
            except Exception:
                # Execution may have mutated state before raising. Preserve
                # previous records and mark this attempt unknown, never retry.
                uncertain.append(call)
                emit("execution", proposal=call, validation=validation,
                     executed=None, latency=self.clock() - before,
                     error="execution_status_unknown")
                return finish(StopReason.EXECUTION_UNKNOWN)
            execution = deepcopy(execution)
            last = execution.observation
            record = CallRecord(call, execution)
            observations.append(record)
            if execution.executed is True:
                executed.append(record)
            elif execution.executed is None:
                uncertain.append(call)
            emit("execution", proposal=call, validation=validation,
                 executed=execution.executed, observation=last,
                 latency=self.clock() - before)
            if execution.executed is None:
                return finish(StopReason.EXECUTION_UNKNOWN)
            if deadline.expired():
                return finish(StopReason.DEADLINE)
            if execution.executed and (validation.is_ask or call.name == "sys.ask_missing_info"):
                return finish(StopReason.ASK)
            if len(executed) >= self.config.max_executed_calls:
                return finish(StopReason.CALL_BUDGET)
        return finish(StopReason.STEP_BUDGET)
