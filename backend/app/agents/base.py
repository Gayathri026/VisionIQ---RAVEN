"""
VisionIQ Specialist Agent Base Architecture
Provides strict typed interfaces, execution status enums, runtime measurements,
error isolation, and provenance tracking for all vision and cognitive agents.
"""

import time
import abc
from pathlib import Path
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    SELECTED = "SELECTED"
    SKIPPED = "SKIPPED"
    EXECUTING = "EXECUTING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    TIMED_OUT = "TIMED_OUT"

class ConfidenceType(str, Enum):
    """
    Epistemic confidence classification.
    Distinguishes calibrated probabilities from raw model logits and heuristic scores.
    """
    CALIBRATED_PROBABILITY = "calibrated_probability"
    RAW_MODEL_CONFIDENCE = "raw_model_confidence"
    HEURISTIC_SCORE = "heuristic_score"

class DetectionState(str, Enum):
    """Three-state visual detection tracking."""
    DETECTED = "DETECTED"
    NOT_DETECTED = "NOT_DETECTED"
    UNCERTAIN = "UNCERTAIN"

class AgentInput(BaseModel):
    session_id: str = Field(..., description="Unique identifier of current analysis session")
    user_query: str = Field(..., description="Raw natural language prompt submitted by the operator")
    image_path: Optional[str] = Field(None, description="Absolute filesystem path to raw or sampled image")
    image_id: Optional[str] = Field(None, description="Logical image identifier for tracking across benchmark manifests")
    frame_timestamp: Optional[float] = Field(None, description="Video frame timestamp in seconds if applicable")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Shared upstream findings and context")
    timeout_seconds: float = Field(default=15.0, description="Maximum execution duration before isolating timeout")

class AgentOutput(BaseModel):
    model_config = {"protected_namespaces": ()}

    agent_name: str = Field(..., description="Unique technical identifier of the specialist agent")
    status: ExecutionStatus = Field(..., description="Final execution lifecycle status")
    model_name: Optional[str] = Field(None, description="Registered model name")
    model_version: Optional[str] = Field(None, description="Exact checkpoint or algorithm version")
    input_image_id: Optional[str] = Field(None, description="Propagated input image identifier")
    measured_runtime_ms: float = Field(0.0, description="Exact measured wall-clock execution duration in milliseconds")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Structured observations, detections, or text")
    evidence_token_ids: List[str] = Field(default_factory=list, description="Evidence token IDs updated by this specialist")
    confidence_type: ConfidenceType = Field(ConfidenceType.RAW_MODEL_CONFIDENCE, description="Classification of primary confidence value")
    primary_confidence: Optional[float] = Field(None, description="Primary aggregate confidence value")
    data: Dict[str, Any] = Field(default_factory=dict, description="Specialist payload and raw metrics")
    error_details: Optional[str] = Field(None, description="Human-readable exception details if execution failed")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source tracking including model checkpoint, parameters, and deterministic rules"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional specialist-specific metadata")

    @property
    def error_message(self) -> Optional[str]:
        """Backward-compatible alias for error_details."""
        return self.error_details

class BaseSpecialistAgent(abc.ABC):
    """
    Abstract Base Class for all VisionIQ specialist agents.
    Enforces deterministic error isolation, execution timing, and typed contracts.
    """

    def __init__(self, agent_name: str, model_version: str = "1.0.0", model_name: Optional[str] = None):
        self.agent_name = agent_name
        self.model_version = model_version
        self.model_name = model_name or agent_name

    @abc.abstractmethod
    def _execute_internal(self, agent_input: AgentInput) -> AgentOutput:
        """Core specialist logic implemented by subclasses."""
        pass

    def execute(self, agent_input: AgentInput) -> AgentOutput:
        """
        Executes specialist logic with strict runtime measurement and error isolation.
        Guarantees that individual agent crashes never fail the pipeline.
        """
        start_time = time.time()
        image_id = agent_input.image_id or (Path(agent_input.image_path).stem if agent_input.image_path else None)
        try:
            output = self._execute_internal(agent_input)
            output.measured_runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            if output.input_image_id is None:
                output.input_image_id = image_id
            if output.model_name is None:
                output.model_name = self.model_name
            if output.model_version is None:
                output.model_version = self.model_version
            if not output.provenance:
                output.provenance = {
                    "agent": self.agent_name,
                    "model_name": self.model_name,
                    "version": self.model_version,
                    "timestamp": time.time()
                }
            return output
        except TimeoutError as te:
            runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.TIMED_OUT,
                model_name=self.model_name,
                model_version=self.model_version,
                input_image_id=image_id,
                measured_runtime_ms=runtime_ms,
                error_details=f"Agent timed out after {agent_input.timeout_seconds}s: {str(te)}",
                provenance={"agent": self.agent_name, "model_name": self.model_name, "version": self.model_version}
            )
        except Exception as exc:
            runtime_ms = round((time.time() - start_time) * 1000.0, 2)
            return AgentOutput(
                agent_name=self.agent_name,
                status=ExecutionStatus.FAILED,
                model_name=self.model_name,
                model_version=self.model_version,
                input_image_id=image_id,
                measured_runtime_ms=runtime_ms,
                error_details=str(exc),
                provenance={"agent": self.agent_name, "model_name": self.model_name, "version": self.model_version}
            )
