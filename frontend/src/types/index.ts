export interface EvidenceRecord {
  evidence_id: string;
  inspection_id: string;
  query_id?: string;
  agent_name: string;
  agent_status: 'SUCCESS' | 'PARTIAL' | 'FAILED' | 'SKIPPED' | string;
  model_id?: string | null;
  model_version?: string | null;
  input_image_ref?: string | null;
  predicted_class?: string | null;
  object_id?: string | null;
  bbox?: [number, number, number, number] | null;
  mask_ref?: string | null;
  raw_confidence?: number | null;
  calibrated_confidence?: number | null;
  timestamp: string;
  processing_time_ms: number;
  evidence_source: string;
  error_info?: string | null;
  provenance?: Record<string, any>;
}

export type RavenDecision = 'ACCEPT' | 'VERIFY' | 'ABSTAIN';

export interface EvidenceToken {
  token_id: string;
  description: string;
  required: boolean;
  present: boolean;
  raw_confidence: number;
  calibrated_confidence: number;
  uncertainty: number;
  source_agent: string;
  metadata?: Record<string, any>;
  status?: 'SATISFIED' | 'INSUFFICIENT_EVIDENCE' | 'CONFIRMED_ABSENT' | 'NOT_APPLICABLE' | 'CONTRADICTORY' | 'MISSING' | 'UNAVAILABLE' | 'UNSUPPORTED' | string;
  status_reason?: string;
}

export interface EvidenceRepertoire {
  intent: string;
  tokens: EvidenceToken[];
  total_required: number;
  total_satisfied: number;
  completeness_score: number;
}

export interface Discrepancy {
  discrepancy_id: string;
  type: string;
  agents_involved: string[];
  entropy_contribution: number;
  details: string;
}

export interface ConflictReport {
  has_conflict: boolean;
  conflict_entropy: number;
  discrepancy_count: number;
  discrepancies: Discrepancy[];
}

export interface HumanEscalationDossier {
  escalation_uid: string;
  urgency_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | string;
  reason: string;
  primary_concerns: string[];
  missing_evidence: string[];
  recommended_human_actions: string[];
  regulatory_clauses: string[];
  bounded_liability_statement?: string;
}

export interface HumanReviewRecord {
  inspection_id: string;
  query: string;
  decision: string;
  reason: string;
  relevant_regions: Array<{
    finding_uid?: string;
    specialist?: string;
    bbox?: [number, number, number, number];
  }>;
  missing_evidence: string[];
  conflicting_findings: Array<{
    conflict_id: string;
    type: string;
    specialists: string[];
    description: string;
    regions?: number[][];
    resolution_status: string;
  }>;
  failed_specialists: string[];
  recommended_actions: string[];
  regulatory_clauses: string[];
  created_at: number;
  disclaimer: string;
}

export interface SequentialStepTelemetry {
  step: number;
  current_deficit: number;
  candidate_agents: string[];
  candidate_predicted_gains: Record<string, number>;
  candidate_costs: Record<string, number>;
  candidate_latencies: Record<string, number>;
  candidate_efficiencies: Record<string, number>;
  selected_specialist: string | null;
  selection_efficiency: number;
  actual_findings_count: number;
  execution_result: 'SUCCESS' | 'FAILED' | 'TIMEOUT' | string;
  cleanup_result: 'CLEAN' | 'CONTAINMENT_FAILED' | string;
  post_execution_deficit: number | null;
  signed_observed_deficit_reduction: number | null;
  prediction_error: number | null;
  remaining_invocation_budget: number;
  remaining_time_budget_ms: number;
  stopping_reason?: string | null;
}

export interface RavenAssessment {
  decision: RavenDecision;
  risk_score: number;
  hazard_severity: number;
  evidence_completeness: number;
  uncertainty_index: number;
  conflict_entropy: number;
  decision_rationale: string;
  evidence_repertoire: EvidenceRepertoire;
  conflict_report: ConflictReport;
  secondary_agents_suggested?: string[];
  escalation_dossier?: HumanEscalationDossier | null;
  human_review_record?: HumanReviewRecord | null;
  safety_veto_applied?: string | null;
  provenance_certified?: boolean;
  verification_timestamp: number;
}

export interface Finding {
  finding_uid: string;
  agent_source: 'object' | 'ppe' | 'defect' | 'ocr' | 'scene' | string;
  label: string;
  bbox?: [number, number, number, number] | null;
  norm_bbox?: [number, number, number, number] | null;
  detection_confidence?: number | null;
  anomaly_score?: number | null;
  routing_score?: number | null;
  observed_evidence: string;
  root_cause_hypothesis?: string | null;
  alternative_explanations?: string | null;
  recommendation?: string | null;
  verification_procedure?: string | null;
  human_review_required: boolean;
  frame_timestamp?: number | null;
  track_id?: number | null;
  model_name?: string | null;
  model_version?: string | null;
  score_interpretation?: string | null;
  person_id?: string | null;
  is_confirmed_defect?: boolean | null;
  spatial_zone?: string | null;
  uncertainty_flag?: string | null;
  calibrated_confidence?: number | null;
  spatial_uncertainty?: number | null;
}

export interface AgentTrace {
  agent_name: string;
  relevance_score: number;
  evidence_capability: number;
  normalized_cost: number;
  expected_latency: number;
  final_routing_score: number;
  decision: 'SELECTED' | 'SKIPPED' | 'MANDATORY_DEPENDENCY' | 'FALLBACK';
  selection_reason?: string;
  execution_status: 'SUCCESS' | 'FAILED' | 'SKIPPED' | 'PENDING';
  actual_latency_ms: number;
  error_message?: string | null;
}

export interface RAGReference {
  document_title: string;
  section_reference: string;
  excerpt: string;
  similarity_score: number;
  source_uri?: string | null;
}

export interface AnalysisResponseData {
  session_id: string;
  created_at: string;
  user_query: string;
  input_type: 'image' | 'video' | 'camera';
  input_path: string;
  input_url?: string;
  annotated_path?: string;
  annotated_url?: string;
  detected_intent?: string;
  intent_confidence: number;
  selected_agents: string[];
  executed_agents: string[];
  skipped_agents: string[];
  failed_agents: string[];
  status: 'completed' | 'partial' | 'failed';
  total_latency_ms: number;
  synthesis_text?: string;
  evidence_support_score: number;
  pdf_report_path?: string;
  pdf_url?: string;
  findings: Finding[];
  traces: AgentTrace[];
  rag_references?: RAGReference[];
  rag_sources?: RAGReference[];
  inspection_id?: string;
  query?: string;
  raven_risk_score?: number;
  raven_decision?: RavenDecision;
  evidence_completeness?: number;
  conflict_entropy?: number;
  raven_assessment?: RavenAssessment;
  raven_dossier?: HumanEscalationDossier;
  human_review_record?: HumanReviewRecord | null;
  evidence_records?: EvidenceRecord[];
  partial_agents?: string[];
  human_review_status?: boolean;
  recommendations?: string[];
  routing_telemetry?: SequentialStepTelemetry[];
  deficit_history?: number[];
  current_deficit?: number;
  stopping_reason?: string | null;
}

export interface RoutingStrategyBenchmark {
  strategy: string;
  sample_count: number;
  mean_invocations: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  mean_task_f1: number;
  mean_task_prec: number;
  mean_task_rec: number;
  mean_deficit: number;
  mean_completeness: number;
  mean_reliability: number;
  abstention_rate: number;
}

export interface BenchmarkMetrics {
  name: string;
  total_queries: number;
  routing_precision: number;
  routing_recall: number;
  routing_f1: number;
  median_latency_ms: number;
  p95_latency_ms: number;
  mean_agents_invoked: number;
  latency_reduction_pct: number;
  failure_recovery_rate: number;
}

export interface BenchmarkSummary {
  timestamp: string;
  total_test_queries: number;
  configurations: {
    fixed_pipeline: BenchmarkMetrics;
    rule_based_routing: BenchmarkMetrics;
    adaptive_visioniq: BenchmarkMetrics;
  };
  ablations: {
    no_cost_penalties: {
      name: string;
      routing_f1: number;
      median_latency_ms: number;
      mean_agents_invoked: number;
    };
    no_rag_grounding: {
      name: string;
      routing_f1: number;
      median_latency_ms: number;
      mean_agents_invoked: number;
    };
  };
  per_query_traces: Array<{
    query_id: string;
    query: string;
    detected_intent: string;
    selected_agents: string[];
    expected_agents: string[];
    precision: number;
    recall: number;
    f1: number;
    estimated_latency_ms: number;
  }>;
}

export interface HistorySession {
  id: string;
  created_at: string;
  user_query: string;
  input_type: string;
  detected_intent: string;
  status: string;
  total_latency_ms: number;
  evidence_support_score: number;
  findings_count?: number;
  pdf_report_path?: string;
}

export interface AssistantResponse {
  status: string;
  query: string;
  title: string;
  answer: string;
  source_references: string[];
  recommended_navigation?: {
    path: string;
    label: string;
  };
  suggested_followups: string[];
  mode: string;
}

export interface AssistantFAQItem {
  id: string;
  question: string;
  category: string;
}

export interface ExperimentComparison {
  pipeline_type: string;
  name: string;
  description: string;
  status: 'completed' | 'not_yet_run';
  note?: string;
  metrics?: any;
  theoretical_mean_agents?: number;
  theoretical_latency_ms?: number;
}

export interface ResearchExperimentsData {
  status: string;
  total_experiments_recorded: number;
  executed_runs: any[];
  pipeline_comparisons: ExperimentComparison[];
  active_models: any[];
  datasets: Array<{
    name: string;
    task: string;
    version: string;
    license: string;
    classes: string[];
    status: string;
  }>;
}
