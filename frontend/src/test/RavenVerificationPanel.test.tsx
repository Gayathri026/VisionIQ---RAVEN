import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RavenVerificationPanel } from '../components/RavenVerificationPanel';
import { RavenAssessment, HumanReviewRecord } from '../types';

describe('RavenVerificationPanel', () => {
  it('renders ACCEPT decision when risk is low and evidence is complete', () => {
    const assessment: RavenAssessment = {
      decision: 'ACCEPT',
      risk_score: 0.12,
      hazard_severity: 0.30,
      evidence_completeness: 1.0,
      uncertainty_index: 0.05,
      conflict_entropy: 0.0,
      decision_rationale: 'All critical requirements verified.',
      evidence_repertoire: {
        intent: 'ppe',
        tokens: [],
        total_required: 2,
        total_satisfied: 2,
        completeness_score: 1.0
      },
      conflict_report: {
        has_conflict: false,
        conflict_entropy: 0.0,
        discrepancy_count: 0,
        discrepancies: []
      },
      verification_timestamp: Date.now()
    };

    render(<RavenVerificationPanel assessment={assessment} />);
    expect(screen.getByText(/RAVEN Tri-State Decision: ACCEPT/)).toBeInTheDocument();
    expect(screen.getByText('0.120')).toBeInTheDocument(); // Risk score
  });

  it('renders VERIFY decision for intermediate ambiguity with bounded pass', () => {
    const assessment: RavenAssessment = {
      decision: 'VERIFY',
      risk_score: 0.45,
      hazard_severity: 0.60,
      evidence_completeness: 0.75,
      uncertainty_index: 0.35,
      conflict_entropy: 0.15,
      decision_rationale: 'Surface defect requires secondary verification pass.',
      evidence_repertoire: {
        intent: 'defect',
        tokens: [],
        total_required: 2,
        total_satisfied: 1,
        completeness_score: 0.50
      },
      conflict_report: {
        has_conflict: false,
        conflict_entropy: 0.0,
        discrepancy_count: 0,
        discrepancies: []
      },
      verification_timestamp: Date.now()
    };

    render(<RavenVerificationPanel assessment={assessment} />);
    expect(screen.getByText(/RAVEN Tri-State Decision: VERIFY/)).toBeInTheDocument();
    expect(screen.getByText('0.450')).toBeInTheDocument();
  });

  it('renders ABSTAIN as safety clearance refusal (never success) with safety veto and human review record', () => {
    const assessment: RavenAssessment = {
      decision: 'ABSTAIN',
      risk_score: 0.88,
      hazard_severity: 0.90,
      evidence_completeness: 0.20,
      uncertainty_index: 0.75,
      conflict_entropy: 0.45,
      decision_rationale: 'Critical PPE missing and severe occlusion detected.',
      safety_veto_applied: 'CRITICAL_SAFETY_GATE_TRIGGERED: Severe Occlusion (R=0.88 >= 0.65)',
      evidence_repertoire: {
        intent: 'ppe',
        tokens: [],
        total_required: 3,
        total_satisfied: 1,
        completeness_score: 0.33
      },
      conflict_report: {
        has_conflict: true,
        conflict_entropy: 0.45,
        discrepancy_count: 1,
        discrepancies: [
          {
            discrepancy_id: 'CONF-01',
            type: 'PPE_SPATIAL_MISMATCH',
            agents_involved: ['ppe', 'object'],
            entropy_contribution: 0.45,
            details: 'PPE helmet detected outside detected human cranial boundary'
          }
        ]
      },
      verification_timestamp: Date.now()
    };

    const reviewRecord: HumanReviewRecord = {
      inspection_id: 'INSP-2026-X99',
      query: 'Check worker helmet compliance',
      decision: 'ABSTAIN',
      reason: 'Mandatory human audit: unvalidated absence of head protection.',
      relevant_regions: [
        {
          finding_uid: 'find_01',
          specialist: 'ppe',
          bbox: [100, 50, 300, 450]
        }
      ],
      missing_evidence: ['tok_helmet_presence'],
      conflicting_findings: [
        {
          conflict_id: 'CONF-01',
          type: 'PPE_SPATIAL_MISMATCH',
          specialists: ['ppe', 'object'],
          description: 'Cranial mismatch',
          resolution_status: 'UNRESOLVED'
        }
      ],
      failed_specialists: [],
      recommended_actions: [
        'Halt perimeter conveyor line',
        'Certified safety supervisor must conduct manual inspection'
      ],
      regulatory_clauses: [
        'OSHA 29 CFR 1926.100: Head Protection',
        'ASTM A568: Surface Quality Standards'
      ],
      created_at: Date.now(),
      disclaimer: 'This automated inspection assessment provides non-binding decision support.'
    };

    render(
      <RavenVerificationPanel
        assessment={assessment}
        humanReviewRecord={reviewRecord}
      />
    );

    // Hard Safety Gate Veto Banner
    expect(screen.getByText('HARD SAFETY GATE VETO TRIGGERED')).toBeInTheDocument();
    expect(screen.getByText(/Severe Occlusion/)).toBeInTheDocument();

    // ABSTAIN Warning - Never shown as success
    expect(screen.getByText(/Safety Clearance Refused — Mandatory Human Audit Mandated/)).toBeInTheDocument();
    expect(screen.getByText(/An ABSTAIN outcome indicates safety refusal, never successful automated sign-off/)).toBeInTheDocument();

    // Forensic Human Review Record
    expect(screen.getByText(/Forensic Human Review Record/)).toBeInTheDocument();
    expect(screen.getByText(/Mandatory human audit: unvalidated absence/)).toBeInTheDocument();
    expect(screen.getByText(/OSHA 29 CFR 1926.100: Head Protection/)).toBeInTheDocument();
    expect(screen.getByText(/ASTM A568: Surface Quality Standards/)).toBeInTheDocument();
    expect(screen.getByText(/tok_helmet_presence/)).toBeInTheDocument();
  });
});
