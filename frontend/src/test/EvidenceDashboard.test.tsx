import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { EvidenceDashboard } from '../components/EvidenceDashboard';
import { EvidenceToken, EvidenceRepertoire } from '../types';

describe('EvidenceDashboard', () => {
  it('renders graceful empty state when repertoire is empty', () => {
    render(<EvidenceDashboard repertoire={null} tokens={[]} />);
    expect(screen.getByText('Evidence Repertoire Dashboard')).toBeInTheDocument();
  });

  it('renders all 8 lifecycle states with explicit badges and explanations', () => {
    const mockTokens: EvidenceToken[] = [
      {
        token_id: 'tok_satisfied',
        description: 'Hardhat detected on cranial region',
        required: true,
        present: true,
        raw_confidence: 0.92,
        calibrated_confidence: 0.89,
        uncertainty: 0.05,
        source_agent: 'ppe',
        status: 'SATISFIED',
        status_reason: 'Cranial safety helmet verified with high confidence'
      },
      {
        token_id: 'tok_missing',
        description: 'High-visibility vest on torso',
        required: true,
        present: false,
        raw_confidence: 0.0,
        calibrated_confidence: 0.0,
        uncertainty: 0.0,
        source_agent: 'ppe',
        status: 'MISSING',
        status_reason: 'No high-vis vest detected on worker torso'
      },
      {
        token_id: 'tok_insufficient',
        description: 'Face shield protection',
        required: false,
        present: false,
        raw_confidence: 0.42,
        calibrated_confidence: 0.38,
        uncertainty: 0.65,
        source_agent: 'ppe',
        status: 'INSUFFICIENT_EVIDENCE',
        status_reason: 'Low contrast occlusion in facial bounding box'
      },
      {
        token_id: 'tok_absent',
        description: 'Surface crack flaw',
        required: true,
        present: false,
        raw_confidence: 0.0,
        calibrated_confidence: 0.0,
        uncertainty: 0.0,
        source_agent: 'defect',
        status: 'CONFIRMED_ABSENT',
        status_reason: 'Absence verified by SiteAbsenceValidationCertificate'
      },
      {
        token_id: 'tok_na',
        description: 'OCR Serial Nameplate',
        required: false,
        present: false,
        raw_confidence: 0.0,
        calibrated_confidence: 0.0,
        uncertainty: 0.0,
        source_agent: 'ocr',
        status: 'NOT_APPLICABLE',
        status_reason: 'Query restricted to PPE compliance inspection'
      },
      {
        token_id: 'tok_conflict',
        description: 'Structural integrity token',
        required: true,
        present: false,
        raw_confidence: 0.70,
        calibrated_confidence: 0.65,
        uncertainty: 0.80,
        source_agent: 'defect',
        status: 'CONTRADICTORY',
        status_reason: 'Defect specialist reported flaw but object detector certified clear'
      },
      {
        token_id: 'tok_unavail',
        description: 'Thermal weld scanner',
        required: false,
        present: false,
        raw_confidence: 0.0,
        calibrated_confidence: 0.0,
        uncertainty: 0.0,
        source_agent: 'thermal',
        status: 'UNAVAILABLE',
        status_reason: 'Specialist service unreachable'
      }
    ];

    const mockRepertoire: EvidenceRepertoire = {
      intent: 'ppe_inspection',
      tokens: mockTokens,
      total_required: 4,
      total_satisfied: 1,
      completeness_score: 0.25
    };

    render(<EvidenceDashboard repertoire={mockRepertoire} />);

    // Header completeness
    expect(screen.getByText('25.0%')).toBeInTheDocument();
    expect(screen.getByText(/7 Total/)).toBeInTheDocument();

    // Verify all 7 distinct statuses rendered
    expect(screen.getByText('SATISFIED')).toBeInTheDocument();
    expect(screen.getByText('MISSING')).toBeInTheDocument();
    expect(screen.getByText('INSUFFICIENT')).toBeInTheDocument();
    expect(screen.getByText('CONFIRMED ABSENT')).toBeInTheDocument();
    expect(screen.getByText('NOT APPLICABLE')).toBeInTheDocument();
    expect(screen.getByText('CONTRADICTORY')).toBeInTheDocument();
    expect(screen.getByText('UNAVAILABLE')).toBeInTheDocument();

    // Verify confidence and scaling indicators
    expect(screen.getByText('89.0%')).toBeInTheDocument(); // Calibrated
    expect(screen.getAllByText('Platt/Beta Scaling').length).toBeGreaterThan(0);
  });
});
