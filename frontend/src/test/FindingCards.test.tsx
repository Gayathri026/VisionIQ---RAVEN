import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { FindingCards } from '../components/FindingCards';
import { Finding } from '../types';

describe('FindingCards', () => {
  it('renders empty state message when no findings are available', () => {
    render(<FindingCards findings={[]} onSelectFinding={() => {}} />);
    expect(screen.getByText('No Specific Inspection Findings')).toBeInTheDocument();
  });

  it('renders calibrated confidence and raw score separately with model information', () => {
    const mockFindings: Finding[] = [
      {
        finding_uid: 'f_ppe_01',
        agent_source: 'ppe',
        label: 'Safety Helmet Compliant',
        detection_confidence: 0.88,
        calibrated_confidence: 0.824,
        spatial_zone: 'Cranial',
        bbox: [100, 200, 150, 260],
        observed_evidence: 'High-contrast yellow helmet secured on head.',
        model_name: 'yolov11-ppe-inspection',
        model_version: 'v2.1.0',
        human_review_required: false
      },
      {
        finding_uid: 'f_def_02',
        agent_source: 'defect',
        label: 'Rolled-in Scale Defect',
        detection_confidence: 0.76,
        calibrated_confidence: 0.71,
        spatial_zone: 'Surface Plane',
        bbox: [320, 410, 480, 520],
        observed_evidence: 'Linear surface scale indentation matching NEU-DET classification.',
        model_name: 'neu-det-surface-classifier',
        model_version: 'v1.4.0',
        is_confirmed_defect: true,
        human_review_required: true
      }
    ];

    render(<FindingCards findings={mockFindings} onSelectFinding={() => {}} />);

    // Header summary ribbon
    expect(screen.getByText('Total Detections')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument();

    // Finding items
    expect(screen.getByText('f_ppe_01')).toBeInTheDocument();
    expect(screen.getByText('Safety Helmet Compliant')).toBeInTheDocument();
    expect(screen.getByText('f_def_02')).toBeInTheDocument();
    expect(screen.getByText('Rolled-in Scale Defect')).toBeInTheDocument();

    // Calibrated and Raw score badges
    expect(screen.getByText(/82.4% \(Calibrated\)/)).toBeInTheDocument();
    expect(screen.getByText(/Raw s: 88.0%/)).toBeInTheDocument();
    expect(screen.getByText(/71.0% \(Calibrated\)/)).toBeInTheDocument();

    // Model and version
    expect(screen.getByText(/yolov11-ppe-inspection/)).toBeInTheDocument();
    expect(screen.getByText(/v2.1.0/)).toBeInTheDocument();
  });
});
