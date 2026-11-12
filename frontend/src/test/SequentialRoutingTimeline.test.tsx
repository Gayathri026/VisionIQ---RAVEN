import React from 'react';
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SequentialRoutingTimeline } from '../components/SequentialRoutingTimeline';
import { SequentialStepTelemetry } from '../types';

describe('SequentialRoutingTimeline', () => {
  it('renders graceful empty state when telemetry is missing or empty', () => {
    render(<SequentialRoutingTimeline telemetry={[]} />);
    expect(screen.getByText('Sequential Evidence-Gain Routing Telemetry')).toBeInTheDocument();
  });

  it('renders step telemetry with candidate gains, post-deficit, and prediction error', () => {
    const mockTelemetry: SequentialStepTelemetry[] = [
      {
        step: 1,
        current_deficit: 0.60,
        candidate_agents: ['ppe', 'defect', 'ocr'],
        candidate_predicted_gains: { ppe: 0.25, defect: 0.10, ocr: 0.05 },
        candidate_costs: { ppe: 0.15, defect: 0.15, ocr: 0.20 },
        candidate_latencies: { ppe: 45, defect: 60, ocr: 90 },
        candidate_efficiencies: { ppe: 1.25, defect: 0.50, ocr: 0.20 },
        selected_specialist: 'ppe',
        selection_efficiency: 1.25,
        actual_findings_count: 2,
        execution_result: 'SUCCESS',
        cleanup_result: 'CLEAN',
        post_execution_deficit: 0.35,
        signed_observed_deficit_reduction: 0.25,
        prediction_error: 0.0,
        remaining_invocation_budget: 2,
        remaining_time_budget_ms: 1955,
        stopping_reason: null
      },
      {
        step: 2,
        current_deficit: 0.35,
        candidate_agents: ['defect', 'ocr'],
        candidate_predicted_gains: { defect: 0.20, ocr: 0.05 },
        candidate_costs: { defect: 0.15, ocr: 0.20 },
        candidate_latencies: { defect: 55, ocr: 90 },
        candidate_efficiencies: { defect: 1.10, ocr: 0.20 },
        selected_specialist: 'defect',
        selection_efficiency: 1.10,
        actual_findings_count: 1,
        execution_result: 'SUCCESS',
        cleanup_result: 'CLEAN',
        post_execution_deficit: 0.15,
        signed_observed_deficit_reduction: 0.20,
        prediction_error: 0.0,
        remaining_invocation_budget: 1,
        remaining_time_budget_ms: 1900,
        stopping_reason: 'DEFICIT_THRESHOLD_REACHED (D <= 0.15)'
      }
    ];

    render(
      <SequentialRoutingTimeline
        telemetry={mockTelemetry}
        currentDeficit={0.15}
        stoppingReason="DEFICIT_THRESHOLD_REACHED (D <= 0.15)"
        targetDeficit={0.15}
      />
    );

    // Header metrics
    expect(screen.getByText('Sequential Evidence-Gain Orchestration Timeline')).toBeInTheDocument();
    expect(screen.getByText('DEFICIT_THRESHOLD_REACHED (D <= 0.15)')).toBeInTheDocument();
    expect(screen.getAllByText('2').length).toBeGreaterThan(0); // 2 steps

    // Step cards
    expect(screen.getByText('Step 1')).toBeInTheDocument();
    expect(screen.getAllByText(/ppe/i).length).toBeGreaterThan(0);
    expect(screen.getByText('Step 2')).toBeInTheDocument();
    expect(screen.getAllByText(/defect/i).length).toBeGreaterThan(0);

    // Prediction and deficit metrics
    expect(screen.getByText('0.600')).toBeInTheDocument(); // Pre-deficit
    expect(screen.getByText('+0.250')).toBeInTheDocument(); // Observed reduction
  });
});
