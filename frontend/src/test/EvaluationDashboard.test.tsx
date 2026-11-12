import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { EvaluationDashboard } from '../pages/EvaluationDashboard';
import { api } from '../services/api';
import { RoutingStrategyBenchmark, BenchmarkSummary } from '../types';

vi.mock('../services/api', () => ({
  api: {
    getRoutingStrategies: vi.fn(),
    runBenchmark: vi.fn()
  }
}));

describe('EvaluationDashboard', () => {
  const mockStrategies: RoutingStrategyBenchmark[] = [
    {
      strategy: 'fixed_pipeline',
      sample_count: 20,
      mean_invocations: 2.4,
      median_latency_ms: 98.46,
      p95_latency_ms: 398.92,
      mean_task_f1: 0.3174,
      mean_task_prec: 0.2602,
      mean_task_rec: 0.425,
      mean_deficit: 0.3081,
      mean_completeness: 0.6043,
      mean_reliability: 0.4489,
      abstention_rate: 0.0
    },
    {
      strategy: 'phase2_weighted',
      sample_count: 20,
      mean_invocations: 1.8,
      median_latency_ms: 101.22,
      p95_latency_ms: 128.14,
      mean_task_f1: 0.3832,
      mean_task_prec: 0.3593,
      mean_task_rec: 0.425,
      mean_deficit: 0.5035,
      mean_completeness: 0.271,
      mean_reliability: 0.305,
      abstention_rate: 0.0
    },
    {
      strategy: 'relevance_dependency',
      sample_count: 20,
      mean_invocations: 2.4,
      median_latency_ms: 103.68,
      p95_latency_ms: 172.63,
      mean_task_f1: 0.3674,
      mean_task_prec: 0.3352,
      mean_task_rec: 0.425,
      mean_deficit: 0.4512,
      mean_completeness: 0.371,
      mean_reliability: 0.3167,
      abstention_rate: 0.0
    },
    {
      strategy: 'sequential_evidence_gain',
      sample_count: 20,
      mean_invocations: 2.2,
      median_latency_ms: 80.25,
      p95_latency_ms: 133.65,
      mean_task_f1: 0.3674,
      mean_task_prec: 0.3352,
      mean_task_rec: 0.425,
      mean_deficit: 0.3661,
      mean_completeness: 0.5043,
      mean_reliability: 0.4085,
      abstention_rate: 0.0
    }
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    (api.getRoutingStrategies as any).mockResolvedValue(mockStrategies);
    (api.runBenchmark as any).mockResolvedValue({
      timestamp: '2026-11-12T12:00:00Z',
      total_test_queries: 20,
      configurations: {
        fixed_pipeline: {
          name: 'Fixed Pipeline',
          total_queries: 20,
          routing_precision: 0.26,
          routing_recall: 0.42,
          routing_f1: 0.32,
          median_latency_ms: 98.5,
          p95_latency_ms: 398.9,
          mean_agents_invoked: 2.4,
          latency_reduction_pct: 0,
          failure_recovery_rate: 0.95
        },
        rule_based_routing: {
          name: 'Rule Based',
          total_queries: 20,
          routing_precision: 0.35,
          routing_recall: 0.42,
          routing_f1: 0.38,
          median_latency_ms: 101.2,
          p95_latency_ms: 128.1,
          mean_agents_invoked: 1.8,
          latency_reduction_pct: 0,
          failure_recovery_rate: 0.95
        },
        adaptive_visioniq: {
          name: 'Adaptive VisionIQ',
          total_queries: 20,
          routing_precision: 0.34,
          routing_recall: 0.42,
          routing_f1: 0.37,
          median_latency_ms: 80.3,
          p95_latency_ms: 133.7,
          mean_agents_invoked: 2.2,
          latency_reduction_pct: 18.5,
          failure_recovery_rate: 1.0
        }
      },
      ablations: {
        no_cost_penalties: { name: 'No Cost', routing_f1: 0.35, median_latency_ms: 120, mean_agents_invoked: 3.2 },
        no_rag_grounding: { name: 'No RAG', routing_f1: 0.34, median_latency_ms: 85, mean_agents_invoked: 2.2 }
      },
      per_query_traces: []
    });
  });

  it('renders the 4-strategy benchmark matrix and offline dataset disambiguation banner', async () => {
    render(<EvaluationDashboard />);

    // Check Offline Dataset Disambiguation
    expect(screen.getByText('Offline Test Split (N=20)')).toBeInTheDocument();
    expect(screen.getByText(/Data Grounding:/)).toBeInTheDocument();

    // Wait for strategies to load
    await waitFor(() => {
      expect(screen.getByText('Sequential Evidence-Gain')).toBeInTheDocument();
      expect(screen.getByText('Fixed Pipeline')).toBeInTheDocument();
      expect(screen.getByText('Phase 2 Weighted')).toBeInTheDocument();
      expect(screen.getByText('Relevance & Dependency')).toBeInTheDocument();
    });

    // Verify lowest median latency for sequential router (80.3 ms)
    expect(screen.getAllByText(/80.3/i).length).toBeGreaterThan(0);
  });
});
