import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { InspectionWorkspacePage } from '../pages/InspectionWorkspacePage';
import { api } from '../services/api';

vi.mock('../services/api', () => ({
  api: {
    getHistory: vi.fn(),
    getSessionDetails: vi.fn(),
    analyzeImage: vi.fn(),
    getPdfReportUrl: vi.fn(() => 'http://localhost:8000/api/v1/reports/sess_123/pdf')
  }
}));

describe('InspectionWorkspacePage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (api.getHistory as any).mockResolvedValue([]);
  });

  it('renders input bar, query input, sample queries, and run analysis button', () => {
    render(
      <MemoryRouter>
        <InspectionWorkspacePage />
      </MemoryRouter>
    );

    expect(screen.getByText('Visual Inspection Command Center')).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Enter inspection query/)).toBeInTheDocument();
    expect(screen.getByText('Run Analysis')).toBeInTheDocument();
    expect(screen.getByText('Load Image')).toBeInTheDocument();
  });

  it('validates missing image on Run Analysis click', async () => {
    render(
      <MemoryRouter>
        <InspectionWorkspacePage />
      </MemoryRouter>
    );

    const runBtn = screen.getByText('Run Analysis');
    fireEvent.click(runBtn);

    expect(await screen.findByText('Please select or drop an industrial image first.')).toBeInTheDocument();
  });

  it('allows tuning sequential routing parameters and toggling the engine', () => {
    render(
      <MemoryRouter>
        <InspectionWorkspacePage />
      </MemoryRouter>
    );

    const configureBtn = screen.getByText('Configure Routing Engine');
    fireEvent.click(configureBtn);

    expect(screen.getByText('Sequential Evidence-Gain Routing (Phase 3 Engine)')).toBeInTheDocument();
    expect(screen.getByText(/Target Deficit/)).toBeInTheDocument();
    expect(screen.getByText(/Max Routing Steps:/)).toBeInTheDocument();
  });
});
