import React, { useState } from 'react';
import { EvidenceToken, EvidenceRepertoire } from '../types';
import {
  Layers,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  HelpCircle,
  ShieldCheck,
  ShieldAlert,
  Info,
  Filter,
  Check,
  Ban,
  Slash,
  AlertOctagon
} from 'lucide-react';

interface Props {
  repertoire?: EvidenceRepertoire | null;
  tokens?: EvidenceToken[];
}

export const EvidenceDashboard: React.FC<Props> = ({
  repertoire,
  tokens: externalTokens = []
}) => {
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [requiredFilter, setRequiredFilter] = useState<string>('all');

  const allTokens: EvidenceToken[] = (repertoire?.tokens && repertoire.tokens.length > 0)
    ? repertoire.tokens
    : externalTokens;

  if (!allTokens || allTokens.length === 0) {
    return (
      <div className="command-panel rounded-xl p-8 text-center text-[#8998AF] space-y-3">
        <Layers className="w-10 h-10 mx-auto text-[#546E9B]" />
        <h3 className="text-base font-bold text-[#F1F5F9]">Evidence Repertoire Dashboard</h3>
        <p className="text-xs text-[#8998AF] max-w-md mx-auto leading-relaxed">
          No evidence tokens recorded yet. Run an inspection analysis to generate and inspect conditioned evidence tokens across all 8 lifecycle states.
        </p>
      </div>
    );
  }

  // Count by lifecycle status
  const totalTokens = allTokens.length;
  const requiredTokens = allTokens.filter(t => t.required);
  const optionalTokens = allTokens.filter(t => !t.required);

  const satisfiedCount = allTokens.filter(t => (t.status === 'SATISFIED' || (!t.status && t.present))).length;
  const missingCount = allTokens.filter(t => t.status === 'MISSING' || (!t.status && !t.present && t.required)).length;
  const insufficientCount = allTokens.filter(t => t.status === 'INSUFFICIENT_EVIDENCE').length;
  const absentCount = allTokens.filter(t => t.status === 'CONFIRMED_ABSENT').length;
  const notApplicableCount = allTokens.filter(t => t.status === 'NOT_APPLICABLE').length;
  const contradictoryCount = allTokens.filter(t => t.status === 'CONTRADICTORY').length;
  const unavailableCount = allTokens.filter(t => t.status === 'UNAVAILABLE' || t.status === 'UNSUPPORTED').length;

  const completeness = repertoire?.completeness_score ?? (
    requiredTokens.length > 0 ? satisfiedCount / requiredTokens.length : 1.0
  );

  // Filter tokens
  const filteredTokens = allTokens.filter(t => {
    // Required Filter
    if (requiredFilter === 'required' && !t.required) return false;
    if (requiredFilter === 'optional' && t.required) return false;

    // Status Filter
    if (statusFilter !== 'all') {
      const effStatus = t.status || (t.present ? 'SATISFIED' : t.required ? 'MISSING' : 'NOT_APPLICABLE');
      if (statusFilter === 'SATISFIED' && effStatus !== 'SATISFIED') return false;
      if (statusFilter === 'MISSING' && effStatus !== 'MISSING') return false;
      if (statusFilter === 'INSUFFICIENT_EVIDENCE' && effStatus !== 'INSUFFICIENT_EVIDENCE') return false;
      if (statusFilter === 'CONFIRMED_ABSENT' && effStatus !== 'CONFIRMED_ABSENT') return false;
      if (statusFilter === 'NOT_APPLICABLE' && effStatus !== 'NOT_APPLICABLE') return false;
      if (statusFilter === 'CONTRADICTORY' && effStatus !== 'CONTRADICTORY') return false;
      if (statusFilter === 'UNAVAILABLE' && effStatus !== 'UNAVAILABLE' && effStatus !== 'UNSUPPORTED') return false;
    }

    return true;
  });

  const getStatusBadge = (t: EvidenceToken) => {
    const status = t.status || (t.present ? 'SATISFIED' : t.required ? 'MISSING' : 'NOT_APPLICABLE');

    switch (status) {
      case 'SATISFIED':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#10B981]/20 text-[#34D399] border border-[#10B981]/40">
            <CheckCircle2 className="w-3 h-3" />
            <span>SATISFIED</span>
          </span>
        );
      case 'INSUFFICIENT_EVIDENCE':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#F59E0B]/20 text-[#FBBF24] border border-[#F59E0B]/40">
            <AlertTriangle className="w-3 h-3" />
            <span>INSUFFICIENT</span>
          </span>
        );
      case 'CONFIRMED_ABSENT':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40">
            <Ban className="w-3 h-3" />
            <span>CONFIRMED ABSENT</span>
          </span>
        );
      case 'NOT_APPLICABLE':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#64748B]/20 text-[#94A3B8] border border-[#64748B]/40">
            <Slash className="w-3 h-3" />
            <span>NOT APPLICABLE</span>
          </span>
        );
      case 'CONTRADICTORY':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#F97316]/20 text-[#FB923C] border border-[#F97316]/40">
            <AlertOctagon className="w-3 h-3" />
            <span>CONTRADICTORY</span>
          </span>
        );
      case 'UNAVAILABLE':
      case 'UNSUPPORTED':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#94A3B8]/20 text-[#CBD5E1] border border-[#94A3B8]/40">
            <HelpCircle className="w-3 h-3" />
            <span>{status}</span>
          </span>
        );
      case 'MISSING':
      default:
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#B91C1C]/20 text-[#F87171] border border-[#B91C1C]/40">
            <XCircle className="w-3 h-3" />
            <span>MISSING</span>
          </span>
        );
    }
  };

  return (
    <div className="space-y-4">
      {/* 1. Completeness & Summary Ribbon */}
      <div className="command-panel p-4 bg-[#10182B] border-[#43D9F5]/40 rounded-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-[#2B3A54]">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-[#43D9F5]/20 text-[#43D9F5]">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-[#43D9F5] font-bold">
                Evidence Life Cycle State Architecture
              </div>
              <h3 className="text-sm font-bold text-white">
                Conditioned Evidence Tokens ({allTokens.length} Total)
              </h3>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="bg-[#151F35] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
              <span className="text-[#8998AF]">Completeness C_ev: </span>
              <strong className="text-[#34D399]">{(completeness * 100).toFixed(1)}%</strong>
            </div>
            <div className="bg-[#151F35] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
              <span className="text-[#8998AF]">Satisfied: </span>
              <strong className="text-white">{satisfiedCount} / {requiredTokens.length} req</strong>
            </div>
          </div>
        </div>

        {/* 8-State Metric Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2 text-center text-xs font-mono">
          <button
            onClick={() => setStatusFilter(statusFilter === 'SATISFIED' ? 'all' : 'SATISFIED')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'SATISFIED' ? 'border-[#34D399] bg-[#10B981]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#34D399]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Satisfied</div>
            <div className="text-base font-extrabold text-[#34D399] mt-0.5">{satisfiedCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'MISSING' ? 'all' : 'MISSING')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'MISSING' ? 'border-[#F87171] bg-[#B91C1C]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#F87171]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Missing</div>
            <div className="text-base font-extrabold text-[#F87171] mt-0.5">{missingCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'INSUFFICIENT_EVIDENCE' ? 'all' : 'INSUFFICIENT_EVIDENCE')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'INSUFFICIENT_EVIDENCE' ? 'border-[#FBBF24] bg-[#F59E0B]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#FBBF24]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Insufficient</div>
            <div className="text-base font-extrabold text-[#FBBF24] mt-0.5">{insufficientCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'CONFIRMED_ABSENT' ? 'all' : 'CONFIRMED_ABSENT')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'CONFIRMED_ABSENT' ? 'border-[#A78BFA] bg-[#8B5CF6]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#A78BFA]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Absent</div>
            <div className="text-base font-extrabold text-[#A78BFA] mt-0.5">{absentCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'CONTRADICTORY' ? 'all' : 'CONTRADICTORY')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'CONTRADICTORY' ? 'border-[#FB923C] bg-[#F97316]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#FB923C]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Conflict</div>
            <div className="text-base font-extrabold text-[#FB923C] mt-0.5">{contradictoryCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'NOT_APPLICABLE' ? 'all' : 'NOT_APPLICABLE')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'NOT_APPLICABLE' ? 'border-[#94A3B8] bg-[#64748B]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#94A3B8]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">N/A</div>
            <div className="text-base font-extrabold text-[#94A3B8] mt-0.5">{notApplicableCount}</div>
          </button>

          <button
            onClick={() => setStatusFilter(statusFilter === 'UNAVAILABLE' ? 'all' : 'UNAVAILABLE')}
            className={`p-2 rounded-lg border transition-all ${
              statusFilter === 'UNAVAILABLE' ? 'border-[#CBD5E1] bg-[#94A3B8]/20' : 'border-[#2B3A54] bg-[#151F35] hover:border-[#CBD5E1]'
            }`}
          >
            <div className="text-[10px] text-[#8998AF]">Unavail</div>
            <div className="text-base font-extrabold text-[#CBD5E1] mt-0.5">{unavailableCount}</div>
          </button>
        </div>
      </div>

      {/* Filter and Table Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center space-x-2">
          <Filter className="w-3.5 h-3.5 text-[#43D9F5]" />
          <span className="text-[#8998AF]">Requirement Filter:</span>
          <select
            value={requiredFilter}
            onChange={(e) => setRequiredFilter(e.target.value)}
            className="bg-[#10182B] border border-[#2B3A54] rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-[#43D9F5]"
          >
            <option value="all">All Tokens ({allTokens.length})</option>
            <option value="required">Required Only ({requiredTokens.length})</option>
            <option value="optional">Optional Only ({optionalTokens.length})</option>
          </select>
        </div>

        {statusFilter !== 'all' && (
          <button
            onClick={() => setStatusFilter('all')}
            className="text-[11px] text-[#43D9F5] hover:underline"
          >
            Reset status filter ({statusFilter})
          </button>
        )}
      </div>

      {/* 2. Detailed Token Cards / Table */}
      <div className="space-y-2.5">
        {filteredTokens.map((tok) => {
          const isReq = tok.required;
          const status = tok.status || (tok.present ? 'SATISFIED' : isReq ? 'MISSING' : 'NOT_APPLICABLE');
          const hasCalibrated = tok.calibrated_confidence != null && tok.calibrated_confidence > 0;
          const hasRaw = tok.raw_confidence != null && tok.raw_confidence > 0;

          return (
            <div
              key={tok.token_id}
              className="command-panel p-3.5 bg-[#151F35] border border-[#2B3A54] rounded-xl text-xs font-mono space-y-2 hover:border-[#3A4D6F] transition-all"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center space-x-2">
                  <span className="text-[#43D9F5] font-extrabold text-sm">{tok.token_id}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    isReq
                      ? 'bg-[#B91C1C]/20 text-[#F87171] border border-[#B91C1C]/40'
                      : 'bg-[#1E293B] text-[#94A3B8] border border-[#334155]'
                  }`}>
                    {isReq ? 'REQUIRED' : 'OPTIONAL'}
                  </span>
                  <span className="text-[#8998AF] uppercase text-[10px]">
                    via <strong className="text-white">{tok.source_agent}</strong>
                  </span>
                </div>

                <div className="flex items-center space-x-2">
                  {getStatusBadge(tok)}
                </div>
              </div>

              {/* Description */}
              <p className="text-[#F1F5F9] font-sans text-xs leading-relaxed">
                {tok.description}
              </p>

              {/* Confidence & Epistemic Uncertainty Breakdown */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 border-t border-[#2B3A54]/50 text-[11px]">
                <div>
                  <span className="text-[#8998AF] block text-[10px]">Calibrated Conf (&sigma;_cal):</span>
                  <strong className={hasCalibrated ? 'text-[#34D399]' : 'text-[#8998AF]'}>
                    {hasCalibrated ? `${(tok.calibrated_confidence * 100).toFixed(1)}%` : '—'}
                  </strong>
                </div>

                <div>
                  <span className="text-[#8998AF] block text-[10px]">Raw Score (s_raw):</span>
                  <strong className={hasRaw ? 'text-[#818CF8]' : 'text-[#8998AF]'}>
                    {hasRaw ? `${(tok.raw_confidence * 100).toFixed(1)}%` : '—'}
                  </strong>
                </div>

                <div>
                  <span className="text-[#8998AF] block text-[10px]">Epistemic Uncertainty:</span>
                  <strong className={tok.uncertainty > 0 ? 'text-[#FBBF24]' : 'text-[#8998AF]'}>
                    {tok.uncertainty > 0 ? tok.uncertainty.toFixed(3) : '0.000'}
                  </strong>
                </div>

                <div>
                  <span className="text-[#8998AF] block text-[10px]">Calibration Applied:</span>
                  <strong className={hasCalibrated ? 'text-[#34D399]' : 'text-[#8998AF]'}>
                    {hasCalibrated ? 'Platt/Beta Scaling' : 'None (Uncalibrated)'}
                  </strong>
                </div>
              </div>

              {/* Status Explanation / Reason */}
              {tok.status_reason && (
                <div className="p-2 rounded bg-[#10182B] border border-[#2B3A54] text-[11px] text-[#B4C0D4] font-sans">
                  <span className="text-[#43D9F5] font-mono font-bold text-[10px] uppercase block mb-0.5">
                    Life Cycle Reason:
                  </span>
                  <span>{tok.status_reason}</span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
