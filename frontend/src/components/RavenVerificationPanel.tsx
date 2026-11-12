import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  AlertCircle,
  HelpCircle,
  UserCheck,
  FileCheck,
  ChevronDown,
  ChevronUp,
  Cpu,
  Layers,
  Scale,
  Activity,
  FileText,
  AlertOctagon,
  BookOpen
} from 'lucide-react';
import { RavenAssessment, HumanEscalationDossier, RavenDecision, HumanReviewRecord } from '../types';

interface Props {
  assessment?: RavenAssessment | null;
  dossier?: HumanEscalationDossier | null;
  humanReviewRecord?: HumanReviewRecord | null;
  riskScore?: number | null;
  decision?: RavenDecision | null;
  evidenceCompleteness?: number | null;
  conflictEntropy?: number | null;
}

export const RavenVerificationPanel: React.FC<Props> = ({
  assessment,
  dossier,
  humanReviewRecord,
  riskScore,
  decision,
  evidenceCompleteness,
  conflictEntropy
}) => {
  const [showFormulaDetails, setShowFormulaDetails] = useState(false);
  const [auditorName, setAuditorName] = useState('');
  const [auditorId, setAuditorId] = useState('');
  const [auditVerdict, setAuditVerdict] = useState<'APPROVED' | 'REJECTED' | ''>('');
  const [auditSigned, setAuditSigned] = useState(false);
  const [signOffSuccess, setSignOffSuccess] = useState(false);

  const effectiveDecision = decision || assessment?.decision || 'ACCEPT';
  const effectiveRisk = riskScore ?? assessment?.risk_score ?? 0.0;
  const effectiveCompleteness = evidenceCompleteness ?? assessment?.evidence_completeness ?? 1.0;
  const effectiveEntropy = conflictEntropy ?? assessment?.conflict_entropy ?? 0.0;
  const effectiveUncertainty = assessment?.uncertainty_index ?? 0.0;
  const effectiveDossier = dossier || assessment?.escalation_dossier;
  const effectiveReviewRecord = humanReviewRecord || assessment?.human_review_record;
  const effectiveSafetyVeto = assessment?.safety_veto_applied;

  const handleSignOff = (e: React.FormEvent) => {
    e.preventDefault();
    if (auditorName && auditorId && auditVerdict && auditSigned) {
      setSignOffSuccess(true);
    }
  };

  return (
    <div className="space-y-4">
      {/* Hard Safety Veto Alert Banner */}
      {effectiveSafetyVeto && (
        <div className="p-3.5 rounded-xl bg-[#7f1d1d] border-2 border-[#ef4444] text-white flex items-center space-x-3 shadow-lg">
          <AlertOctagon className="w-6 h-6 text-[#fca5a5] flex-shrink-0 animate-pulse" />
          <div className="text-xs font-mono">
            <span className="font-extrabold text-[#fca5a5] block uppercase tracking-wider">
              HARD SAFETY GATE VETO TRIGGERED
            </span>
            <span className="text-white font-sans text-xs">
              {effectiveSafetyVeto}
            </span>
          </div>
        </div>
      )}

      {/* 1. Tri-State Decision Policy Hero Banner */}
      {effectiveDecision === 'ABSTAIN' ? (
        <div className="p-4 rounded-xl bg-gradient-to-r from-[#B91C1C]/25 to-[#450a0a]/50 border-2 border-[#B91C1C] shadow-lg shadow-[#B91C1C]/10 space-y-2">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-[#B91C1C] text-white animate-pulse">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-[#F87171] font-bold">
                RAVEN Tri-State Decision: ABSTAIN (Safety Clearance Withheld)
              </div>
              <h3 className="text-sm font-bold text-white">
                Safety Clearance Refused — Mandatory Human Audit Mandated
              </h3>
            </div>
          </div>
          <p className="text-xs text-[#FECACA] leading-relaxed">
            Composite operational risk index <strong className="text-white font-mono">R = {effectiveRisk.toFixed(3)}</strong> exceeds the safety threshold (&tau;<sub>escalate</sub> = 0.650) or a hard safety veto was applied. Automated clearance is strictly withheld; physical inspection is required.
          </p>
          <div className="text-[11px] font-mono text-[#F87171] bg-[#450a0a]/70 px-2.5 py-1 rounded border border-[#B91C1C]/40">
            &bull; Notice: An ABSTAIN outcome indicates safety refusal, never successful automated sign-off.
          </div>
        </div>
      ) : effectiveDecision === 'VERIFY' ? (
        <div className="p-4 rounded-xl bg-gradient-to-r from-[#D97706]/20 to-[#78350f]/40 border border-[#F59E0B] shadow-md space-y-2">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-[#F59E0B] text-[#0B1020]">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-[#FBBF24] font-bold">
                RAVEN Tri-State Decision: VERIFY (Bounded Dynamic Pass)
              </div>
              <h3 className="text-sm font-bold text-white">
                Resolvable Ambiguity Detected — Secondary Verification Executed
              </h3>
            </div>
          </div>
          <p className="text-xs text-[#FDE68A] leading-relaxed">
            Operational risk <strong className="text-white font-mono">R = {effectiveRisk.toFixed(3)}</strong> falls in the intermediate verification band [0.35, 0.65). Additional targeted evidence passes were completed to confirm compliance.
          </p>
        </div>
      ) : (
        <div className="p-4 rounded-xl bg-gradient-to-r from-[#059669]/20 to-[#064e3b]/40 border border-[#10B981] shadow-md space-y-2">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-[#10B981] text-[#0B1020]">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[11px] font-mono uppercase tracking-wider text-[#34D399] font-bold">
                RAVEN Tri-State Decision: ACCEPT
              </div>
              <h3 className="text-sm font-bold text-white">
                Automated Evidence Certified — Low Operational Risk
              </h3>
            </div>
          </div>
          <p className="text-xs text-[#A7F3D0] leading-relaxed">
            Operational risk <strong className="text-white font-mono">R = {effectiveRisk.toFixed(3)}</strong> is below threshold (&tau;<sub>accept</sub> &lt; 0.350). Evidence requirements are satisfied with calibrated certainty.
          </p>
        </div>
      )}

      {/* 2. Key Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] flex flex-col justify-between">
          <div className="flex items-center justify-between text-[11px] font-mono text-[#8998AF]">
            <span>Risk Index R</span>
            <Activity className="w-3.5 h-3.5 text-[#43D9F5]" />
          </div>
          <div className="mt-2 flex items-baseline space-x-1">
            <span className={`text-xl font-mono font-extrabold ${effectiveRisk >= 0.65 ? 'text-[#F87171]' : effectiveRisk >= 0.35 ? 'text-[#FBBF24]' : 'text-[#34D399]'}`}>
              {effectiveRisk.toFixed(3)}
            </span>
            <span className="text-[10px] text-[#8998AF]">/ 1.0</span>
          </div>
          <div className="w-full bg-[#1B2942] h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className={`h-full ${effectiveRisk >= 0.65 ? 'bg-[#F87171]' : effectiveRisk >= 0.35 ? 'bg-[#FBBF24]' : 'bg-[#34D399]'}`}
              style={{ width: `${Math.min(effectiveRisk * 100, 100)}%` }}
            />
          </div>
        </div>

        <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] flex flex-col justify-between">
          <div className="flex items-center justify-between text-[11px] font-mono text-[#8998AF]">
            <span>Completeness C<sub>ev</sub></span>
            <Layers className="w-3.5 h-3.5 text-[#34D399]" />
          </div>
          <div className="mt-2 flex items-baseline space-x-1">
            <span className="text-xl font-mono font-extrabold text-[#34D399]">
              {(effectiveCompleteness * 100).toFixed(1)}%
            </span>
          </div>
          <div className="w-full bg-[#1B2942] h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="h-full bg-[#34D399]"
              style={{ width: `${Math.min(effectiveCompleteness * 100, 100)}%` }}
            />
          </div>
        </div>

        <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] flex flex-col justify-between">
          <div className="flex items-center justify-between text-[11px] font-mono text-[#8998AF]">
            <span>Conflict Entropy H</span>
            <Scale className="w-3.5 h-3.5 text-[#FBBF24]" />
          </div>
          <div className="mt-2 flex items-baseline space-x-1">
            <span className={`text-xl font-mono font-extrabold ${effectiveEntropy > 0.20 ? 'text-[#FBBF24]' : 'text-white'}`}>
              {effectiveEntropy.toFixed(3)}
            </span>
            <span className="text-[10px] text-[#8998AF]">nats</span>
          </div>
          <div className="w-full bg-[#1B2942] h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="h-full bg-[#FBBF24]"
              style={{ width: `${Math.min(effectiveEntropy * 100, 100)}%` }}
            />
          </div>
        </div>

        <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] flex flex-col justify-between">
          <div className="flex items-center justify-between text-[11px] font-mono text-[#8998AF]">
            <span>Epistemic Uncert.</span>
            <Cpu className="w-3.5 h-3.5 text-[#818CF8]" />
          </div>
          <div className="mt-2 flex items-baseline space-x-1">
            <span className="text-xl font-mono font-extrabold text-[#818CF8]">
              {effectiveUncertainty.toFixed(3)}
            </span>
          </div>
          <div className="w-full bg-[#1B2942] h-1.5 rounded-full mt-2 overflow-hidden">
            <div
              className="h-full bg-[#818CF8]"
              style={{ width: `${Math.min(effectiveUncertainty * 100, 100)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Collapsible Mathematical Formulation Callout */}
      <div className="p-3 bg-[#10182B] rounded-lg border border-[#2B3A54] text-xs font-mono text-[#8998AF]">
        <button
          onClick={() => setShowFormulaDetails(!showFormulaDetails)}
          className="w-full flex items-center justify-between hover:text-white transition-colors"
        >
          <div className="flex items-center space-x-2">
            <Scale className="w-3.5 h-3.5 text-[#43D9F5]" />
            <span className="font-bold text-[#43D9F5]">RAVEN Multi-Factor Operational Risk Formulation</span>
          </div>
          {showFormulaDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showFormulaDetails && (
          <div className="mt-3 pt-3 border-t border-[#2B3A54] space-y-2 text-[11px] text-[#B4C0D4]">
            <div className="p-2.5 rounded bg-[#070A14] border border-[#2B3A54] font-mono text-center text-white">
              R(E, q) = w<sub>c</sub> (1 - C<sub>ev</sub>) + w<sub>u</sub> U(E) + w<sub>e</sub> H<sub>conflict</sub> + w<sub>s</sub> S<sub>hazard</sub>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[10px]">
              <div>&bull; w<sub>c</sub> = 0.35 (Incompleteness penalty)</div>
              <div>&bull; w<sub>u</sub> = 0.25 (Calibrated uncertainty)</div>
              <div>&bull; w<sub>e</sub> = 0.20 (Cross-agent disagreement)</div>
              <div>&bull; w<sub>s</sub> = 0.20 (Operational hazard severity)</div>
            </div>
            <div className="text-[10px] text-[#8998AF] border-t border-[#2B3A54]/60 pt-1.5">
              Decision Boundary: R &ge; 0.65 &rarr; <strong>ABSTAIN (Safety Clearance Withheld)</strong>; 0.35 &le; R &lt; 0.65 &rarr; <strong>VERIFY (Cycle bounded &le; 1)</strong>; R &lt; 0.35 &rarr; <strong>ACCEPT</strong>.
            </div>
          </div>
        )}
      </div>

      {/* 3. Query-Conditioned Evidence Repertoire Table */}
      {assessment?.evidence_repertoire && assessment.evidence_repertoire.tokens?.length > 0 && (
        <div className="command-panel p-4 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-[#2B3A54]">
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-[#43D9F5]" />
              <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                Conditioned Evidence Repertoire Satisfaction
              </h4>
            </div>
            <span className="text-[11px] font-mono text-[#8998AF]">
              {assessment.evidence_repertoire.total_satisfied} / {assessment.evidence_repertoire.total_required} Required Satisfied
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-[#2B3A54] text-[#8998AF] text-[10px] uppercase">
                  <th className="py-2 px-2">Token UID</th>
                  <th className="py-2 px-2">Evidence Requirement</th>
                  <th className="py-2 px-2">Specialist</th>
                  <th className="py-2 px-2">Calibrated Conf.</th>
                  <th className="py-2 px-2">Uncertainty</th>
                  <th className="py-2 px-2 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2B3A54]/50">
                {assessment.evidence_repertoire.tokens.map((tok) => (
                  <tr key={tok.token_id} className="hover:bg-[#151F35] transition-colors">
                    <td className="py-2 px-2 font-bold text-[#43D9F5]">{tok.token_id}</td>
                    <td className="py-2 px-2 text-[#F1F5F9]">{tok.description}</td>
                    <td className="py-2 px-2 text-[#8998AF] uppercase">{tok.source_agent}</td>
                    <td className="py-2 px-2 text-[#34D399]">
                      {tok.present ? `${(tok.calibrated_confidence * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 px-2 text-[#818CF8]">
                      {tok.uncertainty > 0 ? tok.uncertainty.toFixed(2) : '—'}
                    </td>
                    <td className="py-2 px-2 text-right">
                      {tok.status ? (
                        <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                          tok.status === 'SATISFIED'
                            ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]/40'
                            : tok.status === 'CONFIRMED_ABSENT'
                            ? 'bg-[#8B5CF6]/20 text-[#A78BFA] border border-[#8B5CF6]/40'
                            : tok.status === 'NOT_APPLICABLE'
                            ? 'bg-[#64748B]/20 text-[#94A3B8] border border-[#64748B]/40'
                            : tok.status === 'INSUFFICIENT_EVIDENCE'
                            ? 'bg-[#F59E0B]/20 text-[#FBBF24] border border-[#F59E0B]/40'
                            : 'bg-[#B91C1C]/20 text-[#F87171] border border-[#B91C1C]/40'
                        }`}>
                          <span>{tok.status}</span>
                        </span>
                      ) : tok.present ? (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#10B981]/20 text-[#34D399] border border-[#10B981]/40">
                          <CheckCircle2 className="w-3 h-3" />
                          <span>Satisfied</span>
                        </span>
                      ) : tok.required ? (
                        <span className="inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold bg-[#B91C1C]/20 text-[#F87171] border border-[#B91C1C]/40">
                          <XCircle className="w-3 h-3" />
                          <span>Missing</span>
                        </span>
                      ) : (
                        <span className="px-2 py-0.5 rounded text-[10px] text-[#8998AF] bg-[#151F35] border border-[#2B3A54]">
                          Optional
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* 4. Cross-Agent Conflict Discrepancies Table */}
      {assessment?.conflict_report?.discrepancies && assessment.conflict_report.discrepancies.length > 0 && (
        <div className="command-panel p-4 space-y-3">
          <div className="flex items-center space-x-2 pb-2 border-b border-[#2B3A54]">
            <Scale className="w-4 h-4 text-[#FBBF24]" />
            <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
              Cross-Agent Epistemic Conflict Discrepancies ({assessment.conflict_report.discrepancies.length})
            </h4>
          </div>

          <div className="space-y-2">
            {assessment.conflict_report.discrepancies.map((disc, idx) => (
              <div key={disc.discrepancy_id || idx} className="p-3 rounded-lg bg-[#151F35] border border-[#F59E0B]/30 text-xs font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-[#FBBF24] font-bold">{disc.discrepancy_id}: {disc.type}</span>
                  <span className="text-[10px] text-[#8998AF]">Agents: {disc.agents_involved?.join(' ↔ ')}</span>
                </div>
                <p className="text-[#B4C0D4] font-sans text-xs">{disc.details}</p>
                <div className="text-[10px] text-[#FBBF24]">Entropy Contribution &Delta;H: {disc.entropy_contribution?.toFixed(3)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 5. Forensic Human Review Record */}
      {effectiveReviewRecord && (
        <div className="command-panel p-4 bg-gradient-to-b from-[#1A1823] to-[#121018] border-2 border-[#B91C1C] rounded-xl space-y-3.5">
          <div className="flex items-center justify-between pb-2 border-b border-[#B91C1C]/40">
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-[#F87171]" />
              <h4 className="text-xs font-mono font-bold text-white uppercase tracking-wider">
                Forensic Human Review Record (Audit UID: {effectiveReviewRecord.inspection_id?.slice(0, 10)})
              </h4>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-extrabold bg-[#B91C1C] text-white">
              {effectiveReviewRecord.decision} ESCALATION
            </span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="p-2.5 bg-[#070A14] rounded-lg border border-[#2B3A54]">
              <span className="text-[#F87171] font-bold block text-[10px] uppercase">Review Trigger Rationale:</span>
              <p className="text-[#F1F5F9] font-sans text-xs mt-0.5">{effectiveReviewRecord.reason}</p>
            </div>

            {effectiveReviewRecord.relevant_regions && effectiveReviewRecord.relevant_regions.length > 0 && (
              <div className="p-2.5 bg-[#070A14] rounded-lg border border-[#2B3A54]">
                <span className="text-[#43D9F5] font-bold block text-[10px] uppercase">
                  Flagged Regions Requiring Physical Verification ({effectiveReviewRecord.relevant_regions.length}):
                </span>
                <div className="space-y-1 mt-1">
                  {effectiveReviewRecord.relevant_regions.map((reg, idx) => (
                    <div key={idx} className="text-[11px] text-[#CBD5E1] flex items-center space-x-2">
                      <span className="text-[#43D9F5]">&bull; {reg.specialist || 'Specialist'}:</span>
                      <span>BBox [{reg.bbox ? reg.bbox.map(v => Math.round(v)).join(', ') : 'Global'}]</span>
                      {reg.finding_uid && <span className="text-[#8998AF]">({reg.finding_uid})</span>}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {effectiveReviewRecord.missing_evidence && effectiveReviewRecord.missing_evidence.length > 0 && (
              <div className="p-2.5 bg-[#070A14] rounded-lg border border-[#2B3A54]">
                <span className="text-[#FBBF24] font-bold block text-[10px] uppercase">
                  Missing Critical Visual Evidence Tokens:
                </span>
                <div className="flex flex-wrap gap-1.5 mt-1">
                  {effectiveReviewRecord.missing_evidence.map((tok, idx) => (
                    <span key={idx} className="px-2 py-0.5 rounded bg-[#B91C1C]/20 border border-[#B91C1C]/40 text-[#F87171] text-[10px]">
                      {tok}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {effectiveReviewRecord.conflicting_findings && effectiveReviewRecord.conflicting_findings.length > 0 && (
              <div className="p-2.5 bg-[#070A14] rounded-lg border border-[#2B3A54]">
                <span className="text-[#FB923C] font-bold block text-[10px] uppercase">
                  Unresolved Contradictions ({effectiveReviewRecord.conflicting_findings.length}):
                </span>
                <div className="space-y-1 mt-1">
                  {effectiveReviewRecord.conflicting_findings.map((cf, idx) => (
                    <div key={idx} className="text-[11px] text-[#CBD5E1]">
                      <strong className="text-[#FB923C]">{cf.conflict_id}: </strong>
                      <span>{cf.description}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {effectiveReviewRecord.regulatory_clauses && effectiveReviewRecord.regulatory_clauses.length > 0 && (
              <div className="p-2.5 bg-[#070A14] rounded-lg border border-[#2B3A54]">
                <div className="flex items-center space-x-1.5 text-[#34D399] font-bold text-[10px] uppercase mb-1">
                  <BookOpen className="w-3.5 h-3.5" />
                  <span>Applicable Statutory & Industrial Standards:</span>
                </div>
                <div className="space-y-0.5 text-[11px] text-[#A7F3D0]">
                  {effectiveReviewRecord.regulatory_clauses.map((clause, idx) => (
                    <div key={idx}>&bull; {clause}</div>
                  ))}
                </div>
              </div>
            )}

            {effectiveReviewRecord.disclaimer && (
              <div className="p-2 bg-[#070A14] rounded border border-[#2B3A54]/60 text-[10px] text-[#8998AF] font-sans">
                <strong>Legal Disclaimer: </strong>
                {effectiveReviewRecord.disclaimer}
              </div>
            )}
          </div>
        </div>
      )}

      {/* 6. Mandatory Human Escalation Dossier & Interactive Physical Audit Sign-Off */}
      {(effectiveDecision === 'ABSTAIN' || effectiveDossier) && (
        <div className="command-panel p-4 bg-gradient-to-b from-[#1E1B24] to-[#14121A] border-2 border-[#B91C1C] rounded-xl space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-[#B91C1C]/40">
            <div className="flex items-center space-x-2.5">
              <ShieldAlert className="w-5 h-5 text-[#F87171]" />
              <h4 className="text-xs font-mono font-extrabold text-white uppercase tracking-wider">
                Physical On-Site Verification Protocol Sign-Off
              </h4>
            </div>
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-extrabold bg-[#B91C1C] text-white">
              {effectiveDossier?.urgency_level || 'CRITICAL'} URGENCY
            </span>
          </div>

          <div className="space-y-3 text-xs">
            {effectiveDossier?.recommended_human_actions && effectiveDossier.recommended_human_actions.length > 0 && (
              <div className="p-3 rounded-lg bg-[#070A14] border border-[#2B3A54]">
                <span className="text-[11px] font-mono text-[#43D9F5] uppercase font-bold block mb-1">
                  Mandated On-Site Inspection Protocol:
                </span>
                <ul className="list-disc list-inside space-y-1 text-[#CBD5E1]">
                  {effectiveDossier.recommended_human_actions.map((act, i) => (
                    <li key={i}>{act}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Interactive Physical Auditor Sign-Off Form */}
            <div className="p-3.5 rounded-lg bg-[#0B1020] border border-[#3A4D6F] space-y-3">
              <div className="flex items-center space-x-2 text-white font-mono font-bold text-xs">
                <UserCheck className="w-4 h-4 text-[#43D9F5]" />
                <span>On-Site Certified Safety Officer Physical Sign-Off</span>
              </div>

              {signOffSuccess ? (
                <div className="p-3 rounded-lg bg-[#10B981]/20 border border-[#10B981] text-[#34D399] font-mono text-xs flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                  <span>
                    Audit recorded: Certified by {auditorName} (ID: {auditorId}). Verdict: {auditVerdict}. Status: Physical Verification Landed.
                  </span>
                </div>
              ) : (
                <form onSubmit={handleSignOff} className="space-y-3 font-mono text-xs">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    <div>
                      <label className="text-[10px] text-[#8998AF] block mb-1">Inspector Full Name</label>
                      <input
                        type="text"
                        required
                        value={auditorName}
                        onChange={(e) => setAuditorName(e.target.value)}
                        placeholder="e.g. John Doe, Lead Safety Inspector"
                        className="w-full px-3 py-1.5 rounded bg-[#151F35] border border-[#2B3A54] text-white text-xs focus:outline-none focus:border-[#43D9F5]"
                      />
                    </div>
                    <div>
                      <label className="text-[10px] text-[#8998AF] block mb-1">Badge / License ID</label>
                      <input
                        type="text"
                        required
                        value={auditorId}
                        onChange={(e) => setAuditorId(e.target.value)}
                        placeholder="e.g. NDT-CERT-8842"
                        className="w-full px-3 py-1.5 rounded bg-[#151F35] border border-[#2B3A54] text-white text-xs focus:outline-none focus:border-[#43D9F5]"
                      />
                    </div>
                  </div>

                  <div className="flex items-center space-x-4 pt-1">
                    <span className="text-[10px] text-[#8998AF]">Physical Audit Verdict:</span>
                    <label className="flex items-center space-x-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="verdict"
                        value="APPROVED"
                        checked={auditVerdict === 'APPROVED'}
                        onChange={() => setAuditVerdict('APPROVED')}
                        className="accent-[#34D399]"
                      />
                      <span className="text-[#34D399] text-xs">Site Verified Compliant</span>
                    </label>
                    <label className="flex items-center space-x-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="verdict"
                        value="REJECTED"
                        checked={auditVerdict === 'REJECTED'}
                        onChange={() => setAuditVerdict('REJECTED')}
                        className="accent-[#F87171]"
                      />
                      <span className="text-[#F87171] text-xs">Remediation Required</span>
                    </label>
                  </div>

                  <div className="pt-1">
                    <label className="flex items-start space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        required
                        checked={auditSigned}
                        onChange={(e) => setAuditSigned(e.target.checked)}
                        className="mt-0.5 accent-[#43D9F5]"
                      />
                      <span className="text-[10px] text-[#8998AF] leading-tight">
                        I confirm that a manual visual / NDT inspection was conducted and physical findings supersede automated decision support.
                      </span>
                    </label>
                  </div>

                  <button
                    type="submit"
                    className="w-full py-2 rounded-lg bg-[#B91C1C] hover:bg-[#991b1b] text-white font-bold text-xs transition-colors flex items-center justify-center space-x-2"
                  >
                    <FileCheck className="w-4 h-4" />
                    <span>Attest Physical Safety Inspection Sign-Off</span>
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
