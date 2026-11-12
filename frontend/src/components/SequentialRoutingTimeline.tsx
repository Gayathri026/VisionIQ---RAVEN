import React from 'react';
import {
  SequentialStepTelemetry
} from '../types';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Legend
} from 'recharts';
import {
  GitCommit,
  CheckCircle2,
  XCircle,
  Clock,
  Cpu,
  Layers,
  ArrowRight,
  TrendingDown,
  AlertCircle,
  HelpCircle,
  Compass,
  Zap,
  Target
} from 'lucide-react';

interface Props {
  telemetry?: SequentialStepTelemetry[];
  deficitHistory?: number[];
  currentDeficit?: number;
  stoppingReason?: string | null;
  targetDeficit?: number;
}

export const SequentialRoutingTimeline: React.FC<Props> = ({
  telemetry = [],
  deficitHistory = [],
  currentDeficit,
  stoppingReason,
  targetDeficit = 0.15
}) => {
  if (!telemetry || telemetry.length === 0) {
    return (
      <div className="command-panel rounded-xl p-8 text-center text-[#8998AF] space-y-3">
        <Compass className="w-10 h-10 mx-auto text-[#546E9B]" />
        <h3 className="text-base font-bold text-[#F1F5F9]">Sequential Evidence-Gain Routing Telemetry</h3>
        <p className="text-xs text-[#8998AF] max-w-md mx-auto leading-relaxed">
          Run an inspection with Sequential Routing enabled to observe iterative deficit reduction, candidate selection efficiency, and marginal gain prediction error.
        </p>
      </div>
    );
  }

  // Build chart data for Deficit Decay Curve
  // If deficitHistory is available, use it; otherwise compute from telemetry
  const deficitCurveData = (deficitHistory && deficitHistory.length > 0)
    ? deficitHistory.map((d, idx) => ({
        step: `Step ${idx}`,
        stepNum: idx,
        deficit: parseFloat(d.toFixed(4)),
        threshold: targetDeficit
      }))
    : [
        {
          step: 'Step 0',
          stepNum: 0,
          deficit: parseFloat(telemetry[0].current_deficit.toFixed(4)),
          threshold: targetDeficit
        },
        ...telemetry.map((t) => ({
          step: `Step ${t.step}`,
          stepNum: t.step,
          deficit: parseFloat((t.post_execution_deficit ?? t.current_deficit).toFixed(4)),
          threshold: targetDeficit
        }))
      ];

  // Build chart data for Predicted vs Observed Reduction
  const predictionComparisonData = telemetry
    .filter(t => t.selected_specialist != null)
    .map(t => {
      const selected = t.selected_specialist || 'specialist';
      const predicted = t.candidate_predicted_gains?.[selected] ?? 0;
      const observed = t.signed_observed_deficit_reduction ?? 0;
      const error = t.prediction_error ?? (predicted - observed);
      return {
        step: `Step ${t.step} (${selected})`,
        predicted: parseFloat(predicted.toFixed(3)),
        observed: parseFloat(observed.toFixed(3)),
        error: parseFloat(error.toFixed(3))
      };
    });

  const finalDeficit = currentDeficit ?? (
    telemetry.length > 0
      ? telemetry[telemetry.length - 1].post_execution_deficit ?? telemetry[telemetry.length - 1].current_deficit
      : 0
  );

  return (
    <div className="space-y-5">
      {/* 1. Header Banner & Stopping Reason Card */}
      <div className="command-panel p-4 bg-[#10182B] border-[#43D9F5]/40 rounded-xl space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-[#2B3A54]">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-lg bg-[#43D9F5]/20 text-[#43D9F5]">
              <Target className="w-5 h-5" />
            </div>
            <div>
              <div className="text-[10px] font-mono uppercase tracking-wider text-[#43D9F5] font-bold">
                Adaptive Decision Process
              </div>
              <h3 className="text-sm font-bold text-white">
                Sequential Evidence-Gain Orchestration Timeline
              </h3>
            </div>
          </div>

          <div className="flex items-center space-x-3 text-xs font-mono">
            <div className="bg-[#151F35] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
              <span className="text-[#8998AF]">Steps: </span>
              <strong className="text-white">{telemetry.length}</strong>
            </div>
            <div className="bg-[#151F35] px-3 py-1.5 rounded-lg border border-[#2B3A54]">
              <span className="text-[#8998AF]">Final Deficit D: </span>
              <strong className={finalDeficit <= targetDeficit ? 'text-[#34D399]' : 'text-[#FBBF24]'}>
                {finalDeficit.toFixed(3)}
              </strong>
              <span className="text-[10px] text-[#8998AF]"> / &tau;={targetDeficit.toFixed(2)}</span>
            </div>
          </div>
        </div>

        {stoppingReason && (
          <div className="flex items-center space-x-2.5 px-3 py-2 rounded-lg bg-[#151F35] border border-[#2B3A54] text-xs font-mono">
            <Compass className="w-4 h-4 text-[#43D9F5] flex-shrink-0" />
            <span className="text-[#8998AF]">Termination Criterion:</span>
            <span className="text-[#34D399] font-bold">{stoppingReason}</span>
          </div>
        )}
      </div>

      {/* 2. Embedded Recharts Visualizations */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Chart A: Deficit Decay Curve */}
        <div className="command-panel p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-mono font-bold text-white">
              <TrendingDown className="w-4 h-4 text-[#43D9F5]" />
              <span>Deficit Decay Curve D(S_t)</span>
            </div>
            <span className="text-[10px] font-mono text-[#8998AF]">&tau; target = {targetDeficit.toFixed(2)}</span>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={deficitCurveData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2B3A54" />
                <XAxis dataKey="step" stroke="#8998AF" fontSize={10} />
                <YAxis stroke="#8998AF" fontSize={10} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#10182B', borderColor: '#2B3A54', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }}
                />
                <ReferenceLine
                  y={targetDeficit}
                  stroke="#F87171"
                  strokeDasharray="4 4"
                  label={{ value: `\u03c4 = ${targetDeficit}`, fill: '#F87171', fontSize: 10, position: 'insideTopRight' }}
                />
                <Line
                  type="monotone"
                  dataKey="deficit"
                  name="Evidence Deficit"
                  stroke="#43D9F5"
                  strokeWidth={2.5}
                  dot={{ r: 4, fill: '#43D9F5' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart B: Predicted vs Observed Reduction */}
        <div className="command-panel p-4 space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2 text-xs font-mono font-bold text-white">
              <Zap className="w-4 h-4 text-[#34D399]" />
              <span>Predicted vs. Observed Gain &Delta;D</span>
            </div>
            <span className="text-[10px] font-mono text-[#8998AF]">Marginal Evidence Gain</span>
          </div>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={predictionComparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2B3A54" />
                <XAxis dataKey="step" stroke="#8998AF" fontSize={10} />
                <YAxis stroke="#8998AF" fontSize={10} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#10182B', borderColor: '#2B3A54', color: '#fff', fontSize: '11px', fontFamily: 'monospace' }}
                />
                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '4px' }} />
                <Bar dataKey="predicted" name="Predicted \u0394D" fill="#818CF8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="observed" name="Observed \u0394D" fill="#34D399" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 3. Step-by-Step Execution Sequence Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs font-mono text-[#8998AF] px-1">
          <span className="font-bold text-white uppercase tracking-wider">Iterative Execution Steps:</span>
          <span>{telemetry.length} Steps Executed</span>
        </div>

        {telemetry.map((step) => {
          const selected = step.selected_specialist;
          const predGain = selected && step.candidate_predicted_gains ? step.candidate_predicted_gains[selected] : 0;
          const obsGain = step.signed_observed_deficit_reduction ?? 0;
          const predErr = step.prediction_error ?? (predGain - obsGain);
          const isSuccess = step.execution_result === 'SUCCESS';

          return (
            <div
              key={step.step}
              className="command-panel p-4 bg-[#151F35] border border-[#2B3A54] rounded-xl space-y-3 text-xs font-mono"
            >
              {/* Step Card Header */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-[#2B3A54]">
                <div className="flex items-center space-x-2.5">
                  <span className="px-2.5 py-1 rounded-md bg-[#43D9F5]/20 text-[#43D9F5] font-extrabold text-xs border border-[#43D9F5]/40">
                    Step {step.step}
                  </span>
                  <div className="flex items-center space-x-1.5 text-white font-bold text-sm">
                    <span className="text-[#8998AF]">Invoked:</span>
                    <span className="text-[#34D399] uppercase">{selected || 'None (Terminated)'}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                    isSuccess
                      ? 'bg-[#10B981]/20 text-[#34D399] border border-[#10B981]/40'
                      : 'bg-[#B91C1C]/20 text-[#F87171] border border-[#B91C1C]/40'
                  }`}>
                    {isSuccess ? <CheckCircle2 className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                    <span>{step.execution_result}</span>
                  </span>

                  <span className="bg-[#10182B] text-[#8998AF] px-2 py-0.5 rounded border border-[#2B3A54] text-[10px]">
                    Findings: <strong className="text-white">{step.actual_findings_count}</strong>
                  </span>
                </div>
              </div>

              {/* Step Deficit Transition & Gain Metrics */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
                <div className="p-2.5 rounded bg-[#10182B] border border-[#2B3A54]">
                  <span className="text-[#8998AF] block text-[10px]">Pre-Step Deficit D(S_t)</span>
                  <strong className="text-[#FBBF24] font-bold text-sm">
                    {step.current_deficit.toFixed(3)}
                  </strong>
                </div>

                <div className="p-2.5 rounded bg-[#10182B] border border-[#2B3A54]">
                  <span className="text-[#8998AF] block text-[10px]">Post-Step Deficit D(S_t+1)</span>
                  <strong className="text-[#34D399] font-bold text-sm">
                    {step.post_execution_deficit != null ? step.post_execution_deficit.toFixed(3) : '—'}
                  </strong>
                </div>

                <div className="p-2.5 rounded bg-[#10182B] border border-[#2B3A54]">
                  <span className="text-[#8998AF] block text-[10px]">Observed Reduction &Delta;D_obs</span>
                  <strong className="text-[#43D9F5] font-bold text-sm">
                    {obsGain >= 0 ? `+${obsGain.toFixed(3)}` : obsGain.toFixed(3)}
                  </strong>
                </div>

                <div className="p-2.5 rounded bg-[#10182B] border border-[#2B3A54]">
                  <span className="text-[#8998AF] block text-[10px]">Prediction Error (\u0394D&#770; - \u0394D)</span>
                  <div className="flex items-center space-x-1">
                    <strong className={`font-bold text-sm ${Math.abs(predErr) < 0.05 ? 'text-[#34D399]' : 'text-[#818CF8]'}`}>
                      {predErr >= 0 ? `+${predErr.toFixed(3)}` : predErr.toFixed(3)}
                    </strong>
                    <span className="text-[9px] px-1 py-0.2 rounded bg-[#151F35] text-[#8998AF]">
                      {Math.abs(predErr) < 0.05 ? 'Accurate' : predErr > 0 ? 'Over' : 'Under'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Candidate Specialists Evaluated at this Step */}
              {step.candidate_agents && step.candidate_agents.length > 0 && (
                <div className="space-y-1.5 pt-1">
                  <div className="text-[10px] text-[#8998AF] uppercase font-bold tracking-wider">
                    Candidate Evaluation at Step {step.step}:
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-[11px] font-mono">
                      <thead>
                        <tr className="border-b border-[#2B3A54] text-[#8998AF] text-[10px]">
                          <th className="py-1 px-1.5">Candidate</th>
                          <th className="py-1 px-1.5 text-right">Pred Gain &Delta;D&#770;</th>
                          <th className="py-1 px-1.5 text-right">Est. Cost c_j</th>
                          <th className="py-1 px-1.5 text-right">Est. Latency L_j</th>
                          <th className="py-1 px-1.5 text-right">Selection Efficiency &eta;_j</th>
                          <th className="py-1 px-1.5 text-center">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#2B3A54]/50">
                        {step.candidate_agents.map((cand) => {
                          const isChosen = cand === selected;
                          const cGain = step.candidate_predicted_gains?.[cand] ?? 0;
                          const cCost = step.candidate_costs?.[cand] ?? 0;
                          const cLat = step.candidate_latencies?.[cand] ?? 0;
                          const cEff = step.candidate_efficiencies?.[cand] ?? 0;

                          return (
                            <tr
                              key={cand}
                              className={isChosen ? 'bg-[#43D9F5]/10 font-bold text-white' : 'text-[#B4C0D4]'}
                            >
                              <td className="py-1 px-1.5">
                                <span className={isChosen ? 'text-[#43D9F5]' : 'text-[#8998AF]'}>
                                  {isChosen ? '\u25b6 ' : '  '}{cand}
                                </span>
                              </td>
                              <td className="py-1 px-1.5 text-right text-[#34D399]">{cGain.toFixed(3)}</td>
                              <td className="py-1 px-1.5 text-right text-[#8998AF]">{cCost.toFixed(2)}</td>
                              <td className="py-1 px-1.5 text-right text-[#8998AF]">{cLat.toFixed(0)}ms</td>
                              <td className="py-1 px-1.5 text-right text-[#FBBF24]">{cEff.toFixed(4)}</td>
                              <td className="py-1 px-1.5 text-center">
                                {isChosen ? (
                                  <span className="px-1.5 py-0.5 rounded text-[9px] bg-[#34D399]/20 text-[#34D399] border border-[#34D399]/40">
                                    Selected
                                  </span>
                                ) : (
                                  <span className="text-[9px] text-[#8998AF]">Deferred</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Budgets Remaining */}
              <div className="flex items-center justify-between text-[10px] text-[#8998AF] pt-1 border-t border-[#2B3A54]/50">
                <span>Remaining Invocations: <strong className="text-white">{step.remaining_invocation_budget}</strong></span>
                <span>Remaining Time Budget: <strong className="text-white">{step.remaining_time_budget_ms.toFixed(0)} ms</strong></span>
                <span>Cleanup State: <strong className="text-[#34D399]">{step.cleanup_result}</strong></span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
