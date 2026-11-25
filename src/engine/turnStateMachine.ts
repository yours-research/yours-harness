import {
  ContextCompactor,
  type CompactionBudgetConfig,
  type CompactionPlan,
} from '../compaction/compactor.js';
import type {
  GpuPacingMode,
  GuardDecision,
  GuardEngine,
  HarnessMessage,
  MemoryProvider,
  MemoryRecord,
  RuntimeAdapter,
  ToolDefinition,
  ToolExecutionResult,
  ToolExecutor,
  ToolInvocationContext,
} from '../contracts.js';
import { createId } from '../utils/id.js';
import type { PrefixSlotCoordinator } from './speculative.js';
import { StreamOutputParser, type ParsedToolCall } from './streamParser.js';

export type TurnState =
  | 'idle'
  | 'prefilling'
  | 'streaming'
  | 'tool_executing'
  | 'compacting'
  | 'awaiting_approval'
  | 'aborted'
  | 'done';

export interface TurnStateTransition {
  from: TurnState;
  to: TurnState;
  timestampMs: number;
  reason?: string;
}

const VALID_TRANSITIONS: Record<TurnState, ReadonlySet<TurnState>> = {
  idle: new Set(['prefilling', 'compacting', 'streaming', 'aborted', 'done']),
  prefilling: new Set(['compacting', 'streaming', 'aborted']),
  compacting: new Set(['prefilling', 'streaming', 'aborted', 'done']),
  streaming: new Set(['tool_executing', 'awaiting_approval', 'compacting', 'done', 'aborted']),
  awaiting_approval: new Set(['tool_executing', 'compacting', 'streaming', 'aborted', 'done']),
  tool_executing: new Set(['compacting', 'streaming', 'done', 'aborted']),
  aborted: new Set(['idle']),
  done: new Set(['idle']),
};

/**
 * Explicit finite state machine governing a single harness turn lifecycle.
 */
export class TurnStateMachine {
  private current: TurnState = 'idle';
  private readonly history: TurnStateTransition[] = [];
  private readonly listeners = new Set<(transition: TurnStateTransition) => void>();

  get state(): TurnState {
    return this.current;
  }

  getHistory(): TurnStateTransition[] {
    return [...this.history];
  }

  subscribe(listener: (transition: TurnStateTransition) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  transition(to: TurnState, reason?: string): TurnStateTransition {
    if (this.current === to) {
      return {
        from: this.current,
        to,
        timestampMs: performance.now(),
        reason,
      };
    }

    const allowed = VALID_TRANSITIONS[this.current];
    if (!allowed.has(to)) {
      throw new Error(`Invalid TurnStateMachine transition: "${this.current}" -> "${to}"`);
    }

    const entry: TurnStateTransition = {
      from: this.current,
      to,
      timestampMs: performance.now(),
      reason,
    };
    this.current = to;
    this.history.push(entry);
    for (const listener of this.listeners) {
      listener(entry);
    }
    return entry;
  }

  reset(): void {
    if (this.current === 'idle') {
      return;
    }
    if (this.current === 'done' || this.current === 'aborted') {
      this.transition('idle', 'reset');
      return;
    }
    this.transition('aborted', 'forced reset');
    this.transition('idle', 'reset');
  }
}

export interface PendingApprovalRequest {
  toolCallId: string;
  toolName: string;
  arguments: Record<string, unknown>;
  decision: GuardDecision;
}

export interface ExecutedToolTrace {
  step: number;
  call: ParsedToolCall;
  result: ToolExecutionResult;
  guardDecision: GuardDecision;
  durationMs: number;
}

export interface TurnTelemetrySnapshot {
  ttftMs?: number;
  tokensPerSec?: number;
  kvTokens?: number;
  speculativeAcceptRate?: number;
  totalDurationMs: number;
  abortLatencyMs?: number;
}

export interface RunHarnessTurnParams {
  sessionId: string;
  environment?: ToolInvocationContext['environment'];
  adapter: RuntimeAdapter;
  messages: readonly HarnessMessage[];
  systemPrompt?: string;
  tools?: readonly ToolDefinition[];
  toolExecutor?: ToolExecutor;
  guard?: GuardEngine;
  memory?: MemoryProvider;
  compactor?: ContextCompactor;
  compactionConfig?: Partial<CompactionBudgetConfig>;
  slotCoordinator?: PrefixSlotCoordinator;
  activeOrigin?: string;
  explicitlyApprovedOrigins?: string[];
  maxToolSteps?: number;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  enableThinking?: boolean;
  pacingMode?: GpuPacingMode;
  stopSequences?: string[];
  previousRollingSummary?: string | null;
  previousRollingSummaryCutoffId?: string | null;
  previousCompiledMessages?: readonly HarnessMessage[];
  signal?: AbortSignal;
  stateMachine?: TurnStateMachine;
  onStateChange?: (transition: TurnStateTransition) => void;
  onTextDelta?: (delta: string, accumulatedText: string, step: number) => void;
  onReasoningDelta?: (delta: string, accumulatedReasoning: string, step: number) => void;
  onToolCallStart?: (info: { step: number; id: string; name: string }) => void;
  onToolCallResult?: (trace: ExecutedToolTrace) => void;
  onApprovalRequired?: (
    request: PendingApprovalRequest,
  ) => Promise<boolean>;
  onCompaction?: (plan: CompactionPlan) => void;
  onTelemetry?: (telemetry: TurnTelemetrySnapshot) => void;
}

export interface RunHarnessTurnResult {
  finalState: 'done' | 'aborted';
  finalText: string;
  reasoningText: string;
  newMessages: HarnessMessage[];
  compiledMessages: HarnessMessage[];
  rollingSummary: string | null;
  rollingSummaryCutoffId: string | null;
  lastCompaction: CompactionPlan | null;
  toolTraces: ExecutedToolTrace[];
  telemetry: TurnTelemetrySnapshot;
  finishReason: 'stop' | 'length' | 'abort' | 'max_steps' | 'denied';
  transitions: TurnStateTransition[];
}

function buildFallbackToolDefinition(name: string): ToolDefinition {
  return {
    name,
    description: `Tool ${name}`,
    parameters: { type: 'object', properties: {} },
    safety: {
      readOnlyHint: false,
      consequentialHint: true,
      networkHint: false,
      untrustedContentHint: true,
    },
  };
}

/**
 * Executes a complete agentic turn using the TurnStateMachine, KV-preserving
 * ContextCompactor, GuardEngine, ToolExecutor, and <100ms AbortController teardown.
 */
export async function runHarnessTurn(params: RunHarnessTurnParams): Promise<RunHarnessTurnResult> {
  const turnStartMs = performance.now();
  const fsm = params.stateMachine ?? new TurnStateMachine();
  if (fsm.state !== 'idle') {
    fsm.reset();
  }
  const unsubscribeFsm = params.onStateChange ? fsm.subscribe(params.onStateChange) : undefined;

  const localController = new AbortController();
  let abortIssuedAt: number | null = null;
  let abortCompletedAt: number | null = null;

  const triggerAbort = () => {
    if (abortIssuedAt === null) {
      abortIssuedAt = performance.now();
    }
    if (!localController.signal.aborted) {
      localController.abort();
    }
  };

  if (params.signal?.aborted) {
    triggerAbort();
  } else {
    params.signal?.addEventListener('abort', triggerAbort, { once: true });
  }

  const caps = await params.adapter.capabilities();
  const compactor =
    params.compactor ??
    new ContextCompactor({
      maxContextTokens: caps.maxContextTokens,
      reservedOutputTokens: params.maxNewTokens ?? 1024,
      countTokens: (msgs) => params.adapter.countTokens(msgs),
      ...params.compactionConfig,
    });

  const tools = params.tools ?? params.toolExecutor?.listTools?.() ?? [];
  let profileMemories: MemoryRecord[] = [];
  if (params.memory) {
    profileMemories = await params.memory.inspect({ tier: 'profile', limit: 16 }).catch(() => []);
  }

  const history: HarnessMessage[] = params.messages.map((m) => ({ ...m }));
  const newMessages: HarnessMessage[] = [];
  const toolTraces: ExecutedToolTrace[] = [];

  let rollingSummary: string | null = params.previousRollingSummary ?? null;
  let rollingSummaryCutoffId: string | null = params.previousRollingSummaryCutoffId ?? null;
  let prevCompiled: readonly HarnessMessage[] | undefined = params.previousCompiledMessages;
  let lastCompaction: CompactionPlan | null = null;
  let finalText = '';
  let reasoningText = '';
  let finishReason: RunHarnessTurnResult['finishReason'] = 'stop';

  const telemetry: TurnTelemetrySnapshot = {
    totalDurationMs: 0,
  };

  const maxToolSteps = Math.max(0, params.maxToolSteps ?? 6);
  let contextHasUntrustedContent = history.some((m) => Boolean(m.tainted));
  let sessionAccessedPrivateData = history.some(
    (m) => m.toolName === 'memory_recall' || m.toolName === 'list_tabs' || m.toolName === 'read_tab',
  );

  try {
    if (localController.signal.aborted) {
      fsm.transition('aborted', 'pre-aborted');
      abortCompletedAt = performance.now();
      telemetry.abortLatencyMs = abortCompletedAt - (abortIssuedAt ?? turnStartMs);
      telemetry.totalDurationMs = abortCompletedAt - turnStartMs;
      return {
        finalState: 'aborted',
        finalText: '',
        reasoningText: '',
        newMessages,
        compiledMessages: history,
        rollingSummary,
        rollingSummaryCutoffId,
        lastCompaction,
        toolTraces,
        telemetry,
        finishReason: 'abort',
        transitions: fsm.getHistory(),
      };
    }

    for (let step = 0; step <= maxToolSteps; step += 1) {
      if (localController.signal.aborted) {
        break;
      }

      // 1. Compact context while keeping Slot 0 immutable
      fsm.transition('compacting', `step-${step}`);
      const plan = await compactor.compact({
        systemPrompt: params.systemPrompt,
        tools,
        profileMemories,
        messages: history,
        previousRollingSummary: rollingSummary,
        previousRollingSummaryCutoffId: rollingSummaryCutoffId,
        previousCompiledMessages: prevCompiled,
        signal: localController.signal,
      });

      rollingSummary = plan.rollingSummary;
      rollingSummaryCutoffId = plan.rollingSummaryCutoffId;
      prevCompiled = plan.compiledMessages;
      lastCompaction = plan;
      params.onCompaction?.(plan);

      if (localController.signal.aborted) {
        break;
      }

      // 2. Determine prefix slot key (Slot 0 or matched cached prefix)
      let prefixSlotKey: string | undefined = plan.slot0.slotKey;
      if (params.slotCoordinator && caps.supportsPrefixSlots) {
        const match = params.slotCoordinator.findBestMatch(plan.compiledMessages);
        if (match.source === 'slot' && match.slotKey) {
          prefixSlotKey = match.slotKey;
        } else if (match.source === 'none') {
          fsm.transition('prefilling', 'slot0-warmup');
          await params.slotCoordinator
            .prefillSlot(plan.slot0.slotKey, plan.slot0.messages, {
              kind: 'base',
              signal: localController.signal,
            })
            .catch(() => {});
        }
      }

      if (localController.signal.aborted) {
        break;
      }

      // 3. Stream from RuntimeAdapter
      fsm.transition('streaming', `step-${step}`);
      const parser = new StreamOutputParser({
        stopSequences: params.stopSequences,
        stopOnToolCall: false,
      });

      const nativeToolCalls: ParsedToolCall[] = [];
      const pendingNativeTools = new Map<string, { id: string; name: string; args: string }>();
      let stepAccumulatedText = '';
      let stepAccumulatedReasoning = '';
      let adapterFinishReason: 'stop' | 'length' | 'tool_calls' | 'abort' = 'stop';

      for await (const delta of params.adapter.generate(
        {
          messages: plan.compiledMessages,
          tools: tools.length > 0 ? [...tools] : undefined,
          maxNewTokens: params.maxNewTokens,
          temperature: params.temperature,
          topP: params.topP,
          topK: params.topK,
          enableThinking: params.enableThinking,
          pacingMode: params.pacingMode,
          prefixSlotKey,
        },
        localController.signal,
      )) {
        if (localController.signal.aborted) {
          adapterFinishReason = 'abort';
          break;
        }

        switch (delta.type) {
          case 'text': {
            const parsed = parser.pushText(delta.delta);
            if (parsed.reasoningDelta) {
              stepAccumulatedReasoning += parsed.reasoningDelta;
              reasoningText += parsed.reasoningDelta;
              params.onReasoningDelta?.(parsed.reasoningDelta, stepAccumulatedReasoning, step);
            }
            if (parsed.textDelta) {
              stepAccumulatedText += parsed.textDelta;
              params.onTextDelta?.(parsed.textDelta, stepAccumulatedText, step);
            }
            if (parsed.pendingToolCall) {
              params.onToolCallStart?.({
                step,
                id: `pending_${step}`,
                name: parsed.pendingToolCall.name,
              });
            }
            break;
          }
          case 'reasoning': {
            stepAccumulatedReasoning += delta.delta;
            reasoningText += delta.delta;
            params.onReasoningDelta?.(delta.delta, stepAccumulatedReasoning, step);
            break;
          }
          case 'tool_call_start': {
            pendingNativeTools.set(delta.id, { id: delta.id, name: delta.name, args: '' });
            params.onToolCallStart?.({ step, id: delta.id, name: delta.name });
            break;
          }
          case 'tool_call_args': {
            const entry = pendingNativeTools.get(delta.id);
            if (entry) {
              entry.args += delta.delta;
            }
            break;
          }
          case 'tool_call_ready': {
            pendingNativeTools.delete(delta.id);
            nativeToolCalls.push({
              id: delta.id,
              name: delta.name,
              arguments: delta.arguments,
              raw: JSON.stringify(delta.arguments),
            });
            break;
          }
          case 'telemetry': {
            if (delta.ttftMs !== undefined) telemetry.ttftMs = delta.ttftMs;
            if (delta.tokensPerSec !== undefined) telemetry.tokensPerSec = delta.tokensPerSec;
            if (delta.kvTokens !== undefined) telemetry.kvTokens = delta.kvTokens;
            if (delta.speculativeAcceptRate !== undefined) {
              telemetry.speculativeAcceptRate = delta.speculativeAcceptRate;
            }
            params.onTelemetry?.({ ...telemetry, totalDurationMs: performance.now() - turnStartMs });
            break;
          }
          case 'done': {
            adapterFinishReason = delta.finishReason;
            break;
          }
        }
      }

      const flushed = parser.flush();
      if (flushed.reasoningDelta) {
        stepAccumulatedReasoning += flushed.reasoningDelta;
        reasoningText += flushed.reasoningDelta;
        params.onReasoningDelta?.(flushed.reasoningDelta, stepAccumulatedReasoning, step);
      }
      if (flushed.textDelta) {
        stepAccumulatedText += flushed.textDelta;
        params.onTextDelta?.(flushed.textDelta, stepAccumulatedText, step);
      }

      if (localController.signal.aborted || adapterFinishReason === 'abort') {
        finishReason = 'abort';
        break;
      }

      const parsedInlineTools = parser.getToolCalls();
      const allToolCalls = nativeToolCalls.length > 0 ? nativeToolCalls : parsedInlineTools;
      const cleanVisible = parser.getVisibleText().trim() || stepAccumulatedText.trim();

      params.slotCoordinator?.setLiveMessages([
        ...plan.compiledMessages,
        ...(cleanVisible
          ? [
              {
                id: createId('live_asst'),
                role: 'assistant' as const,
                content: cleanVisible,
                createdAt: Date.now(),
              },
            ]
          : []),
      ]);

      // No tool calls or reached maxToolSteps -> finalize assistant answer
      if (allToolCalls.length === 0 || !params.toolExecutor || step >= maxToolSteps) {
        finalText = cleanVisible;
        const assistantMsg: HarnessMessage = {
          id: createId('msg'),
          role: 'assistant',
          content: cleanVisible,
          createdAt: Date.now(),
        };
        history.push(assistantMsg);
        newMessages.push(assistantMsg);
        finishReason =
          allToolCalls.length > 0 && step >= maxToolSteps
            ? 'max_steps'
            : adapterFinishReason === 'length'
              ? 'length'
              : 'stop';
        fsm.transition('done', finishReason);
        break;
      }

      // Execute tool calls in this step
      const firstCall = allToolCalls[0]!;
      const assistantToolMsg: HarnessMessage = {
        id: createId('msg'),
        role: 'assistant',
        content: cleanVisible,
        toolName: firstCall.name,
        toolCallId: firstCall.id,
        toolArgs: firstCall.arguments,
        createdAt: Date.now(),
      };
      history.push(assistantToolMsg);
      newMessages.push(assistantToolMsg);

      let deniedByUserOrGuard = false;

      for (const call of allToolCalls) {
        if (localController.signal.aborted) {
          break;
        }

        const toolDef =
          tools.find((t) => t.name === call.name || t.aliases?.includes(call.name)) ??
          params.toolExecutor.getTool?.(call.name) ??
          buildFallbackToolDefinition(call.name);

        let guardDecision: GuardDecision = {
          verdict: 'allow',
          reason: 'No guard engine configured',
        };

        if (params.guard) {
          guardDecision = await params.guard.evaluateToolCall({
            tool: toolDef,
            arguments: call.arguments,
            contextHasUntrustedContent,
            sessionAccessedPrivateData,
            activeOrigin: params.activeOrigin,
            explicitlyApprovedOrigins: params.explicitlyApprovedOrigins,
          });
        }

        if (guardDecision.verdict === 'deny') {
          deniedByUserOrGuard = true;
          const denyResult: ToolExecutionResult = {
            ok: false,
            toolName: call.name,
            formatted: `Tool call "${call.name}" denied by security guard: ${guardDecision.reason}`,
            tainted: false,
            error: guardDecision.reason,
          };
          const trace: ExecutedToolTrace = {
            step,
            call,
            result: denyResult,
            guardDecision,
            durationMs: 0,
          };
          toolTraces.push(trace);
          params.onToolCallResult?.(trace);

          const toolMsg: HarnessMessage = {
            id: createId('msg'),
            role: 'tool',
            toolName: call.name,
            toolCallId: call.id,
            content: denyResult.formatted,
            createdAt: Date.now(),
          };
          history.push(toolMsg);
          newMessages.push(toolMsg);
          continue;
        }

        if (guardDecision.verdict === 'require_approval') {
          fsm.transition('awaiting_approval', `tool:${call.name}`);
          let approved = false;
          if (params.onApprovalRequired) {
            approved = await Promise.race([
              params.onApprovalRequired({
                toolCallId: call.id,
                toolName: call.name,
                arguments: call.arguments,
                decision: guardDecision,
              }),
              new Promise<boolean>((resolve) => {
                if (localController.signal.aborted) {
                  resolve(false);
                } else {
                  localController.signal.addEventListener('abort', () => resolve(false), {
                    once: true,
                  });
                }
              }),
            ]);
          }

          if (localController.signal.aborted) {
            finishReason = 'abort';
            break;
          }

          if (!approved) {
            deniedByUserOrGuard = true;
            const rejectedResult: ToolExecutionResult = {
              ok: false,
              toolName: call.name,
              formatted: `User denied approval for "${call.name}" (${guardDecision.reason}).`,
              tainted: false,
              requiresApproval: true,
              error: 'Approval denied',
            };
            const trace: ExecutedToolTrace = {
              step,
              call,
              result: rejectedResult,
              guardDecision,
              durationMs: 0,
            };
            toolTraces.push(trace);
            params.onToolCallResult?.(trace);

            const toolMsg: HarnessMessage = {
              id: createId('msg'),
              role: 'tool',
              toolName: call.name,
              toolCallId: call.id,
              content: rejectedResult.formatted,
              createdAt: Date.now(),
            };
            history.push(toolMsg);
            newMessages.push(toolMsg);
            continue;
          }
        }

        fsm.transition('tool_executing', `tool:${call.name}`);
        const toolStartedAt = performance.now();

        const rawResult = await params.toolExecutor.execute(call.name, call.arguments, {
          environment: params.environment ?? 'node-test',
          sessionId: params.sessionId,
          origin: params.activeOrigin,
          abortSignal: localController.signal,
        });

        let formattedContent = rawResult.formatted;
        if (rawResult.tainted && params.guard?.sanitizeToolOutput) {
          const sanitized = await params.guard.sanitizeToolOutput(
            call.name,
            rawResult.formatted,
            rawResult.sourceOrigin,
          );
          formattedContent = sanitized.spotlightedPromptBlock || sanitized.cleanText;
        }

        if (rawResult.tainted || toolDef.safety.untrustedContentHint) {
          contextHasUntrustedContent = true;
        }
        if (
          call.name === 'memory_recall' ||
          call.name === 'list_tabs' ||
          call.name === 'read_tab'
        ) {
          sessionAccessedPrivateData = true;
        }

        const trace: ExecutedToolTrace = {
          step,
          call,
          result: { ...rawResult, formatted: formattedContent },
          guardDecision,
          durationMs: performance.now() - toolStartedAt,
        };
        toolTraces.push(trace);
        params.onToolCallResult?.(trace);

        const toolMsg: HarnessMessage = {
          id: createId('msg'),
          role: 'tool',
          toolName: call.name,
          toolCallId: call.id,
          content: formattedContent,
          tainted: rawResult.tainted,
          createdAt: Date.now(),
        };
        history.push(toolMsg);
        newMessages.push(toolMsg);
      }

      if (localController.signal.aborted) {
        finishReason = 'abort';
        break;
      }

      if (deniedByUserOrGuard && step === maxToolSteps) {
        finishReason = 'denied';
        fsm.transition('done', 'denied');
        break;
      }
    }

    if (localController.signal.aborted && fsm.state !== 'aborted') {
      fsm.transition('aborted', 'aborted');
      finishReason = 'abort';
    } else if (fsm.state !== 'done' && fsm.state !== 'aborted') {
      fsm.transition('done', finishReason);
    }

    if (params.memory && fsm.state === 'done' && newMessages.length > 0) {
      await params.memory.consolidate(params.sessionId, newMessages).catch(() => {});
    }

    const endMs = performance.now();
    telemetry.totalDurationMs = endMs - turnStartMs;
    if (abortIssuedAt !== null) {
      abortCompletedAt = endMs;
      telemetry.abortLatencyMs = abortCompletedAt - abortIssuedAt;
    }

    return {
      finalState: fsm.state === 'aborted' ? 'aborted' : 'done',
      finalText,
      reasoningText,
      newMessages,
      compiledMessages: prevCompiled ? [...prevCompiled] : history,
      rollingSummary,
      rollingSummaryCutoffId,
      lastCompaction,
      toolTraces,
      telemetry,
      finishReason,
      transitions: fsm.getHistory(),
    };
  } finally {
    params.signal?.removeEventListener('abort', triggerAbort);
    unsubscribeFsm?.();
  }
}
