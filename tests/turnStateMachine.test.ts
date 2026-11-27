import { describe, expect, it } from 'vitest';
import type {
  GenerateRequest,
  GuardEngine,
  HarnessMessage,
  MemoryProvider,
  MemoryRecord,
  RuntimeAdapter,
  StreamDelta,
  ToolDefinition,
  ToolExecutor,
} from '../src/contracts.js';
import { PrefixSlotCoordinator } from '../src/engine/speculative.js';
import { runHarnessTurn, TurnStateMachine } from '../src/engine/turnStateMachine.js';

function createMockAdapter(
  scriptedTurns: Array<(req: GenerateRequest, signal?: AbortSignal) => AsyncIterable<StreamDelta>>,
): RuntimeAdapter {
  let turnIdx = 0;
  return {
    capabilities: () => ({
      id: 'mock-adapter',
      maxContextTokens: 4096,
      supportsStreaming: true,
      supportsPrefixSlots: false,
      supportsTools: true,
      tier: 'cpu',
    }),
    countTokens: (msgs) => msgs.reduce((acc, m) => acc + Math.ceil(m.content.length / 4) + 4, 0),
    generate(req, signal) {
      const fn = scriptedTurns[turnIdx] ?? scriptedTurns[scriptedTurns.length - 1]!;
      turnIdx += 1;
      return fn(req, signal);
    },
  };
}

const TOOLS: ToolDefinition[] = [
  {
    name: 'read_active_tab',
    description: 'Read current tab',
    parameters: { type: 'object', properties: {} },
    safety: {
      readOnlyHint: true,
      consequentialHint: false,
      networkHint: false,
      untrustedContentHint: true,
    },
  },
  {
    name: 'interact_page',
    description: 'Click element on page',
    parameters: { type: 'object', properties: { selector: { type: 'string' } } },
    safety: {
      readOnlyHint: false,
      consequentialHint: true,
      networkHint: false,
      untrustedContentHint: false,
    },
  },
];

describe('TurnStateMachine & Agentic Turn Runner', () => {
  it('enforces valid FSM state transitions', () => {
    const fsm = new TurnStateMachine();
    expect(fsm.state).toBe('idle');
    fsm.transition('compacting');
    fsm.transition('streaming');
    fsm.transition('awaiting_approval');
    fsm.transition('tool_executing');
    fsm.transition('done');
    expect(() => fsm.transition('streaming')).toThrow(/Invalid TurnStateMachine transition/);
    fsm.reset();
    expect(fsm.state).toBe('idle');
  });

  it('executes tool chain with GuardEngine approval + taint propagation', async () => {
    const guardContexts: Array<{ tool: string; contextHasUntrustedContent: boolean }> = [];

    const guard: GuardEngine = {
      evaluateToolCall(ctx) {
        guardContexts.push({
          tool: ctx.tool.name,
          contextHasUntrustedContent: ctx.contextHasUntrustedContent,
        });
        if (ctx.tool.safety.consequentialHint && ctx.contextHasUntrustedContent) {
          return {
            verdict: 'require_approval',
            reason: 'Consequential action after untrusted content (IDPI protection)',
            lethalTrifectaRisk: true,
          };
        }
        return { verdict: 'allow', reason: 'Read-only tool' };
      },
    };

    const toolExecutor: ToolExecutor = {
      listTools: () => TOOLS,
      async execute(name, args) {
        if (name === 'read_active_tab') {
          return {
            ok: true,
            toolName: 'read_active_tab',
            formatted: 'Page says: Click #buy-now',
            tainted: true,
            sourceOrigin: 'https://shop.example',
          };
        }
        return {
          ok: true,
          toolName: name,
          formatted: `Clicked ${String(args.selector ?? '')}`,
          tainted: false,
        };
      },
    };

    const adapter = createMockAdapter([
      async function* () {
        yield { type: 'text', delta: '<tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>' };
        yield { type: 'done', finishReason: 'stop' };
      },
      async function* () {
        yield { type: 'text', delta: '<tool_call>{"name":"interact_page","arguments":{"selector":"#buy-now"}}</tool_call>' };
        yield { type: 'done', finishReason: 'stop' };
      },
      async function* () {
        yield { type: 'text', delta: 'Done! I clicked #buy-now after your approval.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    ]);

    const statesObserved: string[] = [];
    const userMsg: HarnessMessage = {
      id: 'u1',
      role: 'user',
      content: 'Read the page and click buy',
      createdAt: 1,
    };

    const result = await runHarnessTurn({
      sessionId: 'sess_1',
      adapter,
      messages: [userMsg],
      tools: TOOLS,
      toolExecutor,
      guard,
      onStateChange: (t) => statesObserved.push(t.to),
      onApprovalRequired: async (req) => {
        expect(req.toolName).toBe('interact_page');
        expect(req.decision.lethalTrifectaRisk).toBe(true);
        return true;
      },
    });

    expect(result.finalState).toBe('done');
    expect(result.finalText).toBe('Done! I clicked #buy-now after your approval.');
    expect(result.toolTraces).toHaveLength(2);
    expect(guardContexts).toEqual([
      { tool: 'read_active_tab', contextHasUntrustedContent: false },
      { tool: 'interact_page', contextHasUntrustedContent: true },
    ]);
    expect(statesObserved).toContain('awaiting_approval');
    expect(statesObserved).toContain('tool_executing');
  });

  it('tears down in <100ms when AbortController aborts mid-stream', async () => {
    const slowAdapter = createMockAdapter([
      async function* (_req, signal) {
        yield { type: 'text', delta: 'Starting ' };
        await new Promise<void>((resolve) => {
          const timer = setTimeout(resolve, 5000);
          signal?.addEventListener(
            'abort',
            () => {
              clearTimeout(timer);
              resolve();
            },
            { once: true },
          );
        });
        yield { type: 'done', finishReason: 'abort' };
      },
    ]);

    const controller = new AbortController();
    setTimeout(() => controller.abort(), 15);

    const result = await runHarnessTurn({
      sessionId: 'sess_abort',
      adapter: slowAdapter,
      messages: [{ id: 'u1', role: 'user', content: 'Write an essay', createdAt: 1 }],
      signal: controller.signal,
    });

    expect(result.finalState).toBe('aborted');
    expect(result.finishReason).toBe('abort');
    expect(result.telemetry.abortLatencyMs).toBeDefined();
    expect(result.telemetry.abortLatencyMs!).toBeLessThan(100);
  });

  it('resets onTextDelta on orphan </think> and persists msg.reasoning', async () => {
    const adapter = createMockAdapter([
      async function* () {
        yield { type: 'text', delta: 'Checking ' };
        yield { type: 'text', delta: 'math...</thi' };
        yield { type: 'text', delta: 'nk>\n\nThe answer is 42.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    ]);

    const textDeltas: Array<{ delta: string; acc: string; step: number }> = [];
    const reasoningDeltas: Array<{ delta: string; acc: string; step: number }> = [];

    const result = await runHarnessTurn({
      sessionId: 'sess_orphan_think',
      adapter,
      messages: [{ id: 'u1', role: 'user', content: 'What is 6*7?', createdAt: 1 }],
      onTextDelta: (delta, acc, step) => {
        textDeltas.push({ delta, acc, step });
      },
      onReasoningDelta: (delta, acc, step) => {
        reasoningDeltas.push({ delta, acc, step });
      },
    });

    expect(result.finalState).toBe('done');
    expect(result.finalText).toBe('The answer is 42.');
    expect(result.reasoningText).toBe('Checking math...');
    expect(result.newMessages).toHaveLength(1);
    expect(result.newMessages[0]?.content).toBe('The answer is 42.');
    expect(result.newMessages[0]?.reasoning).toBe('Checking math...');
    expect(textDeltas).toEqual([
      { delta: 'Checking ', acc: 'Checking ', step: 0 },
      { delta: 'math...', acc: 'Checking math...', step: 0 },
      { delta: '', acc: '', step: 0 },
      { delta: 'The answer is 42.', acc: 'The answer is 42.', step: 0 },
    ]);
    expect(reasoningDeltas).toEqual([
      { delta: 'Checking math...', acc: 'Checking math...', step: 0 },
    ]);
  });

  it('persists reasoning onto assistantToolMsg when tool calls follow thinking', async () => {
    const toolExecutor: ToolExecutor = {
      listTools: () => TOOLS,
      async execute(name) {
        return {
          ok: true,
          toolName: name,
          formatted: 'Tab content',
          tainted: false,
        };
      },
    };

    const adapter = createMockAdapter([
      async function* () {
        yield {
          type: 'text',
          delta: '<|channel>thought\nInspect tab first\n<channel|><tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>',
        };
        yield { type: 'done', finishReason: 'stop' };
      },
      async function* () {
        yield { type: 'text', delta: '<think>Got tab</think>\n\nSummary ready.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    ]);

    const result = await runHarnessTurn({
      sessionId: 'sess_tool_think',
      adapter,
      messages: [{ id: 'u1', role: 'user', content: 'Summarize tab', createdAt: 1 }],
      tools: TOOLS,
      toolExecutor,
    });

    expect(result.finalText).toBe('Summary ready.');
    expect(result.newMessages).toHaveLength(3);
    expect(result.newMessages[0]?.role).toBe('assistant');
    expect(result.newMessages[0]?.toolName).toBe('read_active_tab');
    expect(result.newMessages[0]?.reasoning).toBe('Inspect tab first');
    expect(result.newMessages[2]?.role).toBe('assistant');
    expect(result.newMessages[2]?.content).toBe('Summary ready.');
    expect(result.newMessages[2]?.reasoning).toBe('Got tab');
  });

  it('aborts step immediately on stopOnToolCall without aborting turn, clears buffers, and forwards groundingText + rawContent', async () => {
    const requests: GenerateRequest[] = [];
    const stepSignals: AbortSignal[] = [];
    const prefillCalls: string[] = [];

    const adapter: RuntimeAdapter = {
      capabilities: () => ({
        id: 'stop-on-tc-adapter',
        maxContextTokens: 8192,
        supportsStreaming: true,
        supportsPrefixSlots: true,
        supportsTools: true,
        tier: 'tier1-f16-subgroups',
      }),
      countTokens: (msgs) => msgs.reduce((acc, m) => acc + Math.ceil((m.rawContent ?? m.content).length / 4) + 4, 0),
      async prefillSlot(slotKey) {
        prefillCalls.push(slotKey);
        return { tokens: 16, elapsedMs: 1 };
      },
      async *generate(req, signal) {
        requests.push(req);
        if (signal) {
          stepSignals.push(signal);
        }
        if (requests.length === 1) {
          yield {
            type: 'text',
            delta: '<|channel>thought\nNeed tab\n<channel|><|tool_call>call:read_active_tab{}<tool_call|>',
          };
          // Even if adapter would hang waiting for step abort, stepAbortController aborts immediately
          await new Promise<void>((resolve) => {
            if (signal?.aborted) {
              resolve();
            } else {
              signal?.addEventListener('abort', () => resolve(), { once: true });
            }
          });
          yield { type: 'text', delta: 'HALLUCINATED AFTER TOOL' };
          yield { type: 'done', finishReason: 'abort' };
          return;
        }
        yield { type: 'text', delta: '  Final answer with leading space  ' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    const slotCoordinator = new PrefixSlotCoordinator({ adapter });
    const toolExecutor: ToolExecutor = {
      listTools: () => TOOLS,
      async execute(name) {
        return {
          ok: true,
          toolName: name,
          formatted: 'Observed tab fact A',
          tainted: false,
        };
      },
    };

    const textEvents: Array<{ delta: string; acc: string; step: number }> = [];
    const reasoningEvents: Array<{ delta: string; acc: string; step: number }> = [];
    const outerController = new AbortController();

    const result = await runHarnessTurn({
      sessionId: 'sess_stop_tc',
      adapter,
      messages: [{ id: 'u1', role: 'user', content: 'Check tab', createdAt: 1 }],
      contextMessages: [{ id: 'ctx1', role: 'system', content: 'Tab title: Docs', createdAt: 1 }],
      groundingText: 'Seed grounding text',
      tools: TOOLS,
      toolExecutor,
      slotCoordinator,
      signal: outerController.signal,
      onTextDelta: (delta, acc, step) => textEvents.push({ delta, acc, step }),
      onReasoningDelta: (delta, acc, step) => reasoningEvents.push({ delta, acc, step }),
    });

    expect(outerController.signal.aborted).toBe(false);
    expect(stepSignals[0]?.aborted).toBe(true);
    expect(stepSignals[1]?.aborted).toBe(false);
    expect(result.finalState).toBe('done');
    expect(result.finishReason).toBe('stop');
    // Default autoWarmSlot0InTurn=false avoids blocking on Slot 0 prefill before Turn 1
    expect(prefillCalls).toHaveLength(0);

    // Step 0 & Step 1 flags and grounding accumulation
    expect(requests[0]?.stopOnToolCall).toBe(true);
    expect(requests[0]?.groundingText).toBe('Seed grounding text');
    expect(requests[0]?.messages[1]?.id).toBe('ctx1');
    expect(requests[1]?.stopOnToolCall).toBe(true);
    expect(requests[1]?.groundingText).toBe('Seed grounding text\n\nObserved tab fact A');

    // Transient buffers cleared before tool execution in step 0
    const step0ClearTextIdx = textEvents.findIndex((e) => e.step === 0 && e.acc === '');
    const step0ClearReasoningIdx = reasoningEvents.findIndex((e) => e.step === 0 && e.acc === '');
    expect(step0ClearTextIdx).toBeGreaterThanOrEqual(0);
    expect(step0ClearReasoningIdx).toBeGreaterThan(0);

    // rawContent stored on assistantToolMsg & on assistantMsg (when rawStepText !== cleanVisible && !trimmedReasoning)
    expect(result.newMessages).toHaveLength(3);
    const asstToolMsg = result.newMessages[0]!;
    const finalAsstMsg = result.newMessages[2]!;
    expect(asstToolMsg.content).toBe('');
    expect(asstToolMsg.rawContent).toBe(
      '<|channel>thought\nNeed tab\n<channel|><|tool_call>call:read_active_tab{}<tool_call|>',
    );
    expect(finalAsstMsg.content).toBe('Final answer with leading space');
    expect(finalAsstMsg.rawContent).toBe('  Final answer with leading space  ');

    // Live messages preserve rawContent so next turn matches live KV prefix
    const live = slotCoordinator.getLiveMessages();
    expect(live[live.length - 1]?.rawContent).toBe('  Final answer with leading space  ');
    const nextMatch = slotCoordinator.findBestMatch([
      ...live,
      { id: 'u2', role: 'user', content: 'Next?', createdAt: 2 },
    ]);
    expect(nextMatch.source).toBe('live');
    expect(nextMatch.matchedMessages).toBe(live.length);
  });

  it('skips memory.inspect when cachedProfileMemories is supplied and respects waitForMemoryConsolidation', async () => {
    let inspectCalls = 0;
    let consolidateResolved = false;
    let releaseConsolidate: () => void = () => {};

    const memory: MemoryProvider = {
      async remember(input) {
        return {
          id: 'm1',
          ...input,
          createdAt: 1,
          updatedAt: 1,
          lastAccessedAt: 1,
          accessCount: 1,
        };
      },
      async recall() {
        return [];
      },
      async consolidate() {
        await new Promise<void>((resolve) => {
          releaseConsolidate = () => {
            consolidateResolved = true;
            resolve();
          };
        });
        return { added: 1, superseded: 0 };
      },
      async inspect() {
        inspectCalls += 1;
        return [];
      },
      async forget() {
        return true;
      },
      async exportEncrypted() {
        return new Uint8Array();
      },
      async importEncrypted() {
        return { imported: 0 };
      },
      async wipeAll() {},
    };

    const cached: MemoryRecord[] = [
      {
        id: 'prof_1',
        tier: 'profile',
        content: 'User prefers concise TypeScript answers',
        createdAt: 1,
        updatedAt: 1,
        lastAccessedAt: 1,
        accessCount: 1,
      },
    ];

    const requests: GenerateRequest[] = [];
    const adapter = createMockAdapter([
      async function* (req) {
        requests.push(req);
        yield { type: 'text', delta: 'Done.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    ]);

    // Default waitForMemoryConsolidation=false returns immediately without blocking on consolidate()
    const result1 = await runHarnessTurn({
      sessionId: 'sess_mem',
      adapter,
      messages: [{ id: 'u1', role: 'user', content: 'Hi', createdAt: 1 }],
      memory,
      cachedProfileMemories: cached,
    });

    expect(result1.finalState).toBe('done');
    expect(inspectCalls).toBe(0);
    expect(consolidateResolved).toBe(false);
    expect(requests[0]?.messages[0]?.content).toContain('User prefers concise TypeScript answers');
    releaseConsolidate?.();
    await new Promise((r) => setTimeout(r, 5));

    // waitForMemoryConsolidation=true awaits consolidate() before resolving
    consolidateResolved = false;
    let settled = false;
    const p2 = runHarnessTurn({
      sessionId: 'sess_mem_wait',
      adapter,
      messages: [{ id: 'u2', role: 'user', content: 'Hi again', createdAt: 2 }],
      memory,
      cachedProfileMemories: cached,
      waitForMemoryConsolidation: true,
    }).then((res) => {
      settled = true;
      return res;
    });

    await new Promise((r) => setTimeout(r, 10));
    expect(settled).toBe(false);
    releaseConsolidate?.();
    const result2 = await p2;
    expect(settled).toBe(true);
    expect(consolidateResolved).toBe(true);
    expect(result2.finalState).toBe('done');
    expect(inspectCalls).toBe(0);
  });

  it('keeps bare and fenced profile JSON as plain visible text without triggering tool calls or approval (RC1 P3)', async () => {
    const bareJson = '{"name":"Ana","age":31,"city":"Lisbon"}';
    const fencedJson = '```json\n{"name":"Ana","age":31,"city":"Lisbon"}\n```';
    let approvalsRequested = 0;

    for (const payload of [bareJson, fencedJson]) {
      const adapter = createMockAdapter([
        async function* () {
          yield { type: 'text', delta: payload };
          yield { type: 'done', finishReason: 'stop' };
        },
      ]);
      const result = await runHarnessTurn({
        sessionId: 'sess_bare_json',
        adapter,
        messages: [{ id: 'u1', role: 'user', content: 'Return ONLY JSON for Ana', createdAt: 1 }],
        tools: TOOLS,
        toolExecutor: {
          listTools: () => TOOLS,
          async execute() {
            throw new Error('Must not execute any tool for bare JSON');
          },
        },
        onApprovalRequired: async () => {
          approvalsRequested += 1;
          return false;
        },
      });
      expect(result.finalState).toBe('done');
      expect(result.finishReason).toBe('stop');
      expect(result.toolTraces).toHaveLength(0);
      expect(result.finalText).toBe(payload);
    }
    expect(approvalsRequested).toBe(0);
  });
});
