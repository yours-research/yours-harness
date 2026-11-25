import { describe, expect, it } from 'vitest';
import type {
  GenerateRequest,
  GuardEngine,
  HarnessMessage,
  RuntimeAdapter,
  StreamDelta,
  ToolDefinition,
  ToolExecutor,
} from '../src/contracts.js';
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
});
