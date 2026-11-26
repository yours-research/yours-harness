import { describe, expect, it } from 'vitest';
import {
  ChromePromptAiAdapter,
  OpenAiCompatibleAdapter,
  RuntimeRegistry,
  TransformersJsAdapter,
  WebLlmAdapter,
  WgslWorkerAdapter,
  type WorkerPortTransport,
} from '../src/adapters/index.js';
import type { HarnessMessage, StreamDelta } from '../src/contracts.js';

const USER_MSG: HarnessMessage[] = [
  { id: 'u1', role: 'user', content: 'Hello local LLM', createdAt: 1 },
];

async function collectDeltas(iter: AsyncIterable<StreamDelta>): Promise<StreamDelta[]> {
  const out: StreamDelta[] = [];
  for await (const delta of iter) {
    out.push(delta);
  }
  return out;
}

describe('Pluggable Runtime Adapters & Registry', () => {
  it('registers adapters and selects best match by capability criteria', async () => {
    const registry = new RuntimeRegistry();
    const chromeAdapter = new ChromePromptAiAdapter({ id: 'chrome' });
    const wgslAdapter = new WgslWorkerAdapter({
      id: 'wgsl',
      transport: {
        postMessage: () => {},
        addMessageListener: () => () => {},
      },
    });

    registry.register('chrome', chromeAdapter, { setAsDefault: true });
    registry.register('wgsl', wgslAdapter);

    expect(registry.getDefaultId()).toBe('chrome');

    const bestPrefix = await registry.selectBest({
      requirePrefixSlots: true,
      requireTools: true,
    });
    expect(bestPrefix?.id).toBe('wgsl');
  });

  it('bridges WgslWorkerAdapter messages, telemetry, prefillSlot, and abort', async () => {
    const listeners = new Set<(msg: unknown) => void>();
    const posted: Array<Record<string, unknown>> = [];

    const transport: WorkerPortTransport = {
      postMessage(raw) {
        const msg = raw as Record<string, unknown>;
        posted.push(msg);
        queueMicrotask(() => {
          if (msg.type === 'PREFILL_BACKGROUND') {
            listeners.forEach((fn) =>
              fn({
                type: 'PREFILL_COMPLETE',
                requestId: msg.requestId,
                slotKey: msg.slotKey,
                tokens: 128,
                elapsedMs: 11.4,
              }),
            );
          } else if (msg.type === 'GENERATE') {
            listeners.forEach((fn) =>
              fn({ type: 'REASONING_DELTA', requestId: msg.requestId, delta: 'Thinking...' }),
            );
            listeners.forEach((fn) =>
              fn({ type: 'TOKEN_DELTA', requestId: msg.requestId, delta: 'Hello WGSL!' }),
            );
            listeners.forEach((fn) =>
              fn({
                type: 'TELEMETRY_UPDATE',
                requestId: msg.requestId,
                telemetry: { ttftMs: 18, tokensPerSec: 112, kvTokens: 64 },
              }),
            );
            listeners.forEach((fn) =>
              fn({ type: 'GENERATION_COMPLETE', requestId: msg.requestId, finishReason: 'stop' }),
            );
          }
        });
      },
      addMessageListener(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
      },
    };

    const adapter = new WgslWorkerAdapter({ transport, defaultPacingMode: 'smooth-ui' });
    const prefill = await adapter.prefillSlot('base-system:abc', USER_MSG);
    expect(prefill).toEqual({ tokens: 128, elapsedMs: 11.4 });

    const deltas = await collectDeltas(
      adapter.generate({
        messages: USER_MSG,
        prefixSlotKey: 'base-system:abc',
        pacingMode: 'max-speed',
      }),
    );

    expect(deltas.some((d) => d.type === 'reasoning' && d.delta === 'Thinking...')).toBe(true);
    expect(deltas.some((d) => d.type === 'text' && d.delta === 'Hello WGSL!')).toBe(true);
    expect(deltas.some((d) => d.type === 'telemetry' && d.tokensPerSec === 112)).toBe(true);
    expect(deltas[deltas.length - 1]).toEqual({ type: 'done', finishReason: 'stop' });

    const genPost = posted.find((p) => p.type === 'GENERATE');
    expect(genPost?.prefixSlotKey).toBe('base-system:abc');
    expect((genPost?.config as Record<string, unknown>)?.gpuPacingMode).toBe('max-speed');
  });

  it('parses OpenAiCompatibleAdapter SSE stream including reasoning & tool_calls', async () => {
    const ssePayloads = [
      'data: {"choices":[{"delta":{"reasoning_content":"Plan "}}]}\n\n',
      'data: {"choices":[{"delta":{"content":"Calling tool."}}]}\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read_active_tab","arguments":"{\\"mode\\":"}}]}}]}\n',
      'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"article\\"}"}}]},"finish_reason":"tool_calls"}]}\n',
      'data: [DONE]\n\n',
    ];

    const encoder = new TextEncoder();
    const mockFetch: typeof fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            for (const chunk of ssePayloads) {
              controller.enqueue(encoder.encode(chunk));
            }
            controller.close();
          },
        }),
        { status: 200 },
      );

    const adapter = new OpenAiCompatibleAdapter({
      baseUrl: 'http://localhost:11434/v1',
      model: 'qwen3.5:4b',
      fetchFn: mockFetch,
    });

    const deltas = await collectDeltas(adapter.generate({ messages: USER_MSG }));
    expect(deltas.some((d) => d.type === 'reasoning' && d.delta === 'Plan ')).toBe(true);
    expect(deltas.some((d) => d.type === 'text' && d.delta === 'Calling tool.')).toBe(true);
    expect(
      deltas.some(
        (d) =>
          d.type === 'tool_call_ready' &&
          d.name === 'read_active_tab' &&
          d.arguments.mode === 'article',
      ),
    ).toBe(true);
    expect(deltas[deltas.length - 1]).toEqual({ type: 'done', finishReason: 'tool_calls' });
  });

  it('handles ChromePromptAiAdapter cumulative & delta chunks', async () => {
    const adapter = new ChromePromptAiAdapter({
      factory: {
        availability: async () => 'available',
        create: async () => ({
          async *promptStreaming() {
            yield 'Hello';
            yield 'Hello Chrome';
            yield 'Hello Chrome AI!';
          },
        }),
      },
    });

    expect(await adapter.isAvailable()).toBe(true);
    const deltas = await collectDeltas(adapter.generate({ messages: USER_MSG }));
    const text = deltas
      .filter((d): d is Extract<StreamDelta, { type: 'text' }> => d.type === 'text')
      .map((d) => d.delta)
      .join('');
    expect(text).toBe('Hello Chrome AI!');
  });

  it('supports TransformersJsAdapter and WebLlmAdapter shims', async () => {
    const tjs = new TransformersJsAdapter({
      modelId: 'Qwen3.5-0.8B-ONNX',
      runGeneration: async ({ onToken }) => {
        onToken('Fast ');
        onToken('WebGPU!');
        return { finishReason: 'stop', kvTokens: 32 };
      },
    });
    const tjsDeltas = await collectDeltas(tjs.generate({ messages: USER_MSG }));
    expect(
      tjsDeltas
        .filter((d): d is Extract<StreamDelta, { type: 'text' }> => d.type === 'text')
        .map((d) => d.delta)
        .join(''),
    ).toBe('Fast WebGPU!');

    const webllm = new WebLlmAdapter({
      engine: {
        chat: {
          completions: {
            async create() {
              return (async function* () {
                yield { choices: [{ delta: { content: 'MLC ' } }] };
                yield { choices: [{ delta: { content: 'Ready' }, finish_reason: 'stop' }] };
              })();
            },
          },
        },
      },
    });
    const webllmDeltas = await collectDeltas(webllm.generate({ messages: USER_MSG }));
    expect(
      webllmDeltas
        .filter((d): d is Extract<StreamDelta, { type: 'text' }> => d.type === 'text')
        .map((d) => d.delta)
        .join(''),
    ).toBe('MLC Ready');
  });

  it('forwards rawContent, isDraft, stopOnToolCall, and groundingText in WgslWorkerAdapter', async () => {
    const listeners = new Set<(msg: unknown) => void>();
    const posted: Array<Record<string, unknown>> = [];

    const transport: WorkerPortTransport = {
      postMessage(raw) {
        const msg = raw as Record<string, unknown>;
        posted.push(msg);
        queueMicrotask(() => {
          if (msg.type === 'PREFILL_BACKGROUND') {
            listeners.forEach((fn) =>
              fn({
                type: 'PREFILL_COMPLETE',
                requestId: msg.requestId,
                slotKey: msg.slotKey,
                tokens: 64,
                elapsedMs: 4,
              }),
            );
          } else if (msg.type === 'GENERATE') {
            listeners.forEach((fn) =>
              fn({ type: 'GENERATION_COMPLETE', requestId: msg.requestId, finishReason: 'stop' }),
            );
          }
        });
      },
      addMessageListener(fn) {
        listeners.add(fn);
        return () => listeners.delete(fn);
      },
    };

    const adapter = new WgslWorkerAdapter({ transport });
    const rawMsgs: HarnessMessage[] = [
      {
        id: 'a1',
        role: 'assistant',
        content: 'Clean UI text',
        rawContent: '<tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>',
        toolName: 'read_active_tab',
        toolCallId: 'tc_1',
        createdAt: 1,
      },
    ];

    await adapter.prefillSlot('draft:composer', rawMsgs);
    await adapter.prefillSlot('base-system:123', rawMsgs);

    const draftPost = posted.find((p) => p.type === 'PREFILL_BACKGROUND' && p.slotKey === 'draft:composer');
    const basePost = posted.find((p) => p.type === 'PREFILL_BACKGROUND' && p.slotKey === 'base-system:123');
    expect(draftPost?.isDraft).toBe(true);
    expect(basePost?.isDraft).toBe(false);
    expect((draftPost?.messages as Array<{ content: string }>)[0]?.content).toBe(
      '<tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>',
    );

    await collectDeltas(
      adapter.generate({
        messages: rawMsgs,
        stopOnToolCall: true,
        groundingText: 'Grounding facts here',
      }),
    );

    await collectDeltas(
      adapter.generate({
        messages: [{ id: 'u2', role: 'user', content: 'Plain user text', createdAt: 2 }],
      }),
    );

    const genWithFlags = posted.filter((p) => p.type === 'GENERATE')[0]!;
    const genPlain = posted.filter((p) => p.type === 'GENERATE')[1]!;
    expect((genWithFlags.messages as Array<{ content: string }>)[0]?.content).toBe(
      '<tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>',
    );
    expect(genWithFlags.stopOnToolCall).toBe(true);
    expect(genWithFlags.groundingText).toBe('Grounding facts here');

    expect('stopOnToolCall' in genPlain).toBe(false);
    expect('groundingText' in genPlain).toBe(false);
  });
});
