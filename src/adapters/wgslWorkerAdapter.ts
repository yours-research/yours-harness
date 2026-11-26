import type {
  GenerateRequest,
  GpuPacingMode,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
  ToolDefinition,
  WgslComputeTier,
} from '../contracts.js';
import { AsyncPushQueue } from '../utils/asyncQueue.js';
import { createId } from '../utils/id.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface WorkerPortTransport {
  postMessage(message: unknown): void;
  addMessageListener(listener: (message: unknown) => void): () => void;
}

export type WgslWorkerIncomingMessage =
  | {
      type: 'TOKEN_DELTA';
      requestId: string;
      delta: string;
      tokenCount?: number;
      isReasoning?: boolean;
    }
  | {
      type: 'REASONING_DELTA';
      requestId: string;
      delta: string;
    }
  | {
      type: 'TOOL_CALL_START';
      requestId: string;
      id: string;
      name: string;
    }
  | {
      type: 'TOOL_CALL_ARGS';
      requestId: string;
      id: string;
      delta: string;
    }
  | {
      type: 'TOOL_CALL_READY';
      requestId: string;
      id: string;
      name: string;
      arguments: Record<string, unknown>;
    }
  | {
      type: 'TELEMETRY_UPDATE';
      requestId?: string;
      telemetry: {
        ttftMs?: number | null;
        tokensPerSec?: number | null;
        activeContextTokens?: number;
        kvTokens?: number;
        speculativeAcceptRate?: number;
        speculativeStats?: { acceptanceRatePct?: number } | null;
      };
    }
  | {
      type: 'GENERATION_COMPLETE';
      requestId: string;
      finishReason: 'stop' | 'length' | 'tool_calls' | 'abort';
    }
  | {
      type: 'GENERATION_ERROR';
      requestId: string;
      error: string;
    }
  | {
      type: 'PREFILL_COMPLETE';
      requestId: string;
      slotKey: string;
      tokens: number;
      elapsedMs: number;
    }
  | {
      type: 'PREFILL_ERROR';
      requestId: string;
      error: string;
    }
  | {
      type: 'EVICT_COMPLETE';
      requestId: string;
      slotKey: string;
    };

export interface WgslWorkerAdapterOptions {
  id?: string;
  transport: WorkerPortTransport;
  maxContextTokens?: number;
  tier?: WgslComputeTier;
  defaultPacingMode?: GpuPacingMode;
  supportsTools?: boolean;
  countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;
  serializeTools?: (tools: ToolDefinition[]) => unknown;
}

/**
 * Wraps a standard Web Worker (`Worker`) or Chrome MV3 `chrome.runtime.Port`
 * into the uniform WorkerPortTransport interface.
 */
export function createWorkerTransport(
  target:
    | {
        postMessage(message: unknown): void;
        addEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
        removeEventListener(type: 'message', listener: (event: { data: unknown }) => void): void;
      }
    | {
        postMessage(message: unknown): void;
        onMessage: {
          addListener(listener: (message: unknown) => void): void;
          removeListener(listener: (message: unknown) => void): void;
        };
      },
): WorkerPortTransport {
  if ('onMessage' in target) {
    return {
      postMessage: (msg) => target.postMessage(msg),
      addMessageListener: (listener) => {
        target.onMessage.addListener(listener);
        return () => target.onMessage.removeListener(listener);
      },
    };
  }
  return {
    postMessage: (msg) => target.postMessage(msg),
    addMessageListener: (listener) => {
      const wrapped = (event: { data: unknown }) => listener(event.data);
      target.addEventListener('message', wrapped);
      return () => target.removeEventListener('message', wrapped);
    },
  };
}

/**
 * RuntimeAdapter bridging the custom Rust WASM + WGSL WebGPU worker protocol
 * with prefixSlotKey, pacingMode, speculative telemetry, and <100ms abort.
 */
export class WgslWorkerAdapter implements RuntimeAdapter {
  private readonly id: string;
  private readonly transport: WorkerPortTransport;
  private readonly maxContextTokens: number;
  private readonly tier: WgslComputeTier;
  private readonly defaultPacingMode: GpuPacingMode;
  private readonly supportsTools: boolean;
  private readonly countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;
  private readonly serializeTools?: (tools: ToolDefinition[]) => unknown;

  constructor(options: WgslWorkerAdapterOptions) {
    this.id = options.id ?? 'wgsl-worker';
    this.transport = options.transport;
    this.maxContextTokens = options.maxContextTokens ?? 32768;
    this.tier = options.tier ?? 'tier1-f16-subgroups';
    this.defaultPacingMode = options.defaultPacingMode ?? 'balanced';
    this.supportsTools = options.supportsTools ?? true;
    this.countTokensFn = options.countTokensFn;
    this.serializeTools = options.serializeTools;
  }

  capabilities(): RuntimeCapabilities {
    return {
      id: this.id,
      maxContextTokens: this.maxContextTokens,
      supportsStreaming: true,
      supportsPrefixSlots: true,
      supportsTools: this.supportsTools,
      tier: this.tier,
    };
  }

  countTokens(messages: HarnessMessage[]): Promise<number> | number {
    if (this.countTokensFn) {
      return this.countTokensFn(messages);
    }
    return estimateMessagesTokens(messages);
  }

  async prefillSlot(
    slotKey: string,
    messages: HarnessMessage[],
    signal?: AbortSignal,
  ): Promise<{ tokens: number; elapsedMs: number }> {
    if (signal?.aborted) {
      return { tokens: 0, elapsedMs: 0 };
    }

    const requestId = createId('prefill');
    const startedAt = performance.now();

    return new Promise<{ tokens: number; elapsedMs: number }>((resolve, reject) => {
      let settled = false;

      const cleanup = () => {
        unsubscribe();
        signal?.removeEventListener('abort', onAbort);
      };

      const onAbort = () => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        this.transport.postMessage({ type: 'ABORT_BACKGROUND_PREFILL', requestId, slotKey });
        resolve({ tokens: 0, elapsedMs: performance.now() - startedAt });
      };

      const unsubscribe = this.transport.addMessageListener((raw) => {
        if (settled || !raw || typeof raw !== 'object') {
          return;
        }
        const msg = raw as WgslWorkerIncomingMessage;
        if (msg.type === 'PREFILL_COMPLETE' && msg.requestId === requestId) {
          settled = true;
          cleanup();
          resolve({ tokens: msg.tokens, elapsedMs: msg.elapsedMs });
        } else if (msg.type === 'PREFILL_ERROR' && msg.requestId === requestId) {
          settled = true;
          cleanup();
          reject(new Error(msg.error));
        }
      });

      signal?.addEventListener('abort', onAbort, { once: true });

      this.transport.postMessage({
        type: 'PREFILL_BACKGROUND',
        requestId,
        slotKey,
        isDraft: slotKey.startsWith('draft:'),
        messages: messages.map((m) => ({
          id: m.id,
          role: m.role,
          content: m.rawContent ?? m.content,
          toolName: m.toolName,
          toolCallId: m.toolCallId,
        })),
      });
    });
  }

  async evictSlot(slotKey: string): Promise<void> {
    const requestId = createId('evict');
    return new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        unsubscribe();
        resolve();
      }, 250);

      const unsubscribe = this.transport.addMessageListener((raw) => {
        if (!raw || typeof raw !== 'object') {
          return;
        }
        const msg = raw as WgslWorkerIncomingMessage;
        if (msg.type === 'EVICT_COMPLETE' && msg.requestId === requestId) {
          clearTimeout(timer);
          unsubscribe();
          resolve();
        }
      });

      this.transport.postMessage({
        type: 'EVICT_SLOT',
        requestId,
        slotKey,
      });
    });
  }

  async *generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta> {
    const requestId = createId('wgsl_gen');
    const queue = new AsyncPushQueue<StreamDelta>();
    let finished = false;

    const closeWithAbort = () => {
      if (finished) {
        return;
      }
      finished = true;
      this.transport.postMessage({ type: 'ABORT', requestId });
      queue.push({ type: 'done', finishReason: 'abort' });
      queue.close();
    };

    const unsubscribe = this.transport.addMessageListener((raw) => {
      if (finished || !raw || typeof raw !== 'object') {
        return;
      }
      const msg = raw as WgslWorkerIncomingMessage;

      if (msg.type === 'TELEMETRY_UPDATE') {
        if (msg.requestId && msg.requestId !== requestId) {
          return;
        }
        const t = msg.telemetry;
        queue.push({
          type: 'telemetry',
          ttftMs: t.ttftMs ?? undefined,
          tokensPerSec: t.tokensPerSec ?? undefined,
          kvTokens: t.kvTokens ?? t.activeContextTokens,
          speculativeAcceptRate:
            t.speculativeAcceptRate ??
            (t.speculativeStats?.acceptanceRatePct !== undefined
              ? t.speculativeStats.acceptanceRatePct / 100
              : undefined),
        });
        return;
      }

      if (!('requestId' in msg) || msg.requestId !== requestId) {
        return;
      }

      switch (msg.type) {
        case 'TOKEN_DELTA':
          queue.push({
            type: msg.isReasoning ? 'reasoning' : 'text',
            delta: msg.delta,
          });
          break;
        case 'REASONING_DELTA':
          queue.push({ type: 'reasoning', delta: msg.delta });
          break;
        case 'TOOL_CALL_START':
          queue.push({ type: 'tool_call_start', id: msg.id, name: msg.name });
          break;
        case 'TOOL_CALL_ARGS':
          queue.push({ type: 'tool_call_args', id: msg.id, delta: msg.delta });
          break;
        case 'TOOL_CALL_READY':
          queue.push({
            type: 'tool_call_ready',
            id: msg.id,
            name: msg.name,
            arguments: msg.arguments,
          });
          break;
        case 'GENERATION_COMPLETE':
          finished = true;
          queue.push({ type: 'done', finishReason: msg.finishReason });
          queue.close();
          break;
        case 'GENERATION_ERROR':
          finished = true;
          queue.fail(new Error(msg.error));
          break;
      }
    });

    if (signal?.aborted) {
      unsubscribe();
      yield { type: 'done', finishReason: 'abort' };
      return;
    }

    const onAbort = () => {
      closeWithAbort();
    };
    signal?.addEventListener('abort', onAbort, { once: true });

    this.transport.postMessage({
      type: 'GENERATE',
      requestId,
      messages: request.messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.rawContent ?? m.content,
        toolName: m.toolName,
        toolCallId: m.toolCallId,
        toolArgs: m.toolArgs,
      })),
      tools: request.tools
        ? this.serializeTools
          ? this.serializeTools(request.tools)
          : request.tools
        : undefined,
      prefixSlotKey: request.prefixSlotKey,
      ...(request.stopOnToolCall ? { stopOnToolCall: true } : {}),
      ...(request.groundingText ? { groundingText: request.groundingText } : {}),
      config: {
        maxNewTokens: request.maxNewTokens,
        temperature: request.temperature,
        topP: request.topP,
        topK: request.topK,
        enableThinking: request.enableThinking,
        gpuPacingMode: request.pacingMode ?? this.defaultPacingMode,
      },
    });

    try {
      for await (const delta of queue) {
        yield delta;
      }
    } finally {
      finished = true;
      unsubscribe();
      signal?.removeEventListener('abort', onAbort);
    }
  }
}
