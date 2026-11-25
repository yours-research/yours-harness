import type {
  GenerateRequest,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
} from '../contracts.js';
import { AsyncPushQueue } from '../utils/asyncQueue.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface TransformersJsRunOptions {
  messages: Array<{ role: string; content: string }>;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  signal?: AbortSignal;
  onToken: (textDelta: string) => void;
}

export interface TransformersJsAdapterOptions {
  id?: string;
  modelId: string;
  maxContextTokens?: number;
  supportsTools?: boolean;
  runGeneration: (options: TransformersJsRunOptions) => Promise<{
    finishReason?: 'stop' | 'length' | 'abort';
    kvTokens?: number;
  } | void>;
  countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;
}

/**
 * RuntimeAdapter shim for `@huggingface/transformers` (Transformers.js / ORT WebGPU).
 * Zero compile-time or bundle-time dependency on `@huggingface/transformers`.
 */
export class TransformersJsAdapter implements RuntimeAdapter {
  private readonly id: string;
  private readonly maxContextTokens: number;
  private readonly supportsTools: boolean;
  private readonly runGeneration: TransformersJsAdapterOptions['runGeneration'];
  private readonly countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;

  constructor(options: TransformersJsAdapterOptions) {
    this.id = options.id ?? `transformers-js:${options.modelId}`;
    this.maxContextTokens = options.maxContextTokens ?? 16384;
    this.supportsTools = options.supportsTools ?? true;
    this.runGeneration = options.runGeneration;
    this.countTokensFn = options.countTokensFn;
  }

  capabilities(): RuntimeCapabilities {
    return {
      id: this.id,
      maxContextTokens: this.maxContextTokens,
      supportsStreaming: true,
      supportsPrefixSlots: false,
      supportsTools: this.supportsTools,
      tier: 'tier2-f16-shared',
    };
  }

  countTokens(messages: HarnessMessage[]): Promise<number> | number {
    if (this.countTokensFn) {
      return this.countTokensFn(messages);
    }
    return estimateMessagesTokens(messages);
  }

  async *generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta> {
    if (signal?.aborted) {
      yield { type: 'done', finishReason: 'abort' };
      return;
    }

    const queue = new AsyncPushQueue<StreamDelta>();
    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let tokenEvents = 0;
    let aborted = false;

    const onAbort = () => {
      aborted = true;
      queue.push({ type: 'done', finishReason: 'abort' });
      queue.close();
    };
    signal?.addEventListener('abort', onAbort, { once: true });

    const chatMessages = request.messages.map((m) => ({
      role: m.role === 'tool' ? 'user' : m.role,
      content: m.role === 'tool' ? `<tool_result name="${m.toolName ?? 'tool'}">\n${m.content}\n</tool_result>` : m.content,
    }));

    void this.runGeneration({
      messages: chatMessages,
      maxNewTokens: request.maxNewTokens,
      temperature: request.temperature,
      topP: request.topP,
      topK: request.topK,
      signal,
      onToken: (delta) => {
        if (aborted || !delta) {
          return;
        }
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        tokenEvents += 1;
        queue.push({ type: 'text', delta });
      },
    })
      .then((result) => {
        if (aborted) {
          return;
        }
        const elapsedSec = Math.max(0.001, (performance.now() - (firstTokenAt ?? startedAt)) / 1000);
        queue.push({
          type: 'telemetry',
          ttftMs: firstTokenAt !== null ? firstTokenAt - startedAt : undefined,
          tokensPerSec: tokenEvents > 0 ? tokenEvents / elapsedSec : undefined,
          kvTokens: result?.kvTokens,
        });
        queue.push({ type: 'done', finishReason: result?.finishReason ?? 'stop' });
        queue.close();
      })
      .catch((err) => {
        if (aborted || signal?.aborted) {
          queue.push({ type: 'done', finishReason: 'abort' });
          queue.close();
          return;
        }
        queue.fail(err);
      });

    try {
      for await (const delta of queue) {
        yield delta;
      }
    } finally {
      signal?.removeEventListener('abort', onAbort);
    }
  }
}
