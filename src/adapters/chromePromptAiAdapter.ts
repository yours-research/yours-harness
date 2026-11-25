import type {
  GenerateRequest,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
} from '../contracts.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface ChromeLanguageModelSession {
  promptStreaming(input: string, options?: { signal?: AbortSignal }): AsyncIterable<string>;
  countPromptTokens?(input: string): Promise<number>;
  destroy?(): void;
}

export interface ChromeLanguageModelFactory {
  availability?(): Promise<'available' | 'downloadable' | 'downloading' | 'unavailable'>;
  create(options?: {
    systemPrompt?: string;
    initialPrompts?: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
    temperature?: number;
    topK?: number;
    signal?: AbortSignal;
  }): Promise<ChromeLanguageModelSession>;
}

export interface ChromePromptAiAdapterOptions {
  id?: string;
  factory?: ChromeLanguageModelFactory;
  maxContextTokens?: number;
  deltaMode?: 'auto' | 'delta' | 'cumulative';
}

function resolveDefaultChromeFactory(): ChromeLanguageModelFactory | null {
  const g = globalThis as unknown as {
    LanguageModel?: ChromeLanguageModelFactory;
    ai?: { languageModel?: ChromeLanguageModelFactory };
  };
  return g.LanguageModel ?? g.ai?.languageModel ?? null;
}

function formatPromptMessages(messages: readonly HarnessMessage[]): {
  systemPrompt?: string;
  initialPrompts: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>;
  userPrompt: string;
} {
  const systemParts: string[] = [];
  const nonSystem: HarnessMessage[] = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      systemParts.push(msg.content);
    } else {
      nonSystem.push(msg);
    }
  }

  const last = nonSystem[nonSystem.length - 1];
  const history = nonSystem.slice(0, Math.max(0, nonSystem.length - 1));

  const initialPrompts = history.map((m) => ({
    role: (m.role === 'tool' ? 'user' : m.role) as 'user' | 'assistant',
    content: m.role === 'tool' ? `[Tool ${m.toolName ?? 'result'}]\n${m.content}` : m.content,
  }));

  const userPrompt = last
    ? last.role === 'tool'
      ? `[Tool ${last.toolName ?? 'result'}]\n${last.content}`
      : last.content
    : '';

  return {
    systemPrompt: systemParts.length > 0 ? systemParts.join('\n\n') : undefined,
    initialPrompts,
    userPrompt,
  };
}

/**
 * RuntimeAdapter bridging Chrome's Built-in AI (`LanguageModel` / `ai.languageModel`)
 * Prompt API into `@yours/harness`.
 */
export class ChromePromptAiAdapter implements RuntimeAdapter {
  private readonly id: string;
  private readonly factoryOverride?: ChromeLanguageModelFactory;
  private readonly maxContextTokens: number;
  private readonly deltaMode: 'auto' | 'delta' | 'cumulative';

  constructor(options: ChromePromptAiAdapterOptions = {}) {
    this.id = options.id ?? 'chrome-prompt-ai';
    this.factoryOverride = options.factory;
    this.maxContextTokens = options.maxContextTokens ?? 8192;
    this.deltaMode = options.deltaMode ?? 'auto';
  }

  private getFactory(): ChromeLanguageModelFactory {
    const factory = this.factoryOverride ?? resolveDefaultChromeFactory();
    if (!factory) {
      throw new Error(
        'Chrome Built-in AI LanguageModel API is unavailable in this environment.',
      );
    }
    return factory;
  }

  async isAvailable(): Promise<boolean> {
    const factory = this.factoryOverride ?? resolveDefaultChromeFactory();
    if (!factory) {
      return false;
    }
    if (!factory.availability) {
      return true;
    }
    const status = await factory.availability();
    return status === 'available' || status === 'downloadable';
  }

  capabilities(): RuntimeCapabilities {
    return {
      id: this.id,
      maxContextTokens: this.maxContextTokens,
      supportsStreaming: true,
      supportsPrefixSlots: false,
      supportsTools: false,
      tier: 'cpu',
    };
  }

  countTokens(messages: HarnessMessage[]): number {
    return estimateMessagesTokens(messages);
  }

  async *generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta> {
    if (signal?.aborted) {
      yield { type: 'done', finishReason: 'abort' };
      return;
    }

    const factory = this.getFactory();
    const { systemPrompt, initialPrompts, userPrompt } = formatPromptMessages(request.messages);

    const session = await factory.create({
      systemPrompt,
      initialPrompts,
      temperature: request.temperature,
      topK: request.topK,
      signal,
    });

    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let accumulated = '';
    let chunkCount = 0;

    try {
      const stream = session.promptStreaming(userPrompt, { signal });
      for await (const rawChunk of stream) {
        if (signal?.aborted) {
          yield { type: 'done', finishReason: 'abort' };
          return;
        }
        let delta = rawChunk;
        if (this.deltaMode === 'cumulative') {
          delta = rawChunk.startsWith(accumulated) ? rawChunk.slice(accumulated.length) : rawChunk;
          accumulated = rawChunk;
        } else if (this.deltaMode === 'auto') {
          if (accumulated.length > 0 && rawChunk.startsWith(accumulated)) {
            delta = rawChunk.slice(accumulated.length);
            accumulated = rawChunk;
          } else {
            accumulated += rawChunk;
          }
        } else {
          accumulated += rawChunk;
        }

        if (delta) {
          if (firstTokenAt === null) {
            firstTokenAt = performance.now();
          }
          chunkCount += 1;
          yield { type: 'text', delta };
        }
      }

      const elapsedSec = Math.max(0.001, (performance.now() - (firstTokenAt ?? startedAt)) / 1000);
      yield {
        type: 'telemetry',
        ttftMs: firstTokenAt !== null ? firstTokenAt - startedAt : undefined,
        tokensPerSec: chunkCount > 0 ? chunkCount / elapsedSec : undefined,
      };
      yield { type: 'done', finishReason: 'stop' };
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
        yield { type: 'done', finishReason: 'abort' };
        return;
      }
      throw err;
    } finally {
      session.destroy?.();
    }
  }
}
