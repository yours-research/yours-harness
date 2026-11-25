import type {
  GenerateRequest,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
} from '../contracts.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface WebLlmChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
      reasoning_content?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    extra?: {
      decode_tokens_per_s?: number;
      time_to_first_token_s?: number;
    };
  };
}

export interface WebLlmEngineLike {
  chat: {
    completions: {
      create(params: Record<string, unknown>): Promise<AsyncIterable<WebLlmChunk>>;
    };
  };
  interruptGenerate?(): void;
}

export interface WebLlmAdapterOptions {
  id?: string;
  engine: WebLlmEngineLike;
  maxContextTokens?: number;
  supportsTools?: boolean;
  countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;
}

/**
 * RuntimeAdapter shim for MLC `@mlc-ai/web-llm` engines.
 */
export class WebLlmAdapter implements RuntimeAdapter {
  private readonly id: string;
  private readonly engine: WebLlmEngineLike;
  private readonly maxContextTokens: number;
  private readonly supportsTools: boolean;
  private readonly countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;

  constructor(options: WebLlmAdapterOptions) {
    this.id = options.id ?? 'webllm';
    this.engine = options.engine;
    this.maxContextTokens = options.maxContextTokens ?? 32768;
    this.supportsTools = options.supportsTools ?? true;
    this.countTokensFn = options.countTokensFn;
  }

  capabilities(): RuntimeCapabilities {
    return {
      id: this.id,
      maxContextTokens: this.maxContextTokens,
      supportsStreaming: true,
      supportsPrefixSlots: false,
      supportsTools: this.supportsTools,
      tier: 'tier1-f16-subgroups',
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

    const onAbort = () => {
      this.engine.interruptGenerate?.();
    };
    signal?.addEventListener('abort', onAbort, { once: true });

    const messages = request.messages.map((m) => ({
      role: m.role === 'tool' ? 'tool' : m.role,
      content: m.content,
      tool_call_id: m.toolCallId,
      name: m.toolName,
    }));

    const params: Record<string, unknown> = {
      messages,
      stream: true,
      stream_options: { include_usage: true },
    };

    if (request.maxNewTokens !== undefined) {
      params.max_tokens = request.maxNewTokens;
    }
    if (request.temperature !== undefined) {
      params.temperature = request.temperature;
    }
    if (request.topP !== undefined) {
      params.top_p = request.topP;
    }
    if (this.supportsTools && request.tools && request.tools.length > 0) {
      params.tools = request.tools.map((t) => ({
        type: 'function',
        function: {
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        },
      }));
    }

    let finishReason: 'stop' | 'length' | 'tool_calls' = 'stop';
    const activeToolCalls = new Map<number, { id: string; name: string; args: string }>();

    try {
      const stream = await this.engine.chat.completions.create(params);

      for await (const chunk of stream) {
        if (signal?.aborted) {
          yield { type: 'done', finishReason: 'abort' };
          return;
        }

        if (chunk.usage) {
          yield {
            type: 'telemetry',
            kvTokens: chunk.usage.prompt_tokens,
            tokensPerSec: chunk.usage.extra?.decode_tokens_per_s,
            ttftMs:
              chunk.usage.extra?.time_to_first_token_s !== undefined
                ? chunk.usage.extra.time_to_first_token_s * 1000
                : undefined,
          };
        }

        const choice = chunk.choices?.[0];
        if (!choice) {
          continue;
        }

        if (choice.finish_reason === 'length') {
          finishReason = 'length';
        } else if (choice.finish_reason === 'tool_calls') {
          finishReason = 'tool_calls';
        }

        const delta = choice.delta;
        if (delta?.reasoning_content) {
          yield { type: 'reasoning', delta: delta.reasoning_content };
        }
        if (delta?.content) {
          yield { type: 'text', delta: delta.content };
        }
        if (Array.isArray(delta?.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const idx = tc.index ?? 0;
            let acc = activeToolCalls.get(idx);
            if (!acc) {
              acc = { id: tc.id ?? `webllm_tc_${idx}`, name: tc.function?.name ?? '', args: '' };
              activeToolCalls.set(idx, acc);
              if (acc.name) {
                yield { type: 'tool_call_start', id: acc.id, name: acc.name };
              }
            }
            if (tc.function?.arguments) {
              acc.args += tc.function.arguments;
              yield { type: 'tool_call_args', id: acc.id, delta: tc.function.arguments };
            }
          }
        }
      }

      for (const [, call] of activeToolCalls.entries()) {
        if (call.name) {
          let parsedArgs: Record<string, unknown> = {};
          try {
            const raw = JSON.parse(call.args || '{}');
            if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
              parsedArgs = raw as Record<string, unknown>;
            }
          } catch {
            parsedArgs = { raw: call.args };
          }
          yield {
            type: 'tool_call_ready',
            id: call.id,
            name: call.name,
            arguments: parsedArgs,
          };
        }
      }

      yield { type: 'done', finishReason };
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
        yield { type: 'done', finishReason: 'abort' };
        return;
      }
      throw err;
    } finally {
      signal?.removeEventListener('abort', onAbort);
    }
  }
}
