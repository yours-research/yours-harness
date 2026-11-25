import type {
  GenerateRequest,
  HarnessMessage,
  RuntimeAdapter,
  RuntimeCapabilities,
  StreamDelta,
  ToolDefinition,
} from '../contracts.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface OpenAiCompatibleAdapterOptions {
  id?: string;
  baseUrl: string;
  model: string;
  apiKey?: string;
  maxContextTokens?: number;
  supportsTools?: boolean;
  headers?: Record<string, string>;
  fetchFn?: typeof fetch;
  countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;
}

interface OpenAiToolCallChunk {
  index?: number;
  id?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
}

interface OpenAiSseChunk {
  choices?: Array<{
    delta?: {
      content?: string | null;
      reasoning_content?: string | null;
      reasoning?: string | null;
      tool_calls?: OpenAiToolCallChunk[];
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

function mapFinishReason(reason: string | null | undefined): 'stop' | 'length' | 'tool_calls' {
  if (reason === 'length') {
    return 'length';
  }
  if (reason === 'tool_calls' || reason === 'function_call') {
    return 'tool_calls';
  }
  return 'stop';
}

function formatMessagesForOpenAi(messages: readonly HarnessMessage[]): Array<Record<string, unknown>> {
  return messages.map((m) => {
    if (m.role === 'tool') {
      return {
        role: 'tool',
        tool_call_id: m.toolCallId ?? m.id,
        name: m.toolName,
        content: m.content,
      };
    }
    if (m.role === 'assistant' && m.toolName) {
      return {
        role: 'assistant',
        content: m.content || null,
        tool_calls: [
          {
            id: m.toolCallId ?? m.id,
            type: 'function',
            function: {
              name: m.toolName,
              arguments: JSON.stringify(m.toolArgs ?? {}),
            },
          },
        ],
      };
    }
    return {
      role: m.role,
      content: m.content,
    };
  });
}

function formatToolsForOpenAi(tools?: readonly ToolDefinition[]): Array<Record<string, unknown>> | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }
  return tools.map((tool) => ({
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  }));
}

function safeParseJsonObject(raw: string): Record<string, unknown> {
  if (!raw.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { value: parsed };
  } catch {
    return { raw };
  }
}

/**
 * RuntimeAdapter for local Ollama, llama.cpp server, vLLM, LM Studio,
 * or any OpenAI-compatible `/v1/chat/completions` SSE endpoint.
 */
export class OpenAiCompatibleAdapter implements RuntimeAdapter {
  private readonly id: string;
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly apiKey?: string;
  private readonly maxContextTokens: number;
  private readonly supportsTools: boolean;
  private readonly headers: Record<string, string>;
  private readonly fetchFn: typeof fetch;
  private readonly countTokensFn?: (messages: HarnessMessage[]) => Promise<number> | number;

  constructor(options: OpenAiCompatibleAdapterOptions) {
    this.id = options.id ?? `openai-compat:${options.model}`;
    this.baseUrl = options.baseUrl.replace(/\/+$/, '');
    this.model = options.model;
    this.apiKey = options.apiKey;
    this.maxContextTokens = options.maxContextTokens ?? 32768;
    this.supportsTools = options.supportsTools ?? true;
    this.headers = options.headers ?? {};
    this.fetchFn = options.fetchFn ?? globalThis.fetch.bind(globalThis);
    this.countTokensFn = options.countTokensFn;
  }

  capabilities(): RuntimeCapabilities {
    return {
      id: this.id,
      maxContextTokens: this.maxContextTokens,
      supportsStreaming: true,
      supportsPrefixSlots: false,
      supportsTools: this.supportsTools,
      tier: 'http',
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

    const endpoint = this.baseUrl.endsWith('/chat/completions')
      ? this.baseUrl
      : `${this.baseUrl}/chat/completions`;

    const body: Record<string, unknown> = {
      model: this.model,
      messages: formatMessagesForOpenAi(request.messages),
      stream: true,
    };

    const tools = this.supportsTools ? formatToolsForOpenAi(request.tools) : undefined;
    if (tools) {
      body.tools = tools;
    }
    if (request.maxNewTokens !== undefined) {
      body.max_tokens = request.maxNewTokens;
    }
    if (request.temperature !== undefined) {
      body.temperature = request.temperature;
    }
    if (request.topP !== undefined) {
      body.top_p = request.topP;
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.headers,
    };
    if (this.apiKey) {
      headers.Authorization = `Bearer ${this.apiKey}`;
    }

    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let outputChunkCount = 0;

    let response: Response;
    try {
      response = await this.fetchFn(endpoint, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal,
      });
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
        yield { type: 'done', finishReason: 'abort' };
        return;
      }
      throw err;
    }

    if (!response.ok) {
      const errorText = await response.text().catch(() => '');
      throw new Error(
        `OpenAI-compatible endpoint returned HTTP ${response.status}: ${errorText || response.statusText}`,
      );
    }

    if (!response.body) {
      throw new Error('OpenAI-compatible endpoint response contained no readable stream body.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '';
    const activeToolCalls = new Map<number, { id: string; name: string; args: string }>();
    let lastFinishReason: 'stop' | 'length' | 'tool_calls' = 'stop';
    let emittedDone = false;

    const flushToolCalls = (): StreamDelta[] => {
      const deltas: StreamDelta[] = [];
      for (const [, call] of activeToolCalls.entries()) {
        if (call.name) {
          deltas.push({
            type: 'tool_call_ready',
            id: call.id,
            name: call.name,
            arguments: safeParseJsonObject(call.args),
          });
        }
      }
      activeToolCalls.clear();
      return deltas;
    };

    try {
      while (true) {
        if (signal?.aborted) {
          yield { type: 'done', finishReason: 'abort' };
          emittedDone = true;
          break;
        }

        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split(/\r?\n/);
        sseBuffer = lines.pop() ?? '';

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || line.startsWith(':') || !line.startsWith('data:')) {
            continue;
          }

          const payload = line.slice(5).trim();
          if (payload === '[DONE]') {
            for (const tcDelta of flushToolCalls()) {
              yield tcDelta;
            }
            const elapsedSec = Math.max(0.001, (performance.now() - (firstTokenAt ?? startedAt)) / 1000);
            yield {
              type: 'telemetry',
              ttftMs: firstTokenAt !== null ? firstTokenAt - startedAt : undefined,
              tokensPerSec: outputChunkCount > 0 ? outputChunkCount / elapsedSec : undefined,
            };
            yield { type: 'done', finishReason: lastFinishReason };
            emittedDone = true;
            return;
          }

          let chunk: OpenAiSseChunk;
          try {
            chunk = JSON.parse(payload) as OpenAiSseChunk;
          } catch {
            continue;
          }

          if (chunk.usage?.total_tokens !== undefined) {
            yield {
              type: 'telemetry',
              kvTokens: chunk.usage.prompt_tokens,
            };
          }

          const choice = chunk.choices?.[0];
          if (!choice) {
            continue;
          }

          if (choice.finish_reason) {
            lastFinishReason = mapFinishReason(choice.finish_reason);
          }

          const delta = choice.delta;
          if (!delta) {
            continue;
          }

          const reasoning = delta.reasoning_content ?? delta.reasoning;
          if (typeof reasoning === 'string' && reasoning.length > 0) {
            if (firstTokenAt === null) {
              firstTokenAt = performance.now();
            }
            outputChunkCount += 1;
            yield { type: 'reasoning', delta: reasoning };
          }

          if (typeof delta.content === 'string' && delta.content.length > 0) {
            if (firstTokenAt === null) {
              firstTokenAt = performance.now();
            }
            outputChunkCount += 1;
            yield { type: 'text', delta: delta.content };
          }

          if (Array.isArray(delta.tool_calls)) {
            for (const tc of delta.tool_calls) {
              const index = tc.index ?? 0;
              let acc = activeToolCalls.get(index);
              if (!acc) {
                acc = {
                  id: tc.id ?? `call_${index}`,
                  name: tc.function?.name ?? '',
                  args: '',
                };
                activeToolCalls.set(index, acc);
                if (acc.name) {
                  yield { type: 'tool_call_start', id: acc.id, name: acc.name };
                }
              } else {
                if (tc.id) {
                  acc.id = tc.id;
                }
                if (tc.function?.name && !acc.name) {
                  acc.name = tc.function.name;
                  yield { type: 'tool_call_start', id: acc.id, name: acc.name };
                }
              }

              if (typeof tc.function?.arguments === 'string' && tc.function.arguments.length > 0) {
                acc.args += tc.function.arguments;
                yield { type: 'tool_call_args', id: acc.id, delta: tc.function.arguments };
              }
            }
          }
        }
      }
    } catch (err) {
      if (signal?.aborted || (err instanceof Error && err.name === 'AbortError')) {
        if (!emittedDone) {
          yield { type: 'done', finishReason: 'abort' };
        }
        return;
      }
      throw err;
    } finally {
      reader.releaseLock();
    }

    if (!emittedDone) {
      for (const tcDelta of flushToolCalls()) {
        yield tcDelta;
      }
      yield { type: 'done', finishReason: lastFinishReason };
    }
  }
}
