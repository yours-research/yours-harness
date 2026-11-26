export interface ParsedToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  raw: string;
}

export interface StreamParserOptions {
  stopSequences?: string[];
  stopOnToolCall?: boolean;
  /** Allow recovering bare JSON tool call objects without <tool_call> tags (default: true). */
  allowBareJsonToolCall?: boolean;
  /** Initial parser mode (default: 'text'). */
  initialMode?: 'text' | 'think';
  /** Optional deterministic ID factory for parsed tool calls. */
  makeToolCallId?: (index: number) => string;
}

export interface StreamParserStepOutput {
  textDelta: string;
  reasoningDelta: string;
  pendingToolCall: { name: string } | null;
  toolCallsReady: ParsedToolCall[];
  stopSequenceHit: string | null;
  visibleTextReset?: boolean;
}

const DEFAULT_STOP_SEQUENCES = ['<|im_end|>', '<|endoftext|>', '<end_of_turn>'];
const PARTIAL_NAME_RE = /["']?name["']?\s*:\s*["']([^"']+)["']/;
const TOOL_CALL_OPEN_TAG = '<tool_call>';
const TOOL_CALL_CLOSE_TAG = '</tool_call>';
const THINK_OPEN_TAGS = ['<think>', '<thought>', '<|channel>thought\n', '<|channel>thought', '<|thought|>'];
const THINK_CLOSE_TAGS = ['</think>', '</thought>', '<channel|>', '<|/thought|>', '<|end_of_thought|>'];

/**
 * Attempts to parse a JSON object even with common small-model formatting quirks:
 * - trailing commas
 * - unquoted keys
 * - single-quoted keys/values
 * - missing closing braces/brackets
 */
export function parseLenientJsonObject(rawInput: string): Record<string, unknown> | null {
  const trimmed = rawInput.trim();
  if (!trimmed) {
    return null;
  }

  // Strip optional markdown code fences
  let candidate = trimmed
    .replace(/^```(?:json|tool_call)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  const firstBrace = candidate.indexOf('{');
  if (firstBrace === -1) {
    return null;
  }
  candidate = candidate.slice(firstBrace);

  const tryParse = (str: string): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(str);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
      return null;
    } catch {
      return null;
    }
  };

  const direct = tryParse(candidate);
  if (direct) {
    return direct;
  }

  // Normalize common small-model quirks: trailing commas, unquoted keys, single quotes
  let repaired = candidate
    // Quote unquoted keys like { name: "x" } or , arguments: {
    .replace(/([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)/g, '$1"$2"$3')
    // Replace single-quoted keys: {'name': ...} -> {"name": ...}
    .replace(/([{,]\s*)'([^'\\]*(?:\\.[^'\\]*)*)'(\s*:)/g, '$1"$2"$3')
    // Replace single-quoted string values: : 'value' -> : "value"
    .replace(/(:\s*)'([^'\\]*(?:\\.[^'\\]*)*)'/g, (_m, p1: string, val: string) => {
      const escaped = val.replace(/"/g, '\\"');
      return `${p1}"${escaped}"`;
    })
    // Remove trailing commas before } or ]
    .replace(/,\s*([}\]])/g, '$1');

  const repairedDirect = tryParse(repaired);
  if (repairedDirect) {
    return repairedDirect;
  }

  // Balance unclosed strings, brackets, and braces for truncated streams
  let inString = false;
  let escaped = false;
  const stack: string[] = [];

  for (let i = 0; i < repaired.length; i += 1) {
    const ch = repaired[i]!;
    if (escaped) {
      escaped = false;
      continue;
    }
    if (ch === '\\') {
      escaped = true;
      continue;
    }
    if (ch === '"') {
      inString = !inString;
      continue;
    }
    if (!inString) {
      if (ch === '{') {
        stack.push('}');
      } else if (ch === '[') {
        stack.push(']');
      } else if ((ch === '}' || ch === ']') && stack[stack.length - 1] === ch) {
        stack.pop();
      }
    }
  }

  let closed = repaired;
  if (inString) {
    closed += '"';
  }
  // Remove trailing comma after closing partial value
  closed = closed.replace(/,\s*$/, '');
  while (stack.length > 0) {
    closed += stack.pop();
  }
  closed = closed.replace(/,\s*([}\]])/g, '$1');

  return tryParse(closed);
}

/**
 * Extracts a tool call from a raw payload (inside <tool_call> or JSON block).
 */
export function extractToolCallFromPayload(
  payload: string,
  id: string,
): ParsedToolCall | null {
  const parsed = parseLenientJsonObject(payload);
  if (!parsed) {
    return null;
  }
  const rawName = parsed.name ?? parsed.tool ?? parsed.function;
  if (typeof rawName !== 'string' || !rawName.trim()) {
    return null;
  }
  const rawArgs = parsed.arguments ?? parsed.args ?? parsed.parameters ?? {};
  let args: Record<string, unknown> = {};
  if (typeof rawArgs === 'string') {
    args = parseLenientJsonObject(rawArgs) ?? {};
  } else if (rawArgs && typeof rawArgs === 'object' && !Array.isArray(rawArgs)) {
    args = rawArgs as Record<string, unknown>;
  }

  return {
    id,
    name: rawName.trim(),
    arguments: args,
    raw: payload,
  };
}

function findEarliestIndex(haystack: string, needles: readonly string[]): { index: number; tag: string } | null {
  let bestIdx = -1;
  let bestTag = '';
  for (const needle of needles) {
    const idx = haystack.indexOf(needle);
    if (
      idx !== -1 &&
      (bestIdx === -1 || idx < bestIdx || (idx === bestIdx && needle.length > bestTag.length))
    ) {
      bestIdx = idx;
      bestTag = needle;
    }
  }
  return bestIdx === -1 ? null : { index: bestIdx, tag: bestTag };
}

function trailingPartialPrefixLength(buffer: string, tokens: readonly string[]): number {
  let maxKeep = 0;
  for (const token of tokens) {
    const maxCheck = Math.min(buffer.length, token.length - 1);
    for (let len = maxCheck; len >= 1; len -= 1) {
      if (token.startsWith(buffer.slice(buffer.length - len))) {
        if (len > maxKeep) {
          maxKeep = len;
        }
        break;
      }
    }
  }
  return maxKeep;
}

function emptyStepOutput(): StreamParserStepOutput {
  return {
    textDelta: '',
    reasoningDelta: '',
    pendingToolCall: null,
    toolCallsReady: [],
    stopSequenceHit: null,
    visibleTextReset: false,
  };
}

/**
 * Incremental streaming parser tolerant of small local LLM quirks:
 * - Interleaved, pre-opened, or unclosed `<think>` / `<thought>` / `<|channel>thought` / `<|thought|>` reasoning blocks
 * - XML `<tool_call>...</tool_call>` blocks
 * - Malformed/partial JSON inside tool calls
 * - Bare JSON or fenced JSON tool calls on flush
 * - Stop-sequence guards (prevents stop tokens leaking into UI text)
 */
export class StreamOutputParser {
  private readonly stopSequences: string[];
  private readonly stopOnToolCall: boolean;
  private readonly allowBareJsonToolCall: boolean;
  private readonly makeToolCallId: (index: number) => string;

  private mode: 'text' | 'think' | 'tool_call' = 'text';
  private buffer = '';
  private pendingToolPayload = '';
  private pendingPostThinkNewlineTrim = false;
  private toolCallCounter = 0;
  private lastPendingToolName: string | null = null;
  private stopped = false;
  private fullVisibleText = '';
  private fullReasoningText = '';
  private readonly parsedToolCalls: ParsedToolCall[] = [];

  constructor(options: StreamParserOptions = {}) {
    this.stopSequences = options.stopSequences ?? DEFAULT_STOP_SEQUENCES;
    this.stopOnToolCall = options.stopOnToolCall ?? false;
    this.allowBareJsonToolCall = options.allowBareJsonToolCall ?? true;
    this.makeToolCallId = options.makeToolCallId ?? ((idx) => `tc_${idx}`);
    this.mode = options.initialMode ?? 'text';
  }

  getVisibleText(): string {
    return this.fullVisibleText;
  }

  getReasoningText(): string {
    return this.fullReasoningText;
  }

  getToolCalls(): ParsedToolCall[] {
    return [...this.parsedToolCalls];
  }

  pushText(chunk: string): StreamParserStepOutput {
    if (this.stopped || !chunk) {
      return emptyStepOutput();
    }

    this.buffer += chunk;
    return this.drain(false);
  }

  flush(): StreamParserStepOutput {
    if (this.stopped && !this.buffer && !this.pendingToolPayload) {
      return emptyStepOutput();
    }
    return this.drain(true);
  }

  private drain(isFlush: boolean): StreamParserStepOutput {
    let textOut = '';
    let reasoningOut = '';
    const toolCallsReady: ParsedToolCall[] = [];
    let pendingToolCall: { name: string } | null = null;
    let stopSequenceHit: string | null = null;
    let visibleTextReset = false;

    const emitText = (s: string) => {
      if (s) {
        textOut += s;
        this.fullVisibleText += s;
      }
    };
    const emitReasoning = (s: string) => {
      if (s) {
        reasoningOut += s;
        this.fullReasoningText += s;
      }
    };

    const stripPostThinkLeadingNewlines = () => {
      this.buffer = this.buffer.replace(/^(?:\r?\n)+/, '');
      this.pendingPostThinkNewlineTrim =
        this.buffer.length === 0 || (!isFlush && !this.stopped && this.buffer === '\r');
    };

    while (this.buffer.length > 0) {
      // 1. Stop sequence check in text / think modes
      if (!this.stopped && this.mode !== 'tool_call' && this.stopSequences.length > 0) {
        const stopHit = findEarliestIndex(this.buffer, this.stopSequences);
        if (stopHit) {
          const beforeStop = this.buffer.slice(0, stopHit.index);
          this.buffer = beforeStop;
          stopSequenceHit = stopHit.tag;
          this.stopped = true;
          // Continue draining beforeStop cleanly, then stop
        }
      }

      if (this.mode === 'text') {
        if (this.pendingPostThinkNewlineTrim && this.buffer.length > 0) {
          stripPostThinkLeadingNewlines();
          if (this.buffer.length === 0 || this.pendingPostThinkNewlineTrim) {
            break;
          }
        }

        const candidateTags = [
          ...THINK_OPEN_TAGS,
          ...THINK_CLOSE_TAGS,
          TOOL_CALL_OPEN_TAG,
        ];
        const tagMatch = findEarliestIndex(this.buffer, candidateTags);

        if (tagMatch) {
          if (
            !isFlush &&
            !this.stopped &&
            tagMatch.index + tagMatch.tag.length === this.buffer.length &&
            [...candidateTags, ...this.stopSequences].some(
              (candidate) => candidate.length > tagMatch.tag.length && candidate.startsWith(tagMatch.tag),
            )
          ) {
            if (tagMatch.index > 0) {
              emitText(this.buffer.slice(0, tagMatch.index));
              this.buffer = this.buffer.slice(tagMatch.index);
            }
            break;
          }

          if (THINK_CLOSE_TAGS.includes(tagMatch.tag)) {
            const promotedReasoning = this.fullVisibleText + this.buffer.slice(0, tagMatch.index);
            emitReasoning(promotedReasoning);
            this.fullVisibleText = '';
            textOut = '';
            visibleTextReset = true;
            this.buffer = this.buffer.slice(tagMatch.index + tagMatch.tag.length);
            stripPostThinkLeadingNewlines();
            continue;
          }

          emitText(this.buffer.slice(0, tagMatch.index));
          this.buffer = this.buffer.slice(tagMatch.index + tagMatch.tag.length);
          if (tagMatch.tag === TOOL_CALL_OPEN_TAG) {
            this.mode = 'tool_call';
            this.pendingToolPayload = '';
            this.lastPendingToolName = null;
          } else {
            this.mode = 'think';
          }
          continue;
        }

        if (isFlush || this.stopped) {
          emitText(this.buffer);
          this.buffer = '';
          break;
        }

        const holdLen = trailingPartialPrefixLength(this.buffer, [
          ...THINK_OPEN_TAGS,
          ...THINK_CLOSE_TAGS,
          TOOL_CALL_OPEN_TAG,
          ...this.stopSequences,
        ]);
        const safeLen = this.buffer.length - holdLen;
        if (safeLen > 0) {
          emitText(this.buffer.slice(0, safeLen));
          this.buffer = this.buffer.slice(safeLen);
        }
        break;
      }

      if (this.mode === 'think') {
        const closeMatch = findEarliestIndex(this.buffer, THINK_CLOSE_TAGS);
        if (closeMatch) {
          emitReasoning(this.buffer.slice(0, closeMatch.index));
          this.buffer = this.buffer.slice(closeMatch.index + closeMatch.tag.length);
          this.mode = 'text';
          if (this.fullVisibleText === '') {
            stripPostThinkLeadingNewlines();
          }
          continue;
        }

        if (isFlush || this.stopped) {
          emitReasoning(this.buffer);
          this.buffer = '';
          this.mode = 'text';
          break;
        }

        const holdLen = trailingPartialPrefixLength(this.buffer, [
          ...THINK_CLOSE_TAGS,
          ...this.stopSequences,
        ]);
        const safeLen = this.buffer.length - holdLen;
        if (safeLen > 0) {
          emitReasoning(this.buffer.slice(0, safeLen));
          this.buffer = this.buffer.slice(safeLen);
        }
        break;
      }

      if (this.mode === 'tool_call') {
        const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE_TAG);
        if (closeIdx !== -1) {
          this.pendingToolPayload += this.buffer.slice(0, closeIdx);
          this.buffer = this.buffer.slice(closeIdx + TOOL_CALL_CLOSE_TAG.length);

          this.toolCallCounter += 1;
          const id = this.makeToolCallId(this.toolCallCounter);
          const parsed = extractToolCallFromPayload(this.pendingToolPayload, id);
          if (parsed) {
            this.parsedToolCalls.push(parsed);
            toolCallsReady.push(parsed);
          } else {
            // Unrecoverable malformed payload: surface raw text so nothing vanishes silently
            emitText(`${TOOL_CALL_OPEN_TAG}${this.pendingToolPayload}${TOOL_CALL_CLOSE_TAG}`);
          }
          this.pendingToolPayload = '';
          this.lastPendingToolName = null;
          this.mode = 'text';

          if (this.stopOnToolCall && parsed) {
            this.stopped = true;
            this.buffer = '';
            stopSequenceHit = TOOL_CALL_CLOSE_TAG;
            break;
          }
          continue;
        }

        if (isFlush || this.stopped) {
          this.pendingToolPayload += this.buffer;
          this.buffer = '';
          this.toolCallCounter += 1;
          const id = this.makeToolCallId(this.toolCallCounter);
          const parsed = extractToolCallFromPayload(this.pendingToolPayload, id);
          if (parsed) {
            this.parsedToolCalls.push(parsed);
            toolCallsReady.push(parsed);
          } else {
            emitText(`${TOOL_CALL_OPEN_TAG}${this.pendingToolPayload}`);
          }
          this.pendingToolPayload = '';
          this.lastPendingToolName = null;
          this.mode = 'text';
          break;
        }

        const holdLen = trailingPartialPrefixLength(this.buffer, [TOOL_CALL_CLOSE_TAG]);
        const safeLen = this.buffer.length - holdLen;
        if (safeLen > 0) {
          this.pendingToolPayload += this.buffer.slice(0, safeLen);
          this.buffer = this.buffer.slice(safeLen);
        }
        const nameMatch = PARTIAL_NAME_RE.exec(this.pendingToolPayload);
        const detectedName = nameMatch?.[1] ?? 'tool';
        if (detectedName !== this.lastPendingToolName) {
          this.lastPendingToolName = detectedName;
          pendingToolCall = { name: detectedName };
        }
        break;
      }
    }

    // Bare or fenced JSON tool-call recovery on flush when no XML tool_call was parsed
    if (
      isFlush &&
      this.allowBareJsonToolCall &&
      this.parsedToolCalls.length === 0 &&
      this.fullVisibleText.trim().length > 0
    ) {
      const recovered = this.tryRecoverBareJsonToolCall(this.fullVisibleText);
      if (recovered) {
        this.parsedToolCalls.push(recovered.call);
        toolCallsReady.push(recovered.call);
        this.fullVisibleText = recovered.cleanText;
        textOut = '';
      }
    }

    return {
      textDelta: textOut,
      reasoningDelta: reasoningOut,
      pendingToolCall,
      toolCallsReady,
      stopSequenceHit,
      visibleTextReset,
    };
  }

  private tryRecoverBareJsonToolCall(
    text: string,
  ): { call: ParsedToolCall; cleanText: string } | null {
    // Look for fenced ```json { "name": ... } ``` or standalone JSON object with "name"
    const fencedRe = /```(?:json|tool_call)?\s*(\{[\s\S]*?\})\s*```/i;
    const fencedMatch = fencedRe.exec(text);
    if (fencedMatch && fencedMatch[1]) {
      this.toolCallCounter += 1;
      const call = extractToolCallFromPayload(
        fencedMatch[1],
        this.makeToolCallId(this.toolCallCounter),
      );
      if (call) {
        return {
          call,
          cleanText: text.replace(fencedMatch[0], '').trim(),
        };
      }
      this.toolCallCounter -= 1;
    }

    const trimmed = text.trim();
    const braceIdx = trimmed.indexOf('{');
    if (braceIdx !== -1 && PARTIAL_NAME_RE.test(trimmed.slice(braceIdx))) {
      this.toolCallCounter += 1;
      const call = extractToolCallFromPayload(
        trimmed.slice(braceIdx),
        this.makeToolCallId(this.toolCallCounter),
      );
      if (call) {
        return {
          call,
          cleanText: trimmed.slice(0, braceIdx).trim(),
        };
      }
      this.toolCallCounter -= 1;
    }

    return null;
  }
}
