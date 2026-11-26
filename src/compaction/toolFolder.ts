import type { HarnessMessage } from '../contracts.js';
import { estimateTextTokens } from '../utils/tokens.js';

export interface ToolFoldingOptions {
  /** Number of most recent user turns whose tool outputs stay verbatim (default: 1). */
  preserveRecentTurns?: number;
  /** Minimum character length before a stale tool result is folded (default: 300). */
  foldThresholdChars?: number;
  /** Maximum characters kept as semantic excerpt inside folded citation (default: 180). */
  excerptMaxChars?: number;
}

export interface ToolFoldingResult {
  messages: HarnessMessage[];
  foldedCount: number;
  savedTokensEstimate: number;
}

const TOOL_RESULT_TAG_RE = /^<tool_result\s+name="([^"]+)"[^>]*>([\s\S]*?)<\/tool_result>/i;
const URL_RE = /https?:\/\/[^\s)>\]"']+/gi;
const SPOTLIGHT_MARKER_RE = /^\[(?:BEGIN|END)\s+UNTRUSTED CONTENT\b[^\]]*\]$/i;
const SALIENCE_CODE_RE = /\b[A-Z]{2,}[_-][A-Z0-9_-]+\b/g;
const SALIENCE_NUM_RE = /(?:[$€£¥]?\d[\d.,:%-]*|\b\d{3,5}\b)/g;

function scoreToolLineSalience(line: string, index: number): number {
  let score = index === 0 ? 1.5 : 0;
  const codes = line.match(SALIENCE_CODE_RE)?.length ?? 0;
  const nums = line.match(SALIENCE_NUM_RE)?.length ?? 0;
  const urls = line.match(URL_RE)?.length ?? 0;
  score += Math.min(8, codes * 3 + nums * 1.4 + urls * 1.5);
  if (/\b(cookie|privacy policy|skip to content|all rights reserved|subscribe|sign in)\b/i.test(line)) {
    score -= 4;
  }
  if (line.length >= 20 && line.length <= 140) {
    score += 0.8;
  }
  return score;
}

function isToolPayloadMessage(msg: HarnessMessage): boolean {
  if (msg.role === 'tool') {
    return true;
  }
  if (msg.role === 'user' && TOOL_RESULT_TAG_RE.test(msg.content.trim())) {
    return true;
  }
  return false;
}

function extractToolNameAndBody(msg: HarnessMessage): { toolName: string; body: string } {
  if (msg.role === 'tool') {
    return {
      toolName: msg.toolName ?? 'tool',
      body: msg.content,
    };
  }
  const match = TOOL_RESULT_TAG_RE.exec(msg.content.trim());
  if (match) {
    return {
      toolName: match[1] ?? msg.toolName ?? 'tool',
      body: (match[2] ?? '').trim(),
    };
  }
  return {
    toolName: msg.toolName ?? 'tool',
    body: msg.content,
  };
}

function extractCitationSummary(toolName: string, body: string, excerptMaxChars: number): string {
  const urls = Array.from(new Set(body.match(URL_RE) ?? [])).slice(0, 3);
  const contentLines = body
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(
      (line) =>
        line.length > 0 &&
        !SPOTLIGHT_MARKER_RE.test(line) &&
        !/^URL:\s*https?:\/\//i.test(line) &&
        !line.startsWith('<') &&
        !line.startsWith('{'),
    );

  const scored = contentLines
    .map((line, idx) => ({ line, idx, score: scoreToolLineSalience(line, idx) }))
    .sort((a, b) => b.score - a.score || a.idx - b.idx);

  const picked: Array<{ line: string; idx: number }> = [];
  let used = 0;
  const budget = Math.max(80, excerptMaxChars);
  for (const item of scored) {
    const clean = item.line.replace(/\s+/g, ' ');
    const needed = clean.length + (picked.length > 0 ? 3 : 0);
    if (used + needed <= budget) {
      picked.push({ line: clean, idx: item.idx });
      used += needed;
      if (used >= budget - 24) break;
    } else if (picked.length === 0) {
      picked.push({ line: `${clean.slice(0, Math.max(0, budget - 3)).trimEnd()}...`, idx: item.idx });
      used = budget;
      break;
    }
  }

  picked.sort((a, b) => a.idx - b.idx);
  const fallback = body.replace(SPOTLIGHT_MARKER_RE, '').replace(/\s+/g, ' ').trim().slice(0, budget);
  const headline = picked.length > 0 ? picked.map((p) => p.line).join(' | ') : fallback;
  const originalTokens = estimateTextTokens(body);

  const parts = [
    `[Folded ${toolName} output | ~${originalTokens} tokens collapsed]`,
  ];
  if (urls.length > 0) {
    parts.push(`Sources: ${urls.join(', ')}`);
  }
  if (headline) {
    parts.push(`Summary: ${headline}`);
  }
  return parts.join(' ');
}

/**
 * Identifies turn indices (incrementing on genuine user prompts, not tool-result wrappers).
 */
function assignTurnIndices(messages: readonly HarnessMessage[]): number[] {
  const turnIndices: number[] = [];
  let currentTurn = -1;
  for (const msg of messages) {
    if (msg.role === 'user' && !TOOL_RESULT_TAG_RE.test(msg.content.trim())) {
      currentTurn += 1;
    }
    turnIndices.push(Math.max(0, currentTurn));
  }
  return turnIndices;
}

/**
 * Stage 2 Compaction: Transient Tool-Result Folding.
 * Collapses bulky intermediate raw DOM/search/document payloads from completed
 * earlier turns into concise semantic citations while preserving:
 * - all recent turns (`preserveRecentTurns`),
 * - any active/incomplete tool chain at the tail,
 * - all final user-facing assistant answers,
 * - `tainted: true` provenance flags.
 */
export function foldTransientToolResults(
  messages: readonly HarnessMessage[],
  options: ToolFoldingOptions = {},
): ToolFoldingResult {
  const preserveRecentTurns = Math.max(1, options.preserveRecentTurns ?? 1);
  const foldThresholdChars = Math.max(80, options.foldThresholdChars ?? 300);
  const excerptMaxChars = Math.max(40, options.excerptMaxChars ?? 180);

  if (messages.length === 0) {
    return { messages: [], foldedCount: 0, savedTokensEstimate: 0 };
  }

  const turnIndices = assignTurnIndices(messages);
  const maxTurn = turnIndices[turnIndices.length - 1] ?? 0;
  const minTurnToKeepVerbatim = Math.max(0, maxTurn - preserveRecentTurns + 1);

  // Check if a message has a subsequent final assistant message (non-tool-call) in its turn
  const hasFinalAnswerAfter = new Array<boolean>(messages.length).fill(false);
  let seenFinalAssistant = false;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    hasFinalAnswerAfter[i] = seenFinalAssistant;
    const m = messages[i]!;
    if (m.role === 'assistant' && !m.toolName && !m.content.includes('<tool_call>')) {
      seenFinalAssistant = true;
    }
  }

  let foldedCount = 0;
  let savedTokensEstimate = 0;

  const result = messages.map((msg, idx) => {
    const turnIdx = turnIndices[idx] ?? 0;
    if (turnIdx >= minTurnToKeepVerbatim) {
      return msg;
    }
    if (!hasFinalAnswerAfter[idx]) {
      return msg;
    }
    if (!isToolPayloadMessage(msg)) {
      return msg;
    }
    if (msg.content.startsWith('[Folded ')) {
      return msg;
    }

    const { toolName, body } = extractToolNameAndBody(msg);
    if (body.length < foldThresholdChars) {
      return msg;
    }

    const citation = extractCitationSummary(toolName, body, excerptMaxChars);
    const beforeTokens = estimateTextTokens(msg.content);
    const afterTokens = estimateTextTokens(citation);
    if (afterTokens >= beforeTokens) {
      return msg;
    }

    foldedCount += 1;
    savedTokensEstimate += beforeTokens - afterTokens;

    return {
      ...msg,
      content: citation,
      // Preserve taint flag so @yours/guard knows untrusted content entered the thread
      tainted: msg.tainted,
    };
  });

  return {
    messages: result,
    foldedCount,
    savedTokensEstimate,
  };
}
