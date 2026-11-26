import type { HarnessMessage, ToolDefinition } from '../contracts.js';

/**
 * Deterministic 32-bit FNV-1a hex hash for prefix fingerprints, slot keys,
 * and tab content deduplication.
 */
export function hashStringFnv1a(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Fast, deterministic heuristic token estimator (~3.6 chars/token + framing).
 * Used when a runtime adapter does not expose a native BPE tokenizer.
 */
export function estimateTextTokens(text: string): number {
  if (!text) {
    return 0;
  }
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }
  const asciiAndWordBoost = Math.ceil(trimmed.length / 3.6);
  const punctuationMatches = trimmed.match(/[{}\[\]()<>"':;,./\\=+-]/g);
  const punctuationExtra = punctuationMatches ? Math.floor(punctuationMatches.length * 0.2) : 0;
  return Math.max(1, asciiAndWordBoost + punctuationExtra);
}

/**
 * Estimates total tokens across an array of HarnessMessage objects,
 * accounting for role headers and optional tool metadata.
 */
export function estimateMessagesTokens(messages: readonly HarnessMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    total += 4; // <|im_start|>role\n ... <|im_end|>\n
    total += estimateTextTokens(msg.rawContent ?? msg.content);
    if (msg.toolName) {
      total += estimateTextTokens(msg.toolName) + 2;
    }
    if (msg.toolArgs) {
      total += estimateTextTokens(JSON.stringify(msg.toolArgs));
    }
  }
  return total;
}

/**
 * Estimates tokens consumed by serialized tool schemas.
 */
export function estimateToolsTokens(tools?: readonly ToolDefinition[]): number {
  if (!tools || tools.length === 0) {
    return 0;
  }
  let total = 16;
  for (const tool of tools) {
    total += estimateTextTokens(tool.name);
    total += estimateTextTokens(tool.description);
    total += estimateTextTokens(JSON.stringify(tool.parameters));
  }
  return total;
}

/**
 * Computes a deterministic fingerprint over an ordered list of messages.
 */
export function hashMessagesPrefix(messages: readonly HarnessMessage[]): string {
  const serialized = messages
    .map((m) => `${m.role}:${m.toolName ?? ''}:${m.rawContent ?? m.content}`)
    .join('\n\x1e\n');
  return hashStringFnv1a(serialized);
}
