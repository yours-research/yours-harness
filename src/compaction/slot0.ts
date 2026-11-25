import type { HarnessMessage, MemoryRecord, ToolDefinition } from '../contracts.js';
import { estimateMessagesTokens, hashStringFnv1a } from '../utils/tokens.js';

export interface Slot0Options {
  systemPrompt?: string;
  tools?: readonly ToolDefinition[];
  profileMemories?: readonly (Pick<MemoryRecord, 'id' | 'content'> | string)[];
  countTokens?: (messages: HarnessMessage[]) => Promise<number> | number;
}

export interface Slot0Envelope {
  messages: HarnessMessage[];
  prefixHash: string;
  slotKey: string;
  tokenCount: number;
  hasProfileMemory: boolean;
  toolCount: number;
}

export const DEFAULT_BASE_SYSTEM_PROMPT = [
  'You are a fast, accurate local AI assistant.',
  'Be direct and concise. Never fabricate tool output or cite unseen sources.',
  'When calling a tool, emit a single <tool_call>{"name":"tool_name","arguments":{...}}</tool_call> block and wait for the result.',
].join('\n');

function formatToolSignature(tool: ToolDefinition): string {
  const props = tool.parameters?.properties ?? {};
  const required = new Set(tool.parameters?.required ?? []);
  const paramEntries = Object.keys(props)
    .sort()
    .map((key) => {
      const schema = props[key] as { type?: string; enum?: unknown[] } | undefined;
      const typeLabel = schema?.enum
        ? schema.enum.map((v) => JSON.stringify(v)).join(' | ')
        : schema?.type ?? 'unknown';
      const opt = required.has(key) ? '' : '?';
      return `${key}${opt}: ${typeLabel}`;
    });
  return `- ${tool.name}(${paramEntries.join(', ')}): ${tool.description.trim()}`;
}

function normalizeProfileMemories(
  memories?: readonly (Pick<MemoryRecord, 'id' | 'content'> | string)[],
): string[] {
  if (!memories || memories.length === 0) {
    return [];
  }
  const normalized = memories
    .map((entry) =>
      typeof entry === 'string'
        ? { sortKey: entry, content: entry.trim() }
        : { sortKey: `${entry.id}:${entry.content}`, content: entry.content.trim() },
    )
    .filter((entry) => entry.content.length > 0);

  normalized.sort((a, b) => a.sortKey.localeCompare(b.sortKey));
  return normalized.map((entry) => `- ${entry.content}`);
}

/**
 * Builds the immutable pinned Slot 0 prefix containing ONLY deterministic
 * System Prompt + canonically sorted Tool Definitions + sorted Profile Memory.
 * Never embeds dynamic timestamps or random IDs so GPU KV-cache prefix stays hot.
 */
export async function buildSlot0Envelope(options: Slot0Options = {}): Promise<Slot0Envelope> {
  const basePrompt = (options.systemPrompt ?? DEFAULT_BASE_SYSTEM_PROMPT).trim();
  const sortedTools = [...(options.tools ?? [])].sort((a, b) => a.name.localeCompare(b.name));
  const profileLines = normalizeProfileMemories(options.profileMemories);

  const sections: string[] = [basePrompt];

  if (sortedTools.length > 0) {
    sections.push(['Available tools:', ...sortedTools.map(formatToolSignature)].join('\n'));
  }

  if (profileLines.length > 0) {
    sections.push(['[Pinned User Profile Memory]', ...profileLines].join('\n'));
  }

  const content = sections.join('\n\n');
  const prefixHash = hashStringFnv1a(content);
  const slotKey = `base-system:${prefixHash}`;

  const messages: HarnessMessage[] = [
    {
      id: `slot0_${prefixHash}`,
      role: 'system',
      content,
      createdAt: 0,
    },
  ];

  const tokenCount = options.countTokens
    ? await options.countTokens(messages)
    : estimateMessagesTokens(messages);

  return {
    messages,
    prefixHash,
    slotKey,
    tokenCount,
    hasProfileMemory: profileLines.length > 0,
    toolCount: sortedTools.length,
  };
}
