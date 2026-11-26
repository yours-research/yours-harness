import { describe, expect, it } from 'vitest';
import {
  buildSlot0Envelope,
  ContextCompactor,
  foldTransientToolResults,
  measureKvPrefixOverlap,
} from '../src/compaction/index.js';
import { buildSlot0Envelope as buildSlot0FromRoot } from '../src/index.js';
import type { HarnessMessage, ToolDefinition } from '../src/contracts.js';

const SAMPLE_TOOLS: ToolDefinition[] = [
  {
    name: 'read_active_tab',
    description: 'Read the active browser tab.',
    parameters: {
      type: 'object',
      properties: {
        mode: { type: 'string', enum: ['auto', 'article'] },
      },
    },
    safety: {
      readOnlyHint: true,
      consequentialHint: false,
      networkHint: false,
      untrustedContentHint: true,
    },
  },
  {
    name: 'search_web_tab',
    description: 'Search the web.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string' },
      },
      required: ['query'],
    },
    safety: {
      readOnlyHint: true,
      consequentialHint: false,
      networkHint: true,
      untrustedContentHint: true,
    },
  },
];

function msg(
  id: string,
  role: HarnessMessage['role'],
  content: string,
  extra: Partial<HarnessMessage> = {},
): HarnessMessage {
  return { id, role, content, createdAt: 1000, ...extra };
}

describe('KV-Cache-Preserving Context Compaction', () => {
  it('keeps Slot 0 deterministic and byte-identical regardless of tool/memory ordering', async () => {
    const a = await buildSlot0Envelope({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
      profileMemories: [
        { id: 'm2', content: 'Prefers TypeScript' },
        { id: 'm1', content: 'Lives in Seattle' },
      ],
    });

    const b = await buildSlot0Envelope({
      systemPrompt: 'You are Pragma.',
      tools: [...SAMPLE_TOOLS].reverse(),
      profileMemories: [
        { id: 'm1', content: 'Lives in Seattle' },
        { id: 'm2', content: 'Prefers TypeScript' },
      ],
    });

    expect(a.prefixHash).toBe(b.prefixHash);
    expect(a.slotKey).toBe(b.slotKey);
    expect(a.messages[0]?.content).toBe(b.messages[0]?.content);
    expect(a.hasProfileMemory).toBe(true);
    expect(a.toolCount).toBe(2);
  });

  it('folds stale intermediate tool payloads while preserving recent turn & taint flags', () => {
    const longDomPayload = [
      'Documentation page https://yours.sh/docs/wgsl-runtime',
      ...Array.from({ length: 20 }, (_, i) => `Paragraph ${i}: WebGPU subgroup GEMV kernel details.`),
    ].join('\n');

    const messages: HarnessMessage[] = [
      msg('u1', 'user', 'Check the docs on WebGPU runtime'),
      msg('a1_tc', 'assistant', '', { toolName: 'read_active_tab', toolCallId: 'tc1' }),
      msg('t1', 'tool', longDomPayload, { toolName: 'read_active_tab', tainted: true }),
      msg('a1_final', 'assistant', 'The WGSL runtime uses subgroup GEMV kernels.'),
      msg('u2', 'user', 'Now read the second page'),
      msg('a2_tc', 'assistant', '', { toolName: 'read_active_tab', toolCallId: 'tc2' }),
      msg('t2', 'tool', longDomPayload, { toolName: 'read_active_tab', tainted: true }),
    ];

    const folded = foldTransientToolResults(messages, {
      preserveRecentTurns: 1,
      foldThresholdChars: 200,
    });

    expect(folded.foldedCount).toBe(1);
    expect(folded.savedTokensEstimate).toBeGreaterThan(50);

    // Turn 1 tool output is folded into a citation and keeps tainted: true
    expect(folded.messages[2]?.content).toContain('[Folded read_active_tab output');
    expect(folded.messages[2]?.content).toContain('https://yours.sh/docs/wgsl-runtime');
    expect(folded.messages[2]?.tainted).toBe(true);

    // Turn 1 final assistant answer is preserved verbatim
    expect(folded.messages[3]?.content).toBe('The WGSL runtime uses subgroup GEMV kernels.');

    // Most recent turn's tool output stays verbatim
    expect(folded.messages[6]?.content).toBe(longDomPayload);
  });

  it('enforces token budget with rolling summary and preserves KV prefix locality across turns', async () => {
    const compactor = new ContextCompactor({
      maxContextTokens: 800,
      reservedOutputTokens: 200,
      safetyMarginTokens: 50,
      triggerRatio: 0.85,
      preserveRecentMessages: 2,
      summaryMaxTokens: 120,
    });

    const history: HarnessMessage[] = [];
    for (let i = 0; i < 10; i += 1) {
      history.push(
        msg(`u_${i}`, 'user', `Question ${i}: Explain WebGPU storage buffer alignment rule #${i} in detail.`),
      );
      history.push(
        msg(
          `a_${i}`,
          'assistant',
          `Answer ${i}: WebGPU storage buffers require 16-byte struct alignment and proper workgroup sizing for rule #${i}.`,
        ),
      );
    }

    const plan1 = await compactor.compact({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
      messages: history,
    });

    expect(plan1.didCompact).toBe(true);
    expect(plan1.evictedMessageCount).toBeGreaterThan(0);
    expect(plan1.rollingSummary).toBeTruthy();
    expect(plan1.tokensAfter).toBeLessThanOrEqual(800 - 200 - 50);
    expect(plan1.compiledMessages[0]?.id).toBe(plan1.slot0.messages[0]?.id);

    // Append one more turn after compaction: Slot 0 + Rolling Summary + kept messages stay prefixed!
    const nextHistory = [
      ...history,
      msg('u_10', 'user', 'What about subgroup shuffle?'),
    ];

    const plan2 = await compactor.compact({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
      messages: nextHistory,
      previousRollingSummary: plan1.rollingSummary,
      previousRollingSummaryCutoffId: plan1.rollingSummaryCutoffId,
      previousCompiledMessages: plan1.compiledMessages,
    });

    expect(plan2.slot0Preserved).toBe(true);
    expect(plan2.slot0.prefixHash).toBe(plan1.slot0.prefixHash);
    expect(plan2.kvPrefixPreservedMessages).toBeGreaterThanOrEqual(2);
    expect(plan2.kvPrefixPreservedTokens).toBeGreaterThan(0);

    const overlap = measureKvPrefixOverlap(plan1.compiledMessages, plan2.compiledMessages);
    expect(overlap.preservedMessages).toBe(plan2.kvPrefixPreservedMessages);
  });

  it('places deduplicated contextMessages immediately after Slot 0 and accounts for their tokens', async () => {
    expect(buildSlot0FromRoot).toBe(buildSlot0Envelope);

    const compactor = new ContextCompactor({
      maxContextTokens: 650,
      reservedOutputTokens: 150,
      safetyMarginTokens: 50,
      triggerRatio: 0.85,
      preserveRecentMessages: 2,
      summaryMaxTokens: 100,
    });

    const slot0 = await buildSlot0Envelope({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
    });

    const contextMsg1 = msg('ctx_page', 'system', `Active Page Context: ${'WebGPU pipeline '.repeat(12)}`);
    const contextMsgDupInHistory = msg('ctx_dup', 'system', 'Duplicate in history');
    const slot0Dup = msg(slot0.messages[0]!.id, 'system', slot0.messages[0]!.content);

    const history: HarnessMessage[] = [
      contextMsgDupInHistory,
      ...Array.from({ length: 6 }, (_, i) => [
        msg(`u_${i}`, 'user', `Turn ${i}: How does subgroup barrier work in pass ${i}?`),
        msg(`a_${i}`, 'assistant', `Turn ${i}: Workgroup barriers synchronize invocations cleanly.`),
      ]).flat(),
    ];

    const plan1 = await compactor.compact({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
      contextMessages: [slot0Dup, contextMsg1, contextMsg1, contextMsgDupInHistory],
      messages: history,
    });

    // Slot 0 remains at index 0, deduped contextMsg1 immediately follows at index 1
    expect(plan1.compiledMessages[0]?.id).toBe(slot0.messages[0]?.id);
    expect(plan1.compiledMessages[1]?.id).toBe('ctx_page');
    // Duplicates (slot0Dup, second contextMsg1, and contextMsgDupInHistory) were filtered out from prefix
    expect(plan1.compiledMessages.filter((m) => m.id === 'ctx_page')).toHaveLength(1);
    expect(plan1.compiledMessages.filter((m) => m.id === 'ctx_dup').length).toBeLessThanOrEqual(1);
    // Compaction enforced total budget including contextMessages
    expect(plan1.tokensAfter).toBeLessThanOrEqual(650 - 150 - 50);

    // Next turn with unchanged contextMessages preserves both Slot 0 and contextMsg1 in KV prefix
    const plan2 = await compactor.compact({
      systemPrompt: 'You are Pragma.',
      tools: SAMPLE_TOOLS,
      contextMessages: [contextMsg1],
      messages: [...history, msg('u_next', 'user', 'Follow-up question')],
      previousRollingSummary: plan1.rollingSummary,
      previousRollingSummaryCutoffId: plan1.rollingSummaryCutoffId,
      previousCompiledMessages: plan1.compiledMessages,
    });

    expect(plan2.slot0Preserved).toBe(true);
    expect(plan2.compiledMessages[1]?.id).toBe('ctx_page');
    expect(plan2.kvPrefixPreservedMessages).toBeGreaterThanOrEqual(3); // Slot 0 + contextMsg1 + rolling summary
  });

  it('retains early user constraints and concrete tool facts when evicting heavy trailing status chatter', async () => {
    const compactor = new ContextCompactor({
      maxContextTokens: 512,
      reservedOutputTokens: 200,
      safetyMarginTokens: 0,
      triggerRatio: 0.85,
      preserveRecentMessages: 2,
      summaryMaxTokens: 96,
    });

    const history: HarnessMessage[] = [
      msg('u0', 'user', 'Find a USB4 monitor under $350 shipping to 94107.'),
      msg('t0', 'tool', 'Found SKU-9921-BLK at $329 in stock for zip 94107.', { toolName: 'read_tab' }),
      msg('a0', 'assistant', 'Selected SKU-9921-BLK ($329) shipping to 94107.'),
      ...Array.from({ length: 24 }, (_, i) =>
        msg(
          `noise_${i}`,
          i % 2 === 0 ? 'assistant' : 'user',
          `Checking intermediate status step ${i + 1}: waiting for background worker panel synchronization telemetry pass.`,
        ),
      ),
      msg('u_last', 'user', 'What SKU, price, and zip did we confirm earlier?'),
    ];

    const plan = await compactor.compact({
      systemPrompt: 'You are a concise assistant.',
      messages: history,
    });

    expect(plan.didCompact).toBe(true);
    expect(plan.rollingSummary).toContain('SKU-9921-BLK');
    expect(plan.rollingSummary).toContain('$329');
    expect(plan.rollingSummary).toContain('94107');
    expect(plan.tokensAfter).toBeLessThanOrEqual(312);
  });

  it('preserves high-salience buried identifiers when folding bulky spotlighted tool outputs', () => {
    const bulkyPayload = [
      '[BEGIN UNTRUSTED CONTENT id="ug-991a" origin="https://example.org" tainted="true" risk="0.00"]',
      '[ARTICLE] Cloud Infrastructure Overview',
      'URL: https://example.org/docs/cluster-guide',
      ...Array.from({ length: 14 }, (_, i) => `Introductory boilerplate paragraph ${i + 1} describing general cloud networking concepts.`),
      'Critical specification: Cluster code BUDGET-CAP-991 with SLA $7,250 and port 8443.',
      '[END UNTRUSTED CONTENT id="ug-991a"]',
    ].join('\n');

    const messages: HarnessMessage[] = [
      msg('u1', 'user', 'Read the cluster guide page'),
      msg('t1', 'tool', bulkyPayload, { toolName: 'read_tab', tainted: true }),
      msg('a1', 'assistant', 'Noted the cluster guide specification.'),
      msg('u2', 'user', 'Follow-up turn'),
    ];

    const folded = foldTransientToolResults(messages, {
      preserveRecentTurns: 1,
      foldThresholdChars: 360,
    });

    expect(folded.foldedCount).toBe(1);
    const citation = folded.messages[1]?.content ?? '';
    expect(citation).toContain('[Folded read_tab output');
    expect(citation).not.toContain('Summary: [BEGIN UNTRUSTED');
    expect(citation).toContain('BUDGET-CAP-991');
    expect(citation).toContain('8443');
    expect(folded.messages[1]?.tainted).toBe(true);
  });
});
