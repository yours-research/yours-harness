import type { HarnessMessage, MemoryRecord, ToolDefinition } from '../contracts.js';
import { estimateMessagesTokens, estimateTextTokens, hashMessagesPrefix } from '../utils/tokens.js';
import { buildSlot0Envelope, type Slot0Envelope } from './slot0.js';
import { foldTransientToolResults, type ToolFoldingOptions } from './toolFolder.js';

export type CompactionSummarizer = (
  evictedMessages: readonly HarnessMessage[],
  previousSummary: string | null,
  signal?: AbortSignal,
) => Promise<string>;

export interface CompactionBudgetConfig {
  /** Maximum context window supported by the active runtime/model. */
  maxContextTokens: number;
  /** Headroom reserved for reasoning + output tokens (default: 1024). */
  reservedOutputTokens?: number;
  /** Additional safety margin for BPE estimation drift (default: 128). */
  safetyMarginTokens?: number;
  /** Trigger rolling summary when history exceeds this ratio of available budget (default: 0.92). */
  triggerRatio?: number;
  /** Number of recent non-system messages to keep verbatim when compacting (default: 4). */
  preserveRecentMessages?: number;
  /** Target max tokens for rolling summary block (default: 320). */
  summaryMaxTokens?: number;
  /** Tool folding tuning options. */
  toolFolding?: ToolFoldingOptions;
  /** Optional LLM or custom summarizer; falls back to deterministic extractive summary. */
  summarizer?: CompactionSummarizer;
  /** Optional exact tokenizer counter. */
  countTokens?: (messages: HarnessMessage[]) => Promise<number> | number;
}

export interface CompactionInput {
  systemPrompt?: string;
  tools?: readonly ToolDefinition[];
  profileMemories?: readonly (Pick<MemoryRecord, 'id' | 'content'> | string)[];
  /** Optional contextual messages (e.g. active page/tab context) placed immediately after Slot 0. */
  contextMessages?: readonly HarnessMessage[];
  /** Conversation history excluding Slot 0 system message (though stray system msgs are filtered). */
  messages: readonly HarnessMessage[];
  /** Optional previous rolling summary state to extend without re-summarizing old turns. */
  previousRollingSummary?: string | null;
  /** ID of the last message already incorporated into `previousRollingSummary`. */
  previousRollingSummaryCutoffId?: string | null;
  /** Optional compiled prompt from the prior turn to compute KV prefix locality metrics. */
  previousCompiledMessages?: readonly HarnessMessage[];
  signal?: AbortSignal;
}

export interface CompactionPlan {
  slot0: Slot0Envelope;
  compiledMessages: HarnessMessage[];
  rollingSummary: string | null;
  rollingSummaryCutoffId: string | null;
  summaryPrefixHash: string | null;
  tokensBefore: number;
  tokensAfter: number;
  availableHistoryBudget: number;
  reservedOutputTokens: number;
  foldedToolCount: number;
  evictedMessageCount: number;
  kvPrefixPreservedMessages: number;
  kvPrefixPreservedTokens: number;
  slot0Preserved: boolean;
  didCompact: boolean;
}

const ROLLING_SUMMARY_PREFIX = '[Rolling Conversation Summary]\n';

function isRollingSummaryMessage(msg: HarnessMessage): boolean {
  return msg.role === 'system' && msg.content.startsWith(ROLLING_SUMMARY_PREFIX);
}

interface SummaryCandidate {
  order: number;
  text: string;
  skeleton: string;
  fromPrevious: boolean;
  score: number;
}

const ENTITY_CODE_RE = /\b[A-Z]{2,}[_-][A-Z0-9_-]+\b/g;
const HEX_HASH_RE = /\b[0-9a-f]{7,40}\b/gi;
const NUMBER_OR_PRICE_RE = /(?:[$€£¥]?\d[\d.,:%-]*|\bzip\s+\d{5}\b)/gi;
const URL_TOKEN_RE = /https?:\/\/[^\s)>\]]+/gi;
const FILLER_CHATTER_RE =
  /\b(intermediate\s+status|inspected\s+panel|checking\s+step|acknowledged|working\s+on\s+it|continuing\s+analysis)\b/i;

function computeSummarySalience(
  msg: HarnessMessage | null,
  line: string,
  order: number,
  fromPrevious: boolean,
): number {
  let score = fromPrevious ? 3.2 : 0;
  if (msg) {
    if (msg.role === 'tool') {
      score += line.includes('[Folded ') ? 1.5 : 4.0;
    } else if (msg.role === 'user') {
      score += 3.2;
    } else {
      score += 1.8;
    }
  }
  if (order <= 1) {
    score += 3.5;
  }

  const codes = line.match(ENTITY_CODE_RE)?.length ?? 0;
  const hashes = line.match(HEX_HASH_RE)?.length ?? 0;
  const nums = line.match(NUMBER_OR_PRICE_RE)?.length ?? 0;
  const urls = line.match(URL_TOKEN_RE)?.length ?? 0;
  const hyphens = line.match(/\b[a-z]{2,}-[a-z0-9]{2,}\b/gi)?.length ?? 0;

  score += Math.min(9, codes * 2.5 + hashes * 2.0 + nums * 1.1 + urls * 1.5 + hyphens * 1.0);

  if (FILLER_CHATTER_RE.test(line)) {
    score -= 4.5;
  }
  return score;
}

function bulletSkeleton(text: string): string {
  return text.toLowerCase().replace(/\d+/g, '#').replace(/\s+/g, ' ').trim();
}

/**
 * Deterministic extractive summarizer requiring zero extra LLM calls.
 * Prioritizes high-salience constraints, identifiers, and tool findings over
 * repetitive status chatter while fitting cleanly inside `maxTokens`.
 */
export function extractiveSummarize(
  evictedMessages: readonly HarnessMessage[],
  previousSummary: string | null,
  maxTokens = 320,
): string {
  const maxChars = Math.max(160, Math.floor(maxTokens * 3.4));
  const candidates: SummaryCandidate[] = [];

  if (previousSummary && previousSummary.trim()) {
    for (const rawLine of previousSummary.split(/\r?\n/)) {
      const trimmed = rawLine.trim();
      if (!trimmed) continue;
      const line = trimmed.startsWith('- ') ? trimmed : `- ${trimmed}`;
      const order = candidates.length;
      candidates.push({
        order,
        text: line,
        skeleton: bulletSkeleton(line),
        fromPrevious: true,
        score: computeSummarySalience(null, line, order, true),
      });
    }
  }

  for (const msg of evictedMessages) {
    if (msg.role === 'system') {
      continue;
    }
    const clean = msg.content.replace(/\s+/g, ' ').trim();
    if (!clean) {
      continue;
    }
    const label =
      msg.role === 'user'
        ? 'User'
        : msg.role === 'assistant'
          ? 'Assistant'
          : `Tool(${msg.toolName ?? 'tool'})`;
    const snippet = clean.length > 160 ? `${clean.slice(0, 157)}...` : clean;
    const line = `- ${label}: ${snippet}`;
    const order = candidates.length;
    candidates.push({
      order,
      text: line,
      skeleton: bulletSkeleton(line),
      fromPrevious: false,
      score: computeSummarySalience(msg, line, order, false),
    });
  }

  if (candidates.length === 0) {
    return '';
  }

  const skeletonCounts = new Map<string, number>();
  for (const c of candidates) {
    skeletonCounts.set(c.skeleton, (skeletonCounts.get(c.skeleton) ?? 0) + 1);
  }

  // Down-rank and deduplicate repetitive template lines while keeping one exemplar if space allows.
  const seenSkeletons = new Set<string>();
  const ranked = candidates
    .map((c) => {
      const count = skeletonCounts.get(c.skeleton) ?? 1;
      const isRepeatedTemplate = count > 1;
      const penalizedScore = isRepeatedTemplate ? c.score - Math.min(6, count * 1.5) : c.score;
      return { ...c, score: penalizedScore, isRepeatedTemplate };
    })
    .filter((c) => {
      if (!c.isRepeatedTemplate) return true;
      if (seenSkeletons.has(c.skeleton)) return false;
      seenSkeletons.add(c.skeleton);
      return true;
    })
    .sort((a, b) => b.score - a.score || a.order - b.order);

  const selected: SummaryCandidate[] = [];
  let usedChars = 0;

  for (const cand of ranked) {
    const lineCost = cand.text.length + (selected.length > 0 ? 1 : 0);
    if (usedChars + lineCost <= maxChars) {
      selected.push(cand);
      usedChars += lineCost;
      continue;
    }
    const remaining = maxChars - usedChars - (selected.length > 0 ? 1 : 0);
    if (remaining >= 56 && cand.score >= 3) {
      const clipped = `${cand.text.slice(0, remaining - 3).trimEnd()}...`;
      selected.push({ ...cand, text: clipped });
      usedChars += clipped.length + (selected.length > 1 ? 1 : 0);
    }
  }

  if (selected.length === 0 && ranked[0]) {
    const first = ranked[0].text;
    return first.length > maxChars ? `${first.slice(0, Math.max(0, maxChars - 3))}...` : first;
  }

  selected.sort((a, b) => a.order - b.order);
  return selected.map((c) => c.text).join('\n');
}

/**
 * Counts how many leading messages (and tokens) are byte-for-byte identical
 * between the previous turn's compiled prompt and the new compiled prompt.
 */
export function measureKvPrefixOverlap(
  previousMessages: readonly HarnessMessage[] | undefined,
  nextMessages: readonly HarnessMessage[],
): { preservedMessages: number; preservedTokens: number } {
  if (!previousMessages || previousMessages.length === 0) {
    return { preservedMessages: 0, preservedTokens: 0 };
  }
  const limit = Math.min(previousMessages.length, nextMessages.length);
  let matchedCount = 0;
  for (let i = 0; i < limit; i += 1) {
    const prev = previousMessages[i]!;
    const next = nextMessages[i]!;
    if (
      prev.role === next.role &&
      prev.content === next.content &&
      (prev.rawContent ?? prev.content) === (next.rawContent ?? next.content) &&
      prev.toolName === next.toolName
    ) {
      matchedCount += 1;
    } else {
      break;
    }
  }
  if (matchedCount === 0) {
    return { preservedMessages: 0, preservedTokens: 0 };
  }
  const preservedTokens = estimateMessagesTokens(nextMessages.slice(0, matchedCount));
  return {
    preservedMessages: matchedCount,
    preservedTokens,
  };
}

/**
 * 3-Stage KV-Cache–Preserving Context Compactor for fixed-VRAM local models:
 *   1. Immutable Pinned Slot 0 (System prompt + tools + profile memory)
 *   2. Transient Tool-Result Folding (collapses stale DOM/search payloads)
 *   3. Block-Aligned Rolling Summary + Deterministic Budget Enforcer
 */
export class ContextCompactor {
  private readonly config: Required<
    Pick<
      CompactionBudgetConfig,
      | 'maxContextTokens'
      | 'reservedOutputTokens'
      | 'safetyMarginTokens'
      | 'triggerRatio'
      | 'preserveRecentMessages'
      | 'summaryMaxTokens'
    >
  > &
    Pick<CompactionBudgetConfig, 'toolFolding' | 'summarizer' | 'countTokens'>;

  constructor(config: CompactionBudgetConfig) {
    this.config = {
      maxContextTokens: Math.max(512, config.maxContextTokens),
      reservedOutputTokens: Math.max(64, config.reservedOutputTokens ?? 1024),
      safetyMarginTokens: Math.max(0, config.safetyMarginTokens ?? 128),
      triggerRatio: Math.min(1, Math.max(0.5, config.triggerRatio ?? 0.92)),
      preserveRecentMessages: Math.max(1, config.preserveRecentMessages ?? 4),
      summaryMaxTokens: Math.max(64, config.summaryMaxTokens ?? 320),
      toolFolding: config.toolFolding,
      summarizer: config.summarizer,
      countTokens: config.countTokens,
    };
  }

  private async countTokens(messages: HarnessMessage[]): Promise<number> {
    if (this.config.countTokens) {
      return this.config.countTokens(messages);
    }
    return estimateMessagesTokens(messages);
  }

  async compact(input: CompactionInput): Promise<CompactionPlan> {
    // Stage 1: Build immutable Slot 0
    const slot0 = await buildSlot0Envelope({
      systemPrompt: input.systemPrompt,
      tools: input.tools,
      profileMemories: input.profileMemories,
      countTokens: this.config.countTokens,
    });

    // Separate out any pre-existing Slot 0 or rolling summary messages
    let rollingSummary: string | null = input.previousRollingSummary ?? null;
    let rollingSummaryCutoffId: string | null = input.previousRollingSummaryCutoffId ?? null;
    const rawHistory: HarnessMessage[] = [];
    for (const msg of input.messages) {
      if (msg.id.startsWith('slot0_') || (msg.role === 'system' && msg.content === slot0.messages[0]?.content)) {
        continue;
      }
      if (isRollingSummaryMessage(msg)) {
        if (!rollingSummary) {
          rollingSummary = msg.content.slice(ROLLING_SUMMARY_PREFIX.length).trim();
        }
        continue;
      }
      rawHistory.push(msg);
    }

    // Deduplicate contextMessages against Slot 0 and rawHistory, keeping them immediately after Slot 0
    const excludedIds = new Set<string>();
    const excludedSignatures = new Set<string>();
    const markExcluded = (msg: HarnessMessage) => {
      excludedIds.add(msg.id);
      excludedSignatures.add(`${msg.role}\x00${msg.toolName ?? ''}\x00${msg.content}`);
    };
    for (const msg of slot0.messages) markExcluded(msg);
    for (const msg of rawHistory) markExcluded(msg);

    const dedupedContextMessages: HarnessMessage[] = [];
    for (const msg of input.contextMessages ?? []) {
      const sig = `${msg.role}\x00${msg.toolName ?? ''}\x00${msg.content}`;
      if (
        msg.id.startsWith('slot0_') ||
        isRollingSummaryMessage(msg) ||
        excludedIds.has(msg.id) ||
        excludedSignatures.has(sig)
      ) {
        continue;
      }
      markExcluded(msg);
      dedupedContextMessages.push(msg);
    }

    const basePrefixMessages: HarnessMessage[] = [...slot0.messages, ...dedupedContextMessages];
    const basePrefixTokens =
      dedupedContextMessages.length > 0
        ? await this.countTokens(basePrefixMessages)
        : slot0.tokenCount;

    const availableHistoryBudget = Math.max(
      128,
      this.config.maxContextTokens -
        basePrefixTokens -
        this.config.reservedOutputTokens -
        this.config.safetyMarginTokens,
    );

    const rawTokens = await this.countTokens([...basePrefixMessages, ...rawHistory]);

    // Skip messages already folded into previousRollingSummary so we never double-summarize
    // and so [Slot 0, Context, Rolling Summary, ...kept] preserves KV prefix locality on subsequent turns.
    let activeWindow = rawHistory;
    if (rollingSummary && rollingSummaryCutoffId) {
      const cutoffIdx = rawHistory.findIndex((m) => m.id === rollingSummaryCutoffId);
      if (cutoffIdx !== -1) {
        activeWindow = rawHistory.slice(cutoffIdx + 1);
      }
    }

    // Stage 2: Transient Tool-Result Folding on older completed turns
    const folded = foldTransientToolResults(activeWindow, this.config.toolFolding);
    let workingHistory = folded.messages;

    const makeSummaryMessage = (summaryText: string): HarnessMessage => ({
      id: `rolling_summary_${hashMessagesPrefix([{ id: 's', role: 'system', content: summaryText, createdAt: 0 }])}`,
      role: 'system',
      content: `${ROLLING_SUMMARY_PREFIX}${summaryText}`,
      createdAt: 0,
    });

    let currentPrefix: HarnessMessage[] = [...basePrefixMessages];
    if (rollingSummary) {
      currentPrefix.push(makeSummaryMessage(rollingSummary));
    }

    let currentTokens = await this.countTokens([...currentPrefix, ...workingHistory]);
    const triggerLimit =
      basePrefixTokens + Math.floor(availableHistoryBudget * this.config.triggerRatio);

    let evictedMessageCount = 0;

    // Stage 3: Block-Aligned Rolling Summary if over budget or trigger threshold
    if (currentTokens > triggerLimit || currentTokens > basePrefixTokens + availableHistoryBudget) {
      const targetHistoryTokens = Math.floor(availableHistoryBudget * 0.72);
      const minKeep = Math.min(this.config.preserveRecentMessages, workingHistory.length);
      let splitIndex = workingHistory.length - minKeep;

      // Ensure we evict enough to fit under targetHistoryTokens (block-aligned eviction)
      while (splitIndex > 0) {
        const candidateKept = workingHistory.slice(splitIndex);
        const summaryEstimate = this.config.summaryMaxTokens + 16;
        const keptTokens = await this.countTokens(candidateKept);
        if (keptTokens + summaryEstimate <= targetHistoryTokens) {
          break;
        }
        splitIndex -= 1;
      }

      // If keeping minKeep still overflows hard budget, advance splitIndex forward down to 1 message
      if (splitIndex < workingHistory.length - 1) {
        const candidateKept = workingHistory.slice(splitIndex);
        const keptTokens = await this.countTokens(candidateKept);
        if (keptTokens + this.config.summaryMaxTokens + 16 > availableHistoryBudget) {
          let forwardIdx = splitIndex;
          while (forwardIdx < workingHistory.length - 1) {
            const tighterKept = workingHistory.slice(forwardIdx);
            const tighterTokens = await this.countTokens(tighterKept);
            if (tighterTokens + this.config.summaryMaxTokens + 16 <= availableHistoryBudget) {
              break;
            }
            forwardIdx += 1;
          }
          splitIndex = forwardIdx;
        }
      }

      // Advance splitIndex if we didn't evict anything yet but exceeded triggerLimit
      if (splitIndex === 0 && workingHistory.length > 1) {
        splitIndex = Math.max(1, Math.floor(workingHistory.length / 2));
      }

      if (splitIndex > 0) {
        const evicted = workingHistory.slice(0, splitIndex);
        const kept = workingHistory.slice(splitIndex);
        evictedMessageCount = evicted.length;
        rollingSummaryCutoffId = evicted[evicted.length - 1]?.id ?? rollingSummaryCutoffId;

        rollingSummary = this.config.summarizer
          ? await this.config.summarizer(evicted, rollingSummary, input.signal)
          : extractiveSummarize(evicted, rollingSummary, this.config.summaryMaxTokens);

        workingHistory = kept;
        currentPrefix = [...basePrefixMessages, makeSummaryMessage(rollingSummary)];
      }

      // Hard-cap guard if the final remaining message itself exceeds availableHistoryBudget
      const finalCompiled = [...currentPrefix, ...workingHistory];
      currentTokens = await this.countTokens(finalCompiled);
      const hardMaxTotal = basePrefixTokens + availableHistoryBudget;

      if (currentTokens > hardMaxTotal && workingHistory.length > 0) {
        const lastIdx = workingHistory.length - 1;
        const lastMsg = workingHistory[lastIdx]!;
        const prefixOnlyTokens = await this.countTokens([
          ...currentPrefix,
          ...workingHistory.slice(0, lastIdx),
        ]);
        const allowedForLastTokens = Math.max(32, hardMaxTotal - prefixOnlyTokens - 8);
        const allowedChars = Math.max(96, Math.floor(allowedForLastTokens * 3.2));
        if (lastMsg.content.length > allowedChars) {
          workingHistory = [
            ...workingHistory.slice(0, lastIdx),
            {
              ...lastMsg,
              content: `${lastMsg.content.slice(0, allowedChars)}\n[truncated to fit context budget]`,
              rawContent: undefined,
            },
          ];
        }
      }
    }

    const compiledMessages = [...currentPrefix, ...workingHistory];
    const tokensAfter = await this.countTokens(compiledMessages);
    const overlap = measureKvPrefixOverlap(input.previousCompiledMessages, compiledMessages);

    const prevHadSlot0 =
      !input.previousCompiledMessages ||
      input.previousCompiledMessages.length === 0 ||
      input.previousCompiledMessages[0]?.content === slot0.messages[0]?.content;

    const summaryMsg = currentPrefix.find(isRollingSummaryMessage);

    return {
      slot0,
      compiledMessages,
      rollingSummary,
      rollingSummaryCutoffId,
      summaryPrefixHash: summaryMsg
        ? hashMessagesPrefix([...basePrefixMessages, summaryMsg])
        : null,
      tokensBefore: rawTokens,
      tokensAfter,
      availableHistoryBudget,
      reservedOutputTokens: this.config.reservedOutputTokens,
      foldedToolCount: folded.foldedCount,
      evictedMessageCount,
      kvPrefixPreservedMessages: overlap.preservedMessages,
      kvPrefixPreservedTokens: overlap.preservedTokens,
      slot0Preserved: prevHadSlot0,
      didCompact: folded.foldedCount > 0 || evictedMessageCount > 0 || tokensAfter < rawTokens,
    };
  }
}

export { estimateTextTokens };
