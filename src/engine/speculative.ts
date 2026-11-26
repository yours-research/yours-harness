import type { HarnessMessage, RuntimeAdapter } from '../contracts.js';
import { hashMessagesPrefix, hashStringFnv1a } from '../utils/tokens.js';

export const BASE_SYSTEM_SLOT_ID = 'base-system';
export const TURN_CHECKPOINT_SLOT_ID = 'turn-checkpoint';

/**
 * Trims a draft composer string back to the last whitespace boundary so we never
 * speculatively prefill a half-typed trailing word that BPE-tokenizes differently
 * once completed. Returns '' if the stable prefix is shorter than minChars (12).
 */
export function extractStableDraftPrefix(text: string, minChars = 12): string {
  if (text.length < minChars) {
    return '';
  }
  const endsWithWhitespace = /\s$/.test(text);
  const candidate = endsWithWhitespace
    ? text.replace(/\s+$/, '')
    : text.replace(/\S+$/, '').replace(/\s+$/, '');
  return candidate.length >= minChars ? candidate : '';
}

export interface TypingSpeculativePrefillOptions {
  minChars?: number;
  typeDebounceMs?: number;
  pasteDebounceMs?: number;
  pasteThresholdChars?: number;
  slotKey?: string;
  buildMessages: (stableText: string) => HarnessMessage[];
  isEngineIdle: () => boolean;
  onPrefill: (
    slotKey: string,
    messages: HarnessMessage[],
    signal: AbortSignal,
  ) => Promise<void>;
  onAbort?: () => void;
}

/**
 * Debounced speculative composer prefill controller.
 * - Uses longer debounce for normal typing (default 350ms) and fast debounce for paste (50ms).
 * - Immediately cancels via AbortController when text is cleared, edited/diverged, or submitted.
 */
export class TypingSpeculativePrefillController {
  private readonly minChars: number;
  private readonly typeDebounceMs: number;
  private readonly pasteDebounceMs: number;
  private readonly pasteThresholdChars: number;
  private readonly slotKey: string;
  private readonly buildMessages: (stableText: string) => HarnessMessage[];
  private readonly isEngineIdle: () => boolean;
  private readonly onPrefill: (
    slotKey: string,
    messages: HarnessMessage[],
    signal: AbortSignal,
  ) => Promise<void>;
  private readonly onAbort?: () => void;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private activeAbortController: AbortController | null = null;
  private lastText = '';
  private lastPrefilledPrefix = '';
  private prefillInFlight = false;
  private disposed = false;

  constructor(options: TypingSpeculativePrefillOptions) {
    this.minChars = options.minChars ?? 12;
    this.typeDebounceMs = options.typeDebounceMs ?? 350;
    this.pasteDebounceMs = options.pasteDebounceMs ?? 50;
    this.pasteThresholdChars = options.pasteThresholdChars ?? 120;
    this.slotKey = options.slotKey ?? 'draft:composer';
    this.buildMessages = options.buildMessages;
    this.isEngineIdle = options.isEngineIdle;
    this.onPrefill = options.onPrefill;
    this.onAbort = options.onAbort;
  }

  onTextChanged(nextText: string): void {
    if (this.disposed) {
      return;
    }

    const delta = nextText.length - this.lastText.length;
    this.lastText = nextText;
    this.clearTimer();

    const divergedFromPrefilled =
      this.lastPrefilledPrefix.length > 0 && !nextText.startsWith(this.lastPrefilledPrefix);

    if (nextText.length === 0 || divergedFromPrefilled) {
      this.abortActivePrefill();
      if (nextText.length === 0) {
        return;
      }
    }

    const stablePrefix = extractStableDraftPrefix(nextText, this.minChars);
    if (!stablePrefix || stablePrefix === this.lastPrefilledPrefix) {
      return;
    }

    const debounceMs = delta >= this.pasteThresholdChars ? this.pasteDebounceMs : this.typeDebounceMs;

    this.timer = setTimeout(() => {
      this.timer = null;
      if (this.disposed || !this.isEngineIdle()) {
        return;
      }
      if (!this.lastText.startsWith(stablePrefix)) {
        return;
      }

      this.activeAbortController?.abort();
      const controller = new AbortController();
      this.activeAbortController = controller;
      this.lastPrefilledPrefix = stablePrefix;
      this.prefillInFlight = true;

      const messages = this.buildMessages(stablePrefix);
      void this.onPrefill(this.slotKey, messages, controller.signal)
        .catch(() => {})
        .finally(() => {
          if (this.activeAbortController === controller) {
            this.prefillInFlight = false;
            this.activeAbortController = null;
          }
        });
    }, debounceMs);
  }

  cancelForSubmit(): string | null {
    this.clearTimer();
    const matchedSlot = this.lastPrefilledPrefix ? this.slotKey : null;
    this.lastText = '';
    this.abortActivePrefill();
    return matchedSlot;
  }

  isPrefillInFlight(): boolean {
    return this.prefillInFlight;
  }

  getLastPrefilledPrefix(): string {
    return this.lastPrefilledPrefix;
  }

  dispose(): void {
    this.disposed = true;
    this.clearTimer();
    this.abortActivePrefill();
  }

  private abortActivePrefill(): void {
    this.lastPrefilledPrefix = '';
    this.prefillInFlight = false;
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
    this.onAbort?.();
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

export interface ExtractedActiveTab {
  title: string;
  url: string;
  content: string;
  mode?: string;
  truncated?: boolean;
}

export interface ActiveTabSpeculativePrefillOptions {
  debounceMs?: number;
  minContentChars?: number;
  extractActiveTab: () => Promise<ExtractedActiveTab | null>;
  buildMessages: (tab: ExtractedActiveTab & { contentHash: string }) => HarnessMessage[];
  isEngineIdle: () => boolean;
  onPrefill: (
    slotKey: string,
    messages: HarnessMessage[],
    meta: { url: string; contentHash: string },
    signal: AbortSignal,
  ) => Promise<void>;
  onAbort?: () => void;
}

/**
 * Debounced active-tab speculative prefill controller with FNV-1a deduplication
 * and generation-ID fencing to prevent stale async tab extractions from racing.
 */
export class ActiveTabSpeculativePrefillController {
  private readonly debounceMs: number;
  private readonly minContentChars: number;
  private readonly extractActiveTab: () => Promise<ExtractedActiveTab | null>;
  private readonly buildMessages: (
    tab: ExtractedActiveTab & { contentHash: string },
  ) => HarnessMessage[];
  private readonly isEngineIdle: () => boolean;
  private readonly onPrefill: ActiveTabSpeculativePrefillOptions['onPrefill'];
  private readonly onAbort?: () => void;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private activeAbortController: AbortController | null = null;
  private lastUrl = '';
  private lastContentHash = '';
  private prefillInFlight = false;
  private generationId = 0;
  private disposed = false;

  constructor(options: ActiveTabSpeculativePrefillOptions) {
    this.debounceMs = options.debounceMs ?? 250;
    this.minContentChars = options.minContentChars ?? 40;
    this.extractActiveTab = options.extractActiveTab;
    this.buildMessages = options.buildMessages;
    this.isEngineIdle = options.isEngineIdle;
    this.onPrefill = options.onPrefill;
    this.onAbort = options.onAbort;
  }

  scheduleExtraction(): void {
    if (this.disposed) {
      return;
    }
    this.clearTimer();
    const gen = this.generationId;

    this.timer = setTimeout(() => {
      this.timer = null;
      void this.runExtraction(gen);
    }, this.debounceMs);
  }

  notifyTabChanged(reschedule = true): void {
    if (this.disposed) {
      return;
    }
    this.clearTimer();
    this.generationId += 1;
    if (this.prefillInFlight) {
      this.lastUrl = '';
      this.lastContentHash = '';
    }
    this.abortInFlight();
    if (reschedule) {
      this.scheduleExtraction();
    }
  }

  cancelForSubmit(): void {
    this.clearTimer();
    this.generationId += 1;
    if (this.prefillInFlight) {
      this.lastUrl = '';
      this.lastContentHash = '';
    }
    this.abortInFlight();
  }

  isPrefillInFlight(): boolean {
    return this.prefillInFlight;
  }

  dispose(): void {
    this.disposed = true;
    this.generationId += 1;
    this.clearTimer();
    this.abortInFlight();
  }

  private abortInFlight(): void {
    this.prefillInFlight = false;
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
    this.onAbort?.();
  }

  private async runExtraction(gen: number): Promise<void> {
    if (this.disposed || gen !== this.generationId || !this.isEngineIdle()) {
      return;
    }

    const extracted = await this.extractActiveTab().catch(() => null);
    if (this.disposed || gen !== this.generationId || !extracted) {
      return;
    }

    const trimmedContent = extracted.content.trim();
    if (trimmedContent.length < this.minContentChars) {
      return;
    }

    const contentHash = hashStringFnv1a(`${extracted.url}\n${trimmedContent}`);
    if (extracted.url === this.lastUrl && contentHash === this.lastContentHash) {
      return;
    }

    if (!this.isEngineIdle()) {
      return;
    }

    const slotKey = `tab:${hashStringFnv1a(extracted.url)}`;
    const messages = this.buildMessages({
      ...extracted,
      content: trimmedContent,
      contentHash,
    });

    const controller = new AbortController();
    this.activeAbortController = controller;
    this.lastUrl = extracted.url;
    this.lastContentHash = contentHash;
    this.prefillInFlight = true;

    try {
      await this.onPrefill(
        slotKey,
        messages,
        { url: extracted.url, contentHash },
        controller.signal,
      );
    } catch {
      if (gen === this.generationId) {
        this.lastUrl = '';
        this.lastContentHash = '';
      }
    } finally {
      if (gen === this.generationId) {
        this.prefillInFlight = false;
        if (this.activeAbortController === controller) {
          this.activeAbortController = null;
        }
      }
    }
  }

  private clearTimer(): void {
    if (this.timer !== null) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }
}

export type PrefixSlotKind = 'base' | 'tab' | 'draft' | 'checkpoint';

export interface PrefixSlotMeta {
  url?: string;
  contentHash?: string;
}

export interface PrefixSlotRecord {
  slotKey: string;
  kind: PrefixSlotKind;
  prefixHash: string;
  messages: HarnessMessage[];
  tokens: number;
  elapsedMs: number;
  lastUsedAt: number;
  meta?: PrefixSlotMeta;
}

export interface PrefixSlotMatch {
  source: 'live' | 'slot' | 'none';
  slotKey: string | null;
  matchedMessages: number;
  slot?: PrefixSlotRecord;
}

export interface PrefixSlotCoordinatorOptions {
  adapter: Pick<RuntimeAdapter, 'prefillSlot' | 'evictSlot'>;
  maxTabSlots?: number;
  now?: () => number;
}

function messagesMatchPrefix(
  prefix: readonly HarnessMessage[],
  target: readonly HarnessMessage[],
): boolean {
  if (prefix.length === 0 || prefix.length > target.length) {
    return false;
  }
  for (let i = 0; i < prefix.length; i += 1) {
    const a = prefix[i]!;
    const b = target[i]!;
    if (
      a.role !== b.role ||
      a.content !== b.content ||
      (a.rawContent ?? a.content) !== (b.rawContent ?? b.content) ||
      a.toolName !== b.toolName
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Coordinates KV-cache / DeltaNet prefix slots across a RuntimeAdapter.
 * Protects `base-system` (Slot 0) and `turn-checkpoint` while applying LRU eviction
 * to volatile `tab:` and `draft:` speculative slots.
 */
export class PrefixSlotCoordinator {
  private readonly adapter: Pick<RuntimeAdapter, 'prefillSlot' | 'evictSlot'>;
  private readonly maxTabSlots: number;
  private readonly now: () => number;
  private readonly slots = new Map<string, { record: PrefixSlotRecord; touchSeq: number }>();
  private liveMessages: HarnessMessage[] = [];
  private touchCounter = 0;

  constructor(options: PrefixSlotCoordinatorOptions) {
    this.adapter = options.adapter;
    this.maxTabSlots = Math.max(0, Math.floor(options.maxTabSlots ?? 3));
    this.now = options.now ?? (() => Date.now());
  }

  setLiveMessages(messages: readonly HarnessMessage[]): void {
    this.liveMessages = messages.map((m) => ({ ...m }));
  }

  getLiveMessages(): HarnessMessage[] {
    return this.liveMessages.map((m) => ({ ...m }));
  }

  findBestMatch(targetMessages: readonly HarnessMessage[]): PrefixSlotMatch {
    const liveMatched = messagesMatchPrefix(this.liveMessages, targetMessages)
      ? this.liveMessages.length
      : 0;

    let bestSlot: PrefixSlotRecord | undefined;
    let bestMatched = 0;

    for (const { record } of this.slots.values()) {
      if (
        record.messages.length > bestMatched &&
        messagesMatchPrefix(record.messages, targetMessages)
      ) {
        bestSlot = record;
        bestMatched = record.messages.length;
      }
    }

    // Break ties in favor of live GPU state (0ms restore cost)
    if (liveMatched > 0 && liveMatched >= bestMatched) {
      return { source: 'live', slotKey: null, matchedMessages: liveMatched };
    }

    if (bestSlot && bestMatched > 0) {
      bestSlot.lastUsedAt = this.now();
      this.touchCounter += 1;
      const entry = this.slots.get(bestSlot.slotKey);
      if (entry) {
        entry.touchSeq = this.touchCounter;
      }
      return {
        source: 'slot',
        slotKey: bestSlot.slotKey,
        matchedMessages: bestMatched,
        slot: bestSlot,
      };
    }

    return { source: 'none', slotKey: null, matchedMessages: 0 };
  }

  async prefillSlot(
    slotKey: string,
    messages: readonly HarnessMessage[],
    options?: {
      kind?: PrefixSlotKind;
      meta?: PrefixSlotMeta;
      signal?: AbortSignal;
    },
  ): Promise<PrefixSlotRecord> {
    const existing = this.slots.get(slotKey);
    const prefixHash = hashMessagesPrefix(messages);
    if (existing && existing.record.prefixHash === prefixHash) {
      existing.record.lastUsedAt = this.now();
      this.touchCounter += 1;
      existing.touchSeq = this.touchCounter;
      return existing.record;
    }

    const kind: PrefixSlotKind =
      options?.kind ??
      (slotKey === BASE_SYSTEM_SLOT_ID || slotKey.startsWith('base-system:')
        ? 'base'
        : slotKey === TURN_CHECKPOINT_SLOT_ID
          ? 'checkpoint'
          : slotKey.startsWith('draft')
            ? 'draft'
            : 'tab');

    const cloned = messages.map((m) => ({ ...m }));
    const prefillResult = this.adapter.prefillSlot
      ? await this.adapter.prefillSlot(slotKey, cloned, options?.signal)
      : { tokens: 0, elapsedMs: 0 };

    if (options?.signal?.aborted) {
      throw new DOMException('Speculative prefill aborted', 'AbortError');
    }

    this.touchCounter += 1;
    const record: PrefixSlotRecord = {
      slotKey,
      kind,
      prefixHash,
      messages: cloned,
      tokens: prefillResult.tokens,
      elapsedMs: prefillResult.elapsedMs,
      lastUsedAt: this.now(),
      meta: options?.meta ? { ...options.meta } : undefined,
    };

    this.slots.set(slotKey, { record, touchSeq: this.touchCounter });
    await this.evictLruTabSlotsIfNeeded(slotKey);
    return record;
  }

  async clearVolatileSlots(): Promise<void> {
    const toDelete: string[] = [];
    for (const [slotKey, { record }] of this.slots.entries()) {
      if (record.kind === 'base') {
        continue;
      }
      toDelete.push(slotKey);
    }
    for (const slotKey of toDelete) {
      this.slots.delete(slotKey);
      await this.adapter.evictSlot?.(slotKey);
    }
  }

  async clearAll(): Promise<void> {
    const keys = Array.from(this.slots.keys());
    this.slots.clear();
    this.liveMessages = [];
    for (const slotKey of keys) {
      await this.adapter.evictSlot?.(slotKey);
    }
  }

  getStats(): {
    totalSlots: number;
    tabSlotCount: number;
    maxTabSlots: number;
    hasBasePrompt: boolean;
    slots: PrefixSlotRecord[];
  } {
    let tabSlotCount = 0;
    let hasBasePrompt = false;
    const slots: PrefixSlotRecord[] = [];

    for (const { record } of this.slots.values()) {
      if (record.kind === 'base') {
        hasBasePrompt = true;
      } else if (record.kind === 'tab' || record.kind === 'draft') {
        tabSlotCount += 1;
      }
      slots.push(record);
    }

    return {
      totalSlots: this.slots.size,
      tabSlotCount,
      maxTabSlots: this.maxTabSlots,
      hasBasePrompt,
      slots,
    };
  }

  private async evictLruTabSlotsIfNeeded(protectedSlotKey: string): Promise<void> {
    const volatile: Array<{ record: PrefixSlotRecord; touchSeq: number }> = [];
    for (const entry of this.slots.values()) {
      if (entry.record.kind === 'tab' || entry.record.kind === 'draft') {
        volatile.push(entry);
      }
    }

    if (volatile.length <= this.maxTabSlots) {
      return;
    }

    volatile.sort((a, b) => {
      if (a.record.lastUsedAt !== b.record.lastUsedAt) {
        return a.record.lastUsedAt - b.record.lastUsedAt;
      }
      return a.touchSeq - b.touchSeq;
    });

    let remaining = volatile.length;
    for (const candidate of volatile) {
      if (remaining <= this.maxTabSlots) {
        break;
      }
      if (this.maxTabSlots > 0 && candidate.record.slotKey === protectedSlotKey) {
        continue;
      }
      this.slots.delete(candidate.record.slotKey);
      await this.adapter.evictSlot?.(candidate.record.slotKey);
      remaining -= 1;
    }
  }
}
