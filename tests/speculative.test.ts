import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { HarnessMessage, RuntimeAdapter } from '../src/contracts.js';
import {
  ActiveTabSpeculativePrefillController,
  BASE_SYSTEM_SLOT_ID,
  extractStableDraftPrefix,
  PrefixSlotCoordinator,
  TypingSpeculativePrefillController,
} from '../src/engine/speculative.js';

function makeMsg(role: HarnessMessage['role'], content: string): HarnessMessage {
  return { id: `m_${content.slice(0, 8)}`, role, content, createdAt: 100 };
}

describe('Speculative Prefix Controllers & Slot Coordinator', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('extractStableDraftPrefix trims incomplete trailing words and enforces minChars', () => {
    expect(extractStableDraftPrefix('short', 12)).toBe('');
    expect(extractStableDraftPrefix('Explain WebGPU subg', 12)).toBe('Explain WebGPU');
    expect(extractStableDraftPrefix('Explain WebGPU ', 12)).toBe('Explain WebGPU');
  });

  it('debounces typing vs paste and aborts immediately on divergence', async () => {
    const prefilled: Array<{ slotKey: string; text: string; signal: AbortSignal }> = [];
    let abortCount = 0;

    const controller = new TypingSpeculativePrefillController({
      minChars: 12,
      typeDebounceMs: 350,
      pasteDebounceMs: 50,
      pasteThresholdChars: 100,
      buildMessages: (stable) => [makeMsg('user', stable)],
      isEngineIdle: () => true,
      onPrefill: (slotKey, msgs, signal) => {
        prefilled.push({ slotKey, text: msgs[0]!.content, signal });
        // Simulate an active GPU prefill that remains in-flight until aborted
        return new Promise<void>((resolve) => {
          const t = setTimeout(resolve, 1000);
          signal.addEventListener(
            'abort',
            () => {
              clearTimeout(t);
              resolve();
            },
            { once: true },
          );
        });
      },
      onAbort: () => {
        abortCount += 1;
      },
    });

    // Normal typing uses 350ms debounce
    controller.onTextChanged('Explain WebGPU kernels ');
    await vi.advanceTimersByTimeAsync(200);
    expect(prefilled).toHaveLength(0);
    await vi.advanceTimersByTimeAsync(160);
    expect(prefilled).toHaveLength(1);
    expect(prefilled[0]?.text).toBe('Explain WebGPU kernels');
    expect(prefilled[0]?.signal.aborted).toBe(false);

    // Diverging edit aborts previous prefill immediately
    controller.onTextChanged('Explain CUDA kernels ');
    expect(abortCount).toBe(1);
    expect(prefilled[0]?.signal.aborted).toBe(true);

    await vi.advanceTimersByTimeAsync(360);
    expect(prefilled).toHaveLength(2);

    // Large paste (>100 chars) uses fast 50ms debounce
    const largePaste = 'Explain '.repeat(20) + 'detail ';
    controller.onTextChanged(largePaste);
    await vi.advanceTimersByTimeAsync(60);
    expect(prefilled).toHaveLength(3);

    controller.dispose();
  });

  it('deduplicates active tab extraction and fences stale tab switches', async () => {
    let tabTitle = 'Tab A';
    let tabUrl = 'https://a.example';
    let tabContent = 'A'.repeat(60);
    const prefilledUrls: string[] = [];
    let abortCalls = 0;

    const tabController = new ActiveTabSpeculativePrefillController({
      debounceMs: 250,
      extractActiveTab: async () => ({
        title: tabTitle,
        url: tabUrl,
        content: tabContent,
      }),
      buildMessages: (tab) => [makeMsg('system', `${tab.title}: ${tab.content}`)],
      isEngineIdle: () => true,
      onPrefill: async (_slot, _msgs, meta) => {
        prefilledUrls.push(meta.url);
      },
      onAbort: () => {
        abortCalls += 1;
      },
    });

    tabController.scheduleExtraction();
    await vi.advanceTimersByTimeAsync(260);
    expect(prefilledUrls).toEqual(['https://a.example']);

    // Rescheduling with identical content is deduplicated
    tabController.scheduleExtraction();
    await vi.advanceTimersByTimeAsync(260);
    expect(prefilledUrls).toEqual(['https://a.example']);

    // Switching tabs aborts and schedules new extraction
    tabTitle = 'Tab B';
    tabUrl = 'https://b.example';
    tabContent = 'B'.repeat(60);
    tabController.notifyTabChanged(true);
    expect(abortCalls).toBe(1);

    await vi.advanceTimersByTimeAsync(260);
    expect(prefilledUrls).toEqual(['https://a.example', 'https://b.example']);

    tabController.dispose();
  });

  it('coordinates prefix slots with Slot 0 protection and LRU tab eviction', async () => {
    const evicted: string[] = [];
    const mockAdapter: Pick<RuntimeAdapter, 'prefillSlot' | 'evictSlot'> = {
      prefillSlot: async (_slotKey, msgs) => ({ tokens: msgs.length * 10, elapsedMs: 5 }),
      evictSlot: async (slotKey) => {
        evicted.push(slotKey);
      },
    };

    let clock = 1000;
    const coordinator = new PrefixSlotCoordinator({
      adapter: mockAdapter,
      maxTabSlots: 2,
      now: () => {
        clock += 10;
        return clock;
      },
    });

    const sysMsg = makeMsg('system', 'Pinned Slot 0');
    await coordinator.prefillSlot(BASE_SYSTEM_SLOT_ID, [sysMsg], { kind: 'base' });
    await coordinator.prefillSlot('tab:1', [sysMsg, makeMsg('system', 'Tab 1')]);
    await coordinator.prefillSlot('tab:2', [sysMsg, makeMsg('system', 'Tab 2')]);
    await coordinator.prefillSlot('tab:3', [sysMsg, makeMsg('system', 'Tab 3')]);

    // tab:1 should be LRU-evicted, while base-system stays safe
    expect(evicted).toEqual(['tab:1']);
    const stats = coordinator.getStats();
    expect(stats.hasBasePrompt).toBe(true);
    expect(stats.tabSlotCount).toBe(2);

    const match = coordinator.findBestMatch([sysMsg, makeMsg('system', 'Tab 2'), makeMsg('user', 'Hi')]);
    expect(match.source).toBe('slot');
    expect(match.slotKey).toBe('tab:2');
    expect(match.matchedMessages).toBe(2);
  });
});
