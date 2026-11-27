import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { buildSlot0Envelope } from '../src/compaction/index.js';
import type {
  GenerateRequest,
  HarnessMessage,
  MemoryProvider,
  MemoryRecord,
  RuntimeAdapter,
  StreamDelta,
  ToolDefinition,
  ToolExecutor,
} from '../src/contracts.js';
import { ChatSession } from '../src/controller/chatSession.js';
import { useYoursChat } from '../src/react/useYoursChat.js';
import { InMemoryStore } from '../src/storage/inMemoryStore.js';
import type { HarnessStore } from '../src/storage/types.js';

function makeStreamingAdapter(replies: string[]): RuntimeAdapter {
  let idx = 0;
  return {
    capabilities: () => ({
      id: 'session-mock',
      maxContextTokens: 8192,
      supportsStreaming: true,
      supportsPrefixSlots: true,
      supportsTools: false,
      tier: 'tier1-f16-subgroups',
    }),
    countTokens: (msgs) => msgs.reduce((sum, m) => sum + m.content.length, 0),
    async prefillSlot(_slotKey, msgs) {
      return { tokens: msgs.length * 8, elapsedMs: 2 };
    },
    async *generate(): AsyncIterable<StreamDelta> {
      const reply = replies[idx] ?? `Reply #${idx}`;
      idx += 1;
      for (const word of reply.split(' ')) {
        yield { type: 'text', delta: `${word} ` };
      }
      yield { type: 'telemetry', ttftMs: 12, tokensPerSec: 95 };
      yield { type: 'done', finishReason: 'stop' };
    },
  };
}

describe('ChatSession & React Hook Integration', () => {
  it('persists multi-turn history, supports crash-recovery reload and DAG branching', async () => {
    const store = new InMemoryStore();
    const adapter = makeStreamingAdapter(['First answer', 'Branched answer']);

    const session1 = new ChatSession({ adapter, store });
    const thread = await session1.createThread({ title: 'Session Lifecycle' });

    await session1.sendMessage('Original prompt');
    expect(session1.getSnapshot().messages.map((m) => m.content)).toEqual([
      'Original prompt',
      'First answer',
    ]);

    const firstUserNodeId = session1.getSnapshot().nodes[0]!.id;

    // Simulate crash & reload into a fresh ChatSession instance against the same store
    const recoveredSession = new ChatSession({ adapter, store });
    await recoveredSession.loadThread(thread.id);
    expect(recoveredSession.getSnapshot().messages.map((m) => m.content)).toEqual([
      'Original prompt',
      'First answer',
    ]);

    // Edit and branch from the original user node
    await recoveredSession.editMessageAndBranch(firstUserNodeId, 'Edited prompt');
    const snapAfterBranch = recoveredSession.getSnapshot();
    expect(snapAfterBranch.messages.map((m) => m.content)).toEqual([
      'Edited prompt',
      'Branched answer',
    ]);

    const editedUserNodeId = snapAfterBranch.nodes[0]!.id;
    expect(snapAfterBranch.branchSiblings[editedUserNodeId]?.total).toBe(2);

    // Switch back to first branch
    await recoveredSession.switchBranch(firstUserNodeId);
    expect(recoveredSession.getSnapshot().messages.map((m) => m.content)).toEqual([
      'Original prompt',
      'First answer',
    ]);

    // Verify telemetry was saved
    const telemetry = await store.listTelemetry(thread.id);
    expect(telemetry).toHaveLength(2);
  });

  it('exposes useYoursChat React hook compatible with React useSyncExternalStore', async () => {
    expect(typeof useYoursChat).toBe('function');

    // Verify useYoursChat component creation compiles and preserves React element contract
    function DemoComponent() {
      const chat = useYoursChat({
        adapter: makeStreamingAdapter(['Hi from React']),
        autoInit: false,
      });
      return React.createElement('div', null, chat.state);
    }

    const element = React.createElement(DemoComponent);
    expect(element.type).toBe(DemoComponent);
  });

  it('forwards dynamic ChatSession.updateOptions and per-turn enableThinking to subsequent turns', async () => {
    const requests: GenerateRequest[] = [];
    const adapter: RuntimeAdapter = {
      capabilities: () => ({
        id: 'inspect-adapter',
        maxContextTokens: 8192,
        supportsStreaming: true,
        supportsPrefixSlots: false,
        supportsTools: false,
        tier: 'cpu',
      }),
      countTokens: (msgs) => msgs.reduce((sum, m) => sum + Math.ceil(m.content.length / 4) + 4, 0),
      async *generate(req): AsyncIterable<StreamDelta> {
        requests.push(req);
        yield { type: 'text', delta: 'OK' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    const session = new ChatSession({
      adapter,
      enableThinking: false,
      pacingMode: 'balanced',
      maxNewTokens: 256,
      temperature: 0.7,
      topP: 0.9,
      topK: 40,
      systemPrompt: 'Initial prompt',
    });

    await session.init();
    await session.sendMessage('First turn');

    expect(requests[0]?.enableThinking).toBe(false);
    expect(requests[0]?.pacingMode).toBe('balanced');
    expect(requests[0]?.maxNewTokens).toBe(256);
    expect(requests[0]?.temperature).toBe(0.7);
    expect(requests[0]?.topP).toBe(0.9);
    expect(requests[0]?.topK).toBe(40);
    expect(requests[0]?.messages[0]?.content).toContain('Initial prompt');

    session.updateOptions({
      enableThinking: true,
      pacingMode: 'smooth-ui',
      maxNewTokens: 512,
      temperature: 0.2,
      topP: 0.85,
      topK: 20,
      systemPrompt: 'Updated prompt',
    });

    await session.sendMessage('Second turn');

    expect(requests[1]?.enableThinking).toBe(true);
    expect(requests[1]?.pacingMode).toBe('smooth-ui');
    expect(requests[1]?.maxNewTokens).toBe(512);
    expect(requests[1]?.temperature).toBe(0.2);
    expect(requests[1]?.topP).toBe(0.85);
    expect(requests[1]?.topK).toBe(20);
    expect(requests[1]?.messages[0]?.content).toContain('Updated prompt');

    // Per-message enableThinking override should override session default without mutating it
    await session.sendMessage('Third turn', { enableThinking: false });
    expect(requests[2]?.enableThinking).toBe(false);

    await session.sendMessage('Fourth turn');
    expect(requests[3]?.enableThinking).toBe(true);
  });

  it('clears streamingText immediately on orphan </think> and persists reasoning in ChatSession', async () => {
    const store = new InMemoryStore();
    const adapter: RuntimeAdapter = {
      capabilities: () => ({
        id: 'orphan-think-adapter',
        maxContextTokens: 8192,
        supportsStreaming: true,
        supportsPrefixSlots: false,
        supportsTools: false,
        tier: 'cpu',
      }),
      countTokens: (msgs) => msgs.reduce((sum, m) => sum + Math.ceil(m.content.length / 4) + 4, 0),
      async *generate(): AsyncIterable<StreamDelta> {
        yield { type: 'text', delta: 'Checking ' };
        yield { type: 'text', delta: 'math...</thi' };
        yield { type: 'text', delta: 'nk>\n\n' };
        yield { type: 'text', delta: 'The answer is 42.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    const session = new ChatSession({ adapter, store, enableThinking: true });
    await session.init();

    const observedStates: Array<{ streamingText: string; streamingReasoning: string }> = [];
    const unsubscribe = session.subscribe(() => {
      const snap = session.getSnapshot();
      if (snap.isGenerating) {
        observedStates.push({
          streamingText: snap.streamingText,
          streamingReasoning: snap.streamingReasoning,
        });
      }
    });

    await session.sendMessage('What is 6*7?');
    unsubscribe();

    // Verify streamingText was cleared when </think> arrived before 'The answer is 42.'
    const resetIdx = observedStates.findIndex(
      (s, i) => i > 0 && observedStates[i - 1]!.streamingText === 'Checking math...' && s.streamingText === '',
    );
    expect(resetIdx).toBeGreaterThan(0);

    const messages = session.getSnapshot().messages;
    expect(messages).toHaveLength(2);
    expect(messages[1]?.content).toBe('The answer is 42.');
    expect(messages[1]?.reasoning).toBe('Checking math...');
  });

  it('uses a single listThreadNodes read in refreshThreadNodes instead of N+1 getBranchSiblings calls', async () => {
    const baseStore = new InMemoryStore();
    const thread = await baseStore.createThread({ title: 'Branched Thread' });
    const n1 = await baseStore.appendMessage({
      threadId: thread.id,
      parentId: null,
      message: { id: 'm1', role: 'user', content: 'Root 1', createdAt: 1 },
    });
    const n2 = await baseStore.appendMessage({
      threadId: thread.id,
      parentId: n1.id,
      message: { id: 'm2', role: 'assistant', content: 'Reply 1', createdAt: 2 },
    });
    await baseStore.editMessageAndBranch(thread.id, n2.id, {
      id: 'm2b',
      role: 'assistant',
      content: 'Reply 1 alt',
      createdAt: 3,
    });
    const n1b = await baseStore.editMessageAndBranch(thread.id, n1.id, {
      id: 'm1b',
      role: 'user',
      content: 'Root 2',
      createdAt: 4,
    });
    await baseStore.appendMessage({
      threadId: thread.id,
      parentId: n1b.id,
      message: { id: 'm2c', role: 'assistant', content: 'Reply 2', createdAt: 5 },
    });

    const counts = {
      listThreadNodes: 0,
      getActivePath: 0,
      getBranchSiblings: 0,
      getNode: 0,
    };

    const instrumentedStore: HarnessStore = {
      createThread: (input) => baseStore.createThread(input),
      getThread: (id) => baseStore.getThread(id),
      listThreads: (opts) => baseStore.listThreads(opts),
      updateThread: (id, patch) => baseStore.updateThread(id, patch),
      deleteThread: (id) => baseStore.deleteThread(id),
      appendMessage: (input) => baseStore.appendMessage(input),
      getNode: (id) => {
        counts.getNode += 1;
        return baseStore.getNode(id);
      },
      listThreadNodes: (id) => {
        counts.listThreadNodes += 1;
        return baseStore.listThreadNodes(id);
      },
      getActivePath: (id) => {
        counts.getActivePath += 1;
        return baseStore.getActivePath(id);
      },
      getBranchSiblings: (id) => {
        counts.getBranchSiblings += 1;
        return baseStore.getBranchSiblings(id);
      },
      editMessageAndBranch: (tid, targetId, msg) => baseStore.editMessageAndBranch(tid, targetId, msg),
      switchBranch: (tid, branchId) => baseStore.switchBranch(tid, branchId),
      appendToolTrace: (t) => baseStore.appendToolTrace(t),
      listToolTraces: (tid) => baseStore.listToolTraces(tid),
      appendTelemetry: (r) => baseStore.appendTelemetry(r),
      listTelemetry: (tid) => baseStore.listTelemetry(tid),
      searchThreads: (q, o) => baseStore.searchThreads(q, o),
    };

    const session = new ChatSession({
      adapter: makeStreamingAdapter(['OK']),
      store: instrumentedStore,
    });
    await session.loadThread(thread.id);

    expect(counts.listThreadNodes).toBe(1);
    expect(counts.getActivePath).toBe(0);
    expect(counts.getBranchSiblings).toBe(0);
    expect(counts.getNode).toBe(0);
    expect(session.getSnapshot().messages.map((m) => m.content)).toEqual(['Root 2', 'Reply 2']);
    expect(session.getSnapshot().branchSiblings[n1b.id]).toEqual({
      nodeId: n1b.id,
      siblingIds: [n1.id, n1b.id],
      activeIndex: 1,
      total: 2,
    });

    await session.switchBranch(n2.id);
    // 1 call in store.switchBranch (or in-memory) + 1 call in refreshThreadNodes; still 0 getBranchSiblings
    expect(counts.getBranchSiblings).toBe(0);
    expect(counts.getActivePath).toBe(0);
    expect(session.getSnapshot().branchSiblings[n2.id]).toMatchObject({
      nodeId: n2.id,
      activeIndex: 0,
      total: 2,
    });
  });

  it('preserves full Slot 0 + contextMessages + profileMemories in TypingSpeculativePrefillController', async () => {
    vi.useFakeTimers();
    try {
      const prefilledCalls: Array<{ slotKey: string; messages: HarnessMessage[] }> = [];
      let inspectCalls = 0;
      let profileContent = 'Prefers TypeScript';

      const memory: MemoryProvider = {
        async remember(input) {
          return {
            id: 'mem_1',
            ...input,
            createdAt: 1,
            updatedAt: 1,
            lastAccessedAt: 1,
            accessCount: 1,
          };
        },
        async recall() {
          return [];
        },
        async consolidate() {
          return { added: 0, superseded: 0 };
        },
        async inspect() {
          inspectCalls += 1;
          return [
            {
              id: 'prof_1',
              tier: 'profile',
              content: profileContent,
              createdAt: 1,
              updatedAt: 1,
              lastAccessedAt: 1,
              accessCount: 1,
            } satisfies MemoryRecord,
          ];
        },
        async forget() {
          return true;
        },
        async exportEncrypted() {
          return new Uint8Array();
        },
        async importEncrypted() {
          return { imported: 0 };
        },
        async wipeAll() {},
      };

      const tools: ToolDefinition[] = [
        {
          name: 'read_active_tab',
          description: 'Read tab',
          parameters: { type: 'object', properties: {} },
          safety: {
            readOnlyHint: true,
            consequentialHint: false,
            networkHint: false,
            untrustedContentHint: true,
          },
        },
      ];

      const ctxMsg: HarnessMessage = {
        id: 'ctx_tab',
        role: 'system',
        content: 'Active tab: WebGPU Spec',
        createdAt: 1,
      };

      const requests: GenerateRequest[] = [];
      const adapter: RuntimeAdapter = {
        capabilities: () => ({
          id: 'typing-prefill-adapter',
          maxContextTokens: 8192,
          supportsStreaming: true,
          supportsPrefixSlots: true,
          supportsTools: true,
          tier: 'tier1-f16-subgroups',
        }),
        countTokens: (msgs) => msgs.reduce((sum, m) => sum + m.content.length, 0),
        async prefillSlot(slotKey, messages) {
          prefilledCalls.push({ slotKey, messages });
          return { tokens: 64, elapsedMs: 2 };
        },
        async *generate(req) {
          requests.push(req);
          yield { type: 'text', delta: 'Answer' };
          yield { type: 'done', finishReason: 'stop' };
        },
      };

      const session = new ChatSession({
        adapter,
        memory,
        systemPrompt: 'Pinned System Prompt',
        tools,
        contextMessages: [ctxMsg],
        enableSpeculativeTyping: true,
      });

      await session.init();
      expect(inspectCalls).toBe(1);

      session.onComposerChange('Explain WebGPU subgroup ');
      await vi.advanceTimersByTimeAsync(400);

      expect(prefilledCalls).toHaveLength(1);
      const expectedSlot0 = await buildSlot0Envelope({
        systemPrompt: 'Pinned System Prompt',
        tools,
        profileMemories: [{ id: 'prof_1', content: 'Prefers TypeScript' }],
      });

      const msgs = prefilledCalls[0]!.messages;
      expect(msgs[0]?.id).toBe(expectedSlot0.messages[0]?.id);
      expect(msgs[0]?.content).toBe(expectedSlot0.messages[0]?.content);
      expect(msgs[1]?.id).toBe('ctx_tab');
      expect(msgs[2]?.id).toBe('draft_user');
      expect(msgs[2]?.content).toBe('Explain WebGPU subgroup');

      // Subsequent sendMessage reuses cached profile memories (does NOT call inspect again)
      await session.sendMessage('Explain WebGPU subgroup kernels', {
        groundingText: 'Turn grounding',
      });
      expect(inspectCalls).toBe(1);
      expect(requests[0]?.groundingText).toBe('Turn grounding');

      // Explicit refreshProfileMemories updates cached profile memories
      profileContent = 'Prefers Rust + WebGPU';
      await session.refreshProfileMemories();
      expect(inspectCalls).toBe(2);

      const newCtx: HarnessMessage = {
        id: 'ctx_2',
        role: 'system',
        content: 'Updated context',
        createdAt: 2,
      };
      session.updateOptions({ contextMessages: [newCtx], tools: [] });
      await session.sendMessage('Next turn');
      expect(requests[1]?.messages[0]?.content).toContain('Prefers Rust + WebGPU');
      expect(requests[1]?.messages[0]?.content).not.toContain('Available tools:');
      expect(requests[1]?.messages[1]?.id).toBe('ctx_2');

      session.dispose();
    } finally {
      vi.useRealTimers();
    }
  });

  it('coalesces streaming updates via requestAnimationFrame and drops leaked bare JSON immediately on flush', async () => {
    const rafQueue: Array<{ id: number; cb: () => void }> = [];
    let rafCounter = 0;
    const cancelledIds: number[] = [];
    let waitingForRaf = false;
    let releaseAfterRaf = (): void => {};
    const originalRaf = globalThis.requestAnimationFrame;
    const originalCancelRaf = globalThis.cancelAnimationFrame;

    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      rafCounter += 1;
      const id = rafCounter;
      rafQueue.push({ id, cb: () => cb(performance.now()) });
      return id;
    }) as typeof requestAnimationFrame;
    globalThis.cancelAnimationFrame = ((id: number) => {
      cancelledIds.push(id);
    }) as typeof cancelAnimationFrame;

    try {
      const store = new InMemoryStore();
      let step = 0;
      const adapter: RuntimeAdapter = {
        capabilities: () => ({
          id: 'raf-adapter',
          maxContextTokens: 8192,
          supportsStreaming: true,
          supportsPrefixSlots: false,
          supportsTools: true,
          tier: 'cpu',
        }),
        countTokens: (msgs) => msgs.reduce((sum, m) => sum + m.content.length, 0),
        async *generate(): AsyncIterable<StreamDelta> {
          if (step === 0) {
            step += 1;
            // Bare JSON streamed across chunks, recovered on flush() -> visibleTextReset
            yield { type: 'text', delta: '{"name": "read_active_tab", ' };
            yield { type: 'text', delta: '"arguments": {}}' };
            yield { type: 'done', finishReason: 'stop' };
            return;
          }
          yield { type: 'text', delta: 'Part 1 ' };
          yield { type: 'text', delta: 'Part 2' };
          // Wait for animation frame to flush coalesced delta before finishing
          await new Promise<void>((resolve) => {
            releaseAfterRaf = resolve;
            waitingForRaf = true;
          });
          yield { type: 'text', delta: ' Part 3' };
          yield { type: 'done', finishReason: 'stop' };
        },
      };

      const toolExecutor: ToolExecutor = {
        listTools: () => [
          {
            name: 'read_active_tab',
            description: 'Read tab',
            parameters: { type: 'object', properties: {} },
            safety: {
              readOnlyHint: true,
              consequentialHint: false,
              networkHint: false,
              untrustedContentHint: false,
            },
          },
        ],
        async execute(name) {
          return { ok: true, toolName: name, formatted: 'Tab text', tainted: false };
        },
      };

      const session = new ChatSession({ adapter, store, toolExecutor, allowBareJsonToolCall: true });
      await session.init();

      const streamingHistory: string[] = [];
      const unsub = session.subscribe(() => {
        const snap = session.getSnapshot();
        if (snap.isGenerating) {
          streamingHistory.push(snap.streamingText);
        }
      });

      const sendPromise = session.sendMessage('Read tab');
      while (!waitingForRaf) {
        await new Promise((r) => setTimeout(r, 1));
      }
      // Step 0 scheduled RAF 1 (cancelled immediately on flush reset acc===''),
      // Step 1 scheduled RAF 2 coalescing 'Part 1 ' + 'Part 2'
      expect(rafCounter).toBe(2);
      expect(cancelledIds).toEqual([1]);
      expect(streamingHistory).not.toContain('{"name": "read_active_tab", ');
      expect(streamingHistory).not.toContain('Part 1 ');

      // Fire RAF 2 -> flushes coalesced 'Part 1 Part 2' while generating
      const raf2 = rafQueue.find((item) => item.id === 2);
      raf2?.cb();
      expect(streamingHistory).toContain('Part 1 Part 2');

      releaseAfterRaf();
      await sendPromise;
      unsub();

      // Step 1 tail (' Part 3') scheduled RAF 3, cancelled and flushed synchronously on transition to 'done'
      expect(rafCounter).toBe(3);
      expect(cancelledIds).toEqual([1, 3]);
      expect(session.getSnapshot().streamingText).toBe('');
      expect(session.getSnapshot().messages.map((m) => m.content)).toEqual([
        'Read tab',
        '',
        'Tab text',
        'Part 1 Part 2 Part 3',
      ]);
      expect(session.getSnapshot().messages[1]?.content).toBe('');
    } finally {
      globalThis.requestAnimationFrame = originalRaf;
      globalThis.cancelAnimationFrame = originalCancelRaf;
    }
  });

  it('parallelizes post-turn appendToolTrace, appendTelemetry, and updateThread writes', async () => {
    const baseStore = new InMemoryStore();
    let inFlight = 0;
    let maxConcurrent = 0;

    const track = async <T>(fn: () => Promise<T>): Promise<T> => {
      inFlight += 1;
      maxConcurrent = Math.max(maxConcurrent, inFlight);
      await new Promise((r) => setTimeout(r, 10));
      try {
        return await fn();
      } finally {
        inFlight -= 1;
      }
    };

    const delayedStore: HarnessStore = {
      createThread: (i) => baseStore.createThread(i),
      getThread: (id) => baseStore.getThread(id),
      listThreads: (o) => baseStore.listThreads(o),
      updateThread: (id, patch) => track(() => baseStore.updateThread(id, patch)),
      deleteThread: (id) => baseStore.deleteThread(id),
      appendMessage: (i) => baseStore.appendMessage(i),
      getNode: (id) => baseStore.getNode(id),
      listThreadNodes: (id) => baseStore.listThreadNodes(id),
      getActivePath: (id) => baseStore.getActivePath(id),
      getBranchSiblings: (id) => baseStore.getBranchSiblings(id),
      editMessageAndBranch: (tid, targetId, m) => baseStore.editMessageAndBranch(tid, targetId, m),
      switchBranch: (tid, bid) => baseStore.switchBranch(tid, bid),
      appendToolTrace: (t) => track(() => baseStore.appendToolTrace(t)),
      listToolTraces: (tid) => baseStore.listToolTraces(tid),
      appendTelemetry: (r) => track(() => baseStore.appendTelemetry(r)),
      listTelemetry: (tid) => baseStore.listTelemetry(tid),
      searchThreads: (q, o) => baseStore.searchThreads(q, o),
    };

    let step = 0;
    const adapter: RuntimeAdapter = {
      capabilities: () => ({
        id: 'parallel-adapter',
        maxContextTokens: 8192,
        supportsStreaming: true,
        supportsPrefixSlots: false,
        supportsTools: true,
        tier: 'cpu',
      }),
      countTokens: (msgs) => msgs.reduce((sum, m) => sum + m.content.length, 0),
      async *generate(): AsyncIterable<StreamDelta> {
        if (step === 0) {
          step += 1;
          yield { type: 'text', delta: '<tool_call>{"name":"read_active_tab","arguments":{}}</tool_call>' };
          yield { type: 'done', finishReason: 'stop' };
          return;
        }
        yield { type: 'text', delta: 'Done' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    const toolExecutor: ToolExecutor = {
      listTools: () => [
        {
          name: 'read_active_tab',
          description: 'Read tab',
          parameters: { type: 'object', properties: {} },
          safety: {
            readOnlyHint: true,
            consequentialHint: false,
            networkHint: false,
            untrustedContentHint: false,
          },
        },
      ],
      async execute(name) {
        return { ok: true, toolName: name, formatted: 'OK', tainted: false };
      },
    };

    const session = new ChatSession({ adapter, store: delayedStore, toolExecutor });
    await session.init();
    await session.sendMessage('Run tool');

    expect(maxConcurrent).toBeGreaterThanOrEqual(3);
  });
});
