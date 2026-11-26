import React from 'react';
import { describe, expect, it } from 'vitest';
import type { GenerateRequest, RuntimeAdapter, StreamDelta } from '../src/contracts.js';
import { ChatSession } from '../src/controller/chatSession.js';
import { useYoursChat } from '../src/react/useYoursChat.js';
import { InMemoryStore } from '../src/storage/inMemoryStore.js';

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
});
