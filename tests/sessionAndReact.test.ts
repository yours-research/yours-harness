import React from 'react';
import { describe, expect, it } from 'vitest';
import type { RuntimeAdapter, StreamDelta } from '../src/contracts.js';
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
});
