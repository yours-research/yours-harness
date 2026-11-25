import { describe, expect, it } from 'vitest';
import {
  parseLenientJsonObject,
  StreamOutputParser,
} from '../src/engine/streamParser.js';

describe('StreamOutputParser & Malformed JSON Recovery', () => {
  it('separates interleaved <think> blocks from visible text across split chunks', () => {
    const parser = new StreamOutputParser();
    const chunks = ['<thi', 'nk>Checking ', 'math...', '</thi', 'nk>The answer ', 'is 42.<|im_', 'end|>'];

    let visible = '';
    let reasoning = '';
    let stopHit: string | null = null;

    for (const chunk of chunks) {
      const out = parser.pushText(chunk);
      visible += out.textDelta;
      reasoning += out.reasoningDelta;
      if (out.stopSequenceHit) {
        stopHit = out.stopSequenceHit;
      }
    }
    const tail = parser.flush();
    visible += tail.textDelta;
    reasoning += tail.reasoningDelta;

    expect(reasoning).toBe('Checking math...');
    expect(visible).toBe('The answer is 42.');
    expect(stopHit).toBe('<|im_end|>');
  });

  it('handles unclosed <think> blocks gracefully on flush', () => {
    const parser = new StreamOutputParser();
    const a = parser.pushText('<think>Still reasoning when max tokens hit');
    const b = parser.flush();

    expect(a.reasoningDelta + b.reasoningDelta).toBe('Still reasoning when max tokens hit');
    expect(a.textDelta + b.textDelta).toBe('');
  });

  it('parses XML <tool_call> across chunk splits and detects pending tool name', () => {
    const parser = new StreamOutputParser();
    const s1 = parser.pushText('Let me check.\n<tool_call>{"name": "read_act');
    const s2 = parser.pushText('ive_tab", "arguments": {"mode": "article"}}</tool_call>');
    const s3 = parser.flush();

    expect(s1.textDelta).toBe('Let me check.\n');
    expect(s2.pendingToolCall).toBeNull();
    expect(s2.toolCallsReady).toHaveLength(1);
    expect(s2.toolCallsReady[0]?.name).toBe('read_active_tab');
    expect(s2.toolCallsReady[0]?.arguments).toEqual({ mode: 'article' });
    expect(s3.toolCallsReady).toHaveLength(0);
  });

  it('recovers malformed and truncated JSON tool calls', () => {
    // Trailing comma + single quotes + unquoted keys
    const repaired1 = parseLenientJsonObject(`{name: 'search_web_tab', arguments: {query: 'yours.sh',},}`);
    expect(repaired1).toEqual({
      name: 'search_web_tab',
      arguments: { query: 'yours.sh' },
    });

    // Truncated missing closing braces inside <tool_call>
    const parser = new StreamOutputParser();
    parser.pushText(`<tool_call>{"name": "read_tab", "arguments": {"tabId": 7</tool_call>`);
    parser.flush();
    const calls = parser.getToolCalls();
    expect(calls).toHaveLength(1);
    expect(calls[0]?.name).toBe('read_tab');
    expect(calls[0]?.arguments).toEqual({ tabId: 7 });
  });

  it('recovers fenced or bare JSON tool calls without XML tags', () => {
    const parser = new StreamOutputParser();
    parser.pushText('```json\n{"name": "run_sandbox", "arguments": {"expression": "6 * 7"}}\n```');
    const flushed = parser.flush();

    expect(flushed.toolCallsReady).toHaveLength(1);
    expect(flushed.toolCallsReady[0]?.name).toBe('run_sandbox');
    expect(flushed.toolCallsReady[0]?.arguments).toEqual({ expression: '6 * 7' });
    expect(parser.getVisibleText()).toBe('');
  });
});
