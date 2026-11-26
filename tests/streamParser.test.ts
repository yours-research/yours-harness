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

  it('promotes pre-opened Qwen3.5 orphan </think> split across chunks to reasoning without leaking', () => {
    const parser = new StreamOutputParser();
    const chunks = ['Checking ', 'math...</thi', 'nk>\n\nThe answer is 42.'];

    let visible = '';
    let reasoning = '';
    let resetCount = 0;

    for (const chunk of chunks) {
      const out = parser.pushText(chunk);
      if (out.visibleTextReset) {
        visible = '';
        resetCount += 1;
      }
      visible += out.textDelta;
      reasoning += out.reasoningDelta;
    }
    const tail = parser.flush();
    if (tail.visibleTextReset) {
      visible = '';
      resetCount += 1;
    }
    visible += tail.textDelta;
    reasoning += tail.reasoningDelta;

    expect(resetCount).toBe(1);
    expect(reasoning).toBe('Checking math...');
    expect(visible).toBe('The answer is 42.');
    expect(parser.getReasoningText()).toBe('Checking math...');
    expect(parser.getVisibleText()).toBe('The answer is 42.');
    expect(visible).not.toContain('</think>');
    expect(reasoning).not.toContain('</think>');
  });

  it('parses Gemma 4 <|channel>thought...<channel|> and <|thought|>...<|/thought|> blocks cleanly', () => {
    const parser1 = new StreamOutputParser();
    const out1 = parser1.pushText('<|channel>thought\nStep 1\n<channel|>Final answer');
    const tail1 = parser1.flush();

    expect(out1.reasoningDelta + tail1.reasoningDelta).toBe('Step 1\n');
    expect(out1.textDelta + tail1.textDelta).toBe('Final answer');

    // Chunk split between <|channel>thought and trailing newline + split <channel|>
    const parser2 = new StreamOutputParser();
    let vis2 = '';
    let reas2 = '';
    for (const chunk of ['<|channel>thought', '\nStep 1\n<chan', 'nel|>\n\nFinal answer']) {
      const step = parser2.pushText(chunk);
      if (step.visibleTextReset) vis2 = '';
      vis2 += step.textDelta;
      reas2 += step.reasoningDelta;
    }
    const tail2 = parser2.flush();
    vis2 += tail2.textDelta;
    reas2 += tail2.reasoningDelta;
    expect(reas2).toBe('Step 1\n');
    expect(vis2).toBe('Final answer');

    // DeepSeek / custom <|thought|> and <|end_of_thought|>
    const parser3 = new StreamOutputParser();
    const out3 = parser3.pushText('<|thought|>Deep thought<|/thought|>\n\nVisible');
    const tail3 = parser3.flush();
    expect(out3.reasoningDelta + tail3.reasoningDelta).toBe('Deep thought');
    expect(out3.textDelta + tail3.textDelta).toBe('Visible');

    const parser4 = new StreamOutputParser();
    const out4 = parser4.pushText('Pre-opened<|end_of_thought|>\r\nResult');
    const tail4 = parser4.flush();
    expect(out4.visibleTextReset).toBe(true);
    expect(out4.reasoningDelta + tail4.reasoningDelta).toBe('Pre-opened');
    expect(out4.textDelta + tail4.textDelta).toBe('Result');
  });

  it('supports initialMode: think without triggering visibleTextReset', () => {
    const parser = new StreamOutputParser({ initialMode: 'think' });
    const chunks = ['Checking ', 'math...</thi', 'nk>\n\nThe answer is 42.<|im_end|>'];

    let visible = '';
    let reasoning = '';
    let resetSeen = false;

    for (const chunk of chunks) {
      const out = parser.pushText(chunk);
      if (out.visibleTextReset) {
        resetSeen = true;
      }
      visible += out.textDelta;
      reasoning += out.reasoningDelta;
    }
    const tail = parser.flush();
    visible += tail.textDelta;
    reasoning += tail.reasoningDelta;

    expect(resetSeen).toBe(false);
    expect(reasoning).toBe('Checking math...');
    expect(visible).toBe('The answer is 42.');
  });
});
