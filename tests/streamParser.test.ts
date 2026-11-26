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

  it('stops on <turn|> default stop sequence without leaking', () => {
    const parser = new StreamOutputParser();
    const s1 = parser.pushText('Hello from Gemma 4<tu');
    const s2 = parser.pushText('rn|>ignored trailing');
    const tail = parser.flush();

    expect(s1.textDelta + s2.textDelta + tail.textDelta).toBe('Hello from Gemma 4');
    expect(s2.stopSequenceHit).toBe('<turn|>');
    expect(parser.getVisibleText()).toBe('Hello from Gemma 4');
  });

  it('parses Qwen3.5 XML function calls with typed JSON params and pending tool name', () => {
    const parser = new StreamOutputParser();
    const chunks = [
      'Checking now.\n<to',
      'ol_call>\n<function=search_web_',
      'tab>\n<parameter=query>\n  webgpu kernels  \n</parameter>\n',
      '<parameter=limit>10</parameter>\n<parameter=exact>true</parameter>\n',
      '<parameter=filters>{"site":"yours.sh"}</parameter>\n',
      '<parameter=tags>["wgsl","gemv"]</parameter>\n',
      '<parameter=zip>007</parameter>\n',
      '</function>\n</tool_',
      'call>',
    ];

    let visible = '';
    const pendingNames: string[] = [];
    const ready: Array<{ name: string; arguments: Record<string, unknown> }> = [];

    for (const chunk of chunks) {
      const out = parser.pushText(chunk);
      visible += out.textDelta;
      if (out.pendingToolCall) {
        pendingNames.push(out.pendingToolCall.name);
      }
      ready.push(...out.toolCallsReady);
    }
    const tail = parser.flush();
    visible += tail.textDelta;
    ready.push(...tail.toolCallsReady);

    expect(visible).toBe('Checking now.\n');
    expect(pendingNames).toEqual(['tool', 'search_web_tab']);
    expect(ready).toHaveLength(1);
    expect(ready[0]?.name).toBe('search_web_tab');
    expect(ready[0]?.arguments).toEqual({
      query: 'webgpu kernels',
      limit: 10,
      exact: true,
      filters: { site: 'yours.sh' },
      tags: ['wgsl', 'gemv'],
      zip: '007',
    });
  });

  it('parses Gemma 4 <|tool_call> and <|call> native tool calls across 1-2 char splits with zero leakage', () => {
    const input1 =
      'Looking it up.<|tool_call>call:read_tab{tabId:42,mode:<|"|>article "quoted"<|"|>,nested:{ok:true,items:[<|"|>a<|"|>,<|"|>b<|"|>]}}<tool_call|>';
    const parser1 = new StreamOutputParser();
    let visible1 = '';
    const pending1: string[] = [];
    const ready1: Array<{ name: string; arguments: Record<string, unknown> }> = [];

    for (let i = 0; i < input1.length; i += 2) {
      const out = parser1.pushText(input1.slice(i, i + 2));
      visible1 += out.textDelta;
      expect(visible1).not.toContain('<|');
      expect(visible1).not.toContain('call:');
      expect(visible1).not.toContain('<tool_call|>');
      if (out.pendingToolCall) {
        pending1.push(out.pendingToolCall.name);
      }
      ready1.push(...out.toolCallsReady);
    }
    const tail1 = parser1.flush();
    visible1 += tail1.textDelta;
    ready1.push(...tail1.toolCallsReady);

    expect(visible1).toBe('Looking it up.');
    expect(pending1).toEqual(['tool', 'read_tab']);
    expect(ready1).toHaveLength(1);
    expect(ready1[0]?.name).toBe('read_tab');
    expect(ready1[0]?.arguments).toEqual({
      tabId: 42,
      mode: 'article "quoted"',
      nested: { ok: true, items: ['a', 'b'] },
    });

    // Also test <|call>call:...<call|>
    const parser2 = new StreamOutputParser();
    const out2 = parser2.pushText('<|call>call:run_sandbox{expression:<|"|>6 * 7<|"|>}<call|>');
    const tail2 = parser2.flush();
    expect(out2.textDelta + tail2.textDelta).toBe('');
    expect(out2.toolCallsReady).toHaveLength(1);
    expect(out2.toolCallsReady[0]?.name).toBe('run_sandbox');
    expect(out2.toolCallsReady[0]?.arguments).toEqual({ expression: '6 * 7' });
  });

  it('cuts off stream immediately when stopOnToolCall is true and discards trailing hallucination', () => {
    const parser = new StreamOutputParser({ stopOnToolCall: true });
    const s1 = parser.pushText('<|tool_call>call:read_active_tab{}<tool_call|>FAKE TOOL OUTPUT');
    const s2 = parser.pushText('More hallucinated answer<turn|>');
    const tail = parser.flush();

    expect(s1.toolCallsReady).toHaveLength(1);
    expect(s1.toolCallsReady[0]?.name).toBe('read_active_tab');
    expect(s1.stopSequenceHit).toBe('<tool_call|>');
    expect(s1.textDelta).toBe('');
    expect(s2.textDelta).toBe('');
    expect(tail.textDelta).toBe('');
    expect(parser.getVisibleText()).toBe('');
  });

  it('suppresses malformed closed or flushed blocks that contain tool-call markup', () => {
    const parser1 = new StreamOutputParser();
    const out1 = parser1.pushText('Hi <tool_call><function=broken_tool</tool_call>');
    const tail1 = parser1.flush();
    expect(out1.textDelta + tail1.textDelta).toBe('Hi ');
    expect(parser1.getToolCalls()).toHaveLength(0);

    const parser2 = new StreamOutputParser();
    const out2 = parser2.pushText('Prefix <|tool_call>call:bad{???<tool_call|> Suffix');
    const tail2 = parser2.flush();
    expect(out2.textDelta + tail2.textDelta).toBe('Prefix  Suffix');
    expect(parser2.getToolCalls()).toHaveLength(0);

    const parser3 = new StreamOutputParser();
    const out3 = parser3.pushText('Start <tool_call>{"name": "broken_json');
    const tail3 = parser3.flush();
    expect(out3.textDelta + tail3.textDelta).toBe('Start ');
    expect(parser3.getToolCalls()).toHaveLength(0);

    // Non-markup inside <tool_call> still surfaces so non-tool prose is not silently lost
    const parser4 = new StreamOutputParser();
    const out4 = parser4.pushText('<tool_call>plain prose</tool_call>');
    const tail4 = parser4.flush();
    expect(out4.textDelta + tail4.textDelta).toBe('<tool_call>plain prose</tool_call>');
  });

  it('sets visibleTextReset and cleanText on flush when recovering bare or fenced JSON tool calls', () => {
    const parser1 = new StreamOutputParser();
    const p1 = parser1.pushText('{"name": "read_active_tab", "arguments": {"mode": "article"}}');
    expect(p1.textDelta).toContain('read_active_tab');

    const flush1 = parser1.flush();
    expect(flush1.visibleTextReset).toBe(true);
    expect(flush1.textDelta).toBe('');
    expect(flush1.toolCallsReady).toHaveLength(1);
    expect(flush1.toolCallsReady[0]?.name).toBe('read_active_tab');
    expect(parser1.getVisibleText()).toBe('');

    const parser2 = new StreamOutputParser();
    parser2.pushText('Sure, checking now.\n```json\n{"name": "run_sandbox", "arguments": {"expression": "1+1"}}\n```');
    const flush2 = parser2.flush();
    expect(flush2.visibleTextReset).toBe(true);
    expect(flush2.textDelta).toBe('Sure, checking now.');
    expect(parser2.getVisibleText()).toBe('Sure, checking now.');
    expect(flush2.toolCallsReady).toHaveLength(1);
    expect(flush2.toolCallsReady[0]?.name).toBe('run_sandbox');
  });
});
