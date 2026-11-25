import { describe, expect, it } from 'vitest';
import type { WorkerPortTransport } from '../src/adapters/wgslWorkerAdapter.js';
import type { HarnessMessage } from '../src/contracts.js';
import {
  createSqliteWorkerRpcExecutor,
  IndexedDbStore,
  InMemoryStore,
  OpfsSqliteStore,
  type SqlExecutor,
  type SqliteRpcRequest,
  type SqliteRpcResponse,
  type SqlParam,
} from '../src/storage/index.js';

function hMsg(id: string, role: HarnessMessage['role'], content: string, createdAt = 100): HarnessMessage {
  return { id, role, content, createdAt };
}

/**
 * Minimal in-memory SQL executor mocking SQLite behavior for OpfsSqliteStore tests
 * without requiring native SQLite binaries in CI.
 */
function createMockSqliteExecutor(): { executor: SqlExecutor; executedSql: string[] } {
  const tables = {
    threads: new Map<string, Record<string, unknown>>(),
    message_nodes: new Map<string, Record<string, unknown>>(),
    tool_traces: new Map<string, Record<string, unknown>>(),
    telemetry: new Map<string, Record<string, unknown>>(),
  };
  const executedSql: string[] = [];

  const executor: SqlExecutor = {
    async exec(sql: string, params: SqlParam[] = []) {
      executedSql.push(sql);
      if (/CREATE VIRTUAL TABLE/i.test(sql)) {
        throw new Error('fts5 not compiled in mock');
      }
      if (/INSERT INTO threads/i.test(sql)) {
        const [id, title, system_prompt, active_leaf_node_id, rolling_summary, rolling_summary_cutoff_id, metadata_json, created_at, updated_at] = params;
        tables.threads.set(String(id), {
          id,
          title,
          system_prompt,
          active_leaf_node_id,
          rolling_summary,
          rolling_summary_cutoff_id,
          metadata_json,
          created_at,
          updated_at,
        });
      } else if (/UPDATE threads SET active_leaf_node_id = COALESCE/i.test(sql)) {
        const [leafId, updatedAt, id] = params;
        const row = tables.threads.get(String(id));
        if (row) {
          if (leafId !== null) row.active_leaf_node_id = leafId;
          row.updated_at = updatedAt;
        }
      } else if (/UPDATE threads\s+SET title =/i.test(sql)) {
        const [title, system_prompt, active_leaf_node_id, rolling_summary, rolling_summary_cutoff_id, metadata_json, updated_at, id] = params;
        const row = tables.threads.get(String(id));
        if (row) {
          Object.assign(row, {
            title,
            system_prompt,
            active_leaf_node_id,
            rolling_summary,
            rolling_summary_cutoff_id,
            metadata_json,
            updated_at,
          });
        }
      } else if (/INSERT INTO message_nodes/i.test(sql)) {
        const [id, thread_id, parent_id, children_json, message_json, created_at] = params;
        tables.message_nodes.set(String(id), {
          id,
          thread_id,
          parent_id,
          children_json,
          message_json,
          created_at,
        });
      } else if (/UPDATE message_nodes SET children_json =/i.test(sql)) {
        const [children_json, id] = params;
        const row = tables.message_nodes.get(String(id));
        if (row) row.children_json = children_json;
      } else if (/INSERT INTO tool_traces/i.test(sql)) {
        const [id, thread_id, turn_id, node_id, step, tool_name, arguments_json, result_formatted, ok, tainted, guard_verdict, duration_ms, created_at] = params;
        tables.tool_traces.set(String(id), {
          id,
          thread_id,
          turn_id,
          node_id,
          step,
          tool_name,
          arguments_json,
          result_formatted,
          ok,
          tainted,
          guard_verdict,
          duration_ms,
          created_at,
        });
      } else if (/INSERT INTO telemetry/i.test(sql)) {
        const [id, thread_id, turn_id, adapter_id, model_id, ttft_ms, tokens_per_sec, kv_tokens, prompt_tokens, output_tokens, speculative_accept_rate, compacted_saved_tokens, created_at] = params;
        tables.telemetry.set(String(id), {
          id,
          thread_id,
          turn_id,
          adapter_id,
          model_id,
          ttft_ms,
          tokens_per_sec,
          kv_tokens,
          prompt_tokens,
          output_tokens,
          speculative_accept_rate,
          compacted_saved_tokens,
          created_at,
        });
      } else if (/DELETE FROM (\w+) WHERE (thread_id|id) =/i.test(sql)) {
        const match = /DELETE FROM (\w+) WHERE (thread_id|id) =/i.exec(sql)!;
        const tableName = match[1] as keyof typeof tables;
        const col = match[2]!;
        const val = params[0];
        const map = tables[tableName];
        if (map) {
          for (const [k, r] of map.entries()) {
            if (r[col] === val) map.delete(k);
          }
        }
      }
    },
    async query<T = Record<string, unknown>>(sql: string, params: SqlParam[] = []): Promise<T[]> {
      executedSql.push(sql);
      if (/FROM threads WHERE id =/i.test(sql)) {
        const row = tables.threads.get(String(params[0]));
        return (row ? [row] : []) as T[];
      }
      if (/FROM threads ORDER BY/i.test(sql)) {
        return Array.from(tables.threads.values()).sort(
          (a, b) => Number(b.updated_at) - Number(a.updated_at),
        ) as T[];
      }
      if (/FROM message_nodes WHERE id =/i.test(sql)) {
        const row = tables.message_nodes.get(String(params[0]));
        return (row ? [row] : []) as T[];
      }
      if (/FROM message_nodes WHERE thread_id =/i.test(sql)) {
        return Array.from(tables.message_nodes.values())
          .filter((r) => r.thread_id === params[0])
          .sort((a, b) => Number(a.created_at) - Number(b.created_at)) as T[];
      }
      if (/FROM tool_traces WHERE thread_id =/i.test(sql)) {
        return Array.from(tables.tool_traces.values())
          .filter((r) => r.thread_id === params[0]) as T[];
      }
      if (/FROM telemetry WHERE thread_id =/i.test(sql)) {
        return Array.from(tables.telemetry.values())
          .filter((r) => r.thread_id === params[0]) as T[];
      }
      return [];
    },
  };

  return { executor, executedSql };
}

describe('Durable Storage Backends (InMemory, OpfsSqlite, IndexedDB)', () => {
  it('supports branching DAG edit/switch, tool traces, telemetry, and lexical search in InMemoryStore', async () => {
    const store = new InMemoryStore();
    const thread = await store.createThread({ title: 'WebGPU Subgroup Optimization' });

    const n1 = await store.appendMessage({
      threadId: thread.id,
      parentId: null,
      message: hMsg('m1', 'user', 'How fast is WGSL f16 GEMV?', 100),
    });
    const n2 = await store.appendMessage({
      threadId: thread.id,
      parentId: n1.id,
      message: hMsg('m2', 'assistant', 'About 85 tokens/sec on RTX 3090.', 101),
    });

    // Edit n1 into a new branch
    const n1Edit = await store.editMessageAndBranch(
      thread.id,
      n1.id,
      hMsg('m1b', 'user', 'How fast is WGSL int8 KV cache?', 102),
    );
    const n2Edit = await store.appendMessage({
      threadId: thread.id,
      parentId: n1Edit.id,
      message: hMsg('m2b', 'assistant', 'Int8 KV halves VRAM bandwidth.', 103),
    });

    const activeBranch2 = await store.getActivePath(thread.id);
    expect(activeBranch2.map((n) => n.id)).toEqual([n1Edit.id, n2Edit.id]);

    const siblings = await store.getBranchSiblings(n1Edit.id);
    expect(siblings?.total).toBe(2);
    expect(siblings?.activeIndex).toBe(1);

    // Switch back to branch 1
    const branch1Path = await store.switchBranch(thread.id, n1.id);
    expect(branch1Path.map((n) => n.id)).toEqual([n1.id, n2.id]);

    // Tool traces & telemetry
    await store.appendToolTrace({
      id: 'tr_1',
      threadId: thread.id,
      turnId: 'turn_1',
      step: 0,
      toolName: 'run_sandbox',
      arguments: { expression: '2+2' },
      resultFormatted: '4',
      ok: true,
      tainted: false,
      guardVerdict: 'allow',
      durationMs: 2.5,
      createdAt: 105,
    });
    await store.appendTelemetry({
      id: 'tel_1',
      threadId: thread.id,
      turnId: 'turn_1',
      adapterId: 'wgsl-worker',
      ttftMs: 42,
      tokensPerSec: 91,
      promptTokens: 200,
      outputTokens: 30,
      createdAt: 106,
    });

    expect(await store.listToolTraces(thread.id)).toHaveLength(1);
    expect(await store.listTelemetry(thread.id)).toHaveLength(1);

    const searchHits = await store.searchThreads('int8 VRAM');
    expect(searchHits).toHaveLength(1);
    expect(searchHits[0]?.thread.id).toBe(thread.id);
  });

  it('operates OpfsSqliteStore over Worker RPC transport and falls back cleanly without FTS5', async () => {
    const { executor } = createMockSqliteExecutor();
    const listeners = new Set<(msg: unknown) => void>();

    const transport: WorkerPortTransport = {
      postMessage(raw) {
        const req = raw as SqliteRpcRequest;
        void (async () => {
          try {
            if (req.type === 'SQL_EXEC') {
              await executor.exec(req.sql, req.params);
              const res: SqliteRpcResponse = { type: 'SQL_OK', requestId: req.requestId };
              listeners.forEach((fn) => fn(res));
            } else if (req.type === 'SQL_QUERY') {
              const rows = await executor.query(req.sql, req.params);
              const res: SqliteRpcResponse = { type: 'SQL_OK', requestId: req.requestId, rows };
              listeners.forEach((fn) => fn(res));
            }
          } catch (err) {
            const res: SqliteRpcResponse = {
              type: 'SQL_ERROR',
              requestId: req.requestId,
              error: err instanceof Error ? err.message : String(err),
            };
            listeners.forEach((fn) => fn(res));
          }
        })();
      },
      addMessageListener(listener) {
        listeners.add(listener);
        return () => listeners.delete(listener);
      },
    };

    const rpcExecutor = createSqliteWorkerRpcExecutor(transport);
    const store = new OpfsSqliteStore(rpcExecutor);

    const thread = await store.createThread({ title: 'MV3 Offscreen SQLite Thread' });
    const n1 = await store.appendMessage({
      threadId: thread.id,
      parentId: null,
      message: hMsg('s1', 'user', 'Persist DeltaNet checkpoint', 200),
    });
    await store.appendMessage({
      threadId: thread.id,
      parentId: n1.id,
      message: hMsg('s2', 'assistant', 'Saved to OPFS SQLite.', 201),
    });

    const path = await store.getActivePath(thread.id);
    expect(path).toHaveLength(2);
    expect(path[1]?.message.content).toBe('Saved to OPFS SQLite.');

    const hits = await store.searchThreads('DeltaNet');
    expect(hits).toHaveLength(1);
    expect(hits[0]?.thread.id).toBe(thread.id);
  });

  it('IndexedDbStore reports clear error when indexedDB is unavailable', async () => {
    const store = new IndexedDbStore({ dbName: 'test-missing-idb' });
    await expect(store.createThread()).rejects.toThrow(/IndexedDB is not available/);
  });
});
