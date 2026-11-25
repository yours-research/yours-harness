import type { WorkerPortTransport } from '../adapters/wgslWorkerAdapter.js';
import type { GuardVerdict, HarnessMessage } from '../contracts.js';
import { createId } from '../utils/id.js';
import {
  buildActivePathFromMap,
  scoreThreadLexical,
  walkToBranchLeaf,
  type AppendMessageInput,
  type BranchSiblingInfo,
  type CreateThreadInput,
  type HarnessStore,
  type MessageNode,
  type TelemetryRecord,
  type ThreadRecord,
  type ThreadSearchHit,
  type ToolTraceRecord,
} from './types.js';

export type SqlParam = string | number | null;

export interface SqlExecutor {
  exec(sql: string, params?: SqlParam[]): Promise<void>;
  query<T = Record<string, unknown>>(sql: string, params?: SqlParam[]): Promise<T[]>;
}

export const OPFS_SQLITE_SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS threads (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  system_prompt TEXT,
  active_leaf_node_id TEXT,
  rolling_summary TEXT,
  rolling_summary_cutoff_id TEXT,
  metadata_json TEXT,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS message_nodes (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  parent_id TEXT,
  children_json TEXT NOT NULL,
  message_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (thread_id) REFERENCES threads(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_nodes_thread ON message_nodes(thread_id, created_at);

CREATE TABLE IF NOT EXISTS tool_traces (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  node_id TEXT,
  step INTEGER NOT NULL,
  tool_name TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  result_formatted TEXT NOT NULL,
  ok INTEGER NOT NULL,
  tainted INTEGER NOT NULL,
  guard_verdict TEXT NOT NULL,
  duration_ms REAL NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traces_thread ON tool_traces(thread_id, created_at);

CREATE TABLE IF NOT EXISTS telemetry (
  id TEXT PRIMARY KEY,
  thread_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  adapter_id TEXT NOT NULL,
  model_id TEXT,
  ttft_ms REAL,
  tokens_per_sec REAL,
  kv_tokens INTEGER,
  prompt_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  speculative_accept_rate REAL,
  compacted_saved_tokens INTEGER,
  created_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_thread ON telemetry(thread_id, created_at);
`;

export const OPFS_SQLITE_FTS_SQL = `
CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
  node_id UNINDEXED,
  thread_id UNINDEXED,
  content
);
`;

export type SqliteRpcRequest =
  | { type: 'SQL_EXEC'; requestId: string; sql: string; params?: SqlParam[] }
  | { type: 'SQL_QUERY'; requestId: string; sql: string; params?: SqlParam[] };

export type SqliteRpcResponse =
  | { type: 'SQL_OK'; requestId: string; rows?: Array<Record<string, unknown>> }
  | { type: 'SQL_ERROR'; requestId: string; error: string };

/**
 * Bridges a Web Worker / Chrome MV3 Offscreen Port hosting SQLite WASM + OPFS
 * into the `SqlExecutor` interface used by `OpfsSqliteStore`.
 */
export function createSqliteWorkerRpcExecutor(
  transport: WorkerPortTransport,
  timeoutMs = 10_000,
): SqlExecutor {
  const send = <T>(
    payload: { type: 'SQL_EXEC' | 'SQL_QUERY'; sql: string; params?: SqlParam[] },
  ): Promise<T> => {
    const requestId = createId('sql');
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        unsubscribe();
        reject(new Error(`SQLite Worker RPC timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      const unsubscribe = transport.addMessageListener((raw) => {
        if (!raw || typeof raw !== 'object') {
          return;
        }
        const msg = raw as SqliteRpcResponse;
        if (msg.requestId !== requestId) {
          return;
        }
        clearTimeout(timer);
        unsubscribe();
        if (msg.type === 'SQL_OK') {
          resolve((msg.rows ?? []) as T);
        } else {
          reject(new Error(msg.error));
        }
      });

      transport.postMessage({ ...payload, requestId });
    });
  };

  return {
    exec: async (sql, params) => {
      await send<void>({ type: 'SQL_EXEC', sql, params });
    },
    query: <T = Record<string, unknown>>(sql: string, params?: SqlParam[]) =>
      send<T[]>({ type: 'SQL_QUERY', sql, params }),
  };
}

interface ThreadRow {
  id: string;
  title: string;
  system_prompt: string | null;
  active_leaf_node_id: string | null;
  rolling_summary: string | null;
  rolling_summary_cutoff_id: string | null;
  metadata_json: string | null;
  created_at: number;
  updated_at: number;
}

interface NodeRow {
  id: string;
  thread_id: string;
  parent_id: string | null;
  children_json: string;
  message_json: string;
  created_at: number;
}

interface ToolTraceRow {
  id: string;
  thread_id: string;
  turn_id: string;
  node_id: string | null;
  step: number;
  tool_name: string;
  arguments_json: string;
  result_formatted: string;
  ok: number;
  tainted: number;
  guard_verdict: string;
  duration_ms: number;
  created_at: number;
}

interface TelemetryRow {
  id: string;
  thread_id: string;
  turn_id: string;
  adapter_id: string;
  model_id: string | null;
  ttft_ms: number | null;
  tokens_per_sec: number | null;
  kv_tokens: number | null;
  prompt_tokens: number;
  output_tokens: number;
  speculative_accept_rate: number | null;
  compacted_saved_tokens: number | null;
  created_at: number;
}

function mapThreadRow(row: ThreadRow): ThreadRecord {
  return {
    id: row.id,
    title: row.title,
    systemPrompt: row.system_prompt ?? undefined,
    activeLeafNodeId: row.active_leaf_node_id,
    rollingSummary: row.rolling_summary,
    rollingSummaryCutoffId: row.rolling_summary_cutoff_id,
    metadata: row.metadata_json ? (JSON.parse(row.metadata_json) as Record<string, unknown>) : undefined,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function mapNodeRow(row: NodeRow): MessageNode {
  return {
    id: row.id,
    threadId: row.thread_id,
    parentId: row.parent_id,
    childrenIds: JSON.parse(row.children_json) as string[],
    message: JSON.parse(row.message_json) as HarnessMessage,
    createdAt: row.created_at,
  };
}

/**
 * Worker-RPC-friendly SQLite over OPFS `HarnessStore` implementation.
 * Supports FTS5 full-text search with automatic fallback to lexical scoring.
 */
export class OpfsSqliteStore implements HarnessStore {
  private readonly executor: SqlExecutor;
  private initialized = false;
  private ftsEnabled = false;

  constructor(executor: SqlExecutor) {
    this.executor = executor;
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }
    const statements = OPFS_SQLITE_SCHEMA_SQL.split(';')
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
    for (const stmt of statements) {
      await this.executor.exec(stmt);
    }

    try {
      await this.executor.exec(OPFS_SQLITE_FTS_SQL.trim());
      this.ftsEnabled = true;
    } catch {
      this.ftsEnabled = false;
    }

    this.initialized = true;
  }

  async createThread(input: CreateThreadInput = {}): Promise<ThreadRecord> {
    await this.init();
    const now = Date.now();
    const record: ThreadRecord = {
      id: input.id ?? createId('thread'),
      title: input.title ?? 'New Conversation',
      systemPrompt: input.systemPrompt,
      activeLeafNodeId: null,
      rollingSummary: null,
      rollingSummaryCutoffId: null,
      metadata: input.metadata,
      createdAt: now,
      updatedAt: now,
    };

    await this.executor.exec(
      `INSERT INTO threads (id, title, system_prompt, active_leaf_node_id, rolling_summary, rolling_summary_cutoff_id, metadata_json, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        record.id,
        record.title,
        record.systemPrompt ?? null,
        null,
        null,
        null,
        record.metadata ? JSON.stringify(record.metadata) : null,
        record.createdAt,
        record.updatedAt,
      ],
    );

    return record;
  }

  async getThread(threadId: string): Promise<ThreadRecord | null> {
    await this.init();
    const rows = await this.executor.query<ThreadRow>(
      `SELECT * FROM threads WHERE id = ? LIMIT 1`,
      [threadId],
    );
    return rows[0] ? mapThreadRow(rows[0]) : null;
  }

  async listThreads(options?: { limit?: number; offset?: number }): Promise<ThreadRecord[]> {
    await this.init();
    const limit = options?.limit ?? 100;
    const offset = Math.max(0, options?.offset ?? 0);
    const rows = await this.executor.query<ThreadRow>(
      `SELECT * FROM threads ORDER BY updated_at DESC LIMIT ? OFFSET ?`,
      [limit, offset],
    );
    return rows.map(mapThreadRow);
  }

  async updateThread(
    threadId: string,
    patch: Partial<
      Pick<
        ThreadRecord,
        | 'title'
        | 'systemPrompt'
        | 'activeLeafNodeId'
        | 'rollingSummary'
        | 'rollingSummaryCutoffId'
        | 'metadata'
      >
    >,
  ): Promise<ThreadRecord> {
    await this.init();
    const existing = await this.getThread(threadId);
    if (!existing) {
      throw new Error(`Thread not found: ${threadId}`);
    }
    const updated: ThreadRecord = {
      ...existing,
      ...patch,
      updatedAt: Date.now(),
    };
    await this.executor.exec(
      `UPDATE threads
       SET title = ?, system_prompt = ?, active_leaf_node_id = ?, rolling_summary = ?, rolling_summary_cutoff_id = ?, metadata_json = ?, updated_at = ?
       WHERE id = ?`,
      [
        updated.title,
        updated.systemPrompt ?? null,
        updated.activeLeafNodeId,
        updated.rollingSummary ?? null,
        updated.rollingSummaryCutoffId ?? null,
        updated.metadata ? JSON.stringify(updated.metadata) : null,
        updated.updatedAt,
        threadId,
      ],
    );
    return updated;
  }

  async deleteThread(threadId: string): Promise<boolean> {
    await this.init();
    const existing = await this.getThread(threadId);
    if (!existing) {
      return false;
    }
    if (this.ftsEnabled) {
      await this.executor.exec(`DELETE FROM nodes_fts WHERE thread_id = ?`, [threadId]);
    }
    await this.executor.exec(`DELETE FROM message_nodes WHERE thread_id = ?`, [threadId]);
    await this.executor.exec(`DELETE FROM tool_traces WHERE thread_id = ?`, [threadId]);
    await this.executor.exec(`DELETE FROM telemetry WHERE thread_id = ?`, [threadId]);
    await this.executor.exec(`DELETE FROM threads WHERE id = ?`, [threadId]);
    return true;
  }

  async appendMessage(input: AppendMessageInput): Promise<MessageNode> {
    await this.init();
    const thread = await this.getThread(input.threadId);
    if (!thread) {
      throw new Error(`Thread not found: ${input.threadId}`);
    }

    const nodeId = input.nodeId ?? createId('node');
    const createdAt = input.message.createdAt || Date.now();
    const node: MessageNode = {
      id: nodeId,
      threadId: input.threadId,
      parentId: input.parentId,
      childrenIds: [],
      message: { ...input.message },
      createdAt,
    };

    if (input.parentId) {
      const parent = await this.getNode(input.parentId);
      if (!parent || parent.threadId !== input.threadId) {
        throw new Error(`Parent node not found in thread: ${input.parentId}`);
      }
      const nextChildren = [...parent.childrenIds, nodeId];
      await this.executor.exec(
        `UPDATE message_nodes SET children_json = ? WHERE id = ?`,
        [JSON.stringify(nextChildren), parent.id],
      );
    }

    await this.executor.exec(
      `INSERT INTO message_nodes (id, thread_id, parent_id, children_json, message_json, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      [
        node.id,
        node.threadId,
        node.parentId,
        JSON.stringify(node.childrenIds),
        JSON.stringify(node.message),
        node.createdAt,
      ],
    );

    if (this.ftsEnabled) {
      await this.executor.exec(
        `INSERT INTO nodes_fts (node_id, thread_id, content) VALUES (?, ?, ?)`,
        [node.id, node.threadId, node.message.content],
      );
    }

    await this.executor.exec(
      `UPDATE threads SET active_leaf_node_id = COALESCE(?, active_leaf_node_id), updated_at = ? WHERE id = ?`,
      [input.setActive === false ? null : node.id, Date.now(), input.threadId],
    );

    return node;
  }

  async getNode(nodeId: string): Promise<MessageNode | null> {
    await this.init();
    const rows = await this.executor.query<NodeRow>(
      `SELECT * FROM message_nodes WHERE id = ? LIMIT 1`,
      [nodeId],
    );
    return rows[0] ? mapNodeRow(rows[0]) : null;
  }

  async listThreadNodes(threadId: string): Promise<MessageNode[]> {
    await this.init();
    const rows = await this.executor.query<NodeRow>(
      `SELECT * FROM message_nodes WHERE thread_id = ? ORDER BY created_at ASC`,
      [threadId],
    );
    return rows.map(mapNodeRow);
  }

  async getActivePath(threadId: string): Promise<MessageNode[]> {
    const thread = await this.getThread(threadId);
    if (!thread) {
      return [];
    }
    const nodes = await this.listThreadNodes(threadId);
    return buildActivePathFromMap(
      new Map(nodes.map((n) => [n.id, n])),
      thread.activeLeafNodeId,
    );
  }

  async getBranchSiblings(nodeId: string): Promise<BranchSiblingInfo | null> {
    const node = await this.getNode(nodeId);
    if (!node) {
      return null;
    }
    let siblingIds: string[];
    if (node.parentId) {
      const parent = await this.getNode(node.parentId);
      siblingIds = parent ? [...parent.childrenIds] : [node.id];
    } else {
      const all = await this.listThreadNodes(node.threadId);
      siblingIds = all.filter((n) => n.parentId === null).map((n) => n.id);
    }
    const activeIndex = Math.max(0, siblingIds.indexOf(nodeId));
    return {
      nodeId,
      siblingIds,
      activeIndex,
      total: siblingIds.length,
    };
  }

  async editMessageAndBranch(
    threadId: string,
    targetNodeId: string,
    newMessage: HarnessMessage,
  ): Promise<MessageNode> {
    const target = await this.getNode(targetNodeId);
    if (!target || target.threadId !== threadId) {
      throw new Error(`Target node "${targetNodeId}" not found in thread "${threadId}".`);
    }
    return this.appendMessage({
      threadId,
      parentId: target.parentId,
      message: newMessage,
      setActive: true,
    });
  }

  async switchBranch(threadId: string, branchNodeId: string): Promise<MessageNode[]> {
    const nodes = await this.listThreadNodes(threadId);
    const map = new Map(nodes.map((n) => [n.id, n]));
    if (!map.has(branchNodeId)) {
      throw new Error(`Branch node "${branchNodeId}" not found in thread "${threadId}".`);
    }
    const leafId = walkToBranchLeaf(map, branchNodeId);
    await this.updateThread(threadId, { activeLeafNodeId: leafId });
    return buildActivePathFromMap(map, leafId);
  }

  async appendToolTrace(trace: ToolTraceRecord): Promise<void> {
    await this.init();
    await this.executor.exec(
      `INSERT INTO tool_traces
       (id, thread_id, turn_id, node_id, step, tool_name, arguments_json, result_formatted, ok, tainted, guard_verdict, duration_ms, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        trace.id,
        trace.threadId,
        trace.turnId,
        trace.nodeId ?? null,
        trace.step,
        trace.toolName,
        JSON.stringify(trace.arguments),
        trace.resultFormatted,
        trace.ok ? 1 : 0,
        trace.tainted ? 1 : 0,
        trace.guardVerdict,
        trace.durationMs,
        trace.createdAt,
      ],
    );
  }

  async listToolTraces(threadId: string): Promise<ToolTraceRecord[]> {
    await this.init();
    const rows = await this.executor.query<ToolTraceRow>(
      `SELECT * FROM tool_traces WHERE thread_id = ? ORDER BY created_at ASC`,
      [threadId],
    );
    return rows.map((r) => ({
      id: r.id,
      threadId: r.thread_id,
      turnId: r.turn_id,
      nodeId: r.node_id ?? undefined,
      step: r.step,
      toolName: r.tool_name,
      arguments: JSON.parse(r.arguments_json) as Record<string, unknown>,
      resultFormatted: r.result_formatted,
      ok: r.ok === 1,
      tainted: r.tainted === 1,
      guardVerdict: r.guard_verdict as GuardVerdict,
      durationMs: r.duration_ms,
      createdAt: r.created_at,
    }));
  }

  async appendTelemetry(record: TelemetryRecord): Promise<void> {
    await this.init();
    await this.executor.exec(
      `INSERT INTO telemetry
       (id, thread_id, turn_id, adapter_id, model_id, ttft_ms, tokens_per_sec, kv_tokens, prompt_tokens, output_tokens, speculative_accept_rate, compacted_saved_tokens, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        record.id,
        record.threadId,
        record.turnId,
        record.adapterId,
        record.modelId ?? null,
        record.ttftMs ?? null,
        record.tokensPerSec ?? null,
        record.kvTokens ?? null,
        record.promptTokens,
        record.outputTokens,
        record.speculativeAcceptRate ?? null,
        record.compactedSavedTokens ?? null,
        record.createdAt,
      ],
    );
  }

  async listTelemetry(threadId: string): Promise<TelemetryRecord[]> {
    await this.init();
    const rows = await this.executor.query<TelemetryRow>(
      `SELECT * FROM telemetry WHERE thread_id = ? ORDER BY created_at ASC`,
      [threadId],
    );
    return rows.map((r) => ({
      id: r.id,
      threadId: r.thread_id,
      turnId: r.turn_id,
      adapterId: r.adapter_id,
      modelId: r.model_id ?? undefined,
      ttftMs: r.ttft_ms ?? undefined,
      tokensPerSec: r.tokens_per_sec ?? undefined,
      kvTokens: r.kv_tokens ?? undefined,
      promptTokens: r.prompt_tokens,
      outputTokens: r.output_tokens,
      speculativeAcceptRate: r.speculative_accept_rate ?? undefined,
      compactedSavedTokens: r.compacted_saved_tokens ?? undefined,
      createdAt: r.created_at,
    }));
  }

  async searchThreads(
    query: string,
    options?: { limit?: number },
  ): Promise<ThreadSearchHit[]> {
    await this.init();
    const limit = options?.limit ?? 20;

    if (this.ftsEnabled) {
      try {
        const ftsRows = await this.executor.query<{
          thread_id: string;
          node_id: string;
          content: string;
          rank: number;
        }>(
          `SELECT thread_id, node_id, content, bm25(nodes_fts) AS rank
           FROM nodes_fts WHERE nodes_fts MATCH ? ORDER BY rank LIMIT ?`,
          [query.trim(), limit * 2],
        );

        const hits: ThreadSearchHit[] = [];
        const seenThreads = new Set<string>();
        for (const row of ftsRows) {
          if (seenThreads.has(row.thread_id)) {
            continue;
          }
          const thread = await this.getThread(row.thread_id);
          if (!thread) {
            continue;
          }
          seenThreads.add(row.thread_id);
          hits.push({
            thread,
            score: Math.max(0.1, 10 - row.rank),
            matchedSnippet: row.content.slice(0, 140),
            matchedNodeId: row.node_id,
          });
        }
        if (hits.length > 0) {
          return hits.slice(0, limit);
        }
      } catch {
        // Fall through to lexical scan on invalid FTS syntax
      }
    }

    const threads = await this.listThreads({ limit: 500 });
    const hits: ThreadSearchHit[] = [];
    for (const thread of threads) {
      const nodes = await this.listThreadNodes(thread.id);
      const hit = scoreThreadLexical(query, thread, nodes);
      if (hit) {
        hits.push(hit);
      }
    }
    hits.sort((a, b) => b.score - a.score || b.thread.updatedAt - a.thread.updatedAt);
    return hits.slice(0, limit);
  }
}
