import type { HarnessMessage } from '../contracts.js';
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

export interface IndexedDbStoreOptions {
  dbName?: string;
  dbVersion?: number;
  indexedDB?: IDBFactory;
}

const STORE_THREADS = 'threads';
const STORE_NODES = 'nodes';
const STORE_TOOL_TRACES = 'tool_traces';
const STORE_TELEMETRY = 'telemetry';

function idbRequestToPromise<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

function txDone(tx: IDBTransaction): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
  });
}

/**
 * Browser IndexedDB HarnessStore fallback for web demos & environments
 * without OPFS SQLite WASM.
 */
export class IndexedDbStore implements HarnessStore {
  private readonly dbName: string;
  private readonly dbVersion: number;
  private readonly idbFactoryOverride?: IDBFactory;
  private dbPromise: Promise<IDBDatabase> | null = null;

  constructor(options: IndexedDbStoreOptions = {}) {
    this.dbName = options.dbName ?? 'yours-harness-db';
    this.dbVersion = options.dbVersion ?? 1;
    this.idbFactoryOverride = options.indexedDB;
  }

  private getDb(): Promise<IDBDatabase> {
    if (this.dbPromise) {
      return this.dbPromise;
    }
    const factory = this.idbFactoryOverride ?? globalThis.indexedDB;
    if (!factory) {
      return Promise.reject(
        new Error('IndexedDB is not available in this runtime. Use InMemoryStore or OpfsSqliteStore.'),
      );
    }

    this.dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
      const req = factory.open(this.dbName, this.dbVersion);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(STORE_THREADS)) {
          const threads = db.createObjectStore(STORE_THREADS, { keyPath: 'id' });
          threads.createIndex('by_updatedAt', 'updatedAt', { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_NODES)) {
          const nodes = db.createObjectStore(STORE_NODES, { keyPath: 'id' });
          nodes.createIndex('by_threadId', 'threadId', { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_TOOL_TRACES)) {
          const traces = db.createObjectStore(STORE_TOOL_TRACES, { keyPath: 'id' });
          traces.createIndex('by_threadId', 'threadId', { unique: false });
        }
        if (!db.objectStoreNames.contains(STORE_TELEMETRY)) {
          const telemetry = db.createObjectStore(STORE_TELEMETRY, { keyPath: 'id' });
          telemetry.createIndex('by_threadId', 'threadId', { unique: false });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error ?? new Error('Failed to open IndexedDB'));
    });

    return this.dbPromise;
  }

  async createThread(input: CreateThreadInput = {}): Promise<ThreadRecord> {
    const db = await this.getDb();
    const now = Date.now();
    const record: ThreadRecord = {
      id: input.id ?? createId('thread'),
      title: input.title ?? 'New Conversation',
      systemPrompt: input.systemPrompt,
      activeLeafNodeId: null,
      rollingSummary: null,
      rollingSummaryCutoffId: null,
      createdAt: now,
      updatedAt: now,
      metadata: input.metadata,
    };
    const tx = db.transaction(STORE_THREADS, 'readwrite');
    tx.objectStore(STORE_THREADS).put(record);
    await txDone(tx);
    return record;
  }

  async getThread(threadId: string): Promise<ThreadRecord | null> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_THREADS, 'readonly');
    const result = await idbRequestToPromise<ThreadRecord | undefined>(
      tx.objectStore(STORE_THREADS).get(threadId),
    );
    return result ?? null;
  }

  async listThreads(options?: { limit?: number; offset?: number }): Promise<ThreadRecord[]> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_THREADS, 'readonly');
    const all = await idbRequestToPromise<ThreadRecord[]>(
      tx.objectStore(STORE_THREADS).getAll(),
    );
    all.sort((a, b) => b.updatedAt - a.updatedAt);
    const offset = Math.max(0, options?.offset ?? 0);
    const limit = options?.limit ?? all.length;
    return all.slice(offset, offset + limit);
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
    const db = await this.getDb();
    const tx = db.transaction(STORE_THREADS, 'readwrite');
    const store = tx.objectStore(STORE_THREADS);
    const existing = await idbRequestToPromise<ThreadRecord | undefined>(store.get(threadId));
    if (!existing) {
      throw new Error(`Thread not found: ${threadId}`);
    }
    const updated: ThreadRecord = {
      ...existing,
      ...patch,
      updatedAt: Date.now(),
    };
    store.put(updated);
    await txDone(tx);
    return updated;
  }

  async deleteThread(threadId: string): Promise<boolean> {
    const existing = await this.getThread(threadId);
    if (!existing) {
      return false;
    }
    const db = await this.getDb();
    const tx = db.transaction(
      [STORE_THREADS, STORE_NODES, STORE_TOOL_TRACES, STORE_TELEMETRY],
      'readwrite',
    );
    tx.objectStore(STORE_THREADS).delete(threadId);

    for (const storeName of [STORE_NODES, STORE_TOOL_TRACES, STORE_TELEMETRY]) {
      const store = tx.objectStore(storeName);
      const index = store.index('by_threadId');
      const keys = await idbRequestToPromise<IDBValidKey[]>(index.getAllKeys(threadId));
      for (const key of keys) {
        store.delete(key);
      }
    }

    await txDone(tx);
    return true;
  }

  async appendMessage(input: AppendMessageInput): Promise<MessageNode> {
    const db = await this.getDb();
    const tx = db.transaction([STORE_THREADS, STORE_NODES], 'readwrite');
    const threadStore = tx.objectStore(STORE_THREADS);
    const nodeStore = tx.objectStore(STORE_NODES);

    const thread = await idbRequestToPromise<ThreadRecord | undefined>(
      threadStore.get(input.threadId),
    );
    if (!thread) {
      throw new Error(`Thread not found: ${input.threadId}`);
    }

    const nodeId = input.nodeId ?? createId('node');
    const node: MessageNode = {
      id: nodeId,
      threadId: input.threadId,
      parentId: input.parentId,
      childrenIds: [],
      message: { ...input.message },
      createdAt: input.message.createdAt || Date.now(),
    };

    if (input.parentId) {
      const parent = await idbRequestToPromise<MessageNode | undefined>(
        nodeStore.get(input.parentId),
      );
      if (!parent || parent.threadId !== input.threadId) {
        throw new Error(`Parent node not found in thread: ${input.parentId}`);
      }
      parent.childrenIds = [...parent.childrenIds, nodeId];
      nodeStore.put(parent);
    }

    nodeStore.put(node);

    thread.updatedAt = Date.now();
    if (input.setActive !== false) {
      thread.activeLeafNodeId = nodeId;
    }
    threadStore.put(thread);

    await txDone(tx);
    return node;
  }

  async getNode(nodeId: string): Promise<MessageNode | null> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_NODES, 'readonly');
    const node = await idbRequestToPromise<MessageNode | undefined>(
      tx.objectStore(STORE_NODES).get(nodeId),
    );
    return node ?? null;
  }

  async listThreadNodes(threadId: string): Promise<MessageNode[]> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_NODES, 'readonly');
    const index = tx.objectStore(STORE_NODES).index('by_threadId');
    const nodes = await idbRequestToPromise<MessageNode[]>(index.getAll(threadId));
    nodes.sort((a, b) => a.createdAt - b.createdAt);
    return nodes;
  }

  async getActivePath(threadId: string): Promise<MessageNode[]> {
    const thread = await this.getThread(threadId);
    if (!thread) {
      return [];
    }
    const nodes = await this.listThreadNodes(threadId);
    const map = new Map(nodes.map((n) => [n.id, n]));
    return buildActivePathFromMap(map, thread.activeLeafNodeId);
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
    const db = await this.getDb();
    const tx = db.transaction(STORE_TOOL_TRACES, 'readwrite');
    tx.objectStore(STORE_TOOL_TRACES).put(trace);
    await txDone(tx);
  }

  async listToolTraces(threadId: string): Promise<ToolTraceRecord[]> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_TOOL_TRACES, 'readonly');
    const items = await idbRequestToPromise<ToolTraceRecord[]>(
      tx.objectStore(STORE_TOOL_TRACES).index('by_threadId').getAll(threadId),
    );
    items.sort((a, b) => a.createdAt - b.createdAt);
    return items;
  }

  async appendTelemetry(record: TelemetryRecord): Promise<void> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_TELEMETRY, 'readwrite');
    tx.objectStore(STORE_TELEMETRY).put(record);
    await txDone(tx);
  }

  async listTelemetry(threadId: string): Promise<TelemetryRecord[]> {
    const db = await this.getDb();
    const tx = db.transaction(STORE_TELEMETRY, 'readonly');
    const items = await idbRequestToPromise<TelemetryRecord[]>(
      tx.objectStore(STORE_TELEMETRY).index('by_threadId').getAll(threadId),
    );
    items.sort((a, b) => a.createdAt - b.createdAt);
    return items;
  }

  async searchThreads(
    query: string,
    options?: { limit?: number },
  ): Promise<ThreadSearchHit[]> {
    const threads = await this.listThreads();
    const hits: ThreadSearchHit[] = [];
    for (const thread of threads) {
      const nodes = await this.listThreadNodes(thread.id);
      const hit = scoreThreadLexical(query, thread, nodes);
      if (hit) {
        hits.push(hit);
      }
    }
    hits.sort((a, b) => b.score - a.score || b.thread.updatedAt - a.thread.updatedAt);
    return hits.slice(0, options?.limit ?? 20);
  }
}
