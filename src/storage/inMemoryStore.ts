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

/**
 * Instant, zero-dependency in-memory HarnessStore for Vitest suites, SSR,
 * and ephemeral sessions. Implements full DAG branching & lexical search.
 */
export class InMemoryStore implements HarnessStore {
  private readonly threads = new Map<string, ThreadRecord>();
  private readonly nodes = new Map<string, MessageNode>();
  private readonly threadNodeIds = new Map<string, string[]>();
  private readonly toolTraces = new Map<string, ToolTraceRecord[]>();
  private readonly telemetry = new Map<string, TelemetryRecord[]>();

  async createThread(input: CreateThreadInput = {}): Promise<ThreadRecord> {
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
      metadata: input.metadata ? { ...input.metadata } : undefined,
    };
    this.threads.set(record.id, record);
    this.threadNodeIds.set(record.id, []);
    this.toolTraces.set(record.id, []);
    this.telemetry.set(record.id, []);
    return { ...record };
  }

  async getThread(threadId: string): Promise<ThreadRecord | null> {
    const thread = this.threads.get(threadId);
    return thread ? { ...thread } : null;
  }

  async listThreads(options?: { limit?: number; offset?: number }): Promise<ThreadRecord[]> {
    const all = Array.from(this.threads.values()).sort((a, b) => b.updatedAt - a.updatedAt);
    const offset = Math.max(0, options?.offset ?? 0);
    const limit = options?.limit ?? all.length;
    return all.slice(offset, offset + limit).map((t) => ({ ...t }));
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
    const existing = this.threads.get(threadId);
    if (!existing) {
      throw new Error(`Thread not found: ${threadId}`);
    }
    const updated: ThreadRecord = {
      ...existing,
      ...patch,
      updatedAt: Date.now(),
    };
    this.threads.set(threadId, updated);
    return { ...updated };
  }

  async deleteThread(threadId: string): Promise<boolean> {
    const existed = this.threads.delete(threadId);
    if (!existed) {
      return false;
    }
    const nodeIds = this.threadNodeIds.get(threadId) ?? [];
    for (const id of nodeIds) {
      this.nodes.delete(id);
    }
    this.threadNodeIds.delete(threadId);
    this.toolTraces.delete(threadId);
    this.telemetry.delete(threadId);
    return true;
  }

  async appendMessage(input: AppendMessageInput): Promise<MessageNode> {
    const thread = this.threads.get(input.threadId);
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
      const parent = this.nodes.get(input.parentId);
      if (!parent || parent.threadId !== input.threadId) {
        throw new Error(`Parent node not found in thread: ${input.parentId}`);
      }
      parent.childrenIds.push(nodeId);
    }

    this.nodes.set(nodeId, node);
    const list = this.threadNodeIds.get(input.threadId) ?? [];
    list.push(nodeId);
    this.threadNodeIds.set(input.threadId, list);

    if (input.setActive !== false) {
      thread.activeLeafNodeId = nodeId;
    }
    thread.updatedAt = Date.now();

    return { ...node, childrenIds: [...node.childrenIds], message: { ...node.message } };
  }

  async getNode(nodeId: string): Promise<MessageNode | null> {
    const node = this.nodes.get(nodeId);
    return node
      ? { ...node, childrenIds: [...node.childrenIds], message: { ...node.message } }
      : null;
  }

  async listThreadNodes(threadId: string): Promise<MessageNode[]> {
    const ids = this.threadNodeIds.get(threadId) ?? [];
    const out: MessageNode[] = [];
    for (const id of ids) {
      const node = this.nodes.get(id);
      if (node) {
        out.push({ ...node, childrenIds: [...node.childrenIds], message: { ...node.message } });
      }
    }
    return out;
  }

  async getActivePath(threadId: string): Promise<MessageNode[]> {
    const thread = this.threads.get(threadId);
    if (!thread) {
      return [];
    }
    return buildActivePathFromMap(this.nodes, thread.activeLeafNodeId).map((n) => ({
      ...n,
      childrenIds: [...n.childrenIds],
      message: { ...n.message },
    }));
  }

  async getBranchSiblings(nodeId: string): Promise<BranchSiblingInfo | null> {
    const node = this.nodes.get(nodeId);
    if (!node) {
      return null;
    }
    let siblingIds: string[];
    if (node.parentId) {
      const parent = this.nodes.get(node.parentId);
      siblingIds = parent ? [...parent.childrenIds] : [node.id];
    } else {
      const threadIds = this.threadNodeIds.get(node.threadId) ?? [];
      siblingIds = threadIds.filter((id) => this.nodes.get(id)?.parentId === null);
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
    const target = this.nodes.get(targetNodeId);
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
    const thread = this.threads.get(threadId);
    const branchNode = this.nodes.get(branchNodeId);
    if (!thread || !branchNode || branchNode.threadId !== threadId) {
      throw new Error(`Branch node "${branchNodeId}" not found in thread "${threadId}".`);
    }
    thread.activeLeafNodeId = walkToBranchLeaf(this.nodes, branchNodeId);
    thread.updatedAt = Date.now();
    return this.getActivePath(threadId);
  }

  async appendToolTrace(trace: ToolTraceRecord): Promise<void> {
    const list = this.toolTraces.get(trace.threadId) ?? [];
    list.push({ ...trace, arguments: { ...trace.arguments } });
    this.toolTraces.set(trace.threadId, list);
  }

  async listToolTraces(threadId: string): Promise<ToolTraceRecord[]> {
    return (this.toolTraces.get(threadId) ?? []).map((t) => ({
      ...t,
      arguments: { ...t.arguments },
    }));
  }

  async appendTelemetry(record: TelemetryRecord): Promise<void> {
    const list = this.telemetry.get(record.threadId) ?? [];
    list.push({ ...record });
    this.telemetry.set(record.threadId, list);
  }

  async listTelemetry(threadId: string): Promise<TelemetryRecord[]> {
    return (this.telemetry.get(threadId) ?? []).map((r) => ({ ...r }));
  }

  async searchThreads(
    query: string,
    options?: { limit?: number },
  ): Promise<ThreadSearchHit[]> {
    const hits: ThreadSearchHit[] = [];
    for (const thread of this.threads.values()) {
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
