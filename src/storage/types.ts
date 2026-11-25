import type { GuardVerdict, HarnessMessage } from '../contracts.js';

export interface ThreadRecord {
  id: string;
  title: string;
  systemPrompt?: string;
  activeLeafNodeId: string | null;
  rollingSummary?: string | null;
  rollingSummaryCutoffId?: string | null;
  createdAt: number;
  updatedAt: number;
  metadata?: Record<string, unknown>;
}

export interface MessageNode {
  id: string;
  threadId: string;
  parentId: string | null;
  childrenIds: string[];
  message: HarnessMessage;
  createdAt: number;
}

export interface BranchSiblingInfo {
  nodeId: string;
  siblingIds: string[];
  activeIndex: number;
  total: number;
}

export interface ToolTraceRecord {
  id: string;
  threadId: string;
  turnId: string;
  nodeId?: string;
  step: number;
  toolName: string;
  arguments: Record<string, unknown>;
  resultFormatted: string;
  ok: boolean;
  tainted: boolean;
  guardVerdict: GuardVerdict;
  durationMs: number;
  createdAt: number;
}

export interface TelemetryRecord {
  id: string;
  threadId: string;
  turnId: string;
  adapterId: string;
  modelId?: string;
  ttftMs?: number;
  tokensPerSec?: number;
  kvTokens?: number;
  promptTokens: number;
  outputTokens: number;
  speculativeAcceptRate?: number;
  compactedSavedTokens?: number;
  createdAt: number;
}

export interface ThreadSearchHit {
  thread: ThreadRecord;
  score: number;
  matchedSnippet: string;
  matchedNodeId?: string;
}

export interface CreateThreadInput {
  id?: string;
  title?: string;
  systemPrompt?: string;
  metadata?: Record<string, unknown>;
}

export interface AppendMessageInput {
  threadId: string;
  parentId: string | null;
  message: HarnessMessage;
  nodeId?: string;
  setActive?: boolean;
}

export interface HarnessStore {
  createThread(input?: CreateThreadInput): Promise<ThreadRecord>;
  getThread(threadId: string): Promise<ThreadRecord | null>;
  listThreads(options?: { limit?: number; offset?: number }): Promise<ThreadRecord[]>;
  updateThread(
    threadId: string,
    patch: Partial<
      Pick<
        ThreadRecord,
        'title' | 'systemPrompt' | 'activeLeafNodeId' | 'rollingSummary' | 'rollingSummaryCutoffId' | 'metadata'
      >
    >,
  ): Promise<ThreadRecord>;
  deleteThread(threadId: string): Promise<boolean>;

  appendMessage(input: AppendMessageInput): Promise<MessageNode>;
  getNode(nodeId: string): Promise<MessageNode | null>;
  listThreadNodes(threadId: string): Promise<MessageNode[]>;
  getActivePath(threadId: string): Promise<MessageNode[]>;
  getBranchSiblings(nodeId: string): Promise<BranchSiblingInfo | null>;
  editMessageAndBranch(
    threadId: string,
    targetNodeId: string,
    newMessage: HarnessMessage,
  ): Promise<MessageNode>;
  switchBranch(threadId: string, branchNodeId: string): Promise<MessageNode[]>;

  appendToolTrace(trace: ToolTraceRecord): Promise<void>;
  listToolTraces(threadId: string): Promise<ToolTraceRecord[]>;

  appendTelemetry(record: TelemetryRecord): Promise<void>;
  listTelemetry(threadId: string): Promise<TelemetryRecord[]>;

  searchThreads(
    query: string,
    options?: { limit?: number },
  ): Promise<ThreadSearchHit[]>;
}

/**
 * Shared lexical scorer (TF + title boost + snippet extraction) used across
 * InMemoryStore, IndexedDbStore, and SQLite fallback search.
 */
export function scoreThreadLexical(
  query: string,
  thread: ThreadRecord,
  nodes: readonly MessageNode[],
): ThreadSearchHit | null {
  const terms = query
    .toLowerCase()
    .trim()
    .split(/\s+/)
    .filter((t) => t.length > 0);
  if (terms.length === 0) {
    return null;
  }

  let score = 0;
  const titleLower = thread.title.toLowerCase();
  for (const term of terms) {
    if (titleLower.includes(term)) {
      score += 3.0;
    }
  }

  let bestNodeScore = 0;
  let bestSnippet = '';
  let bestNodeId: string | undefined;

  for (const node of nodes) {
    const content = node.message.content;
    const lower = content.toLowerCase();
    let nodeScore = 0;
    let firstHitIdx = -1;

    for (const term of terms) {
      let idx = lower.indexOf(term);
      while (idx !== -1) {
        if (firstHitIdx === -1) {
          firstHitIdx = idx;
        }
        nodeScore += 1.0;
        idx = lower.indexOf(term, idx + term.length);
      }
    }

    if (nodeScore > 0) {
      score += nodeScore;
      if (nodeScore > bestNodeScore) {
        bestNodeScore = nodeScore;
        bestNodeId = node.id;
        const start = Math.max(0, firstHitIdx - 40);
        const end = Math.min(content.length, firstHitIdx + 120);
        bestSnippet =
          (start > 0 ? '...' : '') +
          content.slice(start, end).replace(/\s+/g, ' ').trim() +
          (end < content.length ? '...' : '');
      }
    }
  }

  if (score <= 0) {
    return null;
  }

  return {
    thread,
    score,
    matchedSnippet: bestSnippet || thread.title,
    matchedNodeId: bestNodeId,
  };
}

/**
 * Resolves the active path from root to `activeLeafNodeId` inside a node map.
 */
export function buildActivePathFromMap(
  nodesById: ReadonlyMap<string, MessageNode>,
  activeLeafNodeId: string | null,
): MessageNode[] {
  if (!activeLeafNodeId) {
    return [];
  }
  const path: MessageNode[] = [];
  const visited = new Set<string>();
  let cursor: string | null = activeLeafNodeId;

  while (cursor) {
    if (visited.has(cursor)) {
      break;
    }
    visited.add(cursor);
    const node = nodesById.get(cursor);
    if (!node) {
      break;
    }
    path.push(node);
    cursor = node.parentId;
  }

  path.reverse();
  return path;
}

/**
 * Walks down the newest (last) child chain from a chosen branch node to find its leaf.
 */
export function walkToBranchLeaf(
  nodesById: ReadonlyMap<string, MessageNode>,
  startNodeId: string,
): string {
  let current = nodesById.get(startNodeId);
  const visited = new Set<string>();
  while (current && current.childrenIds.length > 0) {
    if (visited.has(current.id)) {
      break;
    }
    visited.add(current.id);
    const nextId = current.childrenIds[current.childrenIds.length - 1]!;
    const nextNode = nodesById.get(nextId);
    if (!nextNode) {
      break;
    }
    current = nextNode;
  }
  return current?.id ?? startNodeId;
}
