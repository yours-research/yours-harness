import { ContextCompactor, type CompactionBudgetConfig, type CompactionPlan } from '../compaction/compactor.js';
import type {
  GpuPacingMode,
  GuardEngine,
  HarnessMessage,
  MemoryProvider,
  RuntimeAdapter,
  ToolDefinition,
  ToolExecutor,
  ToolInvocationContext,
} from '../contracts.js';
import {
  PrefixSlotCoordinator,
  TypingSpeculativePrefillController,
} from '../engine/speculative.js';
import {
  runHarnessTurn,
  TurnStateMachine,
  type PendingApprovalRequest,
  type TurnState,
  type TurnTelemetrySnapshot,
} from '../engine/turnStateMachine.js';
import { InMemoryStore } from '../storage/inMemoryStore.js';
import type {
  BranchSiblingInfo,
  HarnessStore,
  MessageNode,
  ThreadRecord,
  ThreadSearchHit,
} from '../storage/types.js';
import { createId } from '../utils/id.js';
import { estimateMessagesTokens } from '../utils/tokens.js';

export interface ChatSessionOptions {
  adapter: RuntimeAdapter;
  store?: HarnessStore;
  threadId?: string;
  systemPrompt?: string;
  environment?: ToolInvocationContext['environment'];
  tools?: readonly ToolDefinition[];
  toolExecutor?: ToolExecutor;
  guard?: GuardEngine;
  memory?: MemoryProvider;
  compactor?: ContextCompactor;
  compactionConfig?: Partial<CompactionBudgetConfig>;
  enableSpeculativeTyping?: boolean;
  maxToolSteps?: number;
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  enableThinking?: boolean;
  pacingMode?: GpuPacingMode;
  persistAbortedPartial?: boolean;
}

export interface ChatSessionSnapshot {
  thread: ThreadRecord | null;
  nodes: MessageNode[];
  messages: HarnessMessage[];
  branchSiblings: Record<string, BranchSiblingInfo>;
  state: TurnState;
  isGenerating: boolean;
  streamingText: string;
  streamingReasoning: string;
  pendingToolCalls: Array<{ id: string; name: string }>;
  pendingApproval: PendingApprovalRequest | null;
  telemetry: TurnTelemetrySnapshot | null;
  lastCompaction: CompactionPlan | null;
  error: string | null;
}

function isActiveTurnState(state: TurnState): boolean {
  return (
    state === 'prefilling' ||
    state === 'compacting' ||
    state === 'streaming' ||
    state === 'tool_executing' ||
    state === 'awaiting_approval'
  );
}

/**
 * Headless, observable chat & agent session controller.
 * Compatible with React's `useSyncExternalStore` and vanilla TypeScript apps.
 */
export class ChatSession {
  private readonly adapter: RuntimeAdapter;
  private readonly store: HarnessStore;
  private readonly environment: ToolInvocationContext['environment'];
  private readonly defaultSystemPrompt?: string;
  private readonly tools?: readonly ToolDefinition[];
  private readonly toolExecutor?: ToolExecutor;
  private readonly guard?: GuardEngine;
  private readonly memory?: MemoryProvider;
  private readonly compactor?: ContextCompactor;
  private readonly compactionConfig?: Partial<CompactionBudgetConfig>;
  private readonly maxToolSteps?: number;
  private readonly maxNewTokens?: number;
  private readonly temperature?: number;
  private readonly topP?: number;
  private readonly topK?: number;
  private readonly enableThinking?: boolean;
  private readonly pacingMode?: GpuPacingMode;
  private readonly persistAbortedPartial: boolean;

  private readonly fsm = new TurnStateMachine();
  private readonly listeners = new Set<() => void>();
  private readonly slotCoordinator: PrefixSlotCoordinator;
  private readonly typingPrefill: TypingSpeculativePrefillController | null;

  private activeAbortController: AbortController | null = null;
  private approvalResolver: ((approved: boolean) => void) | null = null;
  private previousCompiledMessages: HarnessMessage[] | undefined;

  private snapshot: ChatSessionSnapshot = {
    thread: null,
    nodes: [],
    messages: [],
    branchSiblings: {},
    state: 'idle',
    isGenerating: false,
    streamingText: '',
    streamingReasoning: '',
    pendingToolCalls: [],
    pendingApproval: null,
    telemetry: null,
    lastCompaction: null,
    error: null,
  };

  constructor(options: ChatSessionOptions) {
    this.adapter = options.adapter;
    this.store = options.store ?? new InMemoryStore();
    this.environment = options.environment ?? 'web-demo';
    this.defaultSystemPrompt = options.systemPrompt;
    this.tools = options.tools;
    this.toolExecutor = options.toolExecutor;
    this.guard = options.guard;
    this.memory = options.memory;
    this.compactor = options.compactor;
    this.compactionConfig = options.compactionConfig;
    this.maxToolSteps = options.maxToolSteps;
    this.maxNewTokens = options.maxNewTokens;
    this.temperature = options.temperature;
    this.topP = options.topP;
    this.topK = options.topK;
    this.enableThinking = options.enableThinking;
    this.pacingMode = options.pacingMode;
    this.persistAbortedPartial = options.persistAbortedPartial ?? false;

    this.slotCoordinator = new PrefixSlotCoordinator({
      adapter: this.adapter,
    });

    this.typingPrefill =
      options.enableSpeculativeTyping && this.adapter.prefillSlot
        ? new TypingSpeculativePrefillController({
            buildMessages: (stableText) => [
              ...this.snapshot.messages,
              {
                id: 'draft_user',
                role: 'user',
                content: stableText,
                createdAt: Date.now(),
              },
            ],
            isEngineIdle: () => !isActiveTurnState(this.snapshot.state),
            onPrefill: async (slotKey, messages, signal) => {
              await this.slotCoordinator.prefillSlot(slotKey, messages, {
                kind: 'draft',
                signal,
              });
            },
          })
        : null;

    this.fsm.subscribe((transition) => {
      this.patchSnapshot({
        state: transition.to,
        isGenerating: isActiveTurnState(transition.to),
      });
    });
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  getSnapshot = (): ChatSessionSnapshot => {
    return this.snapshot;
  };

  async init(threadId?: string): Promise<ThreadRecord> {
    if (threadId) {
      const existing = await this.store.getThread(threadId);
      if (existing) {
        await this.loadThread(existing.id);
        return existing;
      }
    }
    return this.createThread({ id: threadId });
  }

  async createThread(input?: {
    id?: string;
    title?: string;
    systemPrompt?: string;
    metadata?: Record<string, unknown>;
  }): Promise<ThreadRecord> {
    const thread = await this.store.createThread({
      id: input?.id,
      title: input?.title,
      systemPrompt: input?.systemPrompt ?? this.defaultSystemPrompt,
      metadata: input?.metadata,
    });
    this.previousCompiledMessages = undefined;
    this.patchSnapshot({
      thread,
      nodes: [],
      messages: [],
      branchSiblings: {},
      streamingText: '',
      streamingReasoning: '',
      pendingToolCalls: [],
      pendingApproval: null,
      error: null,
    });
    return thread;
  }

  async loadThread(threadId: string): Promise<ThreadRecord> {
    const thread = await this.store.getThread(threadId);
    if (!thread) {
      throw new Error(`Thread not found: ${threadId}`);
    }
    this.previousCompiledMessages = undefined;
    await this.refreshThreadNodes(thread);
    this.patchSnapshot({
      streamingText: '',
      streamingReasoning: '',
      pendingToolCalls: [],
      pendingApproval: null,
      error: null,
    });
    return thread;
  }

  onComposerChange(text: string): void {
    this.typingPrefill?.onTextChanged(text);
  }

  async sendMessage(
    content: string,
    options?: {
      activeOrigin?: string;
      explicitlyApprovedOrigins?: string[];
      titleIfFirst?: string;
    },
  ): Promise<void> {
    const trimmed = content.trim();
    if (!trimmed) {
      return;
    }
    if (isActiveTurnState(this.snapshot.state)) {
      throw new Error('Cannot send a new message while a turn is already in progress.');
    }

    this.typingPrefill?.cancelForSubmit();

    let thread = this.snapshot.thread;
    if (!thread) {
      thread = await this.createThread({
        title: options?.titleIfFirst ?? trimmed.slice(0, 60),
      });
    } else if (this.snapshot.nodes.length === 0 && thread.title === 'New Conversation') {
      thread = await this.store.updateThread(thread.id, {
        title: options?.titleIfFirst ?? trimmed.slice(0, 60),
      });
    }

    const parentId = thread.activeLeafNodeId;
    const userMessage: HarnessMessage = {
      id: createId('msg'),
      role: 'user',
      content: trimmed,
      createdAt: Date.now(),
    };

    // Persist user message immediately for crash recovery
    await this.store.appendMessage({
      threadId: thread.id,
      parentId,
      message: userMessage,
      setActive: true,
    });

    const refreshedThread = await this.store.getThread(thread.id);
    if (refreshedThread) {
      await this.refreshThreadNodes(refreshedThread);
    }

    await this.executeTurnFromCurrentPath(options);
  }

  async editMessageAndBranch(nodeId: string, newContent: string): Promise<void> {
    const thread = this.snapshot.thread;
    if (!thread) {
      throw new Error('No active thread.');
    }
    if (isActiveTurnState(this.snapshot.state)) {
      throw new Error('Cannot branch while a turn is in progress.');
    }

    const target = await this.store.getNode(nodeId);
    if (!target) {
      throw new Error(`Message node not found: ${nodeId}`);
    }

    await this.store.editMessageAndBranch(thread.id, nodeId, {
      id: createId('msg'),
      role: target.message.role,
      content: newContent.trim(),
      createdAt: Date.now(),
    });

    const updatedThread = await this.store.getThread(thread.id);
    if (updatedThread) {
      await this.refreshThreadNodes(updatedThread);
    }

    if (target.message.role === 'user') {
      await this.executeTurnFromCurrentPath();
    }
  }

  async regenerate(assistantNodeId?: string): Promise<void> {
    const thread = this.snapshot.thread;
    if (!thread) {
      throw new Error('No active thread.');
    }
    if (isActiveTurnState(this.snapshot.state)) {
      throw new Error('Cannot regenerate while a turn is in progress.');
    }

    const nodes = this.snapshot.nodes;
    const targetNode = assistantNodeId
      ? nodes.find((n) => n.id === assistantNodeId)
      : [...nodes].reverse().find((n) => n.message.role === 'assistant');

    if (!targetNode) {
      throw new Error('No assistant message found to regenerate.');
    }

    await this.store.updateThread(thread.id, {
      activeLeafNodeId: targetNode.parentId,
    });
    const updatedThread = await this.store.getThread(thread.id);
    if (updatedThread) {
      await this.refreshThreadNodes(updatedThread);
    }

    await this.executeTurnFromCurrentPath();
  }

  async switchBranch(branchNodeId: string): Promise<void> {
    const thread = this.snapshot.thread;
    if (!thread) {
      throw new Error('No active thread.');
    }
    if (isActiveTurnState(this.snapshot.state)) {
      throw new Error('Cannot switch branches while a turn is in progress.');
    }

    await this.store.switchBranch(thread.id, branchNodeId);
    const updatedThread = await this.store.getThread(thread.id);
    if (updatedThread) {
      this.previousCompiledMessages = undefined;
      await this.refreshThreadNodes(updatedThread);
    }
  }

  approvePendingTool(): void {
    if (this.approvalResolver) {
      const resolve = this.approvalResolver;
      this.approvalResolver = null;
      this.patchSnapshot({ pendingApproval: null });
      resolve(true);
    }
  }

  denyPendingTool(): void {
    if (this.approvalResolver) {
      const resolve = this.approvalResolver;
      this.approvalResolver = null;
      this.patchSnapshot({ pendingApproval: null });
      resolve(false);
    }
  }

  abort(): void {
    if (this.approvalResolver) {
      const resolve = this.approvalResolver;
      this.approvalResolver = null;
      resolve(false);
    }
    this.activeAbortController?.abort();
  }

  async listThreads(options?: { limit?: number; offset?: number }): Promise<ThreadRecord[]> {
    return this.store.listThreads(options);
  }

  async searchThreads(query: string, options?: { limit?: number }): Promise<ThreadSearchHit[]> {
    return this.store.searchThreads(query, options);
  }

  dispose(): void {
    this.abort();
    this.typingPrefill?.dispose();
    this.listeners.clear();
  }

  private async executeTurnFromCurrentPath(options?: {
    activeOrigin?: string;
    explicitlyApprovedOrigins?: string[];
  }): Promise<void> {
    const thread = this.snapshot.thread;
    if (!thread) {
      return;
    }

    const controller = new AbortController();
    this.activeAbortController = controller;
    const turnId = createId('turn');

    this.patchSnapshot({
      streamingText: '',
      streamingReasoning: '',
      pendingToolCalls: [],
      pendingApproval: null,
      error: null,
    });

    try {
      const result = await runHarnessTurn({
        sessionId: thread.id,
        environment: this.environment,
        adapter: this.adapter,
        messages: this.snapshot.messages,
        systemPrompt: thread.systemPrompt ?? this.defaultSystemPrompt,
        tools: this.tools,
        toolExecutor: this.toolExecutor,
        guard: this.guard,
        memory: this.memory,
        compactor: this.compactor,
        compactionConfig: this.compactionConfig,
        slotCoordinator: this.slotCoordinator,
        activeOrigin: options?.activeOrigin,
        explicitlyApprovedOrigins: options?.explicitlyApprovedOrigins,
        maxToolSteps: this.maxToolSteps,
        maxNewTokens: this.maxNewTokens,
        temperature: this.temperature,
        topP: this.topP,
        topK: this.topK,
        enableThinking: this.enableThinking,
        pacingMode: this.pacingMode,
        previousRollingSummary: thread.rollingSummary,
        previousRollingSummaryCutoffId: thread.rollingSummaryCutoffId,
        previousCompiledMessages: this.previousCompiledMessages,
        signal: controller.signal,
        stateMachine: this.fsm,
        onTextDelta: (_delta, acc) => {
          this.patchSnapshot({ streamingText: acc });
        },
        onReasoningDelta: (_delta, acc) => {
          this.patchSnapshot({ streamingReasoning: acc });
        },
        onToolCallStart: (info) => {
          const exists = this.snapshot.pendingToolCalls.some((t) => t.id === info.id);
          if (!exists) {
            this.patchSnapshot({
              pendingToolCalls: [...this.snapshot.pendingToolCalls, { id: info.id, name: info.name }],
            });
          }
        },
        onCompaction: (plan) => {
          this.patchSnapshot({ lastCompaction: plan });
        },
        onTelemetry: (telemetry) => {
          this.patchSnapshot({ telemetry });
        },
        onApprovalRequired: (request) => {
          this.patchSnapshot({ pendingApproval: request });
          return new Promise<boolean>((resolve) => {
            this.approvalResolver = resolve;
          });
        },
      });

      this.previousCompiledMessages = result.compiledMessages;

      let parentId = thread.activeLeafNodeId;
      for (const msg of result.newMessages) {
        const saved = await this.store.appendMessage({
          threadId: thread.id,
          parentId,
          message: msg,
          setActive: true,
        });
        parentId = saved.id;
      }

      if (
        result.finalState === 'aborted' &&
        this.persistAbortedPartial &&
        result.newMessages.length === 0 &&
        this.snapshot.streamingText.trim().length > 0
      ) {
        const partialNode = await this.store.appendMessage({
          threadId: thread.id,
          parentId,
          message: {
            id: createId('msg'),
            role: 'assistant',
            content: this.snapshot.streamingText,
            createdAt: Date.now(),
          },
          setActive: true,
        });
        parentId = partialNode.id;
      }

      for (const trace of result.toolTraces) {
        await this.store.appendToolTrace({
          id: createId('trace'),
          threadId: thread.id,
          turnId,
          step: trace.step,
          toolName: trace.call.name,
          arguments: trace.call.arguments,
          resultFormatted: trace.result.formatted,
          ok: trace.result.ok,
          tainted: trace.result.tainted,
          guardVerdict: trace.guardDecision.verdict,
          durationMs: trace.durationMs,
          createdAt: Date.now(),
        });
      }

      const caps = await this.adapter.capabilities();
      await this.store.appendTelemetry({
        id: createId('tel'),
        threadId: thread.id,
        turnId,
        adapterId: caps.id,
        ttftMs: result.telemetry.ttftMs,
        tokensPerSec: result.telemetry.tokensPerSec,
        kvTokens: result.telemetry.kvTokens,
        promptTokens: result.lastCompaction?.tokensAfter ?? estimateMessagesTokens(this.snapshot.messages),
        outputTokens: estimateMessagesTokens(result.newMessages),
        speculativeAcceptRate: result.telemetry.speculativeAcceptRate,
        compactedSavedTokens: result.lastCompaction
          ? Math.max(0, result.lastCompaction.tokensBefore - result.lastCompaction.tokensAfter)
          : 0,
        createdAt: Date.now(),
      });

      await this.store.updateThread(thread.id, {
        rollingSummary: result.rollingSummary,
        rollingSummaryCutoffId: result.rollingSummaryCutoffId,
      });

      const refreshedThread = await this.store.getThread(thread.id);
      if (refreshedThread) {
        await this.refreshThreadNodes(refreshedThread);
      }

      this.patchSnapshot({
        telemetry: result.telemetry,
        lastCompaction: result.lastCompaction,
        pendingToolCalls: [],
        pendingApproval: null,
      });
    } catch (err) {
      this.patchSnapshot({
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      this.activeAbortController = null;
      this.approvalResolver = null;
      this.fsm.reset();
      this.patchSnapshot({
        state: this.fsm.state,
        isGenerating: false,
        streamingText: '',
        streamingReasoning: '',
        pendingToolCalls: [],
        pendingApproval: null,
      });
    }
  }

  private async refreshThreadNodes(thread: ThreadRecord): Promise<void> {
    const nodes = await this.store.getActivePath(thread.id);
    const branchSiblings: Record<string, BranchSiblingInfo> = {};
    for (const node of nodes) {
      const info = await this.store.getBranchSiblings(node.id);
      if (info && info.total > 1) {
        branchSiblings[node.id] = info;
      }
    }
    this.patchSnapshot({
      thread,
      nodes,
      messages: nodes.map((n) => n.message),
      branchSiblings,
    });
  }

  private patchSnapshot(patch: Partial<ChatSessionSnapshot>): void {
    this.snapshot = {
      ...this.snapshot,
      ...patch,
    };
    for (const listener of this.listeners) {
      listener();
    }
  }
}
