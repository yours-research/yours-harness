/**
 * Shared Yours Research Ecosystem Contracts (v1.0 / Tool Schema v1.1).
 * Mirrored from docs/ECOSYSTEM_CONTRACTS.md so @yours/harness interoperates
 * cleanly with @yours/tools, @yours/memory, @yours/guard, and @yours/models.
 */

export type CanonicalToolName =
  | 'read_active_tab'
  | 'list_tabs'
  | 'read_tab'
  | 'interact_page'
  | 'run_sandbox'
  | 'read_document'
  | 'search_web_tab'
  | 'fetch_background_tab'
  | 'memory_recall'
  | 'memory_save';

export interface ToolSafetyHints {
  readOnlyHint: boolean;
  consequentialHint: boolean;
  networkHint: boolean;
  untrustedContentHint: boolean;
}

export interface ToolDefinition {
  name: CanonicalToolName | (string & {});
  aliases?: string[];
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
    additionalProperties?: false;
  };
  safety: ToolSafetyHints;
}

export interface ToolInvocationContext {
  environment: 'chrome-mv3' | 'web-demo' | 'node-test';
  sessionId: string;
  origin?: string;
  abortSignal?: AbortSignal;
}

export interface ToolExecutionResult<TData = unknown> {
  ok: boolean;
  toolName: CanonicalToolName | (string & {});
  data?: TData;
  formatted: string;
  tainted: boolean;
  sourceOrigin?: string;
  error?: string;
  needsPermission?: boolean;
  requiresApproval?: boolean;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
}

export type WgslComputeTier =
  | 'tier1-f16-subgroups'
  | 'tier2-f16-shared'
  | 'tier3-f32-universal';

export type GpuPacingMode = 'max-speed' | 'balanced' | 'smooth-ui';

export interface HarnessMessage {
  id: string;
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  reasoning?: string;
  toolName?: string;
  toolCallId?: string;
  toolArgs?: Record<string, unknown>;
  tainted?: boolean;
  createdAt: number;
}

export interface GenerateRequest {
  messages: HarnessMessage[];
  tools?: ToolDefinition[];
  maxNewTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  enableThinking?: boolean;
  pacingMode?: GpuPacingMode;
  prefixSlotKey?: string;
}

export type StreamDelta =
  | { type: 'text'; delta: string }
  | { type: 'reasoning'; delta: string }
  | { type: 'tool_call_start'; id: string; name: string }
  | { type: 'tool_call_args'; id: string; delta: string }
  | { type: 'tool_call_ready'; id: string; name: string; arguments: Record<string, unknown> }
  | {
      type: 'telemetry';
      ttftMs?: number;
      tokensPerSec?: number;
      kvTokens?: number;
      speculativeAcceptRate?: number;
    }
  | { type: 'done'; finishReason: 'stop' | 'length' | 'tool_calls' | 'abort' };

export interface RuntimeCapabilities {
  id: string;
  maxContextTokens: number;
  supportsStreaming: boolean;
  supportsPrefixSlots: boolean;
  supportsTools: boolean;
  tier?: WgslComputeTier | 'cpu' | 'http';
}

export interface RuntimeAdapter {
  capabilities(): RuntimeCapabilities | Promise<RuntimeCapabilities>;
  generate(request: GenerateRequest, signal?: AbortSignal): AsyncIterable<StreamDelta>;
  prefillSlot?(
    slotKey: string,
    messages: HarnessMessage[],
    signal?: AbortSignal,
  ): Promise<{ tokens: number; elapsedMs: number }>;
  evictSlot?(slotKey: string): Promise<void>;
  countTokens(messages: HarnessMessage[]): Promise<number> | number;
}

export type MemoryTier = 'working' | 'profile' | 'episodic' | 'document';

export interface MemoryRecordInput {
  tier: MemoryTier;
  content: string;
  sourceUrl?: string;
  sourceTitle?: string;
  threadId?: string;
  tags?: string[];
  importance?: number;
  tainted?: boolean;
  supersedesId?: string;
}

export interface MemoryRecord extends MemoryRecordInput {
  id: string;
  createdAt: number;
  updatedAt: number;
  lastAccessedAt: number;
  accessCount: number;
  supersededById?: string;
}

export interface MemoryQuery {
  query: string;
  tiers?: MemoryTier[];
  limit?: number;
  minScore?: number;
  nowMs?: number;
}

export interface MemoryHit {
  record: MemoryRecord;
  score: number;
  bm25Rank?: number;
  vectorRank?: number;
  recencyWeight: number;
}

export interface MemoryProvider {
  remember(input: MemoryRecordInput): Promise<MemoryRecord>;
  recall(query: MemoryQuery): Promise<MemoryHit[]>;
  consolidate(
    threadId: string,
    messages: HarnessMessage[],
  ): Promise<{ added: number; superseded: number }>;
  inspect(filter?: {
    tier?: MemoryTier;
    limit?: number;
    offset?: number;
  }): Promise<MemoryRecord[]>;
  forget(id: string): Promise<boolean>;
  exportEncrypted(passphrase: string): Promise<Uint8Array>;
  importEncrypted(blob: Uint8Array, passphrase: string): Promise<{ imported: number }>;
  wipeAll(): Promise<void>;
}

export interface SanitizedContent {
  cleanText: string;
  spotlightedPromptBlock: string;
  strippedSignals: string[];
  injectionRiskScore: number;
  flaggedPatterns: string[];
  tainted: true;
}

export type GuardVerdict = 'allow' | 'require_approval' | 'deny';

export interface GuardDecision {
  verdict: GuardVerdict;
  reason: string;
  triggeredRule?: string;
  lethalTrifectaRisk?: boolean;
}

export interface GuardPolicyContext {
  tool: ToolDefinition;
  arguments: Record<string, unknown>;
  contextHasUntrustedContent: boolean;
  sessionAccessedPrivateData: boolean;
  activeOrigin?: string;
  explicitlyApprovedOrigins?: string[];
}

export interface GuardEngine {
  evaluateToolCall(context: GuardPolicyContext): Promise<GuardDecision> | GuardDecision;
  sanitizeToolOutput?(
    toolName: string,
    rawText: string,
    sourceOrigin?: string,
  ): Promise<SanitizedContent> | SanitizedContent;
}

export interface ToolExecutor {
  listTools?(): ToolDefinition[];
  getTool?(name: string): ToolDefinition | undefined;
  execute(
    name: string,
    args: Record<string, unknown>,
    context: ToolInvocationContext,
  ): Promise<ToolExecutionResult>;
}
