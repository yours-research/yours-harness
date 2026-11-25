# `@yours/harness` Architecture

`@yours/harness` is designed around a single constraint that separates **local fixed-VRAM inference** from cloud API wrappers: **prefill is expensive and KV memory is finite**. Every subsystem—compaction, speculative scheduling, stream parsing, state transitions, and persistence—is built to preserve GPU KV-cache prefixes and avoid wasted compute.

## 1. Module Layout

```text
src/
├── contracts.ts                 # Canonical TypeScript interfaces (ECOSYSTEM_CONTRACTS.md)
├── utils/
│   ├── asyncQueue.ts            # Callback/port-to-AsyncIterable bridge
│   ├── id.ts                    # Collision-resistant IDs
│   └── tokens.ts                # Deterministic FNV-1a prefix hashing & token estimators
├── adapters/
│   ├── registry.ts              # RuntimeRegistry & capability-based selection
│   ├── wgslWorkerAdapter.ts     # Rust WASM + WGSL WebGPU Worker / MV3 Port bridge
│   ├── openAiCompatibleAdapter.ts # SSE streaming for Ollama / llama.cpp / vLLM
│   ├── chromePromptAiAdapter.ts # Chrome Built-in AI LanguageModel bridge
│   ├── transformersJsAdapter.ts # Transformers.js / ORT WebGPU shim
│   └── webLlmAdapter.ts         # MLC WebLLM engine shim
├── compaction/
│   ├── slot0.ts                 # Immutable Slot 0 envelope builder
│   ├── toolFolder.ts            # Stage 2 transient tool-result citation folding
│   └── compactor.ts             # Stage 3 budget planner & rolling summary
├── engine/
│   ├── streamParser.ts          # <think>, <tool_call>, malformed JSON & stop-seq parser
│   ├── speculative.ts           # Typing & active-tab prefill + PrefixSlotCoordinator
│   └── turnStateMachine.ts      # TurnStateMachine FSM & runHarnessTurn orchestrator
├── storage/
│   ├── types.ts                 # HarnessStore DAG, trace, telemetry, lexical search
│   ├── inMemoryStore.ts         # Instant Vitest/SSR store
│   ├── indexedDbStore.ts        # Web demo IndexedDB store
│   └── opfsSqliteStore.ts       # MV3 Worker-RPC SQLite over OPFS (FTS5 + fallback)
├── controller/
│   └── chatSession.ts           # Observable ChatSession controller
└── react/
    └── useYoursChat.ts          # React 18/19 hook via useSyncExternalStore
```

## 2. KV-Cache–Preserving Context Compaction

1. **Stage 1 — Immutable Pinned Slot 0 (`buildSlot0Envelope`)**:
   - Combines base system instructions, canonically sorted `ToolDefinition` signatures, and sorted `profile` memories into a deterministic system message.
   - Never embeds timestamps, UUIDs, or turn counters so `slot0.prefixHash` stays invariant across turns and sessions.
2. **Stage 2 — Transient Tool-Result Folding (`foldTransientToolResults`)**:
   - Completed earlier tool chains (e.g., raw DOM extractions or search results followed by a final assistant answer) are collapsed into semantic citations containing the tool name, extracted source URLs, token savings, and headline summary.
   - Preserves `tainted: true` provenance flags so `@yours/guard` policy enforcement never loses track of untrusted context.
   - Leaves the most recent `preserveRecentTurns` and any active tool-calling chain verbatim.
3. **Stage 3 — Block-Aligned Rolling Summary (`ContextCompactor`)**:
   - Reserves `reservedOutputTokens + safetyMarginTokens` before computing available history budget.
   - When history exceeds `triggerRatio` of available budget, evicts a contiguous block of older messages into a pinned `[Rolling Conversation Summary]` system message right after Slot 0.
   - Subsequent turns append *after* `[Slot 0, Rolling Summary]`, preserving leading KV prefix tokens (`kvPrefixPreservedTokens`) until the next block compaction boundary.

## 3. Turn State Machine & Security Hooks

`TurnStateMachine` enforces explicit states:
- `idle` → `compacting` → (`prefilling` optional) → `streaming`
- `streaming` → `awaiting_approval` (when `GuardEngine.evaluateToolCall` returns `require_approval`) → `tool_executing` → `compacting`
- Any active state → `aborted` (via `AbortController`, completing teardown in `<100ms`)
- Terminal turn completion → `done`

All tool outputs marked `tainted: true` flip `contextHasUntrustedContent = true` for the remainder of the turn/thread and pass through `GuardEngine.sanitizeToolOutput` when available.

## 4. Branching DAG Persistence

`HarnessStore` models messages as a parent-linked DAG (`MessageNode.parentId`, `childrenIds`, and `ThreadRecord.activeLeafNodeId`):
- Editing a user prompt or regenerating an assistant turn creates a sibling branch under `targetNode.parentId` without destroying prior generations.
- `switchBranch(threadId, branchNodeId)` walks down to the latest leaf of that branch and updates `activeLeafNodeId`.
- `OpfsSqliteStore` executes over an injected `SqlExecutor` or `createSqliteWorkerRpcExecutor(port)` so Chrome MV3 sidepanels can persist directly into an Offscreen/Worker OPFS SQLite database.
