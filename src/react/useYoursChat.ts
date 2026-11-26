import { useCallback, useEffect, useRef, useSyncExternalStore } from 'react';
import {
  ChatSession,
  type ChatSessionOptions,
  type ChatSessionSnapshot,
  type ChatSessionUpdateOptions,
} from '../controller/chatSession.js';
import type { ThreadRecord, ThreadSearchHit } from '../storage/types.js';

export interface UseYoursChatOptions extends ChatSessionOptions {
  /** Existing ChatSession instance if managed externally. */
  session?: ChatSession;
  /** Automatically initialize or load `threadId` on mount (default: true). */
  autoInit?: boolean;
}

export interface UseYoursChatResult extends ChatSessionSnapshot {
  session: ChatSession;
  send: (
    content: string,
    options?: {
      activeOrigin?: string;
      explicitlyApprovedOrigins?: string[];
      titleIfFirst?: string;
      enableThinking?: boolean;
    },
  ) => Promise<void>;
  updateOptions: (patch: ChatSessionUpdateOptions) => void;
  editAndBranch: (nodeId: string, newContent: string) => Promise<void>;
  regenerate: (assistantNodeId?: string) => Promise<void>;
  switchBranch: (branchNodeId: string) => Promise<void>;
  abort: () => void;
  approveTool: () => void;
  denyTool: () => void;
  setComposerText: (text: string) => void;
  createThread: (input?: {
    id?: string;
    title?: string;
    systemPrompt?: string;
    metadata?: Record<string, unknown>;
  }) => Promise<ThreadRecord>;
  loadThread: (threadId: string) => Promise<ThreadRecord>;
  listThreads: (options?: { limit?: number; offset?: number }) => Promise<ThreadRecord[]>;
  searchThreads: (query: string, options?: { limit?: number }) => Promise<ThreadSearchHit[]>;
}

/**
 * Lightweight React hook subscribing to a `@yours/harness` `ChatSession`
 * via `React.useSyncExternalStore` (React 18+ and 19 compatible).
 */
export function useYoursChat(options: UseYoursChatOptions): UseYoursChatResult {
  const sessionRef = useRef<ChatSession | null>(options.session ?? null);
  const ownsSessionRef = useRef<boolean>(!options.session);

  if (!sessionRef.current) {
    sessionRef.current = new ChatSession(options);
    ownsSessionRef.current = true;
  }

  const session = sessionRef.current;

  const snapshot = useSyncExternalStore(
    session.subscribe,
    session.getSnapshot,
    session.getSnapshot,
  );

  useEffect(() => {
    if (!ownsSessionRef.current) {
      return;
    }
    session.updateOptions({
      enableThinking: options.enableThinking,
      pacingMode: options.pacingMode,
      maxNewTokens: options.maxNewTokens,
      temperature: options.temperature,
      topP: options.topP,
      topK: options.topK,
      systemPrompt: options.systemPrompt,
    });
  }, [
    session,
    options.enableThinking,
    options.pacingMode,
    options.maxNewTokens,
    options.temperature,
    options.topP,
    options.topK,
    options.systemPrompt,
  ]);

  useEffect(() => {
    if (options.autoInit === false) {
      return;
    }
    if (!session.getSnapshot().thread) {
      void session.init(options.threadId);
    }
  }, [session, options.autoInit, options.threadId]);

  useEffect(() => {
    return () => {
      if (ownsSessionRef.current) {
        session.dispose();
      }
    };
  }, [session]);

  const send = useCallback(
    (
      content: string,
      sendOptions?: {
        activeOrigin?: string;
        explicitlyApprovedOrigins?: string[];
        titleIfFirst?: string;
        enableThinking?: boolean;
      },
    ) => session.sendMessage(content, sendOptions),
    [session],
  );

  const updateOptions = useCallback(
    (patch: ChatSessionUpdateOptions) => session.updateOptions(patch),
    [session],
  );

  const editAndBranch = useCallback(
    (nodeId: string, newContent: string) => session.editMessageAndBranch(nodeId, newContent),
    [session],
  );

  const regenerate = useCallback(
    (assistantNodeId?: string) => session.regenerate(assistantNodeId),
    [session],
  );

  const switchBranch = useCallback(
    (branchNodeId: string) => session.switchBranch(branchNodeId),
    [session],
  );

  const abort = useCallback(() => session.abort(), [session]);
  const approveTool = useCallback(() => session.approvePendingTool(), [session]);
  const denyTool = useCallback(() => session.denyPendingTool(), [session]);
  const setComposerText = useCallback((text: string) => session.onComposerChange(text), [session]);

  const createThread = useCallback(
    (input?: {
      id?: string;
      title?: string;
      systemPrompt?: string;
      metadata?: Record<string, unknown>;
    }) => session.createThread(input),
    [session],
  );

  const loadThread = useCallback((threadId: string) => session.loadThread(threadId), [session]);
  const listThreads = useCallback(
    (listOptions?: { limit?: number; offset?: number }) => session.listThreads(listOptions),
    [session],
  );
  const searchThreads = useCallback(
    (query: string, searchOptions?: { limit?: number }) =>
      session.searchThreads(query, searchOptions),
    [session],
  );

  return {
    ...snapshot,
    session,
    send,
    updateOptions,
    editAndBranch,
    regenerate,
    switchBranch,
    abort,
    approveTool,
    denyTool,
    setComposerText,
    createThread,
    loadThread,
    listThreads,
    searchThreads,
  };
}
