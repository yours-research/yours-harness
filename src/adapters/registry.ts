import type { RuntimeAdapter, RuntimeCapabilities } from '../contracts.js';

export interface AdapterSelectionCriteria {
  preferredIds?: string[];
  requirePrefixSlots?: boolean;
  requireTools?: boolean;
  requireStreaming?: boolean;
  minContextTokens?: number;
}

/**
 * Registry for pluggable RuntimeAdapter instances. Supports multi-backend
 * registration (e.g. wgsl-worker -> chrome-ai -> ollama) and capability-based selection.
 */
export class RuntimeRegistry {
  private readonly adapters = new Map<string, RuntimeAdapter>();
  private defaultId: string | null = null;

  register(id: string, adapter: RuntimeAdapter, options?: { setAsDefault?: boolean }): this {
    const normalized = id.trim();
    if (!normalized) {
      throw new Error('RuntimeRegistry.register requires a non-empty adapter id.');
    }
    this.adapters.set(normalized, adapter);
    if (options?.setAsDefault || this.defaultId === null) {
      this.defaultId = normalized;
    }
    return this;
  }

  unregister(id: string): boolean {
    const deleted = this.adapters.delete(id);
    if (deleted && this.defaultId === id) {
      const firstRemaining = this.adapters.keys().next();
      this.defaultId = firstRemaining.done ? null : firstRemaining.value;
    }
    return deleted;
  }

  setDefault(id: string): void {
    if (!this.adapters.has(id)) {
      throw new Error(`Cannot set default runtime adapter: "${id}" is not registered.`);
    }
    this.defaultId = id;
  }

  get(id: string): RuntimeAdapter | undefined {
    return this.adapters.get(id);
  }

  getDefault(): RuntimeAdapter {
    if (this.defaultId === null) {
      throw new Error('No RuntimeAdapter is registered in RuntimeRegistry.');
    }
    const adapter = this.adapters.get(this.defaultId);
    if (!adapter) {
      throw new Error(`Default RuntimeAdapter "${this.defaultId}" not found.`);
    }
    return adapter;
  }

  getDefaultId(): string | null {
    return this.defaultId;
  }

  listIds(): string[] {
    return Array.from(this.adapters.keys());
  }

  async listCapabilities(): Promise<Array<{ id: string; capabilities: RuntimeCapabilities }>> {
    const results: Array<{ id: string; capabilities: RuntimeCapabilities }> = [];
    for (const [id, adapter] of this.adapters.entries()) {
      const capabilities = await adapter.capabilities();
      results.push({ id, capabilities });
    }
    return results;
  }

  async selectBest(
    criteria: AdapterSelectionCriteria = {},
  ): Promise<{ id: string; adapter: RuntimeAdapter; capabilities: RuntimeCapabilities } | null> {
    const orderedIds = this.buildOrderedIds(criteria.preferredIds);

    for (const id of orderedIds) {
      const adapter = this.adapters.get(id);
      if (!adapter) {
        continue;
      }
      const caps = await adapter.capabilities();
      if (criteria.requirePrefixSlots && !caps.supportsPrefixSlots) {
        continue;
      }
      if (criteria.requireTools && !caps.supportsTools) {
        continue;
      }
      if (criteria.requireStreaming && !caps.supportsStreaming) {
        continue;
      }
      if (
        criteria.minContextTokens !== undefined &&
        caps.maxContextTokens < criteria.minContextTokens
      ) {
        continue;
      }
      return { id, adapter, capabilities: caps };
    }

    return null;
  }

  private buildOrderedIds(preferredIds?: string[]): string[] {
    const seen = new Set<string>();
    const ordered: string[] = [];
    if (preferredIds) {
      for (const id of preferredIds) {
        if (this.adapters.has(id) && !seen.has(id)) {
          seen.add(id);
          ordered.push(id);
        }
      }
    }
    if (this.defaultId && this.adapters.has(this.defaultId) && !seen.has(this.defaultId)) {
      seen.add(this.defaultId);
      ordered.push(this.defaultId);
    }
    for (const id of this.adapters.keys()) {
      if (!seen.has(id)) {
        seen.add(id);
        ordered.push(id);
      }
    }
    return ordered;
  }
}
