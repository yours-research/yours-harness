let monotonicCounter = 0;

/**
 * Generates a unique, collision-resistant ID without external dependencies.
 */
export function createId(prefix = 'id'): string {
  monotonicCounter = (monotonicCounter + 1) % 1_000_000;
  const cryptoObj = globalThis.crypto;
  if (cryptoObj && typeof cryptoObj.randomUUID === 'function') {
    return `${prefix}_${cryptoObj.randomUUID()}`;
  }
  const timePart = Date.now().toString(36);
  const randPart = Math.random().toString(36).slice(2, 10);
  const seqPart = monotonicCounter.toString(36);
  return `${prefix}_${timePart}_${seqPart}_${randPart}`;
}
