# Contributing to `@yours/harness`

Thank you for contributing to **Yours Research**! All contributions are licensed under the **Apache-2.0** license (`LICENSE`).

## Development Setup

Requirements:
- **Node.js `>= 20.0.0`**
- **npm `>= 10.0.0`**

```bash
npm install
npm run typecheck
npm run build
npm test
```

## Coding & Ecosystem Standards

1. **Zero Heavyweight Runtime Lock-In:** Keep `dependencies` in `package.json` empty or minimal. Runtime integrations (Transformers.js, WebLLM, Chrome AI, SQLite WASM) must work through clean structural interfaces or shims.
2. **Strict TypeScript:** All code must pass `tsc --noEmit` with `"strict": true` and `"moduleResolution": "Bundler"`.
3. **Contract Fidelity:** Any changes touching `src/contracts.ts` must stay aligned with `docs/ECOSYSTEM_CONTRACTS.md`.
4. **Privacy Invariant:** Never introduce unrequested external network calls or telemetry beacons.
5. **Testing:** Every new adapter, compaction heuristic, parser recovery rule, or storage operation should ship with deterministic unit tests under `tests/`.

## Submitting Changes

1. Create a focused branch off `main`.
2. Ensure `npm run typecheck && npm run build && npm test` passes with zero errors.
3. Open a pull request describing the motivation, architectural impact, and verification results.
