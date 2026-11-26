# Experimental Engui lane validation

Tested against upstream public `v2` commit `d376986fd2287e7b56cdca5914e1c4c7838f25b3` on Linux, 2026-11-26.

## Passed

- 76 backend integration/contract checks: durable remote attempts, uncertain-submit replay, cancellation, original-byte recovery, the engine catalog, capability caching and archive routes. The exact-export fixture contains 25 kept images and verifies 25 image/TXT pairs without an extra reference.
- 39 focused generation/backend checks, including the existing local and multi-engine paths.
- Full frontend suites: 5,192 core and 1,307 bundled tests passed; 10 bundled tests skipped. Final affected UI and tuning checks also passed.
- Ruff, ESLint and the production frontend build passed.
- Companion Engui bridge: 46 focused tests and production build passed; TypeScript diagnostics unchanged from its existing baseline.
- Companion Krea worker: 16 CPU tests passed. A separate live GPU check is required before an operator enables the bridge.

## Upstream baseline is not green on this Linux host

The full backend run exposed existing Windows-oriented trainer/plugin fixtures and other platform-dependent failures. The clean upstream comparison, with only the same disabled-plugin loader fix, recorded 11,363 passed, 120 failed, 30 errors and 148 skipped before its stalled final portion was interrupted. This is not a successful full-suite claim.

After correcting the two integration-specific contract issues, all 136 remaining failures/errors in the focused recheck also appeared in that upstream comparison. The corrected contracts are covered by the 76 passing checks above. The Linux disabled-plugin fix ensures an explicitly disabled Windows-only plugin is reported as disabled rather than blocking the installation.

## Scope of the evidence

These checks do not prove character identity quality, do not generate a real 25-image character dataset, and do not train a LoRA. Those require the user's reference, visual curation and captions. Browser-driven UI verification was not available in the deployment environment; frontend tests and HTTP checks do not substitute for that claim.

See [connection and workflow instructions](engui-integration.md). Keep this branch experimental until the operator has qualified its worker and the first real dataset.
