# Engui-backed Dataset Studio

This experimental fork preserves Dataset Studio's shot catalog, subject-specific Krea prompt wrappers, creative suffixes, selection and captions. It adds **Krea 2 Edit · Engui** as a separate remote engine. Local Klein and Krea ComfyUI engines are unchanged.

## Connect

1. In Settings → Image engines → Engui connection, enter the HTTPS server origin (without `/api/mcp`), workspace UUID and a workspace-scoped Engui token with `read`, `generate`, `cancel` permissions.
A same-host installation may use an HTTP loopback origin; other connections require HTTPS.

2. Save the connection, then Test. Enable Krea 2 Edit · Engui and select it in the dataset's Generate panel.
3. Upload/crop the reference, choose the appropriate subject type and the Studio shot cards, then generate candidates. Each candidate is an ordinary billed RunPod job. No local ComfyUI is required.
4. Keep or reject the returned originals. Retry uses the same card's prompt wrapper and current reference/tuning, with a new explicit attempt. Review identity and anatomy; automatic consistency is not a guarantee.
5. Keep exactly 25 images and give every kept image a caption. **Export 25** writes exactly 25 image/TXT pairs, excluding the extra reference. The existing Export ZIP behavior is unchanged and may include the reference as an extra image.

Captions remain Studio's feature: configure a supported captioning backend or edit captions manually. This integration does not install a captioning model or start LoRA training. The engine uses the current Krea identity tuning under Settings; extra local ComfyUI LoRA presets are not uploaded or automatically applied. Target canvases preserve the cards' aspect ratio at approximately 1 MP.

## Recovery and privacy

The server commits the candidate and its attempt before sending it. A restart or interrupted submit retries the same idempotency key; it never invents a replacement paid attempt. Studio dispatches one candidate at a time. Cancellation retains the provider handle until Engui confirms completion/cancellation. Changing the server/workspace while jobs are pending pauses those jobs until their original connection is restored.

The token is stored server-side as `LDS_ENGUI_TOKEN` (environment or Studio's secret store), never in the browser's configuration response. The bridge uses the existing Engui encrypted inference transport and verifies original download size/SHA-256. Reference snapshots and received originals remain in Studio's data directory.

## Engui / worker requirements

The authenticated POST `/api/dataset-studio` bridge must be enabled by the operator with `ENGUI_DATASET_STUDIO_ENABLED=1` and `ENGUI_MCP_ENABLED=1`, and Krea2 Turbo must be operator-approved/configured. It reuses the current upload, immutable-plan, job queue, cancellation and original-artifact storage. The matching worker image must include Krea2 Identity Edit v1.2 weights and pinned nodes at build time. No new GPU endpoint, worker quota, network model disk, or public MCP tool is required. The proxy must route this authenticated bridge path as well as existing MCP upload/artifact paths.

RunPod-native Batch Jobs are deliberately deferred. This feature neither requests them nor promises that separate ordinary jobs will reuse one warm worker. The fork retains the upstream noncommercial license. Commercial permission and later training integration are separate decisions.
