# Token budget and visual quality controls

The factory uses separate model calls for scene analysis, detail generation,
render review, and typed repair. It does not treat a photo-to-JSON call as the
whole job: a review must inspect the rendered image before a scene can enter the
dataset, and a repair must be rendered and reviewed again.

The current runtime keeps complete evidence and hashes on disk for local
validation, but sends compact evidence to the model. Review metadata, material
slot inventories, lighting-fit history, and per-part projections are deduplicated
before transmission. The local validator still receives the complete inventory,
so selector and geometry checks are unchanged.

`pipeline.repair_stagnation_patience` stops before another repair after the
configured number of independently reviewed candidates fail to improve the
selected best candidate. The default is `2`; `0` disables this stop. A stop is
`needs_review`, never acceptance.

Each API record now stores request character/byte counts, image count, schema
bytes, output limit, and provider-reported usage when available. These are
diagnostics, not an invoice or an exact tokenizer estimate. Set
`max_prompt_characters` in a model config to fail locally before credentials,
image encoding, or network access when a prompt exceeds a known provider limit.

The `deepseek-production` template uses separate lower-cost review settings,
bounded output, one validation retry, and the DeepSeek environment variable
`DEEPSEEK_API_KEY`. Configure a funded account before running it; the template
does not contain a credential.

Version 0.3.8 also preserves the native planner's explicit `scene_type` through
asset dispatch. Before this fix, staged analysis forced every scene to `other`,
so an office could receive ordinary dining chairs. Classification is now a
required enum field in new model responses. Existing V3 bundles remain readable;
old staged caches are not silently reinterpreted with a guessed class. `shop`
is preserved at V2 and maps to `other` only in the legacy V1 record.

Built-in detailed monitor screen materials expose stable source ID
`monitor_screen`. The existing material selector can target that ID for one
instance without recoloring its casing/stand or another monitor. Model repairs
must still consult the actual supplied inventory; old renders and imported
assets may have different selectors. This is a binding capability, not a claim
that the reference's screen content has already been reconstructed.

Version 0.3.9 losslessly encodes repeated native-plan records as columns and
rows in detail requests. Every value, all group construction guidance, all
cameras, repeats, material bindings and opening constraints remain available.
The target group remains a complete named record. Small plans use the original
format whenever table headers and decoding instructions would increase prompt
length. Every detail request still receives all original photos in order.
Validators and scene assembly use the complete native plan, not this wire view.
The new code/encoding requires a fresh output directory; it never silently
reuses old staged results. Lossless data encoding does not guarantee identical
model interpretation or visual fidelity; those require a real model regression.

Repair strategy 2 falls back to all custom geometry when any reported issue is
global (framing, lighting or realism), or references an unknown object. Local
well-resolved issues can still use focused geometry. This deliberately trades
some token savings for the context needed to resolve cross-object problems.
Prior partial-context measurements therefore do not apply to every repair.
No savings calculation includes a presumed decrease in image or reasoning
tokens; only the provider can report billed usage.


Version 0.3.10 prevents transparent duplicate generation after an uncertain
POST outcome. Connection refusal before any POST starts can still retry within
the configured budget, and explicit HTTP 429/5xx responses retain their bounded
retry policy. Once a request may have been sent, an interrupted response or
post-send timeout stops automatic replay. Sanitized API records preserve the
transport phase, sent-state flags, and any provider-reported usage. A missing
stream completion marker is never accepted as a complete scene.

The same protection applies when restarting the batch: analysis, detail, review,
visual repair and technical repair check their own prior ambiguous API records
before sending again. Completed committed stages are reused normally. Explicit
rejections such as HTTP 402 and definite pre-send failures remain resumable.
An ambiguous request raises `ambiguous_previous_request`; check the provider's
request history before deliberately starting a new output directory. This can
prevent a second generation, but cannot undo tokens already spent by the first.
Do not delete audit records to bypass the protection. New runtime fingerprints
require fresh output directories for older reconstruction runs.

## Persistent per-scene API limits (0.3.11)

An optional top-level `budget` applies jointly to analysis, concurrent detail
generation, review, visual repair, technical repair and their HTTP retries:

```json
"budget": {
  "max_requests": 24,
  "max_output_tokens": 393216,
  "max_reported_tokens": 600000
}
```

These are the initial DeepSeek production-template limits, not measured optimal
quality settings. Each value is a nonnegative integer; zero disables that limit.
Omitting `budget` preserves the previous behavior. Existing project configs are
not rewritten. The limits apply only to configured model APIs, not legacy Codex
CLI calls or the standalone `probe` command.

- `max_requests` counts transport attempts, including pre-send failures and HTTP
  retries. The next attempt is rejected before transport when the limit is used.
- `max_output_tokens` is the cumulative output capacity. Before each attempt the
  full configured model output cap is reserved atomically, including concurrent
  detail calls. Complete, internally consistent provider input/output usage
  releases unused capacity. Missing, malformed, partial or interrupted usage
  keeps the reservation; these reservations are not asserted to be charges.
  A smaller remaining balance never silently reduces the model output cap.
- `max_reported_tokens` stops new attempts after known reported input plus output
  reaches the threshold. It cannot prevent a current or already in-flight request
  from crossing it. Unknown input/image usage cannot be bounded by this setting.
  Anthropic cache creation/read input is included; totals are not double-counted
  with their components. This is **not a hard currency or total billed-token cap**.

`scene_budget.json` is durably written before sending, under a POSIX process lock,
and reused after restart. Schema/JSON failures still consume reported usage.
Pending reservations after a crash block a new scene session with
`ambiguous_previous_request`: inspect the provider records before deliberately
retrying in a fresh output directory. Completed committed results remain cached.
Limits, scene identity and runtime fingerprints cannot be changed in place to
reset spending. Missing/corrupt ledgers fail closed. POSIX local-file locking is
required; budget-enabled runs fail clearly on unsupported platforms.

`scene-factory usage OUTPUT` and `scene-factory status OUTPUT` expose policy and
summary counters, including unknown-usage and pending attempts. They are offline.
When exhausted, a scene with a completed render stops as `needs_review` with
`stop_reason=scene_budget_exceeded`; analysis without a complete render fails.
No stopped scene gains acceptance or enters the formal dataset. Already reviewed
best candidates and their renders are preserved. Real token savings and photo
fidelity still require a funded provider regression; local tests prove protocol
and budget enforcement only.

## Lossless review and repair context encoding (0.3.12; retained in 0.3.14)

Review and repair prompts can encode repeated homogeneous JSON record arrays as
column/value tables. The encoder preserves row order, every field and nested
value, and keeps the original scene bundle for schema validation, deterministic
edits and Blender generation. The wire format is used only when its aggregate
UTF-8 savings exceed the decoding-note size. Schema-constrained outputs such as
`revised_bundle` remain ordinary JSON arrays and objects. This reduces repeated
text context only; it does not reduce source/render image tokens, output or
reasoning tokens, and does not establish an equal reduction in provider billing
or demonstrate unchanged model interpretation. A real provider regression is
still required to measure usage and compare rendered fidelity.

## Preflight context and repair dependencies (0.3.14)

Pre-render technical repair now uses the same lossless record-table encoding
for the complete scene bundle. Every group and part remains available; measured
geometry/daylight reports remain complete native JSON. View identities, attached
photo order and the typed edit output schema are unchanged. Small inputs retain
native records if savings do not cover the decoder note. Provenance includes
encoding statistics and the context implementation hash; changing that code
invalidates committed repair-cache fingerprints.

Focused visual repair follows support dependencies through base objects and
their explicit repeat families in both directions. A referenced material also
includes users bound via base-material overrides and their support chains.
This prevents omitted geometry from hiding related furniture during a local
correction. Global lighting, framing and realism issues, or unresolved object
references, still retain all geometry. Independent final visual acceptance
continues to be required for dataset export.

Source photos retain their pixel resolution and every supplied view is kept.
No default 1024-pixel upload resize is introduced. EXIF orientation normalization
and texture-crop generation are separate from source-photo downscaling. Render
resolution remains independently configurable. Local round-trip and Blender
tests verify data preservation and the repair protocol; character/UTF-8 savings
are not measured provider-token, cost or photo-fidelity improvements.

## Matched comparison-image inputs (0.3.15)

Review, visual repair and re-render review keep each source photo byte-for-byte
and at its original pixel resolution. If its paired Blender render is larger in
both dimensions and has a matching aspect ratio, only the model-input copy of
that render is downsampled with Lanczos to the source-photo dimensions. The
authoritative render, saved scene, render metadata and acceptance measurements
remain unchanged; no render is enlarged, cropped or distorted. If those size
conditions do not hold, the original render is sent as-is.

The request provenance records original/render/model-input sizes and hashes,
the transform and implementation hash; cache reuse and repair commit validate
those inputs. This can lower image input where a render would otherwise be
uploaded above its paired photo size, but it does not resize source photos or
guarantee a token or billing reduction. Confirm the actual change from provider
`usage` fields on a future authorized API run.
