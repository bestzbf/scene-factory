# Token budget and visual quality controls

## Explicit part arrays and a mode-specific prompt

The opt-in `bigmodel-single-pass-patterns` profile selects
`anchored_patterns_v1`: complete scene generation in one request, original
image bytes, no review, no model repair and no retries. The model declares
prototype part indices, XYZ counts and group-local spacing. The local decoder
expands only these explicit translated copies, preserves topology and materials,
and checks expanded part/topology limits before allocation. Each final part has
a source/repeat receipt. Saved render data remains canonical V3.

This is an expressiveness and serialization option, not a measured model-token
saving. Its dedicated prompt removes the contradictory full-part-field paragraph
and repeated legacy guidance, while preserving coordinate, support, material,
camera and source rules. Existing modes retain their historic prompt. A schema
is still supplied exactly once; arrays add schema overhead and might make very
small scenes larger. No image downsampling or geometry guessing is enabled.

The 2026-10-27 anchored GLM office trial consumed 8,829 prompt plus 10,998
completion tokens (19,827 total) in one request. It also recorded 10,200 reasoning
characters, without a separate reasoning-token counter. The 17,271 output UTF-8
bytes and 36,250-byte pretty-printed local JSON are not billed-token counts.
Do not attribute the entire completion count to the final scene JSON or claim
that a shorter local file demonstrates a provider cost saving. Visual mismatch
remained substantial after successful local recovery and rendering.

Pattern-mode local round trips, mock API resumes and Blender geometry equality
do not establish real model success rate or photo fidelity. Run a bounded matched
provider comparison before selecting the experimental format for production;
unreviewed scenes remain excluded from the accepted dataset.

## Bound asset-geometry corrections (0.3.33)

When a complete native plan passes wire and provider-provenance validation but
fails only the local base-asset room-bounds check, a validation retry requests
numeric edits bound to that exact draft's SHA-256. Only implicated asset IDs
and their position, dimensions, yaw or repeat spacing may change. Full native
validation runs again before any detail stage; there is no automatic clamping,
relaxed geometry gate or scene acceptance. Mixed camera/group errors, malformed
JSON and other failures retain the complete-response retry path.

This uses the existing validation retry and API budget, with all original input
photos. Failed drafts and successful canonical expansions are hash-bound for
resume; unknown POST outcomes still block replay. The correction prompt includes
the complete candidate, so input may increase while repeated output shrinks.
Serialized bytes and hypothetical numeric-patch fixtures are not measured
provider tokens, billing savings or model repair quality.

## Retry backpressure (0.3.31)

Provider `Retry-After` values are minimum waits, not hints to truncate to 30
seconds. If a retry cannot fit the remaining request deadline, the call ends
without another POST; shared batch cooldowns retain the same minimum. Batch
stop interrupts backoff before another reservation or transport attempt.
Existing attempts and unknown usage remain accounted for, and uncertain POST
outcomes remain ineligible for automatic replay. These controls avoid early or
cancelled retries; they do not establish a measured billed-token saving.

## Offline usage by pipeline stage

`scene-factory usage OUTPUT` retains the existing per-model counters and adds
`stages` both globally and under each model. Exact factory output paths classify
`analysis.plan`, `analysis.detail`, `analysis.monolithic`, `review`,
`repair.visual` and `repair.technical`; other paths remain `unknown`. Monolithic
records require the adjacent scene `job.json`. This is output-path attribution,
not a model-inferred role. The report reads only `*.api.json` provenance, not
Codex CLI logs, and makes no API calls.

`transport_attempts` excludes attempts proven not to have started a POST,
including connection failures and cancelled reservations; these appear as
`proven_unsent_attempts`. Conflicting HTTP/usage/send evidence remains counted.
`attempts_without_reported_usage` exposes missing or wholly invalid token
counters. Partial token fields remain partial: having one valid field does not
establish complete billing. Legacy top-level usage is counted once only when no
per-transport usage is available; it is never multiplied across retries or
added to the final transport's duplicate usage.

`failed_call_tokens` contains reported tokens attached to failed logical API
calls. Separately, `local_validation.rejected_tokens` contains API-completed
staged responses subsequently rejected by a local validator, only when the
adjacent validation record matches the attempt number and prompt hash.
Validated stages additionally require the validation record's API artifact
hash. Missing, stale or malformed evidence stays `unknown_calls`; repair and
review currently lack equivalent individual hash-bound verdicts and also stay
unknown. An API failure is not counted again as a completed response rejected
locally. A visual review finding a bad scene is useful review work and is not
classified as invalid JSON or automatically avoidable waste.

Token dictionaries preserve provider field names. Never add `total_tokens` to
its input/output components, or treat response bytes as tokens. All
`elapsed_seconds` values sum logical request durations, including waits and
retries; they are not end-to-end batch wall time or measured throughput.
Unknown usage and provider billing adjustments prevent a complete invoice or
a proven savings estimate. Historical runs can identify expensive stages but
cannot establish matched quality/cost improvements on their own.

## Optional sparse staged-detail responses (0.3.20)

`pipeline.staged.detail_wire_format="native_sparse_parts_v1"` preserves the
same ten numeric values, part order, materials, topology, original photographs
and full scene validation. Primitive records omit unused arrays; curves,
polylines, meshes and curtains put their required topology in a sparse
`geometry` array keyed by zero-based part index. The decoder cannot invent
geometry: missing, duplicate, mismatched and out-of-range records fail local
validation. Exact wire replies and validated canonical expansions are saved
separately, hash-bound to resumable stage records. The default is `canonical`.

This is experimental transport compression, not an established provider-token
saving. The compact-mode prompt replaces its normal per-part-array paragraph;
it does not append a second conflicting instruction or remove spatial limits.
A local replay of five saved GLM office detail responses (132 parts) reduced
serialized **prompt + schema + response** from 77,757 to 74,782 UTF-8 bytes
(2,975 bytes, 3.83%). Every group improved. Original image bytes and pixel
dimensions are unchanged.
These are historical-response serialization measurements, not provider token
counts, generated visual comparisons or real API performance. The option stays
off by default until matched provider usage and visual fidelity are measured;
full planning, images and review costs are outside this measurement.

The explicit `bigmodel-production-cost-saving` project profile opts in alongside
conservative high-confidence image routing. Quality-first profiles remain on the
canonical wire format. The serialized-size reduction motivates this optional
profile only; it does not establish billed-token savings or visual equivalence.

## Provider RPM/TPM admission (0.3.19)

Each model configuration may optionally set:

```json
"rate_limit": {
  "requests_per_minute": 30,
  "tokens_per_minute": 200000,
  "input_tokens_per_request": 40000
}
```

These numbers are examples, not a provider quota or an image tokenizer
measurement. Zero disables that quota. TPM requires a positive input allowance
including prompt, schema and all original images; that allowance plus the
model's maximum output must fit the TPM quota. All active roles on the same
normalized provider/base URL share the strictest positive quotas, including
roles that omit their own RPM setting. Every role sharing TPM must supply its
input allowance. Model names and credentials do not split a shared endpoint.

The batch runner binds all scene/detail/review/repair threads to
`provider_limits.sqlite3` in the output directory. Each transport attempt,
including retries, reserves a slot before sending. Reliable complete usage
settles tokens; missing, partial, invalid or ambiguous usage retains the
conservative allowance. Only a proven unsent attempt refunds its RPM slot.
Queue time counts toward the request deadline and waits respond to batch stop.
The ledger retains at most 61 one-second buckets per endpoint, using an extra
second to keep the rolling-minute approximation conservative.

This shapes requests; it is not a billing cap or proof of provider compliance.
An underestimated image allowance, provider-specific rate windows, external
clients using the same account, and undocumented quotas can still trigger 429.
Use the persistent scene/batch budgets separately for aggregate spending
controls. By default the ledger lives in one output directory. To coordinate
separate local output shards, set `batch.shared_ledger_dir` to a shared parent
and keep every shard output inside it; processes using that same local ledger
then share its state. Separate machines still require external coordination.
A changed policy is rejected while its previous rolling window is active.
Defaults remain disabled and existing configurations are not rewritten. Direct
Python callers that enable quotas must use `provider_limit_session`; otherwise
no request is sent.

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

Version 0.3.9 encodes repeated native-plan records as columns and rows in detail
requests. All physical plan values, construction guidance, cameras, repeats,
material bindings and opening constraints remain available. The target group
remains a complete named record; only its non-target groups' evidence routing
metadata may be omitted as described below. Small plans use the original
format whenever table headers and decoding instructions would increase prompt
length. By default every detail request still receives all original photos in
order. An opt-in `pipeline.staged.route_detail_images=true` routes detail calls
to the planner's explicit group evidence views only when the group is observed
and its `evidence_confidence` meets `detail_view_min_confidence` (default 0.75;
the cost-saving profile sets 0.90). Planning
always receives every original photo; uncertain/unobserved groups fall back to
all photos, and every source view used by a declared photo texture is retained.
No source image is downscaled. This is intended for multi-view scenes and may
save repeated image input only when groups have distinct evidence views. The
global option defaults to false. The explicit `bigmodel-production-cost-saving`
project profile opts in while the quality-first profiles retain all images for
each detail call. Image routing is a cost/quality tradeoff: evidence maps and
confidence are model-authored and the 0.75 threshold is not calibrated. Check
per-group `input_view_ids` provenance and complete a real-provider visual
regression before using routed outputs as a formal dataset. Image counts or
bytes are not token or billing measurements. Single-view jobs and scenes where
every group needs every view get no image-attachment reduction.
When the encoded table plus its decoder note is shorter than the original plan,
the detail projection also omits only `evidence_view_ids` and
`evidence_confidence` from non-target groups. The target group's evidence and
all other physical, support, material and construction fields remain present;
small plans automatically use the complete plan. This is a text-size heuristic,
not a provider token or billing measurement and has no standalone visual
equivalence guarantee.
Imported historic plans without evidence fields retain every source photo;
validation does not add inferred evidence to the stored plan. Fresh API plan
responses must still satisfy the complete strict evidence-field schema.
Validators and scene assembly use the complete native plan, not this wire view.
The new code/encoding requires a fresh output directory; it never silently
reuses old staged results. Compact data encoding does not guarantee identical
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

- `max_requests` counts transport attempts and HTTP retries, except attempts
  durably proven not to have started the POST. The next attempt is rejected
  before transport when the limit is used.
- `max_output_tokens` is the cumulative output capacity. Before each attempt the
  full configured model output cap is reserved atomically, including concurrent
  detail calls. Complete, internally consistent provider input/output usage
  releases unused capacity. Missing, malformed, partial or interrupted usage
  keeps the reservation; these reservations are not asserted to be charges.
  A proven-unsent attempt refunds its request and output reservation.
  A smaller remaining balance never silently reduces the model output cap.
- `max_reported_tokens` stops new attempts after known reported input plus output
  reaches the threshold. It cannot prevent a current or already in-flight request
  from crossing it. Unknown input/image usage cannot be bounded by this setting.
  Anthropic cache creation/read input is included; totals are not double-counted
  with their components. This is **not a hard currency or total billed-token cap**.

`scene_budget.json` is durably written before sending, under a POSIX process lock,
and reused after restart. Version 2 retains allow-listed transport evidence for
each attempt, including refunded attempts. Version 1 ledgers without pending
requests migrate automatically, keeping all historical usage conservative;
their records have no send proof and never gain a retrospective refund.
Schema/JSON failures still consume reported usage.
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


## Typed parts and batch-wide limits (0.3.16)

Set `pipeline.scene_wire_format` to `typed_parts_v1` with
`pipeline.analysis_mode=monolithic` to request one complete scene using strict,
type-specific part records. The default remains `canonical` for provider
compatibility. The model still returns all meaningful geometry, materials,
transforms and cameras. Empty fields used only by other primitive types are
restored locally before normal scene validation and Blender. Mesh vertices and
faces, curtain parameters and ellipsoid dimensions are never invented. The raw
response and format/schema/implementation provenance remain on disk. Providers
using strict output schemas must support nested `anyOf` object branches; no
silent fallback or extra model call is introduced.

On three saved office examples, serialized response bytes fell by 17–21%.
This is NOT provider-token savings: the typed schema is larger and a mesh-heavy
scene may save almost nothing. Source photos stay at their original resolution.
Measure input + output tokens, latency and rendered fidelity on representative
photos before choosing a production default.

Optional top-level configuration adds durable limits shared by every HTTP
attempt, including retries, analysis, review and repair across all scene workers:

```json
"batch_budget": {
  "max_requests": 500,
  "max_output_tokens": 2000000,
  "max_reported_tokens": 5000000,
  "failure_threshold": 5,
  "cooldown_seconds": 10
}
```

These are example pilot limits, not a recommendation for 10,000 scenes. Zero
disables each token/request limit. Each configured API batch uses the SQLite
ledger even when limits are zero; default provider failure protection still
applies. Output capacity is reserved before transport, incomplete/unknown usage
keeps its reservation, and reported-token thresholds cannot cap unknown input
or image costs. Already running requests may finish after a threshold is met.
A local cancellation or connection failure proven not to have started the POST
refunds its batch request/output reservation. Connection failures still count
toward provider cooldown and the shared failure circuit. Missing or conflicting
send evidence keeps the full reservation.

The batch CLI refuses live runs of 1,000 or more scenes unless all three limits
are positive. `--dry-run` remains available without them. This prevents an
accidentally unbounded large run, but the operator must choose caps from the
provider quota and spending ceiling. With the default `shared_ledger_dir=null`,
each output directory has its own ledger. Setting `batch.shared_ledger_dir`
makes the budget and provider-limit ledgers global to all output shards inside
that local root; a shard outside the root is rejected. This does not coordinate
machines or filesystems without reliable SQLite locking. For large inputs,
dry-run reports total scenes/images, the first 20 scenes and which limits a live
run still needs, instead of printing every path.

The optional per-scene `budget` also refunds proven-unsent attempts in its v2
ledger. Migrated v1 entries keep their original conservative accounting because
the older format did not retain transport proof. Refunds never reset provider
failure counts or authorize replay of a request that may already have executed.

HTTP 401/402/403 opens the shared circuit; repeated network/429/5xx failures
have a shared cooldown and eventually stop scheduling new scenes. An explicit
`scene-factory run ... --resume-batch` reopens a service circuit after the
underlying issue is resolved, preserving all counters. It cannot reset a
budget or authorize automatic replay of a request with an unknown outcome.
A process crash with pending reservations blocks automatic resume; inspect
provider records rather than deleting the ledger. CLI-based model calls do
not use the HTTP adapter and are not covered by these API quotas.

Use a fresh output directory when upgrading from a version without a batch
ledger. Existing billed work must not silently acquire zero counters. `status`
and `usage` include the batch controls. A stopped or budget-limited batch never
turns failed/unreviewed outputs into accepted dataset records.

## Structured-output prompt and local verification savings (0.3.23)

When `response_format` is `json_schema`, the provider receives the strict response schema through its native structured-output parameter. The request no longer repeats the generic JSON-only output direction in the text prompt, removing 107 UTF-8 bytes per structured request. This is a byte-count change, not a fixed token or billing estimate. Prompt and `json_object` modes continue to include their explicit schema instructions. Provider-reported `usage` is the only available basis for measuring actual token changes.


## Single-response recovery and typed-schema audit (2026-10-26)

Monolithic analysis now keeps completed transport responses separate from the
local semantic commit. A resume must match the exact prompt, schema, public
model configuration, ordered source-image hashes, completion evidence and raw
response digest before reusing that response. A repeated local validation
failure does not start another model request. Missing or inconsistent evidence
blocks replay. A provably unsent or explicitly rejected attempt can use a new
numbered stage audit without replacing the original usage record. These rules
do not refund billed requests or repair invalid scene data. Failed-attempt counters must match their transport records; empty or contradictory records are not proof of an unsent request. An explicitly selected cache with existing scene evidence but a missing raw response is rejected rather than treated as a cache miss.

Shared pending reservations identify their original invocation with a unique
owner generation and kernel lock. Reacquiring the same shard directory lock
after a crash cannot make an orphan request look live. Legacy pending records
without this evidence remain blocked; request and token counters are retained.

Typed-part schemas merge only branches whose complete constraints are identical
after removing the type enum. This reduces the branch count from 8 to 6 and
saves 1,140 serialized schema bytes on the measured fixture. It does not change
the accepted geometry types or switch the default away from canonical. On the
latest small office response, the old compact encoding saved only 371 output
bytes while its schema and instructions added more input text. Therefore this
optimization is not a demonstrated token or cost reduction; use provider usage
from matched real tests before changing defaults. Source photos retain their
original resolution.

## Explicit placement in a single request (2026-10-27)

The opt-in `bigmodel-single-pass-anchored` profile uses
`anchored_parts_v1`: typed custom parts plus explicit floor, on-top or world
placement for base objects. This removes the need to encode a guessed bottom Z
for declared floor/contact placements; the decoder resolves it deterministically
and records the calculation. It is a placement contract, not a measured token
optimization. Its extra schema/instructions may increase input tokens. This
initial version supports built-in assets only; external asset catalogs are
rejected until their actual contact surfaces can be validated.

The profile allows one analysis request with a 12,288 output-token cap, no
visual review/repair and no transport or validation retries. Original image
files are uploaded without resizing. Failed local checks reuse matching saved
complete responses on resume rather than sending the same paid request again;
invalid scene data still fails. Loopback HTTP and mocked batch tests verify
this accounting contract, not actual provider charges or model fidelity.

The preceding real GLM office test used the older canonical placement contract:
one successful request reported 7,372 input + 3,456 output = 10,828 tokens,
69.415 seconds for analysis and 95.75 seconds end to end. The adjacent failed
material test reported another 12,843 tokens; together those two requests used
23,671 tokens. Neither is a measurement of the new anchored profile. The
successful output remains generated_unreviewed and visibly differs from the
photo. No new paid API call was made to validate anchored placement.
